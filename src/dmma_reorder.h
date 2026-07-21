#ifndef RTT_SPGEMM_DMMA_REORDER_H_
#define RTT_SPGEMM_DMMA_REORDER_H_

#include "dmma_tiles.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system_error.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <vector>

/*
 * Balanced GPU Reorder and Fusion (BGRF-v1).
 *
 * Every non-baseline input follows exactly the same A-only path.  There are
 * no external-algorithm candidates, size/symmetry modes, or matrix-level
 * identity fallbacks.  A single GPU bipartite CM traversal supplies global
 * locality; its joint row/K proposal is evaluated by the same exact A-tile
 * commit rule for every matrix.  One fixed row/K fine pass then packs 8x4
 * Tensor-Core groups and commits only proposals that do not increase A tiles
 * and improve the tile-span/fanout objective.
 */
static constexpr int DMMA_REORDER_SWEEPS = 1;
static constexpr int DMMA_REORDER_ROW_WINDOW = 32;
static constexpr int DMMA_REORDER_INNER_WINDOW = 16;
static constexpr int DMMA_REORDER_DEGREE_BUCKETS = 8;
static constexpr int DMMA_REORDER_FINGERPRINT_WORDS = 2;
static constexpr int DMMA_REORDER_MAX_COARSE_LEVELS = 2048;
static_assert(DMMA_REORDER_ROW_WINDOW % DMMA_TILE_M == 0,
              "row fine window must contain complete A row tiles");
static_assert(DMMA_REORDER_INNER_WINDOW % DMMA_TILE_K == 0,
              "inner fine window must contain complete A K tiles");

/* Public, stable ablation variants.  The default-constructed configuration
 * is deliberately the production BGRF-v1 path used before these controls
 * were introduced. */
enum DmmaReorderVariant
{
    DMMA_REORDER_VARIANT_FULL,
    DMMA_REORDER_VARIANT_COARSE,
    DMMA_REORDER_VARIANT_FINE,
    DMMA_REORDER_VARIANT_COARSE_ROW,
    DMMA_REORDER_VARIANT_COARSE_INNER,
    DMMA_REORDER_VARIANT_UNGUARDED
};

struct DmmaReorderConfig
{
    DmmaReorderVariant variant = DMMA_REORDER_VARIANT_FULL;
};

static inline const char *dmma_reorder_variant_name(
    DmmaReorderVariant variant)
{
    switch (variant)
    {
    case DMMA_REORDER_VARIANT_FULL:
        return "full";
    case DMMA_REORDER_VARIANT_COARSE:
        return "coarse";
    case DMMA_REORDER_VARIANT_FINE:
        return "fine";
    case DMMA_REORDER_VARIANT_COARSE_ROW:
        return "coarse-row";
    case DMMA_REORDER_VARIANT_COARSE_INNER:
        return "coarse-inner";
    case DMMA_REORDER_VARIANT_UNGUARDED:
        return "unguarded";
    }
    return nullptr;
}

static inline bool parse_dmma_reorder_variant(
    const char *text, DmmaReorderVariant *variant)
{
    if (text == nullptr || variant == nullptr)
        return false;
    const DmmaReorderVariant candidates[] = {
        DMMA_REORDER_VARIANT_FULL,
        DMMA_REORDER_VARIANT_COARSE,
        DMMA_REORDER_VARIANT_FINE,
        DMMA_REORDER_VARIANT_COARSE_ROW,
        DMMA_REORDER_VARIANT_COARSE_INNER,
        DMMA_REORDER_VARIANT_UNGUARDED};
    for (DmmaReorderVariant candidate : candidates)
    {
        const char *name = dmma_reorder_variant_name(candidate);
        if (name != nullptr && std::strcmp(text, name) == 0)
        {
            *variant = candidate;
            return true;
        }
    }
    return false;
}

static inline bool resolve_dmma_reorder_variant(
    DmmaReorderVariant variant, bool *enable_coarse,
    bool *enable_row_fine, bool *enable_inner_fine,
    bool *enable_exact_guard)
{
    if (enable_coarse == nullptr || enable_row_fine == nullptr ||
        enable_inner_fine == nullptr || enable_exact_guard == nullptr)
        return false;
    *enable_coarse = variant != DMMA_REORDER_VARIANT_FINE;
    *enable_row_fine = variant != DMMA_REORDER_VARIANT_COARSE &&
                       variant != DMMA_REORDER_VARIANT_COARSE_INNER;
    *enable_inner_fine = variant != DMMA_REORDER_VARIANT_COARSE &&
                         variant != DMMA_REORDER_VARIANT_COARSE_ROW;
    *enable_exact_guard = variant != DMMA_REORDER_VARIANT_UNGUARDED;
    return dmma_reorder_variant_name(variant) != nullptr;
}

struct DmmaAxisProfile
{
    int id = 0;
    int degree = 0;
    int soft_bucket = -1;
    int panel_occupancy = 0;
    int panel_min = -1;
    int panel_max = -1;
    int panel_span = 0;
    uint64_t fingerprint[DMMA_REORDER_FINGERPRINT_WORDS] = {0, 0};
    uint64_t minhash[DMMA_REORDER_FINGERPRINT_WORDS] = {
        ~uint64_t(0), ~uint64_t(0)};
};

struct DmmaTileBlockProfile
{
    int id = 0;
    long long degree = 0;
    int nonempty = 0;
    int panel_occupancy = 0;
    int panel_min = -1;
    int panel_max = -1;
    int panel_span = 0;
    uint64_t fingerprint[DMMA_REORDER_FINGERPRINT_WORDS] = {0, 0};
};

enum DmmaReorderKind
{
    DMMA_REORDER_IDENTITY,
    DMMA_REORDER_UNIFIED,
    DMMA_REORDER_EXTERNAL
};

struct DmmaReorderPlan
{
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    int active_rows = 0;
    int active_inner = 0;
    bool unified = false;
    DmmaReorderKind kind = DMMA_REORDER_IDENTITY;
    DmmaReorderVariant variant = DMMA_REORDER_VARIANT_FULL;
    char algorithm[80] = "identity-baseline";
    int sweeps = 0;
    int row_window = DMMA_REORDER_ROW_WINDOW;
    int inner_window = DMMA_REORDER_INNER_WINDOW;

    int *h_row_old_to_new = nullptr;
    int *h_row_new_to_old = nullptr;
    int *h_inner_old_to_new = nullptr;
    int *h_inner_new_to_old = nullptr;
    int *d_row_old_to_new = nullptr;
    int *d_row_new_to_old = nullptr;
    int *d_inner_old_to_new = nullptr;
    int *d_inner_new_to_old = nullptr;

    unsigned long long moved_rows = 0;
    unsigned long long moved_inner = 0;
    unsigned long long row_displacement = 0;
    unsigned long long inner_displacement = 0;
    unsigned long long accepted_row_windows = 0;
    unsigned long long accepted_inner_windows = 0;
    unsigned long long row_tile_reduction = 0;
    unsigned long long inner_tile_reduction = 0;
    unsigned long long row_fanout_before = 0;
    unsigned long long row_fanout_after = 0;
    unsigned long long inner_fanout_before = 0;
    unsigned long long inner_fanout_after = 0;
    int coarse_components = 0;
    int coarse_levels = 0;
    int coarse_level_budget = DMMA_REORDER_MAX_COARSE_LEVELS;
    bool coarse_candidate_accepted = false;
    unsigned long long coarse_tile_reduction = 0;
    double coarse_ms = 0.0;
    double fine_ms = 0.0;
    std::size_t reorder_peak_workspace_bytes = 0;

    long long num_tiles = 0;
    int active_row_tiles = 0;
    int active_k_tiles = 0;
    long long sparse_tiles = 0;
    long long dense_tiles = 0;
    unsigned long long payload = 0;
};

static inline void destroy_dmma_reorder_plan(DmmaReorderPlan *plan)
{
    if (plan == nullptr)
        return;
    std::free(plan->h_row_old_to_new);
    std::free(plan->h_row_new_to_old);
    std::free(plan->h_inner_old_to_new);
    std::free(plan->h_inner_new_to_old);
    cudaFree(plan->d_row_old_to_new);
    cudaFree(plan->d_row_new_to_old);
    cudaFree(plan->d_inner_old_to_new);
    cudaFree(plan->d_inner_new_to_old);
    *plan = DmmaReorderPlan();
}

/* Once A tiles and the initial B mapping are built, values-only iterations
 * use the host row permutation for C restoration but no device permutation.
 * Dropping these four O(rows+cols) arrays is important for very large graphs. */
static inline void release_dmma_reorder_device_maps(DmmaReorderPlan *plan)
{
    if (plan == nullptr)
        return;
    cudaFree(plan->d_row_old_to_new);
    cudaFree(plan->d_row_new_to_old);
    cudaFree(plan->d_inner_old_to_new);
    cudaFree(plan->d_inner_new_to_old);
    plan->d_row_old_to_new = nullptr;
    plan->d_row_new_to_old = nullptr;
    plan->d_inner_old_to_new = nullptr;
    plan->d_inner_new_to_old = nullptr;
}

namespace dmma_reorder_detail
{

static constexpr int kThreads = 256;

static inline bool cuda_ok(cudaError_t status, const char *label)
{
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "CUDA error in unified DMMA reorder %s: %s\n",
                 label, cudaGetErrorString(status));
    return false;
}

static inline int blocks_for(int count)
{
    return count <= 0 ? 0 : count / kThreads + (count % kThreads != 0);
}

static inline int ceil_div_nonnegative(int count, int divisor)
{
    return count / divisor + (count % divisor != 0);
}

template <typename T>
static inline bool allocate_device(T **pointer, std::size_t count,
                                   const char *label)
{
    *pointer = nullptr;
    if (count == 0)
        return true;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        return false;
    return cuda_ok(cudaMalloc(reinterpret_cast<void **>(pointer),
                              count * sizeof(T)),
                   label);
}

template <typename T>
static inline T *allocate_host(std::size_t count)
{
    if (count == 0)
        return nullptr;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        return nullptr;
    return static_cast<T *>(std::malloc(count * sizeof(T)));
}

__host__ __device__ static inline uint64_t mix64(uint64_t value)
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

__device__ static inline int soft_bucket_device(int degree)
{
    if (degree <= 0)
        return -1;
    int logarithm = 0;
    for (unsigned int value = static_cast<unsigned int>(degree);
         value > 1; value >>= 1)
        ++logarithm;
    return logarithm / 2;
}

__device__ static inline void fingerprint_panel(
    int panel, uint64_t bits[DMMA_REORDER_FINGERPRINT_WORDS])
{
#pragma unroll
    for (int word = 0; word < DMMA_REORDER_FINGERPRINT_WORDS; ++word)
    {
        const uint64_t hash = mix64(
            static_cast<uint64_t>(static_cast<unsigned int>(panel)) +
            static_cast<uint64_t>(word) * 0xd1b54a32d192ed03ull);
        bits[word] |= uint64_t(1) << (hash & 63);
    }
}

__device__ static inline void minhash_panel(
    int panel, uint64_t hashes[DMMA_REORDER_FINGERPRINT_WORDS])
{
#pragma unroll
    for (int word = 0; word < DMMA_REORDER_FINGERPRINT_WORDS; ++word)
    {
        const uint64_t hash = mix64(
            static_cast<uint64_t>(static_cast<unsigned int>(panel)) ^
            (0x243f6a8885a308d3ull +
             static_cast<uint64_t>(word) * 0x9e3779b97f4a7c15ull));
        hashes[word] = min(hashes[word], hash);
    }
}

__device__ static inline int fingerprint_occupancy(
    const uint64_t bits[DMMA_REORDER_FINGERPRINT_WORDS])
{
    int occupancy = 0;
#pragma unroll
    for (int word = 0; word < DMMA_REORDER_FINGERPRINT_WORDS; ++word)
        occupancy += __popcll(bits[word]);
    return occupancy;
}

__host__ __device__ static inline int degree_bucket(int degree)
{
    if (degree <= 1)
        return 0;
    int logarithm = 0;
    for (unsigned int value = static_cast<unsigned int>(degree);
         value > 1; value >>= 1)
        ++logarithm;
    return logarithm < DMMA_REORDER_DEGREE_BUCKETS
               ? logarithm
               : DMMA_REORDER_DEGREE_BUCKETS - 1;
}

__global__ void build_bipartite_degrees_kernel(
    int rows, const int *row_ptr, int *row_degree)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row < rows)
        row_degree[row] = row_ptr[row + 1] - row_ptr[row];
}

__global__ void count_bipartite_columns_kernel(
    int rows, const int *row_ptr, const int *col_idx, int *col_degree)
{
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows)
        return;
    for (int entry = row_ptr[row] + static_cast<int>(threadIdx.x);
         entry < row_ptr[row + 1]; entry += static_cast<int>(blockDim.x))
        atomicAdd(col_degree + col_idx[entry], 1);
}

__global__ void combine_graph_degrees_kernel(
    int rows, int cols, const int *row_degree, const int *col_degree,
    int *graph_degree)
{
    const int node = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (node < rows)
        graph_degree[node] = row_degree[node];
    else if (node < rows + cols)
        graph_degree[node] = col_degree[node - rows];
}

__global__ void generate_bucketed_csr_keys_kernel(
    int rows, int cols, const int *row_ptr, const int *col_idx,
    const int *col_degree, uint64_t *keys, int *neighbors)
{
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows)
        return;
    const uint64_t stride = static_cast<uint64_t>(cols) + 1ull;
    for (int entry = row_ptr[row] + static_cast<int>(threadIdx.x);
         entry < row_ptr[row + 1]; entry += static_cast<int>(blockDim.x))
    {
        const int col = col_idx[entry];
        const uint64_t source_bucket =
            static_cast<uint64_t>(row) * DMMA_REORDER_DEGREE_BUCKETS +
            static_cast<uint64_t>(degree_bucket(col_degree[col]));
        keys[entry] = source_bucket * stride +
                      static_cast<uint64_t>(col);
        neighbors[entry] = col;
    }
}

__global__ void generate_bucketed_csc_keys_kernel(
    int rows, int cols, const int *row_ptr, const int *col_idx,
    const int *row_degree, uint64_t *keys, int *neighbors)
{
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows)
        return;
    const uint64_t stride = static_cast<uint64_t>(rows) + 1ull;
    const int bucket = degree_bucket(row_degree[row]);
    for (int entry = row_ptr[row] + static_cast<int>(threadIdx.x);
         entry < row_ptr[row + 1]; entry += static_cast<int>(blockDim.x))
    {
        const int col = col_idx[entry];
        const uint64_t source_bucket =
            static_cast<uint64_t>(col) * DMMA_REORDER_DEGREE_BUCKETS +
            static_cast<uint64_t>(bucket);
        keys[entry] = source_bucket * stride +
                      static_cast<uint64_t>(row);
        neighbors[entry] = row;
    }
}

struct GraphSeedLess
{
    const int *degree;

    __host__ __device__ bool operator()(int lhs, int rhs) const
    {
        const int lhs_degree = degree[lhs];
        const int rhs_degree = degree[rhs];
        const bool lhs_active = lhs_degree != 0;
        const bool rhs_active = rhs_degree != 0;
        if (lhs_active != rhs_active)
            return lhs_active;
        if (lhs_degree != rhs_degree)
            return lhs_degree < rhs_degree;
        return lhs < rhs;
    }
};

struct IsActiveDegree
{
    const int *degree;
    __host__ __device__ bool operator()(int node) const
    {
        return degree[node] != 0;
    }
};

struct IsUnvisitedActive
{
    const int *degree;
    const unsigned char *visited;
    __host__ __device__ bool operator()(int node) const
    {
        return degree[node] != 0 && visited[node] == 0;
    }
};

struct IsRowNode
{
    int rows;
    __host__ __device__ bool operator()(int node) const
    {
        return node < rows;
    }
};

struct IsInnerNode
{
    int rows;
    __host__ __device__ bool operator()(int node) const
    {
        return node >= rows;
    }
};

struct IsZeroDegree
{
    const int *degree;
    __host__ __device__ bool operator()(int id) const
    {
        return degree[id] == 0;
    }
};

struct IsNonZeroValue
{
    __host__ __device__ bool operator()(int value) const
    {
        return value != 0;
    }
};

struct SubtractRows
{
    int rows;
    __host__ __device__ int operator()(int node) const
    {
        return node - rows;
    }
};

__device__ static inline bool first_unique_neighbor(
    int entry, int begin, const int *neighbors)
{
    return entry == begin || neighbors[entry - 1] != neighbors[entry];
}

__global__ void start_bipartite_component_kernel(
    int seed, int *frontier, int *order, int order_offset,
    unsigned char *visited)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        frontier[0] = seed;
        order[order_offset] = seed;
        visited[seed] = 1;
    }
}

__global__ void claim_bipartite_frontier_kernel(
    int frontier_size, int rows, const int *row_ptr,
    const int *csr_neighbors, const int *col_ptr,
    const int *csc_neighbors, const int *frontier,
    const unsigned char *visited, int *owner)
{
    const int parent_position = static_cast<int>(blockIdx.x);
    if (parent_position >= frontier_size)
        return;
    const int node = frontier[parent_position];
    const bool row_side = node < rows;
    const int local = row_side ? node : node - rows;
    const int begin = row_side ? row_ptr[local] : col_ptr[local];
    const int end = row_side ? row_ptr[local + 1] : col_ptr[local + 1];
    const int *neighbors = row_side ? csr_neighbors : csc_neighbors;
    for (int entry = begin + static_cast<int>(threadIdx.x); entry < end;
         entry += static_cast<int>(blockDim.x))
    {
        if (!first_unique_neighbor(entry, begin, neighbors))
            continue;
        const int neighbor = row_side ? rows + neighbors[entry]
                                      : neighbors[entry];
        if (visited[neighbor] == 0)
            atomicMin(owner + neighbor, parent_position);
    }
}

__global__ void count_owned_bipartite_neighbors_kernel(
    int frontier_size, int rows, const int *row_ptr,
    const int *csr_neighbors, const int *col_ptr,
    const int *csc_neighbors, const int *frontier,
    const int *owner, int *counts)
{
    const int parent_position = static_cast<int>(blockIdx.x);
    if (parent_position >= frontier_size)
        return;
    const int node = frontier[parent_position];
    const bool row_side = node < rows;
    const int local = row_side ? node : node - rows;
    const int begin = row_side ? row_ptr[local] : col_ptr[local];
    const int end = row_side ? row_ptr[local + 1] : col_ptr[local + 1];
    const int *neighbors = row_side ? csr_neighbors : csc_neighbors;
    int own = 0;
    for (int entry = begin + static_cast<int>(threadIdx.x); entry < end;
         entry += static_cast<int>(blockDim.x))
    {
        if (!first_unique_neighbor(entry, begin, neighbors))
            continue;
        const int neighbor = row_side ? rows + neighbors[entry]
                                      : neighbors[entry];
        own += owner[neighbor] == parent_position;
    }
    __shared__ int reduction[kThreads];
    reduction[threadIdx.x] = own;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
            reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        counts[parent_position] = reduction[0];
}

__global__ void write_owned_bipartite_neighbors_kernel(
    int frontier_size, int rows, const int *row_ptr,
    const int *csr_neighbors, const int *col_ptr,
    const int *csc_neighbors, const int *frontier, const int *offsets,
    int order_offset, int *next_frontier, int *order,
    unsigned char *visited, int *owner)
{
    const int parent_position = static_cast<int>(blockIdx.x);
    if (parent_position >= frontier_size)
        return;
    const int node = frontier[parent_position];
    const bool row_side = node < rows;
    const int local = row_side ? node : node - rows;
    const int begin = row_side ? row_ptr[local] : col_ptr[local];
    const int end = row_side ? row_ptr[local + 1] : col_ptr[local + 1];
    const int *neighbors = row_side ? csr_neighbors : csc_neighbors;
    __shared__ int scan[kThreads];
    __shared__ int batch_base;
    if (threadIdx.x == 0)
        batch_base = 0;
    __syncthreads();
    for (int base = begin; base < end; base += kThreads)
    {
        const int entry = base + static_cast<int>(threadIdx.x);
        int neighbor = -1;
        int flag = 0;
        if (entry < end && first_unique_neighbor(entry, begin, neighbors))
        {
            neighbor = row_side ? rows + neighbors[entry]
                                : neighbors[entry];
            flag = owner[neighbor] == parent_position;
        }
        scan[threadIdx.x] = flag;
        __syncthreads();
        for (int offset = 1; offset < kThreads; offset <<= 1)
        {
            const int add = threadIdx.x >= offset
                                ? scan[threadIdx.x - offset]
                                : 0;
            __syncthreads();
            if (threadIdx.x >= offset)
                scan[threadIdx.x] += add;
            __syncthreads();
        }
        const int local_base = batch_base;
        if (flag != 0)
        {
            const int destination = offsets[parent_position] + local_base +
                                    scan[threadIdx.x] - 1;
            next_frontier[destination] = neighbor;
            order[order_offset + destination] = neighbor;
            visited[neighbor] = 1;
            owner[neighbor] = INT_MAX;
        }
        __syncthreads();
        if (threadIdx.x == 0)
            batch_base += scan[kThreads - 1];
        __syncthreads();
    }
}

__global__ void invert_permutation_kernel(int count,
                                           const int *new_to_old,
                                           int *old_to_new);

struct BgrfTensorBlockRank
{
    int id = 0;
    unsigned long long rank_sum = 0;
};

struct BgrfTensorBlockRankLess
{
    __host__ __device__ bool operator()(const BgrfTensorBlockRank &lhs,
                                        const BgrfTensorBlockRank &rhs) const
    {
        if (lhs.rank_sum != rhs.rank_sum)
            return lhs.rank_sum < rhs.rank_sum;
        return lhs.id < rhs.id;
    }
};

template <int Group>
__global__ void rank_original_tensor_blocks_kernel(
    int complete_blocks, const int *scalar_old_to_new,
    BgrfTensorBlockRank *ranks)
{
    const int block = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (block >= complete_blocks)
        return;
    BgrfTensorBlockRank rank;
    rank.id = block;
    for (int slot = 0; slot < Group; ++slot)
        rank.rank_sum += static_cast<unsigned long long>(
            scalar_old_to_new[block * Group + slot]);
    ranks[block] = rank;
}

template <int Group>
__global__ void project_rcm_to_tensor_blocks_kernel(
    int count, int complete_blocks, const BgrfTensorBlockRank *ranks,
    int *new_to_old)
{
    const int next = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (next >= count)
        return;
    const int destination_block = next / Group;
    const int slot = next % Group;
    if (destination_block < complete_blocks)
        new_to_old[next] = ranks[destination_block].id * Group + slot;
    else
        new_to_old[next] = next;
}

__global__ void build_rcm_window_sort_keys_kernel(
    int count, int window, const int *current_old_to_new,
    const int *rcm_old_to_new, uint64_t *keys, int *ids)
{
    const int old = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (old >= count)
        return;
    const int current = current_old_to_new[old];
    const int coarse_window = current / window;
    keys[old] = static_cast<uint64_t>(
                    static_cast<unsigned int>(coarse_window)) *
                    (static_cast<uint64_t>(count) + 1ull) +
                static_cast<uint64_t>(
                    static_cast<unsigned int>(rcm_old_to_new[old]));
    ids[old] = old;
}

static inline bool build_rcm_window_proposal(
    int count, int window, const thrust::device_vector<int> &current_old_to_new,
    const thrust::device_vector<int> &rcm_old_to_new,
    thrust::device_vector<uint64_t> &sort_keys,
    thrust::device_vector<int> &proposal_new_to_old, cudaStream_t stream)
{
    if (count == 0)
        return true;
    auto policy = thrust::cuda::par.on(stream);
    build_rcm_window_sort_keys_kernel<<<blocks_for(count), kThreads, 0,
                                        stream>>>(
        count, window,
        thrust::raw_pointer_cast(current_old_to_new.data()),
        thrust::raw_pointer_cast(rcm_old_to_new.data()),
        thrust::raw_pointer_cast(sort_keys.data()),
        thrust::raw_pointer_cast(proposal_new_to_old.data()));
    if (!cuda_ok(cudaGetLastError(), "build BGRF RCM-window keys"))
        return false;
    thrust::sort_by_key(policy, sort_keys.begin(),
                        sort_keys.begin() + count,
                        proposal_new_to_old.begin());
    return cuda_ok(cudaGetLastError(), "sort BGRF RCM-window proposal");
}

static inline bool build_bgrf_global_coarse(
    int rows, int cols, int nnz, const int *d_row_ptr,
    const int *d_col_idx, thrust::device_vector<int> &row_new_to_old,
    thrust::device_vector<int> &row_old_to_new,
    thrust::device_vector<int> &inner_new_to_old,
    thrust::device_vector<int> &inner_old_to_new,
    thrust::device_vector<int> &rcm_row_old_to_new,
    thrust::device_vector<int> &rcm_inner_old_to_new,
    int *active_rows_out, int *active_inner_out, int *components_out,
    int *levels_out, std::size_t *peak_workspace_out,
    cudaStream_t stream)
{
    if (active_rows_out == nullptr || active_inner_out == nullptr ||
        components_out == nullptr || levels_out == nullptr ||
        peak_workspace_out == nullptr)
        return false;
    *active_rows_out = 0;
    *active_inner_out = 0;
    *components_out = 0;
    *levels_out = 0;
    *peak_workspace_out = 0;
    if (rows > INT_MAX - cols)
    {
        std::fprintf(stderr,
                     "BGRF bipartite graph exceeds the 32-bit vertex limit.\n");
        return false;
    }
    const int vertices = rows + cols;
    auto policy = thrust::cuda::par.on(stream);
    thrust::sequence(policy, row_new_to_old.begin(), row_new_to_old.end());
    thrust::sequence(policy, row_old_to_new.begin(), row_old_to_new.end());
    thrust::sequence(policy, inner_new_to_old.begin(), inner_new_to_old.end());
    thrust::sequence(policy, inner_old_to_new.begin(), inner_old_to_new.end());
    thrust::sequence(policy, rcm_row_old_to_new.begin(),
                     rcm_row_old_to_new.end());
    thrust::sequence(policy, rcm_inner_old_to_new.begin(),
                     rcm_inner_old_to_new.end());
    if (vertices == 0 || nnz == 0)
        return cuda_ok(cudaStreamSynchronize(stream),
                       "complete empty BGRF coarse permutation");

    const unsigned __int128 row_key_limit =
        static_cast<unsigned __int128>(rows) *
        DMMA_REORDER_DEGREE_BUCKETS *
        (static_cast<unsigned __int128>(cols) + 1);
    const unsigned __int128 col_key_limit =
        static_cast<unsigned __int128>(cols) *
        DMMA_REORDER_DEGREE_BUCKETS *
        (static_cast<unsigned __int128>(rows) + 1);
    if (row_key_limit > std::numeric_limits<uint64_t>::max() ||
        col_key_limit > std::numeric_limits<uint64_t>::max())
    {
        std::fprintf(stderr, "BGRF adjacency key space exceeds uint64_t.\n");
        return false;
    }

    thrust::device_vector<int> row_degree(rows, 0);
    thrust::device_vector<int> col_degree(cols, 0);
    thrust::device_vector<int> graph_degree(vertices, 0);
    thrust::device_vector<int> col_ptr(static_cast<std::size_t>(cols) + 1, 0);
    thrust::device_vector<int> csr_neighbors(nnz);
    thrust::device_vector<int> csc_neighbors(nnz);
    thrust::device_vector<uint64_t> adjacency_keys(nnz);
    const std::size_t permutation_bytes =
        static_cast<std::size_t>(4) *
        (static_cast<std::size_t>(rows) + static_cast<std::size_t>(cols)) *
        sizeof(int);
    const std::size_t adjacency_peak = permutation_bytes +
        (static_cast<std::size_t>(rows) + static_cast<std::size_t>(cols) +
         static_cast<std::size_t>(vertices) +
         static_cast<std::size_t>(cols) + 1) * sizeof(int) +
        static_cast<std::size_t>(nnz) *
            (2 * sizeof(int) + sizeof(uint64_t));
    *peak_workspace_out = adjacency_peak;

    if (rows > 0)
    {
        build_bipartite_degrees_kernel<<<blocks_for(rows), kThreads, 0,
                                           stream>>>(
            rows, d_row_ptr, thrust::raw_pointer_cast(row_degree.data()));
        count_bipartite_columns_kernel<<<rows, kThreads, 0, stream>>>(
            rows, d_row_ptr, d_col_idx,
            thrust::raw_pointer_cast(col_degree.data()));
        if (!cuda_ok(cudaGetLastError(), "build BGRF degrees"))
            return false;
    }
    if (cols > 0)
    {
        thrust::copy(policy, col_degree.begin(), col_degree.end(),
                     col_ptr.begin());
        thrust::fill(policy, col_ptr.end() - 1, col_ptr.end(), 0);
        thrust::exclusive_scan(policy, col_ptr.begin(), col_ptr.end(),
                               col_ptr.begin());
    }
    combine_graph_degrees_kernel<<<blocks_for(vertices), kThreads, 0,
                                    stream>>>(
        rows, cols, thrust::raw_pointer_cast(row_degree.data()),
        thrust::raw_pointer_cast(col_degree.data()),
        thrust::raw_pointer_cast(graph_degree.data()));
    if (!cuda_ok(cudaGetLastError(), "combine BGRF graph degrees"))
        return false;

    generate_bucketed_csr_keys_kernel<<<rows, kThreads, 0, stream>>>(
        rows, cols, d_row_ptr, d_col_idx,
        thrust::raw_pointer_cast(col_degree.data()),
        thrust::raw_pointer_cast(adjacency_keys.data()),
        thrust::raw_pointer_cast(csr_neighbors.data()));
    if (!cuda_ok(cudaGetLastError(), "generate BGRF CSR bucket keys"))
        return false;
    thrust::sort_by_key(policy, adjacency_keys.begin(), adjacency_keys.end(),
                        csr_neighbors.begin());
    generate_bucketed_csc_keys_kernel<<<rows, kThreads, 0, stream>>>(
        rows, cols, d_row_ptr, d_col_idx,
        thrust::raw_pointer_cast(row_degree.data()),
        thrust::raw_pointer_cast(adjacency_keys.data()),
        thrust::raw_pointer_cast(csc_neighbors.data()));
    if (!cuda_ok(cudaGetLastError(), "generate BGRF CSC bucket keys"))
        return false;
    thrust::sort_by_key(policy, adjacency_keys.begin(), adjacency_keys.end(),
                        csc_neighbors.begin());
    thrust::device_vector<uint64_t>().swap(adjacency_keys);

    const int active_rows = static_cast<int>(thrust::count_if(
        policy, row_degree.begin(), row_degree.end(), IsNonZeroValue()));
    const int active_inner = static_cast<int>(thrust::count_if(
        policy, col_degree.begin(), col_degree.end(), IsNonZeroValue()));
    const int active_vertices = active_rows + active_inner;
    *active_rows_out = active_rows;
    *active_inner_out = active_inner;
    if (active_vertices == 0)
        return cuda_ok(cudaStreamSynchronize(stream),
                       "complete zero-degree BGRF graph");

    thrust::device_vector<int> seed_order(vertices);
    thrust::device_vector<int> frontier(vertices);
    thrust::device_vector<int> next_frontier(vertices);
    thrust::device_vector<int> traversal_order(active_vertices);
    thrust::device_vector<int> owner(vertices, INT_MAX);
    thrust::device_vector<int> offsets(static_cast<std::size_t>(vertices) + 1,
                                       0);
    thrust::device_vector<unsigned char> visited(vertices, 0);
    thrust::sequence(policy, seed_order.begin(), seed_order.end());
    thrust::sort(policy, seed_order.begin(), seed_order.end(),
                 GraphSeedLess{
                     thrust::raw_pointer_cast(graph_degree.data())});
    const std::size_t bfs_peak = permutation_bytes +
        (static_cast<std::size_t>(rows) + static_cast<std::size_t>(cols) +
         static_cast<std::size_t>(vertices) +
         static_cast<std::size_t>(cols) + 1 +
         static_cast<std::size_t>(2) * nnz +
         static_cast<std::size_t>(6) * vertices + 1) * sizeof(int) +
        static_cast<std::size_t>(vertices) * sizeof(unsigned char);
    *peak_workspace_out = std::max(*peak_workspace_out, bfs_peak);

    int order_size = 0;
    int seed_cursor = 0;
    int components = 0;
    int levels = 0;
    bool traversal_budget_reached = false;
    while (order_size < active_vertices)
    {
        auto seed_it = thrust::find_if(
            policy, seed_order.begin() + seed_cursor, seed_order.end(),
            IsUnvisitedActive{
                thrust::raw_pointer_cast(graph_degree.data()),
                thrust::raw_pointer_cast(visited.data())});
        if (seed_it == seed_order.end())
        {
            std::fprintf(stderr,
                         "BGRF traversal ended before all active vertices.\n");
            return false;
        }
        seed_cursor = static_cast<int>(seed_it - seed_order.begin()) + 1;
        int seed = -1;
        if (!cuda_ok(cudaMemcpyAsync(
                         &seed, thrust::raw_pointer_cast(seed_order.data()) +
                                    seed_cursor - 1,
                         sizeof(int), cudaMemcpyDeviceToHost, stream),
                     "copy BGRF component seed") ||
            !cuda_ok(cudaStreamSynchronize(stream),
                     "select BGRF component seed"))
            return false;
        start_bipartite_component_kernel<<<1, 1, 0, stream>>>(
            seed, thrust::raw_pointer_cast(frontier.data()),
            thrust::raw_pointer_cast(traversal_order.data()), order_size,
            thrust::raw_pointer_cast(visited.data()));
        if (!cuda_ok(cudaGetLastError(), "start BGRF component"))
            return false;
        ++components;
        ++order_size;
        int frontier_size = 1;
        while (frontier_size > 0)
        {
            claim_bipartite_frontier_kernel
                <<<frontier_size, kThreads, 0, stream>>>(
                    frontier_size, rows, d_row_ptr,
                    thrust::raw_pointer_cast(csr_neighbors.data()),
                    thrust::raw_pointer_cast(col_ptr.data()),
                    thrust::raw_pointer_cast(csc_neighbors.data()),
                    thrust::raw_pointer_cast(frontier.data()),
                    thrust::raw_pointer_cast(visited.data()),
                    thrust::raw_pointer_cast(owner.data()));
            count_owned_bipartite_neighbors_kernel
                <<<frontier_size, kThreads, 0, stream>>>(
                    frontier_size, rows, d_row_ptr,
                    thrust::raw_pointer_cast(csr_neighbors.data()),
                    thrust::raw_pointer_cast(col_ptr.data()),
                    thrust::raw_pointer_cast(csc_neighbors.data()),
                    thrust::raw_pointer_cast(frontier.data()),
                    thrust::raw_pointer_cast(owner.data()),
                    thrust::raw_pointer_cast(offsets.data()));
            if (!cuda_ok(cudaGetLastError(), "count BGRF next frontier"))
                return false;
            thrust::fill(policy, offsets.begin() + frontier_size,
                         offsets.begin() + frontier_size + 1, 0);
            thrust::exclusive_scan(policy, offsets.begin(),
                                   offsets.begin() + frontier_size + 1,
                                   offsets.begin());
            int next_size = 0;
            if (!cuda_ok(cudaMemcpyAsync(
                             &next_size,
                             thrust::raw_pointer_cast(offsets.data()) +
                                 frontier_size,
                             sizeof(int), cudaMemcpyDeviceToHost, stream),
                         "copy BGRF frontier size") ||
                !cuda_ok(cudaStreamSynchronize(stream),
                         "complete BGRF frontier scan"))
                return false;
            if (next_size > 0)
            {
                write_owned_bipartite_neighbors_kernel
                    <<<frontier_size, kThreads, 0, stream>>>(
                        frontier_size, rows, d_row_ptr,
                        thrust::raw_pointer_cast(csr_neighbors.data()),
                        thrust::raw_pointer_cast(col_ptr.data()),
                        thrust::raw_pointer_cast(csc_neighbors.data()),
                        thrust::raw_pointer_cast(frontier.data()),
                        thrust::raw_pointer_cast(offsets.data()), order_size,
                        thrust::raw_pointer_cast(next_frontier.data()),
                        thrust::raw_pointer_cast(traversal_order.data()),
                        thrust::raw_pointer_cast(visited.data()),
                        thrust::raw_pointer_cast(owner.data()));
                if (!cuda_ok(cudaGetLastError(), "write BGRF next frontier"))
                    return false;
            }
            ++levels;
            order_size += next_size;
            frontier.swap(next_frontier);
            frontier_size = next_size;
            if (levels >= DMMA_REORDER_MAX_COARSE_LEVELS)
            {
                if (order_size < active_vertices)
                {
                    /* Keep the coarse work independent of graph diameter.
                     * Residual active vertices are appended in reverse seed
                     * order so the final RCM reversal exposes the same
                     * deterministic low-degree bucket order. */
                    const auto residual_begin =
                        thrust::make_reverse_iterator(seed_order.end());
                    const auto residual_end =
                        thrust::make_reverse_iterator(seed_order.begin());
                    auto tail_end = thrust::copy_if(
                        policy, residual_begin, residual_end,
                        traversal_order.begin() + order_size,
                        IsUnvisitedActive{
                            thrust::raw_pointer_cast(graph_degree.data()),
                            thrust::raw_pointer_cast(visited.data())});
                    const int residual = static_cast<int>(
                        tail_end - (traversal_order.begin() + order_size));
                    if (order_size + residual != active_vertices)
                    {
                        std::fprintf(stderr,
                                     "BGRF bounded traversal residual count "
                                     "mismatch.\n");
                        return false;
                    }
                    order_size += residual;
                }
                traversal_budget_reached = true;
                frontier_size = 0;
            }
        }
        if (traversal_budget_reached)
            break;
    }

    const auto reverse_begin = thrust::make_reverse_iterator(
        traversal_order.begin() + active_vertices);
    const auto reverse_end =
        thrust::make_reverse_iterator(traversal_order.begin());
    auto row_end = thrust::copy_if(policy, reverse_begin, reverse_end,
                                   row_new_to_old.begin(), IsRowNode{rows});
    auto inner_end = thrust::copy_if(policy, reverse_begin, reverse_end,
                                     inner_new_to_old.begin(),
                                     IsInnerNode{rows});
    if (row_end - row_new_to_old.begin() != active_rows ||
        inner_end - inner_new_to_old.begin() != active_inner)
    {
        std::fprintf(stderr, "BGRF row/K projection count mismatch.\n");
        return false;
    }
    thrust::transform(policy, inner_new_to_old.begin(), inner_end,
                      inner_new_to_old.begin(), SubtractRows{rows});
    const auto row_ids = thrust::make_counting_iterator<int>(0);
    thrust::copy_if(policy, row_ids, row_ids + rows,
                    row_new_to_old.begin() + active_rows,
                    IsZeroDegree{
                        thrust::raw_pointer_cast(row_degree.data())});
    const auto inner_ids = thrust::make_counting_iterator<int>(0);
    thrust::copy_if(policy, inner_ids, inner_ids + cols,
                    inner_new_to_old.begin() + active_inner,
                    IsZeroDegree{
                        thrust::raw_pointer_cast(col_degree.data())});

    /* Preserve the scalar RCM ranks for the single Tensor-window fine pass. */
    invert_permutation_kernel<<<blocks_for(rows), kThreads, 0, stream>>>(
        rows, thrust::raw_pointer_cast(row_new_to_old.data()),
        thrust::raw_pointer_cast(row_old_to_new.data()));
    invert_permutation_kernel<<<blocks_for(cols), kThreads, 0, stream>>>(
        cols, thrust::raw_pointer_cast(inner_new_to_old.data()),
        thrust::raw_pointer_cast(inner_old_to_new.data()));
    thrust::copy(policy, row_old_to_new.begin(), row_old_to_new.end(),
                 rcm_row_old_to_new.begin());
    thrust::copy(policy, inner_old_to_new.begin(), inner_old_to_new.end(),
                 rcm_inner_old_to_new.begin());
    /* The Tensor-safe incumbent is identity.  The caller evaluates the joint
     * scalar RCM row/K proposal once against the exact global A-tile count. */
    thrust::sequence(policy, row_new_to_old.begin(), row_new_to_old.end());
    thrust::sequence(policy, row_old_to_new.begin(), row_old_to_new.end());
    thrust::sequence(policy, inner_new_to_old.begin(), inner_new_to_old.end());
    thrust::sequence(policy, inner_old_to_new.begin(), inner_old_to_new.end());
    *components_out = components;
    *levels_out = levels;
    return cuda_ok(cudaStreamSynchronize(stream),
                   "complete BGRF global coarse traversal");
}

/* Variants that disable either fine axis do not have its DmmaAxisProfile
 * array available for the final ActivePrefix reduction.  Compute the exact
 * nonempty prefixes directly from CSR and the selected permutations.  One
 * block handles one source row and emits at most one inner-axis atomicMax. */
__global__ void compute_active_prefixes_kernel(
    int rows, const int *row_ptr, const int *col_idx,
    const int *row_old_to_new, const int *inner_old_to_new,
    int *active_prefixes)
{
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows)
        return;
    const int begin = row_ptr[row];
    const int end = row_ptr[row + 1];
    if (threadIdx.x == 0 && begin < end)
        atomicMax(active_prefixes, row_old_to_new[row] + 1);

    int local_inner_prefix = 0;
    for (int entry = begin + static_cast<int>(threadIdx.x);
         entry < end; entry += static_cast<int>(blockDim.x))
        local_inner_prefix = max(
            local_inner_prefix,
            inner_old_to_new[col_idx[entry]] + 1);
    __shared__ int block_inner_prefix[kThreads];
    block_inner_prefix[threadIdx.x] = local_inner_prefix;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            block_inner_prefix[threadIdx.x] = max(
                block_inner_prefix[threadIdx.x],
                block_inner_prefix[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0 && block_inner_prefix[0] > 0)
        atomicMax(active_prefixes + 1, block_inner_prefix[0]);
}

static inline bool compute_active_prefixes(
    int rows, const int *d_row_ptr, const int *d_col_idx,
    const thrust::device_vector<int> &row_old_to_new,
    const thrust::device_vector<int> &inner_old_to_new,
    int *active_rows, int *active_inner, cudaStream_t stream)
{
    if (active_rows == nullptr || active_inner == nullptr)
        return false;
    *active_rows = 0;
    *active_inner = 0;
    if (rows == 0)
        return true;
    thrust::device_vector<int> prefixes(2, 0);
    compute_active_prefixes_kernel<<<rows, kThreads, 0, stream>>>(
        rows, d_row_ptr, d_col_idx,
        thrust::raw_pointer_cast(row_old_to_new.data()),
        thrust::raw_pointer_cast(inner_old_to_new.data()),
        thrust::raw_pointer_cast(prefixes.data()));
    int host_prefixes[2] = {0, 0};
    if (!cuda_ok(cudaGetLastError(), "compute BGRF active prefixes") ||
        !cuda_ok(cudaMemcpyAsync(
                     host_prefixes,
                     thrust::raw_pointer_cast(prefixes.data()),
                     sizeof(host_prefixes), cudaMemcpyDeviceToHost,
                     stream),
                 "copy BGRF active prefixes") ||
        !cuda_ok(cudaStreamSynchronize(stream),
                 "complete BGRF active prefixes"))
        return false;
    *active_rows = host_prefixes[0];
    *active_inner = host_prefixes[1];
    return true;
}

__global__ void extract_row_profiles_kernel(
    int rows, const int *row_ptr, const int *col_idx,
    const int *inner_old_to_new, DmmaAxisProfile *profiles)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows)
        return;

    const int begin = row_ptr[row];
    const int end = row_ptr[row + 1];
    uint64_t fingerprint[DMMA_REORDER_FINGERPRINT_WORDS] = {0, 0};
    uint64_t minhash[DMMA_REORDER_FINGERPRINT_WORDS] = {
        ~uint64_t(0), ~uint64_t(0)};
    int panel_min = INT_MAX;
    int panel_max = -1;
    for (int entry = begin; entry < end; ++entry)
    {
        const int old_col = col_idx[entry];
        const int new_col = inner_old_to_new[old_col];
        const int panel = new_col / DMMA_TILE_K;
        panel_min = min(panel_min, panel);
        panel_max = max(panel_max, panel);
        fingerprint_panel(panel, fingerprint);
        minhash_panel(panel, minhash);
    }

    DmmaAxisProfile profile;
    profile.id = row;
    profile.degree = end - begin;
    profile.soft_bucket = soft_bucket_device(profile.degree);
    profile.panel_min = profile.degree == 0 ? -1 : panel_min;
    profile.panel_max = panel_max;
    profile.panel_span =
        profile.degree == 0 ? 0 : panel_max - panel_min + 1;
#pragma unroll
    for (int word = 0; word < DMMA_REORDER_FINGERPRINT_WORDS; ++word)
    {
        profile.fingerprint[word] = fingerprint[word];
        profile.minhash[word] = minhash[word];
    }
    profile.panel_occupancy = fingerprint_occupancy(fingerprint);
    profiles[row] = profile;
}

__global__ void initialize_profiles_kernel(int count,
                                            DmmaAxisProfile *profiles)
{
    const int id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (id >= count)
        return;
    DmmaAxisProfile profile;
    profile.id = id;
    profile.panel_min = INT_MAX;
    profile.panel_max = -1;
    profiles[id] = profile;
}

__global__ void accumulate_inner_profiles_kernel(
    int rows, const int *row_ptr, const int *col_idx,
    const int *row_old_to_new, DmmaAxisProfile *profiles)
{
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows)
        return;
    const int panel = row_old_to_new[row] / DMMA_TILE_M;
    uint64_t bits[4] = {0, 0, 0, 0};
    uint64_t minhash[DMMA_REORDER_FINGERPRINT_WORDS] = {
        ~uint64_t(0), ~uint64_t(0)};
    fingerprint_panel(panel, bits);
    minhash_panel(panel, minhash);
    for (int entry = row_ptr[row]; entry < row_ptr[row + 1]; ++entry)
    {
        DmmaAxisProfile *profile = profiles + col_idx[entry];
        atomicAdd(&profile->degree, 1);
        atomicMin(&profile->panel_min, panel);
        atomicMax(&profile->panel_max, panel);
#pragma unroll
        for (int word = 0; word < DMMA_REORDER_FINGERPRINT_WORDS; ++word)
        {
            atomicOr(reinterpret_cast<unsigned long long *>(
                         &profile->fingerprint[word]),
                     static_cast<unsigned long long>(bits[word]));
            atomicMin(reinterpret_cast<unsigned long long *>(
                          &profile->minhash[word]),
                      static_cast<unsigned long long>(minhash[word]));
        }
    }
}

__global__ void finalize_profiles_kernel(int count,
                                          DmmaAxisProfile *profiles)
{
    const int id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (id >= count)
        return;
    DmmaAxisProfile &profile = profiles[id];
    profile.soft_bucket = soft_bucket_device(profile.degree);
    if (profile.degree == 0)
    {
        profile.panel_min = -1;
        profile.panel_max = -1;
        profile.panel_span = 0;
    }
    else
    {
        profile.panel_span = profile.panel_max - profile.panel_min + 1;
    }
    profile.panel_occupancy =
        fingerprint_occupancy(profile.fingerprint);
}

struct CoarseProfileLess
{
    __host__ __device__ bool operator()(const DmmaAxisProfile &a,
                                        const DmmaAxisProfile &b) const
    {
        const bool a_nonempty = a.degree != 0;
        const bool b_nonempty = b.degree != 0;
        if (a_nonempty != b_nonempty)
            return a_nonempty;
        if (a.soft_bucket != b.soft_bucket)
            return a.soft_bucket > b.soft_bucket;
        if (a.panel_occupancy != b.panel_occupancy)
            return a.panel_occupancy > b.panel_occupancy;
        if (a.panel_min != b.panel_min)
            return a.panel_min < b.panel_min;
        if (a.panel_span != b.panel_span)
            return a.panel_span < b.panel_span;
        if (a.degree != b.degree)
            return a.degree > b.degree;
#pragma unroll
        for (int word = 0; word < DMMA_REORDER_FINGERPRINT_WORDS; ++word)
            if (a.fingerprint[word] != b.fingerprint[word])
                return a.fingerprint[word] < b.fingerprint[word];
        return a.id < b.id;
    }
};

struct CurrentPositionLess
{
    const int *old_to_new;

    __host__ __device__ bool operator()(const DmmaAxisProfile &a,
                                        const DmmaAxisProfile &b) const
    {
        return old_to_new[a.id] < old_to_new[b.id];
    }
};

template <int Window>
struct TensorWindowProfileLess
{
    const int *old_to_new;

    __host__ __device__ bool operator()(const DmmaAxisProfile &lhs,
                                        const DmmaAxisProfile &rhs) const
    {
        const int lhs_window = old_to_new[lhs.id] / Window;
        const int rhs_window = old_to_new[rhs.id] / Window;
        if (lhs_window != rhs_window)
            return lhs_window < rhs_window;
        return CoarseProfileLess()(lhs, rhs);
    }
};

struct ProfileId
{
    __host__ __device__ int operator()(const DmmaAxisProfile &profile) const
    {
        return profile.id;
    }
};

struct ScalarWithinTileLess
{
    __host__ __device__ bool operator()(const DmmaAxisProfile &a,
                                        const DmmaAxisProfile &b) const
    {
        const bool a_nonempty = a.degree != 0;
        const bool b_nonempty = b.degree != 0;
        if (a_nonempty != b_nonempty)
            return a_nonempty;
        if (a.degree != b.degree)
            return a.degree > b.degree;
        if (a.panel_occupancy != b.panel_occupancy)
            return a.panel_occupancy > b.panel_occupancy;
        if (a.panel_min != b.panel_min)
            return a.panel_min < b.panel_min;
        if (a.panel_span != b.panel_span)
            return a.panel_span < b.panel_span;
        return a.id < b.id;
    }
};

struct TileBlockLess
{
    __host__ __device__ bool operator()(const DmmaTileBlockProfile &a,
                                        const DmmaTileBlockProfile &b) const
    {
        const bool a_nonempty = a.degree != 0;
        const bool b_nonempty = b.degree != 0;
        if (a_nonempty != b_nonempty)
            return a_nonempty;
        /* Preserve the input's macro order.  Only complete empty blocks move
         * to the tail globally; degree/locality ordering stays inside the
         * fixed Tensor windows below. */
        return a.id < b.id;
    }
};

struct PackScore
{
    int bucket_distance = INT_MAX;
    int new_bits = INT_MAX;
    int overlap = -1;
    int union_bits = 1;
    int span_growth = INT_MAX;
    int degree_difference = INT_MAX;
    int id = INT_MAX;
};

__device__ static inline int abs_difference(int lhs, int rhs)
{
    return lhs >= rhs ? lhs - rhs : rhs - lhs;
}

__device__ static inline bool better_pack_score(const PackScore &lhs,
                                                 const PackScore &rhs)
{
    if (lhs.bucket_distance != rhs.bucket_distance)
        return lhs.bucket_distance < rhs.bucket_distance;
    if (lhs.new_bits != rhs.new_bits)
        return lhs.new_bits < rhs.new_bits;
    const int lhs_ratio = lhs.overlap * rhs.union_bits;
    const int rhs_ratio = rhs.overlap * lhs.union_bits;
    if (lhs_ratio != rhs_ratio)
        return lhs_ratio > rhs_ratio;
    if (lhs.overlap != rhs.overlap)
        return lhs.overlap > rhs.overlap;
    if (lhs.span_growth != rhs.span_growth)
        return lhs.span_growth < rhs.span_growth;
    if (lhs.degree_difference != rhs.degree_difference)
        return lhs.degree_difference < rhs.degree_difference;
    return lhs.id < rhs.id;
}

/* Coarse movement is tile preserving by construction.  Scalar axes are
 * sorted only inside their current physical Tensor-Core tile, then complete
 * tiles are moved as indivisible objects.  Consequently this phase merely
 * renumbers A tiles: it cannot create a new nonzero 8x4 tile.  The final
 * partial tile is kept in place because it is not interchangeable with a
 * complete tile without changing tile membership. */
template <int Group>
__global__ void build_coarse_blocks_kernel(
    const DmmaAxisProfile *ordered, int count, int *local_new_to_old,
    DmmaTileBlockProfile *blocks)
{
    if (threadIdx.x != 0)
        return;
    const int block = static_cast<int>(blockIdx.x);
    const int begin = block * Group;
    if (begin >= count)
        return;
    const int local_count = min(Group, count - begin);
    unsigned char used[Group];
#pragma unroll
    for (int slot = 0; slot < Group; ++slot)
        used[slot] = 0;

    DmmaTileBlockProfile aggregate;
    aggregate.id = block;
    aggregate.panel_min = INT_MAX;
    aggregate.panel_max = -1;
    for (int slot = 0; slot < local_count; ++slot)
    {
        const DmmaAxisProfile &profile = ordered[begin + slot];
        aggregate.degree += static_cast<long long>(profile.degree);
        aggregate.nonempty += profile.degree != 0;
        if (profile.panel_min >= 0)
        {
            aggregate.panel_min = min(aggregate.panel_min,
                                      profile.panel_min);
            aggregate.panel_max = max(aggregate.panel_max,
                                      profile.panel_max);
        }
#pragma unroll
        for (int word = 0; word < DMMA_REORDER_FINGERPRINT_WORDS; ++word)
            aggregate.fingerprint[word] |= profile.fingerprint[word];
    }
    if (aggregate.degree == 0)
    {
        aggregate.panel_min = -1;
        aggregate.panel_max = -1;
        aggregate.panel_span = 0;
    }
    else
    {
        aggregate.panel_span = aggregate.panel_max - aggregate.panel_min + 1;
    }
    aggregate.panel_occupancy =
        fingerprint_occupancy(aggregate.fingerprint);
    blocks[block] = aggregate;

    for (int output = 0; output < local_count; ++output)
    {
        int best = -1;
        for (int slot = 0; slot < local_count; ++slot)
            if (used[slot] == 0 &&
                (best < 0 ||
                 ScalarWithinTileLess()(ordered[begin + slot],
                                        ordered[begin + best])))
                best = slot;
        used[best] = 1;
        local_new_to_old[begin + output] = ordered[begin + best].id;
    }
}

template <int Group>
__global__ void assemble_coarse_permutation_kernel(
    int count, int complete_blocks, const DmmaTileBlockProfile *sorted_blocks,
    const int *local_new_to_old, int *proposal_new_to_old)
{
    const int next = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (next >= count)
        return;
    const int destination_block = next / Group;
    const int slot = next % Group;
    const int source_block = destination_block < complete_blocks
                                 ? sorted_blocks[destination_block].id
                                 : destination_block;
    proposal_new_to_old[next] =
        local_new_to_old[source_block * Group + slot];
}

template <int Window, int Group>
__global__ void pack_windows_kernel(const DmmaAxisProfile *coarse,
                                    int count, int *new_to_old)
{
    if (threadIdx.x != 0)
        return;
    const int begin = static_cast<int>(blockIdx.x) * Window;
    if (begin >= count)
        return;
    const int end = begin + min(Window, count - begin);
    const int local_count = end - begin;
    unsigned char used[Window];
#pragma unroll
    for (int i = 0; i < Window; ++i)
        used[i] = 0;

    int written = 0;
    while (written < local_count)
    {
        int anchor_local = -1;
        for (int local = 0; local < local_count; ++local)
            if (used[local] == 0)
            {
                if (anchor_local < 0 ||
                    CoarseProfileLess()(coarse[begin + local],
                                        coarse[begin + anchor_local]))
                    anchor_local = local;
            }
        if (anchor_local < 0)
            break;

        const DmmaAxisProfile &anchor = coarse[begin + anchor_local];
        used[anchor_local] = 1;
        new_to_old[begin + written++] = anchor.id;
        uint64_t group_fingerprint[DMMA_REORDER_FINGERPRINT_WORDS];
#pragma unroll
        for (int word = 0; word < DMMA_REORDER_FINGERPRINT_WORDS; ++word)
            group_fingerprint[word] = anchor.fingerprint[word];
        int group_min = anchor.panel_min;
        int group_max = anchor.panel_max;

        for (int slot = 1; slot < Group && written < local_count; ++slot)
        {
            int best_local = -1;
            PackScore best;
            const int current_span = group_min < 0 ? 0 : group_max - group_min + 1;
            for (int local = 0; local < local_count; ++local)
            {
                if (used[local] != 0)
                    continue;
                const DmmaAxisProfile &candidate = coarse[begin + local];
                PackScore score;
                score.bucket_distance = abs_difference(
                    anchor.soft_bucket, candidate.soft_bucket);
                int overlap = 0;
                int additions = 0;
                int union_bits = 0;
#pragma unroll
                for (int word = 0; word < DMMA_REORDER_FINGERPRINT_WORDS; ++word)
                {
                    overlap += __popcll(group_fingerprint[word] &
                                        candidate.fingerprint[word]);
                    additions += __popcll((~group_fingerprint[word]) &
                                          candidate.fingerprint[word]);
                    union_bits += __popcll(group_fingerprint[word] |
                                           candidate.fingerprint[word]);
                }
                score.new_bits = additions;
                score.overlap = overlap;
                score.union_bits = max(union_bits, 1);
                int next_min = group_min;
                int next_max = group_max;
                if (candidate.panel_min >= 0)
                {
                    next_min = group_min < 0
                                   ? candidate.panel_min
                                   : min(group_min, candidate.panel_min);
                    next_max = max(group_max, candidate.panel_max);
                }
                const int next_span = next_min < 0 ? 0 : next_max - next_min + 1;
                score.span_growth = next_span - current_span;
                score.degree_difference = abs_difference(
                    anchor.degree, candidate.degree);
                score.id = candidate.id;
                if (best_local < 0 || better_pack_score(score, best))
                {
                    best_local = local;
                    best = score;
                }
            }
            if (best_local < 0)
                break;
            const DmmaAxisProfile &selected = coarse[begin + best_local];
            used[best_local] = 1;
            new_to_old[begin + written++] = selected.id;
            if (selected.panel_min >= 0)
            {
                group_min = group_min < 0
                                ? selected.panel_min
                                : min(group_min, selected.panel_min);
                group_max = max(group_max, selected.panel_max);
            }
#pragma unroll
            for (int word = 0; word < DMMA_REORDER_FINGERPRINT_WORDS; ++word)
                group_fingerprint[word] |= selected.fingerprint[word];
        }
    }
}

__global__ void invert_permutation_kernel(int count,
                                           const int *new_to_old,
                                           int *old_to_new)
{
    const int next = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (next < count)
        old_to_new[new_to_old[next]] = next;
}

static constexpr uint64_t kProposalTag = uint64_t(1) << 63;

__global__ void invert_old_to_new_kernel(int count, const int *old_to_new,
                                          int *new_to_old)
{
    const int old = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (old < count)
        new_to_old[old_to_new[old]] = old;
}

__global__ void generate_joint_global_tile_keys_kernel(
    int rows, int nnz, const int *row_ptr, const int *col_idx,
    const int *current_row_old_to_new,
    const int *proposal_row_old_to_new,
    const int *current_inner_old_to_new,
    const int *proposal_inner_old_to_new, int inner_tiles,
    uint64_t *keys)
{
    const int old_row = static_cast<int>(blockIdx.x);
    if (old_row >= rows)
        return;
    const int current_row_tile =
        current_row_old_to_new[old_row] / DMMA_TILE_M;
    const int proposal_row_tile =
        proposal_row_old_to_new[old_row] / DMMA_TILE_M;
    for (int entry = row_ptr[old_row] + static_cast<int>(threadIdx.x);
         entry < row_ptr[old_row + 1]; entry += static_cast<int>(blockDim.x))
    {
        const int old_col = col_idx[entry];
        const uint64_t current =
            static_cast<uint64_t>(current_row_tile) * inner_tiles +
            static_cast<uint64_t>(
                current_inner_old_to_new[old_col] / DMMA_TILE_K);
        const uint64_t proposal =
            static_cast<uint64_t>(proposal_row_tile) * inner_tiles +
            static_cast<uint64_t>(
                proposal_inner_old_to_new[old_col] / DMMA_TILE_K);
        keys[entry] = current;
        keys[static_cast<std::size_t>(nnz) + entry] =
            kProposalTag | proposal;
    }
}

static inline bool exact_joint_global_commit(
    int rows, int cols, int nnz, const int *d_row_ptr,
    const int *d_col_idx,
    thrust::device_vector<int> &current_row_new_to_old,
    thrust::device_vector<int> &current_row_old_to_new,
    thrust::device_vector<int> &proposal_row_new_to_old,
    thrust::device_vector<int> &proposal_row_old_to_new,
    thrust::device_vector<int> &current_inner_new_to_old,
    thrust::device_vector<int> &current_inner_old_to_new,
    thrust::device_vector<int> &proposal_inner_new_to_old,
    thrust::device_vector<int> &proposal_inner_old_to_new,
    thrust::device_vector<uint64_t> &exact_keys, bool *accepted,
    unsigned long long *tile_reduction, cudaStream_t stream)
{
    *accepted = false;
    *tile_reduction = 0;
    if (nnz == 0)
        return true;
    auto policy = thrust::cuda::par.on(stream);
    const int inner_tiles = ceil_div_nonnegative(cols, DMMA_TILE_K);
    generate_joint_global_tile_keys_kernel<<<rows, kThreads, 0, stream>>>(
        rows, nnz, d_row_ptr, d_col_idx,
        thrust::raw_pointer_cast(current_row_old_to_new.data()),
        thrust::raw_pointer_cast(proposal_row_old_to_new.data()),
        thrust::raw_pointer_cast(current_inner_old_to_new.data()),
        thrust::raw_pointer_cast(proposal_inner_old_to_new.data()),
        inner_tiles, thrust::raw_pointer_cast(exact_keys.data()));
    if (!cuda_ok(cudaGetLastError(), "generate joint BGRF tile keys"))
        return false;
    const std::size_t key_count = static_cast<std::size_t>(2) * nnz;
    thrust::sort(policy, exact_keys.begin(), exact_keys.begin() + key_count);
    auto unique_end = thrust::unique(policy, exact_keys.begin(),
                                     exact_keys.begin() + key_count);
    auto proposal_begin = thrust::lower_bound(
        policy, exact_keys.begin(), unique_end, kProposalTag);
    const std::size_t current_tiles = static_cast<std::size_t>(
        proposal_begin - exact_keys.begin());
    const std::size_t proposal_tiles = static_cast<std::size_t>(
        unique_end - proposal_begin);
    if (proposal_tiles < current_tiles)
    {
        current_row_new_to_old.swap(proposal_row_new_to_old);
        current_row_old_to_new.swap(proposal_row_old_to_new);
        current_inner_new_to_old.swap(proposal_inner_new_to_old);
        current_inner_old_to_new.swap(proposal_inner_old_to_new);
        *accepted = true;
        *tile_reduction = static_cast<unsigned long long>(
            current_tiles - proposal_tiles);
    }
    return cuda_ok(cudaGetLastError(), "complete joint BGRF tile decision");
}

/* Each emitted key is exactly one possible A tile, not one scalar entry:
 *
 *   (state, fixed two-tile window, proposed 0/1 tile group, opposite tile).
 *
 * Sorting and uniquing these keys therefore gives an exact tile count even
 * for duplicate CSR entries and rows/columns of arbitrary degree. */
template <bool ReorderRows>
__global__ void generate_exact_tile_keys_kernel(
    int rows, int nnz, const int *row_ptr, const int *col_idx,
    const int *current_row_old_to_new,
    const int *proposal_row_old_to_new,
    const int *current_inner_old_to_new,
    const int *proposal_inner_old_to_new,
    int opposite_tile_count, int groups_per_window, uint64_t *keys)
{
    const int old_row = static_cast<int>(blockIdx.x);
    if (old_row >= rows)
        return;
    const int current_row = current_row_old_to_new[old_row];
    const int proposal_row = ReorderRows
                                 ? proposal_row_old_to_new[old_row]
                                 : current_row;
    for (std::size_t entry =
             static_cast<std::size_t>(row_ptr[old_row]) + threadIdx.x;
         entry < static_cast<std::size_t>(row_ptr[old_row + 1]);
         entry += static_cast<std::size_t>(blockDim.x))
    {
        const int old_col = col_idx[entry];
        const int current_col = current_inner_old_to_new[old_col];
        const int proposal_col = ReorderRows
                                     ? current_col
                                     : proposal_inner_old_to_new[old_col];
        int window = 0;
        int opposite_tile = 0;
        int current_group = 0;
        int proposal_group = 0;
        if constexpr (ReorderRows)
        {
            window = current_row / DMMA_REORDER_ROW_WINDOW;
            opposite_tile = current_col / DMMA_TILE_K;
            current_group =
                (current_row % DMMA_REORDER_ROW_WINDOW) / DMMA_TILE_M;
            proposal_group =
                (proposal_row % DMMA_REORDER_ROW_WINDOW) / DMMA_TILE_M;
        }
        else
        {
            window = current_col / DMMA_REORDER_INNER_WINDOW;
            opposite_tile = current_row / DMMA_TILE_M;
            current_group =
                (current_col % DMMA_REORDER_INNER_WINDOW) / DMMA_TILE_K;
            proposal_group =
                (proposal_col % DMMA_REORDER_INNER_WINDOW) / DMMA_TILE_K;
        }
        const uint64_t pair =
            static_cast<uint64_t>(static_cast<unsigned int>(window)) *
                static_cast<uint64_t>(
                    static_cast<unsigned int>(opposite_tile_count)) +
            static_cast<uint64_t>(static_cast<unsigned int>(opposite_tile));
        keys[entry] = pair * static_cast<uint64_t>(groups_per_window) +
                      static_cast<uint64_t>(current_group);
        keys[static_cast<std::size_t>(nnz) +
             entry] =
            kProposalTag |
            (pair * static_cast<uint64_t>(groups_per_window) +
             static_cast<uint64_t>(proposal_group));
    }
}

__global__ void count_exact_tiles_per_window_kernel(
    std::size_t unique_count, const uint64_t *unique_keys,
    int opposite_tile_count, int windows, int groups_per_window, int *counts,
    int *span_min, int *span_max)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= unique_count)
        return;
    const uint64_t key = unique_keys[index];
    const int state = (key & kProposalTag) != 0;
    const uint64_t packed = key & ~kProposalTag;
    const uint64_t pair = packed /
                          static_cast<uint64_t>(groups_per_window);
    const int group = static_cast<int>(
        packed % static_cast<uint64_t>(groups_per_window));
    const int opposite_tile = static_cast<int>(
        pair % static_cast<uint64_t>(opposite_tile_count));
    const int window = static_cast<int>(
        pair / static_cast<uint64_t>(
                   static_cast<unsigned int>(opposite_tile_count)));
    if (window < windows)
    {
        atomicAdd(counts + state * windows + window, 1);
        const int group_index =
            (state * windows + window) * groups_per_window + group;
        atomicMin(span_min + group_index, opposite_tile);
        atomicMax(span_max + group_index, opposite_tile);
    }
}

__global__ void finalize_fanout_proxy_kernel(
    int windows, int groups_per_window, const int *span_min,
    const int *span_max, int *fanout)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= 2 * windows)
        return;
    int proxy = 0;
    for (int group = 0; group < groups_per_window; ++group)
    {
        const int group_index = index * groups_per_window + group;
        if (span_max[group_index] >= 0)
            proxy += span_max[group_index] - span_min[group_index] + 1;
    }
    fanout[index] = proxy;
}

template <int Window>
__global__ void commit_balanced_windows_kernel(
    int count, const int *counts, const int *fanout,
    const int *proposal_new_to_old,
    int *current_new_to_old, unsigned long long *accepted,
    unsigned long long *tile_reduction,
    unsigned long long *fanout_before,
    unsigned long long *fanout_after)
{
    if (threadIdx.x != 0)
        return;
    const int window = static_cast<int>(blockIdx.x);
    const int begin = window * Window;
    if (begin >= count)
        return;
    const int windows = count / Window + (count % Window != 0);
    const int current_tiles = counts[window];
    const int proposal_tiles = counts[windows + window];
    const int current_fanout = fanout[window];
    const int proposal_fanout = fanout[windows + window];
    const bool accept = proposal_tiles < current_tiles ||
                        (proposal_tiles == current_tiles &&
                         proposal_fanout < current_fanout);
    atomicAdd(fanout_before,
              static_cast<unsigned long long>(current_fanout));
    atomicAdd(fanout_after,
              static_cast<unsigned long long>(
                  accept ? proposal_fanout : current_fanout));
    if (!accept)
        return;
    const int end = begin + min(Window, count - begin);
    for (int next = begin; next < end; ++next)
        current_new_to_old[next] = proposal_new_to_old[next];
    atomicAdd(accepted, 1ull);
    if (current_tiles > proposal_tiles)
        atomicAdd(tile_reduction,
                  static_cast<unsigned long long>(current_tiles -
                                                  proposal_tiles));
}

struct ActivePrefix
{
    const int *old_to_new;
    __host__ __device__ int operator()(const DmmaAxisProfile &profile) const
    {
        return profile.degree == 0 ? 0 : old_to_new[profile.id] + 1;
    }
};

template <bool ReorderRows, int Window>
static inline bool exact_monotone_commit(
    int rows, int cols, int nnz, const int *d_row_ptr,
    const int *d_col_idx,
    thrust::device_vector<int> &current_row_new_to_old,
    thrust::device_vector<int> &current_row_old_to_new,
    thrust::device_vector<int> &proposal_row_new_to_old,
    thrust::device_vector<int> &proposal_row_old_to_new,
    thrust::device_vector<int> &current_inner_new_to_old,
    thrust::device_vector<int> &current_inner_old_to_new,
    thrust::device_vector<int> &proposal_inner_new_to_old,
    thrust::device_vector<int> &proposal_inner_old_to_new,
    thrust::device_vector<uint64_t> &exact_keys,
    thrust::device_vector<int> &exact_counts,
    thrust::device_vector<int> &exact_span_min,
    thrust::device_vector<int> &exact_span_max,
    thrust::device_vector<int> &exact_fanout,
    unsigned long long *d_accepted,
    unsigned long long *d_tile_reduction,
    unsigned long long *d_fanout_before,
    unsigned long long *d_fanout_after,
    cudaStream_t stream)
{
    auto policy = thrust::cuda::par.on(stream);
    const int axis_count = ReorderRows ? rows : cols;
    const int windows = axis_count == 0
                            ? 0
                            : ceil_div_nonnegative(axis_count, Window);
    const int opposite_tiles = ReorderRows
                                   ? ceil_div_nonnegative(cols, DMMA_TILE_K)
                                   : ceil_div_nonnegative(rows, DMMA_TILE_M);
    constexpr int group = ReorderRows ? DMMA_TILE_M : DMMA_TILE_K;
    constexpr int groups_per_window = Window / group;
    if (windows == 0)
        return true;
    thrust::fill(policy, exact_counts.begin(),
                 exact_counts.begin() + static_cast<std::size_t>(2) * windows,
                 0);
    const std::size_t group_slots = static_cast<std::size_t>(2) * windows *
                                    groups_per_window;
    thrust::fill(policy, exact_span_min.begin(),
                 exact_span_min.begin() + group_slots, INT_MAX);
    thrust::fill(policy, exact_span_max.begin(),
                 exact_span_max.begin() + group_slots, -1);
    std::size_t unique_count = 0;
    if (nnz > 0)
    {
        generate_exact_tile_keys_kernel<ReorderRows>
            <<<rows, kThreads, 0, stream>>>(
                rows, nnz, d_row_ptr, d_col_idx,
                thrust::raw_pointer_cast(current_row_old_to_new.data()),
                thrust::raw_pointer_cast(proposal_row_old_to_new.data()),
                thrust::raw_pointer_cast(current_inner_old_to_new.data()),
                thrust::raw_pointer_cast(proposal_inner_old_to_new.data()),
                opposite_tiles, groups_per_window,
                thrust::raw_pointer_cast(exact_keys.data()));
        if (!cuda_ok(cudaGetLastError(), "generate exact A-tile keys"))
            return false;
        const std::size_t key_count = static_cast<std::size_t>(2) * nnz;
        thrust::sort(policy, exact_keys.begin(),
                     exact_keys.begin() + key_count);
        auto unique_end = thrust::unique(
            policy, exact_keys.begin(), exact_keys.begin() + key_count);
        unique_count = static_cast<std::size_t>(
            unique_end - exact_keys.begin());
        const int unique_blocks = static_cast<int>(
            unique_count / kThreads + (unique_count % kThreads != 0));
        count_exact_tiles_per_window_kernel
            <<<unique_blocks, kThreads, 0, stream>>>(
                unique_count, thrust::raw_pointer_cast(exact_keys.data()),
                opposite_tiles, windows, groups_per_window,
                thrust::raw_pointer_cast(exact_counts.data()),
                thrust::raw_pointer_cast(exact_span_min.data()),
                thrust::raw_pointer_cast(exact_span_max.data()));
        if (!cuda_ok(cudaGetLastError(), "count exact A tiles per window"))
            return false;
    }
    finalize_fanout_proxy_kernel
        <<<blocks_for(2 * windows), kThreads, 0, stream>>>(
            windows, groups_per_window,
            thrust::raw_pointer_cast(exact_span_min.data()),
            thrust::raw_pointer_cast(exact_span_max.data()),
            thrust::raw_pointer_cast(exact_fanout.data()));
    if (!cuda_ok(cudaGetLastError(), "finalize BGRF fanout proxy"))
        return false;
    if constexpr (ReorderRows)
    {
        commit_balanced_windows_kernel<Window>
            <<<windows, 1, 0, stream>>>(
                rows, thrust::raw_pointer_cast(exact_counts.data()),
                thrust::raw_pointer_cast(exact_fanout.data()),
                thrust::raw_pointer_cast(proposal_row_new_to_old.data()),
                thrust::raw_pointer_cast(current_row_new_to_old.data()),
                d_accepted, d_tile_reduction, d_fanout_before,
                d_fanout_after);
        if (!cuda_ok(cudaGetLastError(), "commit exact row windows"))
            return false;
        invert_permutation_kernel<<<blocks_for(rows), kThreads, 0, stream>>>(
            rows,
            thrust::raw_pointer_cast(current_row_new_to_old.data()),
            thrust::raw_pointer_cast(current_row_old_to_new.data()));
    }
    else
    {
        commit_balanced_windows_kernel<Window>
            <<<windows, 1, 0, stream>>>(
                cols, thrust::raw_pointer_cast(exact_counts.data()),
                thrust::raw_pointer_cast(exact_fanout.data()),
                thrust::raw_pointer_cast(proposal_inner_new_to_old.data()),
                thrust::raw_pointer_cast(current_inner_new_to_old.data()),
                d_accepted, d_tile_reduction, d_fanout_before,
                d_fanout_after);
        if (!cuda_ok(cudaGetLastError(), "commit exact inner windows"))
            return false;
        invert_permutation_kernel<<<blocks_for(cols), kThreads, 0, stream>>>(
            cols,
            thrust::raw_pointer_cast(current_inner_new_to_old.data()),
            thrust::raw_pointer_cast(current_inner_old_to_new.data()));
    }
    return cuda_ok(cudaGetLastError(), "invert exact accepted permutation");
}

static inline bool copy_device_permutation_to_owned(
    const thrust::device_vector<int> &new_to_old,
    const thrust::device_vector<int> &old_to_new,
    int **h_new_to_old, int **h_old_to_new,
    int **d_new_to_old, int **d_old_to_new, cudaStream_t stream)
{
    const std::size_t count = new_to_old.size();
    *h_new_to_old = allocate_host<int>(count);
    *h_old_to_new = allocate_host<int>(count);
    if (count != 0 && (*h_new_to_old == nullptr || *h_old_to_new == nullptr))
        return false;
    if (!allocate_device(d_new_to_old, count,
                         "allocate unified new-to-old permutation") ||
        !allocate_device(d_old_to_new, count,
                         "allocate unified old-to-new permutation"))
        return false;
    if (count == 0)
        return true;
    const std::size_t bytes = count * sizeof(int);
    return cuda_ok(cudaMemcpyAsync(*d_new_to_old,
                                   thrust::raw_pointer_cast(new_to_old.data()),
                                   bytes, cudaMemcpyDeviceToDevice, stream),
                   "copy unified new-to-old device permutation") &&
           cuda_ok(cudaMemcpyAsync(*d_old_to_new,
                                   thrust::raw_pointer_cast(old_to_new.data()),
                                   bytes, cudaMemcpyDeviceToDevice, stream),
                   "copy unified old-to-new device permutation") &&
           cuda_ok(cudaMemcpyAsync(*h_new_to_old,
                                   thrust::raw_pointer_cast(new_to_old.data()),
                                   bytes, cudaMemcpyDeviceToHost, stream),
                   "copy unified new-to-old host permutation") &&
           cuda_ok(cudaMemcpyAsync(*h_old_to_new,
                                   thrust::raw_pointer_cast(old_to_new.data()),
                                   bytes, cudaMemcpyDeviceToHost, stream),
                   "copy unified old-to-new host permutation");
}

static inline bool read_external_axis_order(
    const char *path, int count, std::vector<int> *new_to_old,
    std::vector<int> *old_to_new)
{
    if (path == nullptr || count < 0 || new_to_old == nullptr ||
        old_to_new == nullptr)
        return false;

    std::ifstream file(path);
    if (!file.is_open())
    {
        std::fprintf(stderr, "Unable to open external reorder file %s.\n",
                     path);
        return false;
    }

    new_to_old->assign(static_cast<std::size_t>(count), -1);
    old_to_new->assign(static_cast<std::size_t>(count), -1);
    std::vector<unsigned char> seen(static_cast<std::size_t>(count), 0);
    std::string token;
    for (int next = 0; next < count; ++next)
    {
        if (!(file >> token))
        {
            std::fprintf(stderr,
                         "External reorder file %s is too short: expected "
                         "%d entries, found %d.\n",
                         path, count, next);
            return false;
        }
        char *end = nullptr;
        errno = 0;
        const long long parsed = std::strtoll(token.c_str(), &end, 10);
        if (errno == ERANGE || end == token.c_str() || *end != '\0')
        {
            std::fprintf(stderr,
                         "External reorder file %s has a non-integer token "
                         "at new id %d: %s.\n",
                         path, next, token.c_str());
            return false;
        }
        if (parsed < 0 || parsed >= count)
        {
            std::fprintf(stderr,
                         "External reorder file %s maps new id %d to "
                         "out-of-range old id %lld (axis size %d).\n",
                         path, next, parsed, count);
            return false;
        }
        const int old = static_cast<int>(parsed);
        if (seen[static_cast<std::size_t>(old)] != 0)
        {
            std::fprintf(stderr,
                         "External reorder file %s contains duplicate old "
                         "id %d (at new id %d).\n",
                         path, old, next);
            return false;
        }
        seen[static_cast<std::size_t>(old)] = 1;
        (*new_to_old)[static_cast<std::size_t>(next)] = old;
        (*old_to_new)[static_cast<std::size_t>(old)] = next;
    }

    if (file >> token)
    {
        std::fprintf(stderr,
                     "External reorder file %s has an extra token after "
                     "%d entries: %s.\n",
                     path, count, token.c_str());
        return false;
    }
    if (file.bad())
    {
        std::fprintf(stderr, "I/O failure while reading external reorder "
                             "file %s.\n",
                     path);
        return false;
    }
    return true;
}

static inline bool copy_host_permutation_to_owned(
    const std::vector<int> &new_to_old,
    const std::vector<int> &old_to_new, int **h_new_to_old,
    int **h_old_to_new, int **d_new_to_old, int **d_old_to_new,
    cudaStream_t stream)
{
    if (new_to_old.size() != old_to_new.size() ||
        h_new_to_old == nullptr || h_old_to_new == nullptr ||
        d_new_to_old == nullptr || d_old_to_new == nullptr)
        return false;
    const std::size_t count = new_to_old.size();
    *h_new_to_old = allocate_host<int>(count);
    *h_old_to_new = allocate_host<int>(count);
    if (count != 0 && (*h_new_to_old == nullptr || *h_old_to_new == nullptr))
        return false;
    if (!allocate_device(d_new_to_old, count,
                         "allocate external new-to-old permutation") ||
        !allocate_device(d_old_to_new, count,
                         "allocate external old-to-new permutation"))
        return false;
    if (count == 0)
        return true;

    const std::size_t bytes = count * sizeof(int);
    std::memcpy(*h_new_to_old, new_to_old.data(), bytes);
    std::memcpy(*h_old_to_new, old_to_new.data(), bytes);
    return cuda_ok(cudaMemcpyAsync(*d_new_to_old, new_to_old.data(), bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "upload external new-to-old permutation") &&
           cuda_ok(cudaMemcpyAsync(*d_old_to_new, old_to_new.data(), bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "upload external old-to-new permutation");
}

static inline void summarize_permutation(const int *old_to_new, int count,
                                         unsigned long long *moved,
                                         unsigned long long *displacement)
{
    *moved = 0;
    *displacement = 0;
    for (int old = 0; old < count; ++old)
    {
        const int next = old_to_new[old];
        *moved += next != old;
        *displacement += static_cast<unsigned long long>(
            next >= old ? next - old : old - next);
    }
}

} // namespace dmma_reorder_detail

/* Import two canonical external axis orders.  Each file contains exactly one
 * zero-based old id for every new id (new-to-old semantics).  No proposal,
 * guard, or identity fallback is applied: a valid external permutation is
 * always used in full. */
static inline bool build_external_dmma_reorder_plan(
    const SMatrix &host, const char *row_order_path,
    const char *inner_order_path, const char *reorder_name,
    DmmaReorderPlan *out, cudaStream_t stream = 0)
{
    using namespace dmma_reorder_detail;
    if (out == nullptr || row_order_path == nullptr ||
        inner_order_path == nullptr || reorder_name == nullptr ||
        reorder_name[0] == '\0' || std::strlen(reorder_name) > 63 ||
        host.m < 0 || host.n < 0 ||
        host.nnz < 0 || host.rowpointer == nullptr ||
        (host.nnz > 0 && host.columnindex == nullptr))
        return false;

    if (host.rowpointer[0] != 0 || host.rowpointer[host.m] != host.nnz)
    {
        std::fprintf(stderr,
                     "Host CSR endpoints are invalid while importing an "
                     "external reorder.\n");
        return false;
    }
    for (int row = 0; row < host.m; ++row)
    {
        const int begin = host.rowpointer[row];
        const int end = host.rowpointer[row + 1];
        if (begin < 0 || begin > end || end > host.nnz)
        {
            std::fprintf(stderr,
                         "Host CSR row %d is invalid while importing an "
                         "external reorder.\n",
                         row);
            return false;
        }
    }

    DmmaReorderPlan result;
    result.rows = host.m;
    result.cols = host.n;
    result.nnz = host.nnz;
    result.unified = false;
    result.kind = DMMA_REORDER_EXTERNAL;
    std::snprintf(result.algorithm, sizeof(result.algorithm),
                  "external:%s", reorder_name);
    result.sweeps = 0;
    result.row_window = 0;
    result.inner_window = 0;

    try
    {
        std::vector<int> row_new_to_old;
        std::vector<int> row_old_to_new;
        std::vector<int> inner_new_to_old;
        std::vector<int> inner_old_to_new;
        if (!read_external_axis_order(row_order_path, host.m,
                                      &row_new_to_old,
                                      &row_old_to_new))
            return false;
        if (host.m == host.n &&
            std::strcmp(row_order_path, inner_order_path) == 0)
        {
            /* RW-SpGEMM's square-matrix adapters intentionally use one P
             * for both axes.  Parse that potentially large file only once;
             * the two plan axes still receive independent owned arrays. */
            inner_new_to_old = row_new_to_old;
            inner_old_to_new = row_old_to_new;
        }
        else if (!read_external_axis_order(inner_order_path, host.n,
                                           &inner_new_to_old,
                                           &inner_old_to_new))
        {
            return false;
        }

        int active_rows = 0;
        int active_inner = 0;
        for (int old_row = 0; old_row < host.m; ++old_row)
        {
            const int begin = host.rowpointer[old_row];
            const int end = host.rowpointer[old_row + 1];
            if (begin != end)
                active_rows = std::max(
                    active_rows,
                    row_old_to_new[static_cast<std::size_t>(old_row)] + 1);
            for (int entry = begin; entry < end; ++entry)
            {
                const int old_col = host.columnindex[entry];
                if (old_col < 0 || old_col >= host.n)
                {
                    std::fprintf(stderr,
                                 "Host CSR column %d is invalid while "
                                 "importing an external reorder.\n",
                                 old_col);
                    return false;
                }
                active_inner = std::max(
                    active_inner,
                    inner_old_to_new[static_cast<std::size_t>(old_col)] + 1);
            }
        }
        result.active_rows = active_rows;
        result.active_inner = active_inner;

        if (!copy_host_permutation_to_owned(
                row_new_to_old, row_old_to_new,
                &result.h_row_new_to_old, &result.h_row_old_to_new,
                &result.d_row_new_to_old, &result.d_row_old_to_new,
                stream) ||
            !copy_host_permutation_to_owned(
                inner_new_to_old, inner_old_to_new,
                &result.h_inner_new_to_old,
                &result.h_inner_old_to_new,
                &result.d_inner_new_to_old,
                &result.d_inner_old_to_new, stream) ||
            !cuda_ok(cudaStreamSynchronize(stream),
                     "complete external permutation import"))
        {
            destroy_dmma_reorder_plan(&result);
            return false;
        }

        summarize_permutation(result.h_row_old_to_new, host.m,
                              &result.moved_rows,
                              &result.row_displacement);
        summarize_permutation(result.h_inner_old_to_new, host.n,
                              &result.moved_inner,
                              &result.inner_displacement);
    }
    catch (const std::bad_alloc &)
    {
        std::fprintf(stderr,
                     "External reorder permutation allocation failed.\n");
        destroy_dmma_reorder_plan(&result);
        return false;
    }

    destroy_dmma_reorder_plan(out);
    *out = result;
    return true;
}

#if 0
static inline bool build_legacy_dmma_reorder_plan(
    int rows, int cols, int nnz, const int *d_row_ptr, const int *d_col_idx,
    int dense_threshold, DmmaReorderPlan *out, cudaStream_t stream = 0)
{
    using namespace dmma_reorder_detail;
    if (out == nullptr || rows < 0 || cols < 0 || nnz < 0 ||
        dense_threshold < 1 || dense_threshold > DMMA_INPUT_ELEMS ||
        d_row_ptr == nullptr || (nnz > 0 && d_col_idx == nullptr))
        return false;

    DmmaReorderPlan result;
    result.rows = rows;
    result.cols = cols;
    result.nnz = nnz;
    result.unified = true;
    result.kind = DMMA_REORDER_UNIFIED;
    std::snprintf(result.algorithm, sizeof(result.algorithm), "unified");
    result.sweeps = DMMA_REORDER_SWEEPS;

    try
    {
        auto policy = thrust::cuda::par.on(stream);
        thrust::device_vector<int> row_new_to_old(rows);
        thrust::device_vector<int> row_old_to_new(rows);
        thrust::device_vector<int> proposal_row_new_to_old(rows);
        thrust::device_vector<int> proposal_row_old_to_new(rows);
        thrust::device_vector<int> local_row_new_to_old(rows);
        thrust::device_vector<int> inner_new_to_old(cols);
        thrust::device_vector<int> inner_old_to_new(cols);
        thrust::device_vector<int> rcm_row_old_to_new(rows);
        thrust::device_vector<int> rcm_inner_old_to_new(cols);
        thrust::device_vector<int> proposal_inner_new_to_old(cols);
        thrust::device_vector<int> proposal_inner_old_to_new(cols);
        thrust::device_vector<int> local_inner_new_to_old(cols);
        thrust::device_vector<DmmaAxisProfile> row_profiles(rows);
        thrust::device_vector<DmmaAxisProfile> inner_profiles(cols);
        const int row_tile_blocks = ceil_div_nonnegative(rows, DMMA_TILE_M);
        const int inner_tile_blocks =
            ceil_div_nonnegative(cols, DMMA_TILE_K);
        thrust::device_vector<DmmaTileBlockProfile> row_blocks(
            row_tile_blocks);
        thrust::device_vector<DmmaTileBlockProfile> inner_blocks(
            inner_tile_blocks);
        thrust::device_vector<uint64_t> exact_keys(
            static_cast<std::size_t>(2) * static_cast<std::size_t>(nnz));
        const int max_windows = std::max(
            ceil_div_nonnegative(rows, DMMA_REORDER_ROW_WINDOW),
            ceil_div_nonnegative(cols, DMMA_REORDER_INNER_WINDOW));
        thrust::device_vector<int> exact_counts(
            static_cast<std::size_t>(2) * max_windows);
        thrust::device_vector<unsigned long long> exact_stats(4, 0);
        unsigned long long h_exact_stats[4] = {0, 0, 0, 0};
        thrust::sequence(policy, row_new_to_old.begin(), row_new_to_old.end());
        thrust::sequence(policy, row_old_to_new.begin(), row_old_to_new.end());
        thrust::sequence(policy, proposal_row_new_to_old.begin(),
                         proposal_row_new_to_old.end());
        thrust::sequence(policy, proposal_row_old_to_new.begin(),
                         proposal_row_old_to_new.end());
        thrust::sequence(policy, inner_new_to_old.begin(), inner_new_to_old.end());
        thrust::sequence(policy, inner_old_to_new.begin(), inner_old_to_new.end());
        thrust::sequence(policy, proposal_inner_new_to_old.begin(),
                         proposal_inner_new_to_old.end());
        thrust::sequence(policy, proposal_inner_old_to_new.begin(),
                         proposal_inner_old_to_new.end());

        for (int sweep = 0; sweep < DMMA_REORDER_SWEEPS; ++sweep)
        {
            if (rows > 0)
            {
                extract_row_profiles_kernel<<<blocks_for(rows), kThreads, 0,
                                                stream>>>(
                    rows, d_row_ptr, d_col_idx,
                    thrust::raw_pointer_cast(inner_old_to_new.data()),
                    thrust::raw_pointer_cast(row_profiles.data()));
                if (!cuda_ok(cudaGetLastError(), "extract row profiles"))
                    goto failure;
                thrust::sort(policy, row_profiles.begin(), row_profiles.end(),
                             CurrentPositionLess{
                                 thrust::raw_pointer_cast(
                                     row_old_to_new.data())});
                build_coarse_blocks_kernel<DMMA_TILE_M>
                    <<<row_tile_blocks, 1, 0, stream>>>(
                        thrust::raw_pointer_cast(row_profiles.data()), rows,
                        thrust::raw_pointer_cast(local_row_new_to_old.data()),
                        thrust::raw_pointer_cast(row_blocks.data()));
                if (!cuda_ok(cudaGetLastError(),
                             "build coarse row-tile profiles"))
                    goto failure;
                const int complete_row_blocks = rows / DMMA_TILE_M;
                thrust::sort(policy, row_blocks.begin(),
                             row_blocks.begin() + complete_row_blocks,
                             TileBlockLess());
                assemble_coarse_permutation_kernel<DMMA_TILE_M>
                    <<<blocks_for(rows), kThreads, 0, stream>>>(
                        rows, complete_row_blocks,
                        thrust::raw_pointer_cast(row_blocks.data()),
                        thrust::raw_pointer_cast(local_row_new_to_old.data()),
                        thrust::raw_pointer_cast(
                            proposal_row_new_to_old.data()));
                if (!cuda_ok(cudaGetLastError(),
                             "assemble coarse row permutation"))
                    goto failure;
                invert_permutation_kernel<<<blocks_for(rows), kThreads, 0,
                                             stream>>>(
                    rows,
                    thrust::raw_pointer_cast(
                        proposal_row_new_to_old.data()),
                    thrust::raw_pointer_cast(
                        proposal_row_old_to_new.data()));
                if (!cuda_ok(cudaGetLastError(),
                             "invert coarse row permutation"))
                    goto failure;
                row_new_to_old.swap(proposal_row_new_to_old);
                row_old_to_new.swap(proposal_row_old_to_new);

                /* Re-establish physical order after the tile-preserving
                 * coarse move, then form one deterministic fine proposal
                 * inside each adjacent pair of row tiles. */
                thrust::sort(policy, row_profiles.begin(), row_profiles.end(),
                             CurrentPositionLess{
                                 thrust::raw_pointer_cast(
                                     row_old_to_new.data())});
                const int windows = ceil_div_nonnegative(
                    rows, DMMA_REORDER_ROW_WINDOW);
                pack_windows_kernel<DMMA_REORDER_ROW_WINDOW, DMMA_TILE_M>
                    <<<windows, 1, 0, stream>>>(
                        thrust::raw_pointer_cast(row_profiles.data()), rows,
                        thrust::raw_pointer_cast(
                            proposal_row_new_to_old.data()));
                if (!cuda_ok(cudaGetLastError(), "pack 8-row windows"))
                    goto failure;
                invert_permutation_kernel<<<blocks_for(rows), kThreads, 0,
                                             stream>>>(
                    rows,
                    thrust::raw_pointer_cast(
                        proposal_row_new_to_old.data()),
                    thrust::raw_pointer_cast(
                        proposal_row_old_to_new.data()));
                if (!cuda_ok(cudaGetLastError(),
                             "invert proposed row permutation") ||
                    !exact_monotone_commit<true,
                                           DMMA_REORDER_ROW_WINDOW>(
                        rows, cols, nnz, d_row_ptr, d_col_idx,
                        row_new_to_old, row_old_to_new,
                        proposal_row_new_to_old,
                        proposal_row_old_to_new, inner_new_to_old,
                        inner_old_to_new, proposal_inner_new_to_old,
                        proposal_inner_old_to_new, exact_keys, exact_counts,
                        thrust::raw_pointer_cast(exact_stats.data()),
                        thrust::raw_pointer_cast(exact_stats.data()) + 1,
                        stream))
                    goto failure;
            }

            if (cols > 0)
            {
                initialize_profiles_kernel<<<blocks_for(cols), kThreads, 0,
                                               stream>>>(
                    cols, thrust::raw_pointer_cast(inner_profiles.data()));
                if (!cuda_ok(cudaGetLastError(), "initialize inner profiles"))
                    goto failure;
                if (rows > 0)
                {
                    accumulate_inner_profiles_kernel
                        <<<blocks_for(rows), kThreads, 0, stream>>>(
                            rows, d_row_ptr, d_col_idx,
                            thrust::raw_pointer_cast(row_old_to_new.data()),
                            thrust::raw_pointer_cast(inner_profiles.data()));
                    if (!cuda_ok(cudaGetLastError(),
                                 "accumulate inner profiles"))
                        goto failure;
                }
                finalize_profiles_kernel<<<blocks_for(cols), kThreads, 0,
                                             stream>>>(
                    cols, thrust::raw_pointer_cast(inner_profiles.data()));
                if (!cuda_ok(cudaGetLastError(), "finalize inner profiles"))
                    goto failure;
                thrust::sort(policy, inner_profiles.begin(),
                             inner_profiles.end(),
                             CurrentPositionLess{
                                 thrust::raw_pointer_cast(
                                     inner_old_to_new.data())});
                build_coarse_blocks_kernel<DMMA_TILE_K>
                    <<<inner_tile_blocks, 1, 0, stream>>>(
                        thrust::raw_pointer_cast(inner_profiles.data()), cols,
                        thrust::raw_pointer_cast(
                            local_inner_new_to_old.data()),
                        thrust::raw_pointer_cast(inner_blocks.data()));
                if (!cuda_ok(cudaGetLastError(),
                             "build coarse inner-tile profiles"))
                    goto failure;
                const int complete_inner_blocks = cols / DMMA_TILE_K;
                thrust::sort(policy, inner_blocks.begin(),
                             inner_blocks.begin() + complete_inner_blocks,
                             TileBlockLess());
                assemble_coarse_permutation_kernel<DMMA_TILE_K>
                    <<<blocks_for(cols), kThreads, 0, stream>>>(
                        cols, complete_inner_blocks,
                        thrust::raw_pointer_cast(inner_blocks.data()),
                        thrust::raw_pointer_cast(
                            local_inner_new_to_old.data()),
                        thrust::raw_pointer_cast(
                            proposal_inner_new_to_old.data()));
                if (!cuda_ok(cudaGetLastError(),
                             "assemble coarse inner permutation"))
                    goto failure;
                invert_permutation_kernel<<<blocks_for(cols), kThreads, 0,
                                             stream>>>(
                    cols,
                    thrust::raw_pointer_cast(
                        proposal_inner_new_to_old.data()),
                    thrust::raw_pointer_cast(
                        proposal_inner_old_to_new.data()));
                if (!cuda_ok(cudaGetLastError(),
                             "invert coarse inner permutation"))
                    goto failure;
                inner_new_to_old.swap(proposal_inner_new_to_old);
                inner_old_to_new.swap(proposal_inner_old_to_new);

                thrust::sort(policy, inner_profiles.begin(),
                             inner_profiles.end(),
                             CurrentPositionLess{
                                 thrust::raw_pointer_cast(
                                     inner_old_to_new.data())});
                const int windows = ceil_div_nonnegative(
                    cols, DMMA_REORDER_INNER_WINDOW);
                pack_windows_kernel<DMMA_REORDER_INNER_WINDOW, DMMA_TILE_K>
                    <<<windows, 1, 0, stream>>>(
                        thrust::raw_pointer_cast(inner_profiles.data()), cols,
                        thrust::raw_pointer_cast(
                            proposal_inner_new_to_old.data()));
                if (!cuda_ok(cudaGetLastError(), "pack 4-column windows"))
                    goto failure;
                invert_permutation_kernel<<<blocks_for(cols), kThreads, 0,
                                             stream>>>(
                    cols,
                    thrust::raw_pointer_cast(
                        proposal_inner_new_to_old.data()),
                    thrust::raw_pointer_cast(
                        proposal_inner_old_to_new.data()));
                if (!cuda_ok(cudaGetLastError(),
                             "invert proposed inner permutation") ||
                    !exact_monotone_commit<false,
                                           DMMA_REORDER_INNER_WINDOW>(
                        rows, cols, nnz, d_row_ptr, d_col_idx,
                        row_new_to_old, row_old_to_new,
                        proposal_row_new_to_old,
                        proposal_row_old_to_new, inner_new_to_old,
                        inner_old_to_new, proposal_inner_new_to_old,
                        proposal_inner_old_to_new, exact_keys, exact_counts,
                        thrust::raw_pointer_cast(exact_stats.data()) + 2,
                        thrust::raw_pointer_cast(exact_stats.data()) + 3,
                        stream))
                    goto failure;
            }
        }

        result.active_rows = rows == 0
                                 ? 0
                                 : thrust::transform_reduce(
                                       policy, row_profiles.begin(),
                                       row_profiles.end(),
                                       ActivePrefix{thrust::raw_pointer_cast(
                                           row_old_to_new.data())},
                                       0, thrust::maximum<int>());
        result.active_inner = cols == 0
                                  ? 0
                                  : thrust::transform_reduce(
                                        policy, inner_profiles.begin(),
                                        inner_profiles.end(),
                                        ActivePrefix{thrust::raw_pointer_cast(
                                            inner_old_to_new.data())},
                                        0, thrust::maximum<int>());
        if (!cuda_ok(cudaMemcpyAsync(
                         h_exact_stats,
                         thrust::raw_pointer_cast(exact_stats.data()),
                         sizeof(h_exact_stats), cudaMemcpyDeviceToHost,
                         stream),
                     "copy exact-monotone statistics") ||
            !copy_device_permutation_to_owned(
                row_new_to_old, row_old_to_new,
                &result.h_row_new_to_old, &result.h_row_old_to_new,
                &result.d_row_new_to_old, &result.d_row_old_to_new, stream) ||
            !copy_device_permutation_to_owned(
                inner_new_to_old, inner_old_to_new,
                &result.h_inner_new_to_old, &result.h_inner_old_to_new,
                &result.d_inner_new_to_old, &result.d_inner_old_to_new,
                stream) ||
            !cuda_ok(cudaStreamSynchronize(stream),
                     "complete unified permutation"))
            goto failure;

        result.accepted_row_windows = h_exact_stats[0];
        result.row_tile_reduction = h_exact_stats[1];
        result.accepted_inner_windows = h_exact_stats[2];
        result.inner_tile_reduction = h_exact_stats[3];

        summarize_permutation(result.h_row_old_to_new, rows,
                              &result.moved_rows,
                              &result.row_displacement);
        summarize_permutation(result.h_inner_old_to_new, cols,
                              &result.moved_inner,
                              &result.inner_displacement);
    }
    catch (const std::bad_alloc &)
    {
        std::fprintf(stderr, "Unified DMMA reorder allocation failed.\n");
        goto failure;
    }
    catch (const thrust::system_error &error)
    {
        std::fprintf(stderr, "Unified DMMA reorder Thrust failure: %s\n",
                     error.what());
        goto failure;
    }

    destroy_dmma_reorder_plan(out);
    *out = result;
    return true;

failure:
    destroy_dmma_reorder_plan(&result);
    return false;
}
#endif

static inline bool build_dmma_reorder_plan(
    int rows, int cols, int nnz, const int *d_row_ptr, const int *d_col_idx,
    int dense_threshold, DmmaReorderPlan *out, cudaStream_t stream = 0,
    const DmmaReorderConfig *config = nullptr)
{
    using namespace dmma_reorder_detail;
    if (out == nullptr || rows < 0 || cols < 0 || nnz < 0 ||
        dense_threshold < 1 || dense_threshold > DMMA_INPUT_ELEMS ||
        d_row_ptr == nullptr || (nnz > 0 && d_col_idx == nullptr))
        return false;

    const DmmaReorderVariant variant =
        config == nullptr ? DMMA_REORDER_VARIANT_FULL : config->variant;
    bool enable_coarse = false;
    bool enable_row_fine = false;
    bool enable_inner_fine = false;
    bool enable_exact_guard = false;
    if (!resolve_dmma_reorder_variant(
            variant, &enable_coarse, &enable_row_fine,
            &enable_inner_fine, &enable_exact_guard))
    {
        std::fprintf(stderr, "Invalid BGRF reorder variant.\n");
        return false;
    }
    const bool enable_any_fine = enable_row_fine || enable_inner_fine;

    DmmaReorderPlan result;
    result.rows = rows;
    result.cols = cols;
    result.nnz = nnz;
    result.unified = true;
    result.kind = DMMA_REORDER_UNIFIED;
    result.variant = variant;
    if (variant == DMMA_REORDER_VARIANT_FULL)
        std::snprintf(result.algorithm, sizeof(result.algorithm), "bgrf-v1");
    else
        std::snprintf(result.algorithm, sizeof(result.algorithm),
                      "bgrf-v1-%s", dmma_reorder_variant_name(variant));
    result.sweeps = DMMA_REORDER_SWEEPS;
    const auto reorder_start = std::chrono::steady_clock::now();

    try
    {
        auto policy = thrust::cuda::par.on(stream);
        thrust::device_vector<int> row_new_to_old(rows);
        thrust::device_vector<int> row_old_to_new(rows);
        thrust::device_vector<int> inner_new_to_old(cols);
        thrust::device_vector<int> inner_old_to_new(cols);
        thrust::device_vector<int> rcm_row_old_to_new(
            enable_coarse ? rows : 0);
        thrust::device_vector<int> rcm_inner_old_to_new(
            enable_coarse ? cols : 0);
        std::size_t coarse_peak = 0;
        if (enable_coarse)
        {
            if (!build_bgrf_global_coarse(
                    rows, cols, nnz, d_row_ptr, d_col_idx,
                    row_new_to_old, row_old_to_new, inner_new_to_old,
                    inner_old_to_new, rcm_row_old_to_new,
                    rcm_inner_old_to_new, &result.active_rows,
                    &result.active_inner, &result.coarse_components,
                    &result.coarse_levels, &coarse_peak, stream))
                goto failure;
        }
        else
        {
            thrust::sequence(policy, row_new_to_old.begin(),
                             row_new_to_old.end());
            thrust::sequence(policy, row_old_to_new.begin(),
                             row_old_to_new.end());
            thrust::sequence(policy, inner_new_to_old.begin(),
                             inner_new_to_old.end());
            thrust::sequence(policy, inner_old_to_new.begin(),
                             inner_old_to_new.end());
        }
        const bool need_proposals = enable_coarse || enable_any_fine;
        thrust::device_vector<int> proposal_row_new_to_old(
            need_proposals ? rows : 0);
        thrust::device_vector<int> proposal_row_old_to_new(
            need_proposals ? rows : 0);
        thrust::device_vector<int> proposal_inner_new_to_old(
            need_proposals ? cols : 0);
        thrust::device_vector<int> proposal_inner_old_to_new(
            need_proposals ? cols : 0);
        thrust::device_vector<DmmaAxisProfile> row_profiles(
            enable_row_fine ? rows : 0);
        thrust::device_vector<DmmaAxisProfile> inner_profiles(
            enable_inner_fine ? cols : 0);
        thrust::device_vector<uint64_t> exact_keys(
            enable_exact_guard && need_proposals
                ? static_cast<std::size_t>(2) *
                      static_cast<std::size_t>(nnz)
                : 0);
        const int max_windows = std::max(
            enable_row_fine
                ? ceil_div_nonnegative(rows, DMMA_REORDER_ROW_WINDOW)
                : 0,
            enable_inner_fine
                ? ceil_div_nonnegative(cols, DMMA_REORDER_INNER_WINDOW)
                : 0);
        constexpr int max_groups = std::max(
            DMMA_REORDER_ROW_WINDOW / DMMA_TILE_M,
            DMMA_REORDER_INNER_WINDOW / DMMA_TILE_K);
        thrust::device_vector<int> exact_counts(
            enable_exact_guard && enable_any_fine
                ? static_cast<std::size_t>(2) * max_windows
                : 0,
            0);
        thrust::device_vector<int> exact_span_min(
            enable_exact_guard && enable_any_fine
                ? static_cast<std::size_t>(2) * max_windows * max_groups
                : 0,
            INT_MAX);
        thrust::device_vector<int> exact_span_max(
            enable_exact_guard && enable_any_fine
                ? static_cast<std::size_t>(2) * max_windows * max_groups
                : 0,
            -1);
        thrust::device_vector<int> exact_fanout(
            enable_exact_guard && enable_any_fine
                ? static_cast<std::size_t>(2) * max_windows
                : 0,
            0);
        thrust::device_vector<unsigned long long> exact_stats(
            enable_exact_guard && enable_any_fine ? 8 : 0, 0);
        unsigned long long h_exact_stats[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        const std::size_t proposal_bytes =
            (need_proposals ? static_cast<std::size_t>(4) : 0) *
                (static_cast<std::size_t>(rows) +
                 static_cast<std::size_t>(cols)) * sizeof(int);
        const std::size_t rcm_bytes =
            (enable_coarse ? static_cast<std::size_t>(2) : 0) *
                (static_cast<std::size_t>(rows) +
                 static_cast<std::size_t>(cols)) * sizeof(int);
        const std::size_t profile_bytes =
            ((enable_row_fine ? static_cast<std::size_t>(rows) : 0) +
             (enable_inner_fine ? static_cast<std::size_t>(cols) : 0)) *
            sizeof(DmmaAxisProfile);
        const std::size_t exact_key_bytes =
            enable_exact_guard
                ? static_cast<std::size_t>(2) * nnz * sizeof(uint64_t)
                : 0;
        const std::size_t exact_window_bytes =
            enable_exact_guard && enable_any_fine
                ? static_cast<std::size_t>(4) * max_windows * sizeof(int) +
                      static_cast<std::size_t>(4) * max_windows *
                          max_groups * sizeof(int) +
                      static_cast<std::size_t>(8) *
                          sizeof(unsigned long long)
                : 0;
        const std::size_t fine_peak =
            proposal_bytes + rcm_bytes + profile_bytes + exact_key_bytes +
            exact_window_bytes;
        result.reorder_peak_workspace_bytes =
            std::max(coarse_peak, fine_peak);

        if (enable_coarse)
        {
            thrust::copy(policy, rcm_row_old_to_new.begin(),
                         rcm_row_old_to_new.end(),
                         proposal_row_old_to_new.begin());
            thrust::copy(policy, rcm_inner_old_to_new.begin(),
                         rcm_inner_old_to_new.end(),
                         proposal_inner_old_to_new.begin());
            if (rows > 0)
                invert_old_to_new_kernel<<<blocks_for(rows), kThreads, 0,
                                           stream>>>(
                    rows,
                    thrust::raw_pointer_cast(
                        proposal_row_old_to_new.data()),
                    thrust::raw_pointer_cast(
                        proposal_row_new_to_old.data()));
            if (cols > 0)
                invert_old_to_new_kernel<<<blocks_for(cols), kThreads, 0,
                                           stream>>>(
                    cols,
                    thrust::raw_pointer_cast(
                        proposal_inner_old_to_new.data()),
                    thrust::raw_pointer_cast(
                        proposal_inner_new_to_old.data()));
            if (!cuda_ok(cudaGetLastError(),
                         "materialize joint RCM proposal"))
                goto failure;
            if (enable_exact_guard)
            {
                if (!exact_joint_global_commit(
                        rows, cols, nnz, d_row_ptr, d_col_idx,
                        row_new_to_old, row_old_to_new,
                        proposal_row_new_to_old,
                        proposal_row_old_to_new, inner_new_to_old,
                        inner_old_to_new, proposal_inner_new_to_old,
                        proposal_inner_old_to_new, exact_keys,
                        &result.coarse_candidate_accepted,
                        &result.coarse_tile_reduction, stream))
                    goto failure;
            }
            else
            {
                row_new_to_old.swap(proposal_row_new_to_old);
                row_old_to_new.swap(proposal_row_old_to_new);
                inner_new_to_old.swap(proposal_inner_new_to_old);
                inner_old_to_new.swap(proposal_inner_old_to_new);
                result.coarse_candidate_accepted = nnz > 0;
            }
        }
        const auto fine_start = std::chrono::steady_clock::now();
        result.coarse_ms =
            enable_coarse
                ? std::chrono::duration<double, std::milli>(
                      fine_start - reorder_start)
                      .count()
                : 0.0;

        if (enable_row_fine && rows > 0)
        {
            extract_row_profiles_kernel<<<blocks_for(rows), kThreads, 0,
                                            stream>>>(
                rows, d_row_ptr, d_col_idx,
                thrust::raw_pointer_cast(inner_old_to_new.data()),
                thrust::raw_pointer_cast(row_profiles.data()));
            if (!cuda_ok(cudaGetLastError(), "extract BGRF row profiles"))
                goto failure;
            thrust::sort(policy, row_profiles.begin(), row_profiles.end(),
                         CurrentPositionLess{
                             thrust::raw_pointer_cast(
                                 row_old_to_new.data())});
            const int row_windows = ceil_div_nonnegative(
                rows, DMMA_REORDER_ROW_WINDOW);
            pack_windows_kernel<DMMA_REORDER_ROW_WINDOW, DMMA_TILE_M>
                <<<row_windows, 1, 0, stream>>>(
                    thrust::raw_pointer_cast(row_profiles.data()), rows,
                    thrust::raw_pointer_cast(
                        proposal_row_new_to_old.data()));
            invert_permutation_kernel<<<blocks_for(rows), kThreads, 0,
                                         stream>>>(
                rows,
                thrust::raw_pointer_cast(proposal_row_new_to_old.data()),
                thrust::raw_pointer_cast(proposal_row_old_to_new.data()));
            if (!cuda_ok(cudaGetLastError(),
                         "propose BGRF row packing"))
                goto failure;
            if (enable_exact_guard)
            {
                if (!exact_monotone_commit<
                        true, DMMA_REORDER_ROW_WINDOW>(
                        rows, cols, nnz, d_row_ptr, d_col_idx,
                        row_new_to_old, row_old_to_new,
                        proposal_row_new_to_old,
                        proposal_row_old_to_new, inner_new_to_old,
                        inner_old_to_new, proposal_inner_new_to_old,
                        proposal_inner_old_to_new, exact_keys,
                        exact_counts, exact_span_min, exact_span_max,
                        exact_fanout,
                        thrust::raw_pointer_cast(exact_stats.data()),
                        thrust::raw_pointer_cast(exact_stats.data()) + 1,
                        thrust::raw_pointer_cast(exact_stats.data()) + 4,
                        thrust::raw_pointer_cast(exact_stats.data()) + 5,
                        stream))
                    goto failure;
            }
            else
            {
                row_new_to_old.swap(proposal_row_new_to_old);
                row_old_to_new.swap(proposal_row_old_to_new);
                result.accepted_row_windows = static_cast<unsigned long long>(
                    ceil_div_nonnegative(
                        rows, DMMA_REORDER_ROW_WINDOW));
            }
        }

        if (enable_inner_fine && cols > 0)
        {
            initialize_profiles_kernel<<<blocks_for(cols), kThreads, 0,
                                           stream>>>(
                cols, thrust::raw_pointer_cast(inner_profiles.data()));
            if (rows > 0)
            {
                accumulate_inner_profiles_kernel
                    <<<blocks_for(rows), kThreads, 0, stream>>>(
                        rows, d_row_ptr, d_col_idx,
                        thrust::raw_pointer_cast(row_old_to_new.data()),
                        thrust::raw_pointer_cast(inner_profiles.data()));
            }
            finalize_profiles_kernel<<<blocks_for(cols), kThreads, 0,
                                        stream>>>(
                cols, thrust::raw_pointer_cast(inner_profiles.data()));
            if (!cuda_ok(cudaGetLastError(),
                         "build BGRF inner profiles"))
                goto failure;
            thrust::sort(policy, inner_profiles.begin(), inner_profiles.end(),
                         CurrentPositionLess{
                             thrust::raw_pointer_cast(
                                 inner_old_to_new.data())});
            const int inner_windows = ceil_div_nonnegative(
                cols, DMMA_REORDER_INNER_WINDOW);
            pack_windows_kernel<DMMA_REORDER_INNER_WINDOW, DMMA_TILE_K>
                <<<inner_windows, 1, 0, stream>>>(
                    thrust::raw_pointer_cast(inner_profiles.data()), cols,
                    thrust::raw_pointer_cast(
                        proposal_inner_new_to_old.data()));
            invert_permutation_kernel<<<blocks_for(cols), kThreads, 0,
                                         stream>>>(
                cols,
                thrust::raw_pointer_cast(proposal_inner_new_to_old.data()),
                thrust::raw_pointer_cast(proposal_inner_old_to_new.data()));
            if (!cuda_ok(cudaGetLastError(),
                         "propose BGRF inner packing"))
                goto failure;
            if (enable_exact_guard)
            {
                if (!exact_monotone_commit<
                        false, DMMA_REORDER_INNER_WINDOW>(
                        rows, cols, nnz, d_row_ptr, d_col_idx,
                        row_new_to_old, row_old_to_new,
                        proposal_row_new_to_old,
                        proposal_row_old_to_new, inner_new_to_old,
                        inner_old_to_new, proposal_inner_new_to_old,
                        proposal_inner_old_to_new, exact_keys,
                        exact_counts, exact_span_min, exact_span_max,
                        exact_fanout,
                        thrust::raw_pointer_cast(exact_stats.data()) + 2,
                        thrust::raw_pointer_cast(exact_stats.data()) + 3,
                        thrust::raw_pointer_cast(exact_stats.data()) + 6,
                        thrust::raw_pointer_cast(exact_stats.data()) + 7,
                        stream))
                    goto failure;
            }
            else
            {
                inner_new_to_old.swap(proposal_inner_new_to_old);
                inner_old_to_new.swap(proposal_inner_old_to_new);
                result.accepted_inner_windows =
                    static_cast<unsigned long long>(
                        ceil_div_nonnegative(
                            cols, DMMA_REORDER_INNER_WINDOW));
            }
        }

        int fallback_active_rows = 0;
        int fallback_active_inner = 0;
        if ((!enable_row_fine || !enable_inner_fine) &&
            !compute_active_prefixes(
                rows, d_row_ptr, d_col_idx, row_old_to_new,
                inner_old_to_new, &fallback_active_rows,
                &fallback_active_inner, stream))
            goto failure;
        result.active_rows =
            !enable_row_fine
                ? fallback_active_rows
                : (rows == 0
                       ? 0
                       : thrust::transform_reduce(
                             policy, row_profiles.begin(),
                             row_profiles.end(),
                             ActivePrefix{thrust::raw_pointer_cast(
                                 row_old_to_new.data())},
                             0, thrust::maximum<int>()));
        result.active_inner =
            !enable_inner_fine
                ? fallback_active_inner
                : (cols == 0
                       ? 0
                       : thrust::transform_reduce(
                             policy, inner_profiles.begin(),
                             inner_profiles.end(),
                             ActivePrefix{thrust::raw_pointer_cast(
                                 inner_old_to_new.data())},
                             0, thrust::maximum<int>()));

        if (enable_exact_guard && enable_any_fine &&
            !cuda_ok(cudaMemcpyAsync(
                         h_exact_stats,
                         thrust::raw_pointer_cast(exact_stats.data()),
                         sizeof(h_exact_stats), cudaMemcpyDeviceToHost,
                         stream),
                     "copy BGRF fine statistics"))
            goto failure;
        if (!copy_device_permutation_to_owned(
                row_new_to_old, row_old_to_new,
                &result.h_row_new_to_old, &result.h_row_old_to_new,
                &result.d_row_new_to_old, &result.d_row_old_to_new, stream) ||
            !copy_device_permutation_to_owned(
                inner_new_to_old, inner_old_to_new,
                &result.h_inner_new_to_old, &result.h_inner_old_to_new,
                &result.d_inner_new_to_old, &result.d_inner_old_to_new,
                stream) ||
            !cuda_ok(cudaStreamSynchronize(stream),
                     "complete BGRF permutation"))
            goto failure;

        if (enable_exact_guard && enable_any_fine)
        {
            result.accepted_row_windows = h_exact_stats[0];
            result.row_tile_reduction = h_exact_stats[1];
            result.accepted_inner_windows = h_exact_stats[2];
            result.inner_tile_reduction = h_exact_stats[3];
            result.row_fanout_before = h_exact_stats[4];
            result.row_fanout_after = h_exact_stats[5];
            result.inner_fanout_before = h_exact_stats[6];
            result.inner_fanout_after = h_exact_stats[7];
        }
        result.fine_ms =
            enable_any_fine
                ? std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - fine_start)
                      .count()
                : 0.0;
        summarize_permutation(result.h_row_old_to_new, rows,
                              &result.moved_rows,
                              &result.row_displacement);
        summarize_permutation(result.h_inner_old_to_new, cols,
                              &result.moved_inner,
                              &result.inner_displacement);
    }
    catch (const std::bad_alloc &)
    {
        std::fprintf(stderr, "BGRF reorder allocation failed.\n");
        goto failure;
    }
    catch (const thrust::system_error &error)
    {
        std::fprintf(stderr, "BGRF reorder Thrust failure: %s\n",
                     error.what());
        goto failure;
    }

    destroy_dmma_reorder_plan(out);
    *out = result;
    return true;

failure:
    destroy_dmma_reorder_plan(&result);
    return false;
}

static inline bool dump_dmma_reorder_plan(
    const DmmaReorderPlan &plan, const char *stem, const int *d_row_ptr,
    const int *d_col_idx, bool dump_reordered_matrix = true,
    cudaStream_t stream = 0)
{
    if (stem == nullptr ||
        (plan.rows > 0 && (plan.h_row_old_to_new == nullptr ||
                           plan.h_row_new_to_old == nullptr)) ||
        (plan.cols > 0 && (plan.h_inner_old_to_new == nullptr ||
                           plan.h_inner_new_to_old == nullptr)))
        return false;
    const std::string prefix(stem);
    const std::string permutation_path = prefix + "_permutations.csv";
    const std::string stats_path = prefix + "_unified_stats.csv";
    const std::string matrix_path = prefix + "_A_reordered.mtx";

    FILE *file = std::fopen(permutation_path.c_str(), "w");
    if (file == nullptr)
        return false;
    std::fprintf(file, "axis,old_id,new_id,new_to_old_at_new_id\n");
    for (int old = 0; old < plan.rows; ++old)
    {
        const int next = plan.h_row_old_to_new[old];
        std::fprintf(file, "row,%d,%d,%d\n", old, next,
                     plan.h_row_new_to_old[next]);
    }
    for (int old = 0; old < plan.cols; ++old)
    {
        const int next = plan.h_inner_old_to_new[old];
        std::fprintf(file, "inner,%d,%d,%d\n", old, next,
                     plan.h_inner_new_to_old[next]);
    }
    if (std::fclose(file) != 0)
        return false;

    file = std::fopen(stats_path.c_str(), "w");
    if (file == nullptr)
        return false;
    std::fprintf(file, "key,value\nalgorithm,%s\n",
                 plan.kind == DMMA_REORDER_IDENTITY
                     ? "identity_baseline"
                     : plan.algorithm);
    std::fprintf(file,
                 "sweeps,%d\nrow_window,%d\ninner_window,%d\n"
                 "active_rows,%d\nactive_inner,%d\n"
                 "moved_rows,%llu\nmoved_inner,%llu\n"
                 "row_displacement_sum,%llu\n"
                 "inner_displacement_sum,%llu\n"
                 "accepted_row_windows,%llu\n"
                 "accepted_inner_windows,%llu\n"
                 "row_tile_reduction,%llu\n"
                 "inner_tile_reduction,%llu\n"
                 "row_fanout_before,%llu\n"
                 "row_fanout_after,%llu\n"
                 "inner_fanout_before,%llu\n"
                 "inner_fanout_after,%llu\n"
                 "coarse_components,%d\ncoarse_levels,%d\n"
                 "coarse_level_budget,%d\n"
                 "coarse_candidate_accepted,%d\n"
                 "coarse_tile_reduction,%llu\n"
                 "coarse_ms,%.6f\nfine_ms,%.6f\n"
                 "reorder_peak_workspace_bytes,%zu\n"
                 "num_tiles,%lld\nactive_row_tiles,%d\n"
                 "active_k_tiles,%d\nsparse_tiles,%lld\n"
                 "dense_tiles,%lld\npayload,%llu\n",
                 plan.sweeps, plan.row_window, plan.inner_window,
                 plan.active_rows, plan.active_inner,
                 plan.moved_rows, plan.moved_inner,
                 plan.row_displacement, plan.inner_displacement,
                 plan.accepted_row_windows,
                 plan.accepted_inner_windows,
                 plan.row_tile_reduction,
                 plan.inner_tile_reduction,
                 plan.row_fanout_before, plan.row_fanout_after,
                 plan.inner_fanout_before, plan.inner_fanout_after,
                 plan.coarse_components, plan.coarse_levels,
                 plan.coarse_level_budget,
                 plan.coarse_candidate_accepted ? 1 : 0,
                 plan.coarse_tile_reduction,
                 plan.coarse_ms, plan.fine_ms,
                 plan.reorder_peak_workspace_bytes,
                 plan.num_tiles, plan.active_row_tiles,
                 plan.active_k_tiles, plan.sparse_tiles,
                 plan.dense_tiles, plan.payload);
    if (std::fclose(file) != 0)
        return false;

    if (!dump_reordered_matrix)
        return true;
    if ((plan.rows > 0 && d_row_ptr == nullptr) ||
        (plan.nnz > 0 && d_col_idx == nullptr))
        return false;
    std::vector<int> row_ptr(static_cast<std::size_t>(plan.rows) + 1);
    std::vector<int> col_idx(plan.nnz);
    if ((plan.rows >= 0 &&
         !dmma_reorder_detail::cuda_ok(
             cudaMemcpyAsync(row_ptr.data(), d_row_ptr,
                             row_ptr.size() * sizeof(int),
                             cudaMemcpyDeviceToHost, stream),
             "dump row pointer")) ||
        (plan.nnz > 0 &&
         !dmma_reorder_detail::cuda_ok(
             cudaMemcpyAsync(col_idx.data(), d_col_idx,
                             static_cast<std::size_t>(plan.nnz) * sizeof(int),
                             cudaMemcpyDeviceToHost, stream),
             "dump columns")) ||
        !dmma_reorder_detail::cuda_ok(cudaStreamSynchronize(stream),
                                      "dump CSR copy"))
        return false;
    file = std::fopen(matrix_path.c_str(), "w");
    if (file == nullptr)
        return false;
    std::fprintf(file, "%%%%MatrixMarket matrix coordinate pattern general\n");
    std::fprintf(file, "%% Unified 8x4 row/inner reordered A\n");
    std::fprintf(file, "%d %d %d\n", plan.rows, plan.cols, plan.nnz);
    for (int old_row = 0; old_row < plan.rows; ++old_row)
        for (int entry = row_ptr[old_row]; entry < row_ptr[old_row + 1];
             ++entry)
            std::fprintf(file, "%d %d\n",
                         plan.h_row_old_to_new[old_row] + 1,
                         plan.h_inner_old_to_new[col_idx[entry]] + 1);
    return std::fclose(file) == 0;
}

#endif // RTT_SPGEMM_DMMA_REORDER_H_
