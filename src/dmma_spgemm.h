#ifndef RTT_SPGEMM_DMMA_SPGEMM_H_
#define RTT_SPGEMM_DMMA_SPGEMM_H_

#include "common.h"
#include "dmma_tiles.h"

#include <cuda_runtime.h>
#include <mma.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

/* nsparse defines a legacy WARP macro, so parse CUB/Thrust first. */
#include "spgemm_nsparse_kernel.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

static constexpr int DMMA_WARPS_PER_BLOCK = 4;
static constexpr int DMMA_THREADS_PER_BLOCK = DMMA_WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int DMMA_SPA_WORDS_PER_WARP = 512;
static constexpr int DMMA_SPA_MAX_TILE_COLUMNS =
    DMMA_SPA_WORDS_PER_WARP * 32;

struct DmmaDeviceTiles
{
    int rows;
    int cols;
    int tile_rows;
    int tile_cols;
    int tile_row_count;
    int tile_col_count;
    int num_tiles;
    int payload_size;
    int dense_tiles;
    int sparse_tiles;
    const int *tile_row_ptr;
    const int *tile_col_idx;
    const int *value_offsets;
    const uint32_t *masks;
    const MAT_VAL_TYPE *values;
    const int *tile_col_ptr;
    const int *tile_row_idx;
    const int *csc_tile_ids;
};

struct DmmaOwnedDeviceTiles
{
    DmmaDeviceTiles view{};
    int *tile_row_ptr = nullptr;
    int *tile_col_idx = nullptr;
    int *value_offsets = nullptr;
    uint32_t *masks = nullptr;
    MAT_VAL_TYPE *values = nullptr;
    int *tile_col_ptr = nullptr;
    int *tile_row_idx = nullptr;
    int *csc_tile_ids = nullptr;
};

struct DmmaSpGemmStats
{
    double candidate_ms = 0.0;
    double symbolic_ms = 0.0;
    double numeric_ms = 0.0;
    double total_ms = 0.0;
    double allocation_ms = 0.0;
    double output_copy_ms = 0.0;
    int candidate_tiles = 0;
    int output_tiles = 0;
    int output_nnz = 0;
};

static inline double dmma_elapsed_ms(const timeval &begin, const timeval &end)
{
    return (end.tv_sec - begin.tv_sec) * 1000.0 +
           (end.tv_usec - begin.tv_usec) / 1000.0;
}

static inline bool dmma_cuda_ok(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "CUDA error in %s: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

static inline void destroy_device_tiles(DmmaOwnedDeviceTiles *tiles)
{
    if (tiles == nullptr)
        return;
    cudaFree(tiles->tile_row_ptr);
    cudaFree(tiles->tile_col_idx);
    cudaFree(tiles->value_offsets);
    cudaFree(tiles->masks);
    cudaFree(tiles->values);
    cudaFree(tiles->tile_col_ptr);
    cudaFree(tiles->tile_row_idx);
    cudaFree(tiles->csc_tile_ids);
    *tiles = DmmaOwnedDeviceTiles();
}

template <typename T>
static inline bool dmma_upload_array(T **device, const T *host,
                                     std::size_t count, const char *label)
{
    if (count == 0)
    {
        *device = nullptr;
        return true;
    }
    if (!dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(device),
                                 count * sizeof(T)),
                      label))
        return false;
    if (!dmma_cuda_ok(cudaMemcpy(*device, host, count * sizeof(T),
                                 cudaMemcpyHostToDevice),
                      label))
        return false;
    return true;
}

static inline bool upload_hybrid_tiles(const HybridTileMatrix &host,
                                       bool require_csc,
                                       DmmaOwnedDeviceTiles *device)
{
    if (device == nullptr ||
        (require_csc &&
         (host.tile_col_ptr == nullptr ||
          (host.num_tiles && (host.tile_row_idx == nullptr ||
                              host.csc_tile_ids == nullptr)))))
        return false;

    DmmaOwnedDeviceTiles result;
    const std::size_t tiles = static_cast<std::size_t>(host.num_tiles);
    const std::size_t payload = static_cast<std::size_t>(host.payload_size);
    if (!dmma_upload_array(&result.tile_row_ptr, host.tile_row_ptr,
                           static_cast<std::size_t>(host.tile_row_count) + 1,
                           "upload tile row pointer") ||
        !dmma_upload_array(&result.tile_col_idx, host.tile_col_idx, tiles,
                           "upload tile column index") ||
        !dmma_upload_array(&result.value_offsets, host.value_offsets, tiles + 1,
                           "upload tile value offsets") ||
        !dmma_upload_array(&result.masks, host.masks, tiles,
                           "upload tile masks") ||
        !dmma_upload_array(&result.values, host.values, payload,
                           "upload tile values"))
    {
        destroy_device_tiles(&result);
        return false;
    }

    if (require_csc &&
        (!dmma_upload_array(&result.tile_col_ptr, host.tile_col_ptr,
                            static_cast<std::size_t>(host.tile_col_count) + 1,
                            "upload tile column pointer") ||
         !dmma_upload_array(&result.tile_row_idx, host.tile_row_idx, tiles,
                            "upload tile row index") ||
         !dmma_upload_array(&result.csc_tile_ids, host.csc_tile_ids, tiles,
                            "upload CSC tile IDs")))
    {
        destroy_device_tiles(&result);
        return false;
    }

    result.view.rows = host.rows;
    result.view.cols = host.cols;
    result.view.tile_rows = host.tile_rows;
    result.view.tile_cols = host.tile_cols;
    result.view.tile_row_count = host.tile_row_count;
    result.view.tile_col_count = host.tile_col_count;
    result.view.num_tiles = host.num_tiles;
    result.view.payload_size = host.payload_size;
    result.view.dense_tiles = host.dense_tiles;
    result.view.sparse_tiles = host.sparse_tiles;
    result.view.tile_row_ptr = result.tile_row_ptr;
    result.view.tile_col_idx = result.tile_col_idx;
    result.view.value_offsets = result.value_offsets;
    result.view.masks = result.masks;
    result.view.values = result.values;
    result.view.tile_col_ptr = result.tile_col_ptr;
    result.view.tile_row_idx = result.tile_row_idx;
    result.view.csc_tile_ids = result.csc_tile_ids;
    destroy_device_tiles(device);
    *device = result;
    return true;
}

__device__ __forceinline__ int dmma_warp_sum(int value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__global__ void dmma_candidate_count_spa_kernel(
    const int *__restrict__ a_row_ptr, const int *__restrict__ a_col_idx,
    int a_tile_rows, const int *__restrict__ b_row_ptr,
    const int *__restrict__ b_col_idx, int b_tile_cols, int *c_row_counts)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    __shared__ uint32_t words[DMMA_WARPS_PER_BLOCK *
                              DMMA_SPA_WORDS_PER_WARP];
    if (warp >= a_tile_rows)
        return;

    uint32_t *local = words + local_warp * DMMA_SPA_WORDS_PER_WARP;
    const int word_count = (b_tile_cols + 31) / 32;
    for (int word = lane; word < word_count; word += WARP_SIZE)
        local[word] = 0;
    __syncwarp();

    for (int a = a_row_ptr[warp]; a < a_row_ptr[warp + 1]; ++a)
    {
        const int k_tile = a_col_idx[a];
        for (int b = b_row_ptr[k_tile] + lane; b < b_row_ptr[k_tile + 1];
             b += WARP_SIZE)
        {
            const int col = b_col_idx[b];
            atomicOr(local + col / 32, uint32_t(1) << (col & 31));
        }
    }
    __syncwarp();

    int count = 0;
    for (int word = lane; word < word_count; word += WARP_SIZE)
        count += __popc(local[word]);
    count = dmma_warp_sum(count);
    if (lane == 0)
        c_row_counts[warp] = count;
}

__global__ void dmma_candidate_fill_spa_kernel(
    const int *__restrict__ a_row_ptr, const int *__restrict__ a_col_idx,
    int a_tile_rows, const int *__restrict__ b_row_ptr,
    const int *__restrict__ b_col_idx, int b_tile_cols,
    const int *__restrict__ c_row_ptr, int *c_row_idx, int *c_col_idx)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    __shared__ uint32_t words[DMMA_WARPS_PER_BLOCK *
                              DMMA_SPA_WORDS_PER_WARP];
    if (warp >= a_tile_rows)
        return;

    uint32_t *local = words + local_warp * DMMA_SPA_WORDS_PER_WARP;
    const int word_count = (b_tile_cols + 31) / 32;
    const int rounded_words = ((word_count + WARP_SIZE - 1) / WARP_SIZE) *
                              WARP_SIZE;
    for (int word = lane; word < rounded_words; word += WARP_SIZE)
        local[word] = 0;
    __syncwarp();

    for (int a = a_row_ptr[warp]; a < a_row_ptr[warp + 1]; ++a)
    {
        const int k_tile = a_col_idx[a];
        for (int b = b_row_ptr[k_tile] + lane; b < b_row_ptr[k_tile + 1];
             b += WARP_SIZE)
        {
            const int col = b_col_idx[b];
            atomicOr(local + col / 32, uint32_t(1) << (col & 31));
        }
    }
    __syncwarp();

    int running = 0;
    for (int word = lane; word < rounded_words; word += WARP_SIZE)
    {
        const uint32_t mask = local[word];
        const int own_count = __popc(mask);
        int prefix = own_count;
#pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
        {
            const int other = __shfl_up_sync(0xffffffffu, prefix, offset);
            if (lane >= offset)
                prefix += other;
        }
        const int batch_count = __shfl_sync(0xffffffffu, prefix, 31);
        const int exclusive = prefix - own_count + running;
        int local_offset = 0;
#pragma unroll
        for (int bit = 0; bit < 32; ++bit)
        {
            if ((mask & (uint32_t(1) << bit)) != 0)
            {
                const int output = c_row_ptr[warp] + exclusive + local_offset;
                c_row_idx[output] = warp;
                c_col_idx[output] = word * 32 + bit;
                ++local_offset;
            }
        }
        running += batch_count;
    }
}

__device__ __forceinline__ int dmma_first_col_after(
    const int *__restrict__ columns, int begin, int end, int previous)
{
    while (begin < end)
    {
        const int middle = begin + (end - begin) / 2;
        if (columns[middle] <= previous)
            begin = middle + 1;
        else
            end = middle;
    }
    return begin;
}

__device__ __forceinline__ int dmma_warp_min(int value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = min(value,
                    __shfl_down_sync(0xffffffffu, value, offset));
    return __shfl_sync(0xffffffffu, value, 0);
}

/*
 * The legacy seven-bin hash path is retained for counting wide symbolic
 * rows.  Its historical fill kernels compact hash tables in place, which can
 * overwrite slots that another lane has not read yet.  Fill those counted
 * rows deterministically instead: each warp performs a sorted multiway union
 * of the relevant B tile rows.  This path is used only beyond the 16K-column
 * shared-memory SPA limit, where correctness is more important than the
 * extra binary searches.
 */
__global__ void dmma_candidate_fill_wide_kernel(
    const int *__restrict__ a_row_ptr, const int *__restrict__ a_col_idx,
    int a_tile_rows, const int *__restrict__ b_row_ptr,
    const int *__restrict__ b_col_idx, const int *__restrict__ c_row_ptr,
    int *c_row_idx, int *c_col_idx, int *count_mismatch)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (warp >= a_tile_rows)
        return;

    const int output_begin = c_row_ptr[warp];
    const int output_end = c_row_ptr[warp + 1];
    int previous = -1;
    for (int output = output_begin; output < output_end; ++output)
    {
        int lane_minimum = INT_MAX;
        for (int a = a_row_ptr[warp] + lane; a < a_row_ptr[warp + 1];
             a += WARP_SIZE)
        {
            const int k_tile = a_col_idx[a];
            const int b_begin = b_row_ptr[k_tile];
            const int b_end = b_row_ptr[k_tile + 1];
            const int position =
                dmma_first_col_after(b_col_idx, b_begin, b_end, previous);
            if (position < b_end)
                lane_minimum = min(lane_minimum, b_col_idx[position]);
        }

        const int next = dmma_warp_min(lane_minimum);
        if (next == INT_MAX)
        {
            if (lane == 0)
                atomicExch(count_mismatch, 1);
            return;
        }
        if (lane == 0)
        {
            c_row_idx[output] = warp;
            c_col_idx[output] = next;
        }
        previous = next;
    }

    int lane_minimum = INT_MAX;
    for (int a = a_row_ptr[warp] + lane; a < a_row_ptr[warp + 1];
         a += WARP_SIZE)
    {
        const int k_tile = a_col_idx[a];
        const int b_begin = b_row_ptr[k_tile];
        const int b_end = b_row_ptr[k_tile + 1];
        const int position =
            dmma_first_col_after(b_col_idx, b_begin, b_end, previous);
        if (position < b_end)
            lane_minimum = min(lane_minimum, b_col_idx[position]);
    }
    const int remaining = dmma_warp_min(lane_minimum);
    if (lane == 0 && remaining != INT_MAX)
        atomicExch(count_mismatch, 1);
}

__device__ __forceinline__ unsigned long long dmma_warp_or(
    unsigned long long value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value |= __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__global__ void dmma_exact_mask_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int candidate_count,
    const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols, uint64_t *candidate_masks,
    int *candidate_nnz, int *candidate_keep)
{
    const int global_warp =
        (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (global_warp >= candidate_count)
        return;

    const int tile_row = candidate_rows[global_warp];
    const int tile_col = candidate_cols[global_warp];
    int pa = a.tile_row_ptr[tile_row];
    const int a_end = a.tile_row_ptr[tile_row + 1];
    int pb = b.tile_col_ptr[tile_col];
    const int b_end = b.tile_col_ptr[tile_col + 1];
    unsigned long long local_output = 0;

    while (pa < a_end && pb < b_end)
    {
        const int ka = a.tile_col_idx[pa];
        const int kb = b.tile_row_idx[pb];
        if (ka == kb)
        {
            const uint32_t mask_a = a.masks[pa];
            const int tile_b = b.csc_tile_ids[pb];
            const uint32_t mask_b = b.masks[tile_b];
            for (int output_pos = lane; output_pos < DMMA_OUTPUT_ELEMS;
                 output_pos += WARP_SIZE)
            {
                const int row = output_pos / DMMA_TILE_N;
                const int col = output_pos % DMMA_TILE_N;
                bool present = false;
#pragma unroll
                for (int k = 0; k < DMMA_TILE_K; ++k)
                {
                    const int pos_a = row * DMMA_TILE_K + k;
                    const int pos_b = col * DMMA_TILE_K + k;
                    present |= ((mask_a >> pos_a) & 1u) != 0 &&
                               ((mask_b >> pos_b) & 1u) != 0;
                }
                if (present)
                    local_output |= 1ull << output_pos;
            }
            ++pa;
            ++pb;
        }
        else if (ka < kb)
        {
            ++pa;
        }
        else
        {
            ++pb;
        }
    }

    const unsigned long long output_mask = dmma_warp_or(local_output);
    if (lane == 0)
    {
        candidate_masks[global_warp] = output_mask;
        const int nnz = __popcll(output_mask);
        candidate_nnz[global_warp] = nnz;
        candidate_keep[global_warp] = nnz != 0;
    }
}

__global__ void dmma_compact_candidates_kernel(
    int candidate_count, const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols,
    const uint64_t *__restrict__ candidate_masks,
    const int *__restrict__ candidate_nnz,
    const int *__restrict__ candidate_positions, int *output_rows,
    int *output_cols, uint64_t *output_masks, int *output_nnz)
{
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count || candidate_masks[candidate] == 0)
        return;
    const int output = candidate_positions[candidate];
    output_rows[output] = candidate_rows[candidate];
    output_cols[output] = candidate_cols[candidate];
    output_masks[output] = candidate_masks[candidate];
    output_nnz[output] = candidate_nnz[candidate];
}

__global__ void dmma_count_output_rows_kernel(int tile_count,
                                               const int *tile_rows,
                                               int *row_counts)
{
    const int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile < tile_count)
        atomicAdd(row_counts + tile_rows[tile], 1);
}

__device__ __forceinline__ MAT_VAL_TYPE dmma_decode_value(
    DmmaDeviceTiles matrix, int tile, int physical)
{
    const int begin = matrix.value_offsets[tile];
    const int span = matrix.value_offsets[tile + 1] - begin;
    if (span == DMMA_INPUT_ELEMS)
        return matrix.values[begin + physical];
    const uint32_t mask = matrix.masks[tile];
    if ((mask & (uint32_t(1) << physical)) == 0)
        return MAT_VAL_TYPE(0);
    const uint32_t lower = physical == 0
                               ? 0u
                               : mask & ((uint32_t(1) << physical) - 1u);
    return matrix.values[begin + __popc(lower)];
}

__global__ void dmma_numeric_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int output_tile_count,
    const int *__restrict__ output_rows, const int *__restrict__ output_cols,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets, unsigned char *output_row_ptr,
    unsigned char *output_col_idx, MAT_VAL_TYPE *output_values)
{
#if __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;
    const int global_warp =
        (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    if (global_warp >= output_tile_count)
        return;

    __shared__ MAT_VAL_TYPE shared_a[DMMA_WARPS_PER_BLOCK *
                                     DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE shared_b[DMMA_WARPS_PER_BLOCK *
                                     DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE shared_c[DMMA_WARPS_PER_BLOCK *
                                     DMMA_OUTPUT_ELEMS];
    MAT_VAL_TYPE *tile_a_values = shared_a + local_warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_b_values = shared_b + local_warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_c_values = shared_c + local_warp * DMMA_OUTPUT_ELEMS;

    wmma::fragment<wmma::matrix_a, DMMA_TILE_M, DMMA_TILE_N, DMMA_TILE_K,
                   MAT_VAL_TYPE, wmma::row_major>
        fragment_a;
    wmma::fragment<wmma::matrix_b, DMMA_TILE_M, DMMA_TILE_N, DMMA_TILE_K,
                   MAT_VAL_TYPE, wmma::col_major>
        fragment_b;
    wmma::fragment<wmma::accumulator, DMMA_TILE_M, DMMA_TILE_N, DMMA_TILE_K,
                   MAT_VAL_TYPE>
        fragment_c;
    wmma::fill_fragment(fragment_c, MAT_VAL_TYPE(0));

    const int tile_row = output_rows[global_warp];
    const int tile_col = output_cols[global_warp];
    int pa = a.tile_row_ptr[tile_row];
    const int a_end = a.tile_row_ptr[tile_row + 1];
    int pb = b.tile_col_ptr[tile_col];
    const int b_end = b.tile_col_ptr[tile_col + 1];

    while (pa < a_end && pb < b_end)
    {
        const int ka = a.tile_col_idx[pa];
        const int kb = b.tile_row_idx[pb];
        if (ka == kb)
        {
            const int tile_b = b.csc_tile_ids[pb];
            tile_a_values[lane] = dmma_decode_value(a, pa, lane);
            tile_b_values[lane] = dmma_decode_value(b, tile_b, lane);
            __syncwarp();
            wmma::load_matrix_sync(fragment_a, tile_a_values, DMMA_TILE_K);
            wmma::load_matrix_sync(fragment_b, tile_b_values, DMMA_TILE_K);
            wmma::mma_sync(fragment_c, fragment_a, fragment_b, fragment_c);
            __syncwarp();
            ++pa;
            ++pb;
        }
        else if (ka < kb)
        {
            ++pa;
        }
        else
        {
            ++pb;
        }
    }

    wmma::store_matrix_sync(tile_c_values, fragment_c, DMMA_TILE_N,
                            wmma::mem_row_major);
    __syncwarp();

    const uint64_t mask = output_masks[global_warp];
    const int output_begin = output_offsets[global_warp];
    if (lane < DMMA_TILE_M)
    {
        const uint64_t preceding =
            lane == 0 ? 0ull : mask & ((1ull << (lane * DMMA_TILE_N)) - 1ull);
        output_row_ptr[global_warp * DMMA_TILE_M + lane] =
            static_cast<unsigned char>(__popcll(preceding));
    }
    for (int position = lane; position < DMMA_OUTPUT_ELEMS;
         position += WARP_SIZE)
    {
        if ((mask & (1ull << position)) == 0)
            continue;
        const uint64_t lower =
            position == 0 ? 0ull : mask & ((1ull << position) - 1ull);
        const int rank = __popcll(lower);
        output_col_idx[output_begin + rank] =
            static_cast<unsigned char>(position % DMMA_TILE_N);
        output_values[output_begin + rank] = tile_c_values[position];
    }
#endif
}

static inline bool dmma_copy_output_to_host(
    int rows, int cols, int tile_rows, int tile_cols, int tile_count,
    int nnz, const int *d_row_ptr, const int *d_cols,
    const int *d_value_offsets, const unsigned char *d_tile_row_ptr,
    const unsigned char *d_value_cols, const MAT_VAL_TYPE *d_values,
    SMatrix *output)
{
    if (output == nullptr)
        return false;
    SMatrix result{};
    result.m = rows;
    result.n = cols;
    result.nnz = nnz;
    result.tilem = tile_rows;
    result.tilen = tile_cols;
    result.numtile = tile_count;
    result.tile_ptr = static_cast<MAT_PTR_TYPE *>(std::malloc(
        (static_cast<std::size_t>(tile_rows) + 1) * sizeof(MAT_PTR_TYPE)));
    result.tile_columnidx = static_cast<int *>(
        std::malloc(static_cast<std::size_t>(tile_count) * sizeof(int)));
    result.tile_nnz = static_cast<int *>(
        std::malloc((static_cast<std::size_t>(tile_count) + 1) * sizeof(int)));
    result.tile_csr_Ptr = static_cast<unsigned char *>(
        std::malloc(static_cast<std::size_t>(tile_count) * DMMA_TILE_M));
    result.tile_csr_Col = static_cast<unsigned char *>(
        std::malloc(static_cast<std::size_t>(nnz)));
    result.tile_csr_Value = static_cast<MAT_VAL_TYPE *>(
        std::malloc(static_cast<std::size_t>(nnz) * sizeof(MAT_VAL_TYPE)));
    if (result.tile_ptr == nullptr || result.tile_nnz == nullptr ||
        (tile_count && (result.tile_columnidx == nullptr ||
                        result.tile_csr_Ptr == nullptr)) ||
        (nnz && (result.tile_csr_Col == nullptr ||
                 result.tile_csr_Value == nullptr)))
        goto host_copy_failure;

    if (!dmma_cuda_ok(
            cudaMemcpy(result.tile_ptr, d_row_ptr,
                       (static_cast<std::size_t>(tile_rows) + 1) *
                           sizeof(MAT_PTR_TYPE),
                       cudaMemcpyDeviceToHost),
            "copy C tile row pointer") ||
        !dmma_cuda_ok(
            cudaMemcpy(result.tile_nnz, d_value_offsets,
                       (static_cast<std::size_t>(tile_count) + 1) *
                           sizeof(int),
                       cudaMemcpyDeviceToHost),
            "copy C value offsets"))
        goto host_copy_failure;
    if (tile_count &&
        (!dmma_cuda_ok(cudaMemcpy(result.tile_columnidx, d_cols,
                                  static_cast<std::size_t>(tile_count) *
                                      sizeof(int),
                                  cudaMemcpyDeviceToHost),
                       "copy C tile columns") ||
         !dmma_cuda_ok(cudaMemcpy(result.tile_csr_Ptr, d_tile_row_ptr,
                                  static_cast<std::size_t>(tile_count) *
                                      DMMA_TILE_M,
                                  cudaMemcpyDeviceToHost),
                       "copy C tile row offsets")))
        goto host_copy_failure;
    if (nnz &&
        (!dmma_cuda_ok(cudaMemcpy(result.tile_csr_Col, d_value_cols,
                                  static_cast<std::size_t>(nnz),
                                  cudaMemcpyDeviceToHost),
                       "copy C value columns") ||
         !dmma_cuda_ok(cudaMemcpy(result.tile_csr_Value, d_values,
                                  static_cast<std::size_t>(nnz) *
                                      sizeof(MAT_VAL_TYPE),
                                  cudaMemcpyDeviceToHost),
                       "copy C values")))
        goto host_copy_failure;

    *output = result;
    return true;

host_copy_failure:
    std::free(result.tile_ptr);
    std::free(result.tile_columnidx);
    std::free(result.tile_nnz);
    std::free(result.tile_csr_Ptr);
    std::free(result.tile_csr_Col);
    std::free(result.tile_csr_Value);
    return false;
}

/*
 * Structural discovery remains adaptive: a shared-memory SPA handles up to
 * 16K output tile columns, while the existing hash-bin implementation handles
 * wider matrices.  Numeric work is always one uniform DMMA kernel.
 */
static inline bool dmma_tilespgemm(const DmmaDeviceTiles &a,
                                   const DmmaDeviceTiles &b,
                                   SMatrix *output, DmmaSpGemmStats *stats)
{
    if (output == nullptr || stats == nullptr ||
        a.cols != b.rows ||
        a.tile_col_count != b.tile_row_count ||
        a.tile_rows != DMMA_TILE_M ||
        a.tile_cols != DMMA_TILE_K ||
        b.tile_rows != DMMA_TILE_K ||
        b.tile_cols != DMMA_TILE_N ||
        a.tile_row_ptr == nullptr || b.tile_row_ptr == nullptr ||
        b.tile_col_ptr == nullptr)
        return false;

    *stats = DmmaSpGemmStats();
    timeval total_begin{}, total_end{}, begin{}, end{};
    gettimeofday(&total_begin, nullptr);

    int *d_candidate_row_ptr = nullptr;
    int *d_candidate_rows = nullptr;
    int *d_candidate_cols = nullptr;
    int *d_candidate_count_mismatch = nullptr;
    int candidate_count = 0;
    uint64_t *d_candidate_masks = nullptr;
    int *d_candidate_nnz = nullptr;
    int *d_candidate_keep = nullptr;
    int *d_output_rows = nullptr;
    int *d_output_cols = nullptr;
    uint64_t *d_output_masks = nullptr;
    int *d_output_nnz = nullptr;
    int *d_output_row_ptr = nullptr;
    int output_tile_count = 0;
    int output_nnz = 0;
    unsigned char *d_output_tile_row_ptr = nullptr;
    unsigned char *d_output_value_cols = nullptr;
    MAT_VAL_TYPE *d_output_values = nullptr;
    sfBIN bin{};
    bool hash_bin_active = false;
    gettimeofday(&begin, nullptr);
    if (!dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_candidate_row_ptr),
                                 (static_cast<std::size_t>(a.tile_row_count) +
                                  1) *
                                     sizeof(int)),
                      "allocate candidate row pointer") ||
        !dmma_cuda_ok(cudaMemset(d_candidate_row_ptr, 0,
                                 (static_cast<std::size_t>(a.tile_row_count) +
                                  1) *
                                     sizeof(int)),
                      "clear candidate row pointer"))
        goto failure;

    if (a.tile_row_count == 0)
    {
        candidate_count = 0;
    }
    else if (b.tile_col_count <= DMMA_SPA_MAX_TILE_COLUMNS)
    {
        const int blocks =
            (a.tile_row_count + DMMA_WARPS_PER_BLOCK - 1) /
            DMMA_WARPS_PER_BLOCK;
        dmma_candidate_count_spa_kernel<<<blocks, DMMA_THREADS_PER_BLOCK>>>(
            a.tile_row_ptr, a.tile_col_idx, a.tile_row_count,
            b.tile_row_ptr, b.tile_col_idx, b.tile_col_count,
            d_candidate_row_ptr);
        if (!dmma_cuda_ok(cudaGetLastError(), "launch candidate count") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(), "candidate count"))
            goto failure;
        thrust::device_ptr<int> row_pointer(d_candidate_row_ptr);
        thrust::exclusive_scan(row_pointer,
                               row_pointer + a.tile_row_count + 1,
                               row_pointer);
        if (!dmma_cuda_ok(cudaMemcpy(&candidate_count,
                                     d_candidate_row_ptr +
                                         a.tile_row_count,
                                     sizeof(int), cudaMemcpyDeviceToHost),
                          "read candidate count"))
            goto failure;
    }
    else
    {
        init_bin(&bin, a.tile_row_count);
        hash_bin_active = true;
        set_max_bin(const_cast<int *>(a.tile_row_ptr),
                    const_cast<int *>(a.tile_col_idx),
                    const_cast<int *>(b.tile_row_ptr), &bin,
                    a.tile_row_count);
        set_row_nnz(const_cast<int *>(a.tile_row_ptr),
                    const_cast<int *>(a.tile_col_idx),
                    const_cast<int *>(b.tile_row_ptr),
                    const_cast<int *>(b.tile_col_idx), d_candidate_row_ptr, &bin,
                    a.tile_row_count, &candidate_count);
        if (!dmma_cuda_ok(cudaGetLastError(), "hash candidate count") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(),
                          "hash candidate count completion"))
            goto failure;
    }

    if (candidate_count < 0)
    {
        std::fprintf(stderr, "Invalid negative symbolic candidate count.\n");
        goto failure;
    }

    if (candidate_count > 0)
    {
        if (!dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_candidate_rows),
                                     static_cast<std::size_t>(candidate_count) *
                                         sizeof(int)),
                          "allocate candidate rows") ||
            !dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_candidate_cols),
                                     static_cast<std::size_t>(candidate_count) *
                                         sizeof(int)),
                          "allocate candidate columns"))
            goto failure;
    }
    if (candidate_count > 0 &&
        b.tile_col_count <= DMMA_SPA_MAX_TILE_COLUMNS)
    {
        const int blocks =
            (a.tile_row_count + DMMA_WARPS_PER_BLOCK - 1) /
            DMMA_WARPS_PER_BLOCK;
        dmma_candidate_fill_spa_kernel<<<blocks, DMMA_THREADS_PER_BLOCK>>>(
            a.tile_row_ptr, a.tile_col_idx, a.tile_row_count,
            b.tile_row_ptr, b.tile_col_idx, b.tile_col_count,
            d_candidate_row_ptr, d_candidate_rows, d_candidate_cols);
        if (!dmma_cuda_ok(cudaGetLastError(), "launch candidate fill"))
            goto failure;
    }
    else if (a.tile_row_count > 0 &&
             b.tile_col_count > DMMA_SPA_MAX_TILE_COLUMNS)
    {
        if (!dmma_cuda_ok(
                cudaMalloc(reinterpret_cast<void **>(
                               &d_candidate_count_mismatch),
                           sizeof(int)),
                "allocate wide candidate count check") ||
            !dmma_cuda_ok(cudaMemset(d_candidate_count_mismatch, 0,
                                     sizeof(int)),
                          "clear wide candidate count check"))
            goto failure;
        const int blocks =
            (a.tile_row_count + DMMA_WARPS_PER_BLOCK - 1) /
            DMMA_WARPS_PER_BLOCK;
        dmma_candidate_fill_wide_kernel<<<blocks, DMMA_THREADS_PER_BLOCK>>>(
            a.tile_row_ptr, a.tile_col_idx, a.tile_row_count,
            b.tile_row_ptr, b.tile_col_idx, d_candidate_row_ptr,
            d_candidate_rows, d_candidate_cols,
            d_candidate_count_mismatch);
    }
    if (!dmma_cuda_ok(cudaDeviceSynchronize(), "candidate discovery"))
        goto failure;
    if (d_candidate_count_mismatch != nullptr)
    {
        int count_mismatch = 0;
        if (!dmma_cuda_ok(cudaMemcpy(&count_mismatch,
                                     d_candidate_count_mismatch, sizeof(int),
                                     cudaMemcpyDeviceToHost),
                          "read wide candidate count check") ||
            count_mismatch != 0)
        {
            if (count_mismatch != 0)
                std::fprintf(stderr,
                             "Wide symbolic candidate count/fill mismatch.\n");
            goto failure;
        }
        cudaFree(d_candidate_count_mismatch);
        d_candidate_count_mismatch = nullptr;
    }
    if (hash_bin_active)
    {
        release_bin(bin);
        hash_bin_active = false;
    }
    gettimeofday(&end, nullptr);
    stats->candidate_ms = dmma_elapsed_ms(begin, end);
    stats->candidate_tiles = candidate_count;

    if (candidate_count == 0)
    {
        gettimeofday(&total_end, nullptr);
        stats->total_ms = dmma_elapsed_ms(total_begin, total_end);
        gettimeofday(&begin, nullptr);
        if (!dmma_copy_output_to_host(
                a.rows, b.cols, a.tile_row_count,
                b.tile_col_count, 0, 0, d_candidate_row_ptr, nullptr,
                d_candidate_row_ptr, nullptr, nullptr, nullptr, output))
            goto failure;
        gettimeofday(&end, nullptr);
        stats->output_copy_ms = dmma_elapsed_ms(begin, end);
        cudaFree(d_candidate_row_ptr);
        return true;
    }

    gettimeofday(&begin, nullptr);
    if (!dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_candidate_masks),
                                 static_cast<std::size_t>(candidate_count) *
                                     sizeof(uint64_t)),
                      "allocate candidate masks") ||
        !dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_candidate_nnz),
                                 static_cast<std::size_t>(candidate_count) *
                                     sizeof(int)),
                      "allocate candidate nnz") ||
        !dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_candidate_keep),
                                 (static_cast<std::size_t>(candidate_count) + 1) *
                                     sizeof(int)),
                      "allocate candidate keep scan") ||
        !dmma_cuda_ok(cudaMemset(d_candidate_keep, 0,
                                 (static_cast<std::size_t>(candidate_count) + 1) *
                                     sizeof(int)),
                      "clear candidate keep scan"))
        goto symbolic_failure;
    {
        const int warps = candidate_count;
        const int blocks =
            (warps + DMMA_WARPS_PER_BLOCK - 1) / DMMA_WARPS_PER_BLOCK;
        dmma_exact_mask_kernel<<<blocks, DMMA_THREADS_PER_BLOCK>>>(
            a, b, candidate_count, d_candidate_rows,
            d_candidate_cols, d_candidate_masks, d_candidate_nnz,
            d_candidate_keep);
        if (!dmma_cuda_ok(cudaGetLastError(), "launch exact C masks") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(), "exact C masks"))
            goto symbolic_failure;
    }
    {
        thrust::device_ptr<int> keep(d_candidate_keep);
        thrust::exclusive_scan(keep, keep + candidate_count + 1, keep);
        if (!dmma_cuda_ok(cudaMemcpy(&output_tile_count,
                                     d_candidate_keep + candidate_count,
                                     sizeof(int), cudaMemcpyDeviceToHost),
                          "read exact C tile count"))
            goto symbolic_failure;
    }

    if (output_tile_count == 0)
    {
        if (!dmma_cuda_ok(
                cudaMemset(d_candidate_row_ptr, 0,
                           (static_cast<std::size_t>(a.tile_row_count) +
                            1) * sizeof(int)),
                "clear empty exact C row pointer"))
            goto symbolic_failure;
        gettimeofday(&end, nullptr);
        stats->symbolic_ms = dmma_elapsed_ms(begin, end);
        stats->output_tiles = 0;
        stats->output_nnz = 0;
        gettimeofday(&total_end, nullptr);
        stats->total_ms = dmma_elapsed_ms(total_begin, total_end);
        gettimeofday(&begin, nullptr);
        if (!dmma_copy_output_to_host(
                a.rows, b.cols, a.tile_row_count,
                b.tile_col_count, 0, 0, d_candidate_row_ptr, nullptr,
                d_candidate_keep, nullptr, nullptr, nullptr, output))
            goto symbolic_failure;
        gettimeofday(&end, nullptr);
        stats->output_copy_ms = dmma_elapsed_ms(begin, end);
        cudaFree(d_candidate_masks);
        cudaFree(d_candidate_nnz);
        cudaFree(d_candidate_keep);
        cudaFree(d_candidate_row_ptr);
        cudaFree(d_candidate_rows);
        cudaFree(d_candidate_cols);
        return true;
    }

    if (!dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_output_rows),
                                 static_cast<std::size_t>(output_tile_count) *
                                     sizeof(int)),
                      "allocate output rows") ||
        !dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_output_cols),
                                 static_cast<std::size_t>(output_tile_count) *
                                     sizeof(int)),
                      "allocate output columns") ||
        !dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_output_masks),
                                 static_cast<std::size_t>(output_tile_count) *
                                     sizeof(uint64_t)),
                      "allocate output masks") ||
        !dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_output_nnz),
                                 (static_cast<std::size_t>(output_tile_count) + 1) *
                                     sizeof(int)),
                      "allocate output nnz") ||
        !dmma_cuda_ok(cudaMemset(d_output_nnz, 0,
                                 (static_cast<std::size_t>(output_tile_count) + 1) *
                                     sizeof(int)),
                      "clear output nnz") ||
        !dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_output_row_ptr),
                                 (static_cast<std::size_t>(a.tile_row_count) +
                                  1) *
                                     sizeof(int)),
                      "allocate exact C row pointer") ||
        !dmma_cuda_ok(cudaMemset(d_output_row_ptr, 0,
                                 (static_cast<std::size_t>(a.tile_row_count) +
                                  1) *
                                     sizeof(int)),
                      "clear exact C row pointer"))
        goto symbolic_failure;
    {
        const int threads = 256;
        const int blocks = (candidate_count + threads - 1) / threads;
        dmma_compact_candidates_kernel<<<blocks, threads>>>(
            candidate_count, d_candidate_rows, d_candidate_cols,
            d_candidate_masks, d_candidate_nnz, d_candidate_keep,
            d_output_rows, d_output_cols, d_output_masks, d_output_nnz);
        const int output_blocks = (output_tile_count + threads - 1) / threads;
        dmma_count_output_rows_kernel<<<output_blocks, threads>>>(
            output_tile_count, d_output_rows, d_output_row_ptr);
        if (!dmma_cuda_ok(cudaGetLastError(), "compact exact C tiles") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(), "compact exact C tiles"))
            goto symbolic_failure;
        thrust::device_ptr<int> row_pointer(d_output_row_ptr);
        thrust::exclusive_scan(row_pointer,
                               row_pointer + a.tile_row_count + 1,
                               row_pointer);
    }
    {
        thrust::device_ptr<int> nnz_pointer(d_output_nnz);
        thrust::exclusive_scan(nnz_pointer,
                               nnz_pointer + output_tile_count + 1,
                               nnz_pointer);
        if (!dmma_cuda_ok(cudaMemcpy(&output_nnz,
                                     d_output_nnz + output_tile_count,
                                     sizeof(int), cudaMemcpyDeviceToHost),
                          "read C nnz"))
            goto symbolic_failure;
    }
    gettimeofday(&end, nullptr);
    stats->symbolic_ms = dmma_elapsed_ms(begin, end);
    stats->output_tiles = output_tile_count;
    stats->output_nnz = output_nnz;

    gettimeofday(&begin, nullptr);
    if (!dmma_cuda_ok(cudaMalloc(
                          reinterpret_cast<void **>(&d_output_tile_row_ptr),
                          static_cast<std::size_t>(output_tile_count) *
                              DMMA_TILE_M),
                      "allocate C tile row offsets") ||
        !dmma_cuda_ok(cudaMalloc(
                          reinterpret_cast<void **>(&d_output_value_cols),
                          static_cast<std::size_t>(output_nnz)),
                      "allocate C value columns") ||
        !dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_output_values),
                                 static_cast<std::size_t>(output_nnz) *
                                     sizeof(MAT_VAL_TYPE)),
                      "allocate C values"))
        goto numeric_failure;
    gettimeofday(&end, nullptr);
    stats->allocation_ms += dmma_elapsed_ms(begin, end);
    gettimeofday(&begin, nullptr);
    {
        const int blocks =
            (output_tile_count + DMMA_WARPS_PER_BLOCK - 1) /
            DMMA_WARPS_PER_BLOCK;
        dmma_numeric_kernel<<<blocks, DMMA_THREADS_PER_BLOCK>>>(
            a, b, output_tile_count, d_output_rows, d_output_cols,
            d_output_masks, d_output_nnz, d_output_tile_row_ptr,
            d_output_value_cols, d_output_values);
        if (!dmma_cuda_ok(cudaGetLastError(), "launch uniform DMMA kernel") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(), "uniform DMMA kernel"))
            goto numeric_failure;
    }
    gettimeofday(&end, nullptr);
    stats->numeric_ms = dmma_elapsed_ms(begin, end);

    gettimeofday(&total_end, nullptr);
    stats->total_ms = dmma_elapsed_ms(total_begin, total_end);
    gettimeofday(&begin, nullptr);
    if (!dmma_copy_output_to_host(
            a.rows, b.cols, a.tile_row_count,
            b.tile_col_count, output_tile_count, output_nnz,
            d_output_row_ptr, d_output_cols, d_output_nnz,
            d_output_tile_row_ptr, d_output_value_cols, d_output_values,
            output))
        goto numeric_failure;
    gettimeofday(&end, nullptr);
    stats->output_copy_ms = dmma_elapsed_ms(begin, end);
    cudaFree(d_output_tile_row_ptr);
    cudaFree(d_output_value_cols);
    cudaFree(d_output_values);
    cudaFree(d_candidate_masks);
    cudaFree(d_candidate_nnz);
    cudaFree(d_candidate_keep);
    cudaFree(d_output_rows);
    cudaFree(d_output_cols);
    cudaFree(d_output_masks);
    cudaFree(d_output_nnz);
    cudaFree(d_output_row_ptr);
    cudaFree(d_candidate_row_ptr);
    cudaFree(d_candidate_rows);
    cudaFree(d_candidate_cols);
    cudaFree(d_candidate_count_mismatch);
    return true;

numeric_failure:
    cudaFree(d_output_tile_row_ptr);
    cudaFree(d_output_value_cols);
    cudaFree(d_output_values);
symbolic_failure:
    cudaFree(d_candidate_masks);
    cudaFree(d_candidate_nnz);
    cudaFree(d_candidate_keep);
    cudaFree(d_output_rows);
    cudaFree(d_output_cols);
    cudaFree(d_output_masks);
    cudaFree(d_output_nnz);
    cudaFree(d_output_row_ptr);
failure:
    if (hash_bin_active)
        release_bin(bin);
    cudaFree(d_candidate_row_ptr);
    cudaFree(d_candidate_rows);
    cudaFree(d_candidate_cols);
    cudaFree(d_candidate_count_mismatch);
    return false;
}

/* Test/benchmark compatibility path.  Production code prepares tiles on the
 * GPU and calls the device-view overload above directly. */
static inline bool dmma_tilespgemm(const HybridTileMatrix &host_a,
                                   const HybridTileMatrix &host_b,
                                   SMatrix *output, DmmaSpGemmStats *stats)
{
    DmmaOwnedDeviceTiles a;
    DmmaOwnedDeviceTiles b;
    if (!upload_hybrid_tiles(host_a, false, &a) ||
        !upload_hybrid_tiles(host_b, true, &b))
    {
        destroy_device_tiles(&a);
        destroy_device_tiles(&b);
        return false;
    }
    const bool ok = dmma_tilespgemm(a.view, b.view, output, stats);
    destroy_device_tiles(&a);
    destroy_device_tiles(&b);
    return ok;
}

#endif // RTT_SPGEMM_DMMA_SPGEMM_H_
