#ifndef RTT_SPGEMM_DMMA_SPGEMM_H_
#define RTT_SPGEMM_DMMA_SPGEMM_H_

#include "common.h"
#include "dmma_tiles.h"

#include <cuda_runtime.h>
#include <mma.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sys/time.h>
#include <vector>

static constexpr int DMMA_WARPS_PER_BLOCK = 4;
static constexpr int DMMA_THREADS_PER_BLOCK = DMMA_WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int DMMA_SPA_WORDS_PER_WARP = 512;
static constexpr int DMMA_SPA_MAX_TILE_COLUMNS =
    DMMA_SPA_WORDS_PER_WARP * 32;
static constexpr std::size_t DMMA_WIDE_BITSET_SCRATCH_BYTES =
    std::size_t(128) * 1024 * 1024;
static constexpr int DMMA_WIDE_BLOCK_THREADS = 256;

#ifdef DMMA_ENABLE_TIMELINE_TRACE
static inline bool dmma_cuda_ok(cudaError_t status, const char *operation);

struct DmmaTimelineView
{
    unsigned long long *warp_start = nullptr;
    unsigned long long *warp_end = nullptr;
    uint32_t *sm_id = nullptr;
    unsigned int sample_shift = 0;
    unsigned int sample_phase = 0;
};

__device__ __forceinline__ unsigned long long dmma_read_globaltimer()
{
    unsigned long long value = 0;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(value) :: "memory");
    return value;
}

__device__ __forceinline__ uint32_t dmma_read_smid()
{
    uint32_t value = 0;
    asm volatile("mov.u32 %0, %smid;" : "=r"(value) :: "memory");
    return value;
}

static inline unsigned int dmma_timeline_env_uint(
    const char *name, unsigned int fallback, unsigned int maximum)
{
    const char *text = std::getenv(name);
    if (text == nullptr || *text == '\0')
        return fallback;
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || value > maximum)
    {
        std::fprintf(stderr, "Ignoring invalid %s=%s.\n", name, text);
        return fallback;
    }
    return static_cast<unsigned int>(value);
}

static inline bool dmma_write_timeline_trace(
    const char *path, const DmmaTimelineView &view, std::size_t slots,
    unsigned int grid_blocks, int output_tile_count)
{
    if (path == nullptr || *path == '\0' || slots == 0 ||
        view.warp_start == nullptr || view.warp_end == nullptr ||
        view.sm_id == nullptr)
        return true;

    std::vector<unsigned long long> start(slots);
    std::vector<unsigned long long> end(slots);
    std::vector<uint32_t> smid(slots);
    if (!dmma_cuda_ok(cudaMemcpy(start.data(), view.warp_start,
                                 slots * sizeof(start[0]),
                                 cudaMemcpyDeviceToHost),
                      "copy timeline warp starts") ||
        !dmma_cuda_ok(cudaMemcpy(end.data(), view.warp_end,
                                 slots * sizeof(end[0]),
                                 cudaMemcpyDeviceToHost),
                      "copy timeline warp ends") ||
        !dmma_cuda_ok(cudaMemcpy(smid.data(), view.sm_id,
                                 slots * sizeof(smid[0]),
                                 cudaMemcpyDeviceToHost),
                      "copy timeline SM IDs"))
        return false;

    int device = 0;
    cudaDeviceProp properties{};
    if (!dmma_cuda_ok(cudaGetDevice(&device), "read timeline CUDA device") ||
        !dmma_cuda_ok(cudaGetDeviceProperties(&properties, device),
                      "read timeline CUDA properties"))
        return false;

    FILE *file = std::fopen(path, "wb");
    if (file == nullptr)
    {
        std::fprintf(stderr, "Cannot open DMMA timeline output: %s\n", path);
        return false;
    }
    const char magic[8] = {'D', 'M', 'T', 'L', '0', '0', '0', '1'};
    const uint32_t version = 1;
    const uint32_t sm_count = static_cast<uint32_t>(properties.multiProcessorCount);
    const uint32_t warps_per_block = DMMA_WARPS_PER_BLOCK;
    const uint32_t shift = view.sample_shift;
    const uint32_t phase = view.sample_phase;
    const uint64_t output_tiles = static_cast<uint64_t>(output_tile_count);
    const uint64_t blocks = static_cast<uint64_t>(grid_blocks);
    const uint64_t slot_count = static_cast<uint64_t>(slots);
    bool ok = true;
    ok = ok && std::fwrite(magic, sizeof(magic), 1, file) == 1;
    ok = ok && std::fwrite(&version, sizeof(version), 1, file) == 1;
    ok = ok && std::fwrite(&sm_count, sizeof(sm_count), 1, file) == 1;
    ok = ok && std::fwrite(&warps_per_block, sizeof(warps_per_block), 1, file) == 1;
    ok = ok && std::fwrite(&shift, sizeof(shift), 1, file) == 1;
    ok = ok && std::fwrite(&phase, sizeof(phase), 1, file) == 1;
    ok = ok && std::fwrite(&output_tiles, sizeof(output_tiles), 1, file) == 1;
    ok = ok && std::fwrite(&blocks, sizeof(blocks), 1, file) == 1;
    ok = ok && std::fwrite(&slot_count, sizeof(slot_count), 1, file) == 1;
    ok = ok && std::fwrite(start.data(), sizeof(start[0]), slots, file) == slots;
    ok = ok && std::fwrite(end.data(), sizeof(end[0]), slots, file) == slots;
    ok = ok && std::fwrite(smid.data(), sizeof(smid[0]), slots, file) == slots;
    if (std::fclose(file) != 0)
        ok = false;
    if (!ok)
        std::fprintf(stderr, "Failed while writing DMMA timeline: %s\n", path);
    else
        std::printf("DMMA_TIMELINE_TRACE=%s slots=%zu stride=%u phase=%u\n",
                    path, slots, 1u << shift, phase);
    return ok;
}
#endif

/* Production always materialises up to the largest count supported by the
 * legacy int-indexed output.  Tests may lower this compile-time boundary to
 * exercise the otherwise multi-billion-candidate fused path on tiny inputs. */
#ifdef DMMA_TEST_CANDIDATE_MATERIALIZE_LIMIT
static constexpr unsigned long long DMMA_CANDIDATE_MATERIALIZE_LIMIT =
    DMMA_TEST_CANDIDATE_MATERIALIZE_LIMIT;
#else
static constexpr unsigned long long DMMA_CANDIDATE_MATERIALIZE_LIMIT =
    static_cast<unsigned long long>(INT_MAX);
#endif

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
    unsigned long long candidate_tiles = 0;
    int output_tiles = 0;
    int output_nnz = 0;
    bool wide_output_unrepresentable = false;
    unsigned long long wide_output_tiles = 0;
    unsigned long long wide_output_nnz = 0;
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

/* Keep launch arithmetic wide even though matrix/tile counts remain int.
 * In particular, adding items_per_block - 1 to a count close to INT_MAX
 * must not overflow before the CUDA grid dimension is formed. */
static inline bool dmma_launch_blocks(std::size_t item_count,
                                      std::size_t items_per_block,
                                      unsigned int *block_count,
                                      const char *operation)
{
    if (block_count == nullptr || items_per_block == 0)
        return false;
    const std::size_t blocks =
        item_count / items_per_block +
        (item_count % items_per_block != 0 ? 1u : 0u);
    if (blocks == 0 || blocks > static_cast<std::size_t>(INT_MAX))
    {
        std::fprintf(stderr,
                     "Invalid CUDA grid size (%zu blocks) for %s.\n",
                     blocks, operation);
        return false;
    }
    *block_count = static_cast<unsigned int>(blocks);
    return true;
}

static inline bool dmma_exclusive_scan_int(int *data, std::size_t count,
                                           const char *operation)
{
    try
    {
        thrust::device_ptr<int> pointer(data);
        thrust::exclusive_scan(pointer, pointer + count, pointer);
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "Thrust error in %s: %s\n", operation,
                     error.what());
        cudaGetLastError();
        return false;
    }
    catch (...)
    {
        std::fprintf(stderr, "Unknown Thrust error in %s.\n", operation);
        cudaGetLastError();
        return false;
    }
    return dmma_cuda_ok(cudaGetLastError(), operation);
}

static inline bool dmma_reduce_int64(const int *data, std::size_t count,
                                     long long *result,
                                     const char *operation)
{
    if (result == nullptr)
        return false;
    try
    {
        thrust::device_ptr<const int> pointer(data);
        *result = thrust::reduce(pointer, pointer + count, 0ll,
                                 thrust::plus<long long>());
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "Thrust error in %s: %s\n", operation,
                     error.what());
        cudaGetLastError();
        return false;
    }
    catch (...)
    {
        std::fprintf(stderr, "Unknown Thrust error in %s.\n", operation);
        cudaGetLastError();
        return false;
    }
    return dmma_cuda_ok(cudaGetLastError(), operation);
}

static inline bool dmma_reduce_uint64(const unsigned long long *data,
                                      std::size_t count,
                                      unsigned long long *result,
                                      const char *operation)
{
    if (result == nullptr)
        return false;
    try
    {
        thrust::device_ptr<const unsigned long long> pointer(data);
        *result = thrust::reduce(pointer, pointer + count, 0ull,
                                 thrust::plus<unsigned long long>());
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "Thrust error in %s: %s\n", operation,
                     error.what());
        cudaGetLastError();
        return false;
    }
    catch (...)
    {
        std::fprintf(stderr, "Unknown Thrust error in %s.\n", operation);
        cudaGetLastError();
        return false;
    }
    return dmma_cuda_ok(cudaGetLastError(), operation);
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

/* CUDA's dim3 members are 32-bit. Cast before multiplication so a launch
 * covering more than 2^32 threads cannot wrap its candidate/output index. */
__device__ __forceinline__ std::size_t dmma_global_thread_index()
{
    return static_cast<std::size_t>(blockIdx.x) *
               static_cast<std::size_t>(blockDim.x) +
           static_cast<std::size_t>(threadIdx.x);
}

__global__ void dmma_candidate_count_spa_kernel(
    const int *__restrict__ a_row_ptr, const int *__restrict__ a_col_idx,
    int a_tile_rows, const int *__restrict__ b_row_ptr,
    const int *__restrict__ b_col_idx, int b_tile_cols, int *c_row_counts)
{
    const std::size_t warp = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    __shared__ uint32_t words[DMMA_WARPS_PER_BLOCK *
                              DMMA_SPA_WORDS_PER_WARP];
    if (warp >= static_cast<std::size_t>(a_tile_rows))
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
    const std::size_t warp = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    __shared__ uint32_t words[DMMA_WARPS_PER_BLOCK *
                              DMMA_SPA_WORDS_PER_WARP];
    if (warp >= static_cast<std::size_t>(a_tile_rows))
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
                c_row_idx[output] = static_cast<int>(warp);
                c_col_idx[output] = word * 32 + bit;
                ++local_offset;
            }
        }
        running += batch_count;
    }
}

__device__ __forceinline__ unsigned long long dmma_warp_sum_u64(
    unsigned long long value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return __shfl_sync(0xffffffffu, value, 0);
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
        value = min(value, __shfl_down_sync(0xffffffffu, value, offset));
    return __shfl_sync(0xffffffffu, value, 0);
}

/* Select only rows whose multiway union is likely more expensive than a
 * dense bitset pass.  Ratios to word_count keep ultra-wide, low-degree graph
 * rows on the sparse union path instead of turning symbolic work into
 * O(tile_rows * tile_cols / 32). */
__global__ void dmma_classify_wide_rows_kernel(
    const int *__restrict__ a_row_ptr, const int *__restrict__ a_col_idx,
    int a_tile_rows, const int *__restrict__ b_row_ptr, int word_count,
    int *bitset_flags, int *bitset_positions)
{
    const std::size_t row = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (row >= static_cast<std::size_t>(a_tile_rows))
        return;

    unsigned long long work = 0;
    for (int a = a_row_ptr[row] + lane; a < a_row_ptr[row + 1];
         a += WARP_SIZE)
    {
        const int k_tile = a_col_idx[a];
        work += static_cast<unsigned long long>(b_row_ptr[k_tile + 1] -
                                                b_row_ptr[k_tile]);
    }
    work = dmma_warp_sum_u64(work);
    if (lane == 0)
    {
        const unsigned long long words =
            static_cast<unsigned long long>(word_count);
        const int a_degree = a_row_ptr[row + 1] - a_row_ptr[row];
        const bool use_bitset =
            work >= 2ull * words ||
            (a_degree >= 32 && work >= (words + 1ull) / 2ull) ||
            (a_degree >= 128 && work >= (words + 7ull) / 8ull);
        const int flag = use_bitset ? 1 : 0;
        bitset_flags[row] = flag;
        bitset_positions[row] = flag;
    }
}

__global__ void dmma_compact_wide_rows_kernel(
    int row_count, const int *__restrict__ flags,
    const int *__restrict__ positions, int *rows)
{
    const std::size_t row = dmma_global_thread_index();
    if (row < static_cast<std::size_t>(row_count) && flags[row] != 0)
        rows[positions[row]] = static_cast<int>(row);
}

/* Low-work wide rows use a bounded-memory sorted multiway union. */
__global__ void dmma_candidate_count_wide_sparse_kernel(
    const int *__restrict__ a_row_ptr, const int *__restrict__ a_col_idx,
    int a_tile_rows, const int *__restrict__ b_row_ptr,
    const int *__restrict__ b_col_idx,
    const int *__restrict__ bitset_flags, int *c_row_counts)
{
    const std::size_t row = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (row >= static_cast<std::size_t>(a_tile_rows) ||
        bitset_flags[row] != 0)
        return;

    int previous = -1;
    int count = 0;
    while (true)
    {
        int lane_minimum = INT_MAX;
        for (int a = a_row_ptr[row] + lane; a < a_row_ptr[row + 1];
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
            break;
        previous = next;
        ++count;
    }
    if (lane == 0)
        c_row_counts[row] = count;
}

__global__ void dmma_candidate_fill_wide_sparse_kernel(
    const int *__restrict__ a_row_ptr, const int *__restrict__ a_col_idx,
    int a_tile_rows, const int *__restrict__ b_row_ptr,
    const int *__restrict__ b_col_idx,
    const int *__restrict__ bitset_flags,
    const int *__restrict__ c_row_ptr, int *c_row_idx, int *c_col_idx,
    int *count_mismatch)
{
    const std::size_t row = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (row >= static_cast<std::size_t>(a_tile_rows) ||
        bitset_flags[row] != 0)
        return;

    const int output_begin = c_row_ptr[row];
    const int output_end = c_row_ptr[row + 1];
    int previous = -1;
    for (int output = output_begin; output < output_end; ++output)
    {
        int lane_minimum = INT_MAX;
        for (int a = a_row_ptr[row] + lane; a < a_row_ptr[row + 1];
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
            c_row_idx[output] = static_cast<int>(row);
            c_col_idx[output] = next;
        }
        previous = next;
    }

    int lane_minimum = INT_MAX;
    for (int a = a_row_ptr[row] + lane; a < a_row_ptr[row + 1];
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

/* High-work rows are compacted and processed in batches with one global
 * bitset per resident row. Count and fill call the same mark helper. */
__device__ __forceinline__ void dmma_wide_clear_and_mark(
    const int *__restrict__ a_row_ptr, const int *__restrict__ a_col_idx,
    int row, const int *__restrict__ b_row_ptr,
    const int *__restrict__ b_col_idx, int word_count,
    uint32_t *__restrict__ bitset)
{
    for (int word = threadIdx.x; word < word_count; word += blockDim.x)
        bitset[word] = 0;
    __syncthreads();

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int block_warps = blockDim.x / WARP_SIZE;
    for (int a = a_row_ptr[row] + warp; a < a_row_ptr[row + 1];
         a += block_warps)
    {
        const int k_tile = a_col_idx[a];
        for (int b = b_row_ptr[k_tile] + lane; b < b_row_ptr[k_tile + 1];
             b += WARP_SIZE)
        {
            const int col = b_col_idx[b];
            atomicOr(bitset + col / 32, uint32_t(1) << (col & 31));
        }
    }
    __syncthreads();
}

__global__ void dmma_candidate_count_wide_bitset_kernel(
    const int *__restrict__ a_row_ptr, const int *__restrict__ a_col_idx,
    const int *__restrict__ bitset_rows, int batch_row_begin,
    int batch_row_count,
    const int *__restrict__ b_row_ptr, const int *__restrict__ b_col_idx,
    int word_count, uint32_t *__restrict__ bitsets, int *c_row_counts)
{
    const int batch_row = blockIdx.x;
    if (batch_row >= batch_row_count)
        return;
    const int row = bitset_rows[batch_row_begin + batch_row];
    uint32_t *bitset = bitsets +
                       static_cast<std::size_t>(batch_row) * word_count;
    dmma_wide_clear_and_mark(a_row_ptr, a_col_idx, row, b_row_ptr,
                             b_col_idx, word_count, bitset);

    __shared__ int sums[DMMA_WIDE_BLOCK_THREADS];
    int sum = 0;
    for (int word = threadIdx.x; word < word_count; word += blockDim.x)
        sum += __popc(bitset[word]);
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        c_row_counts[row] = sums[0];
}

__global__ void dmma_candidate_fill_wide_bitset_kernel(
    const int *__restrict__ a_row_ptr, const int *__restrict__ a_col_idx,
    const int *__restrict__ bitset_rows, int batch_row_begin,
    int batch_row_count,
    const int *__restrict__ b_row_ptr, const int *__restrict__ b_col_idx,
    int word_count, uint32_t *__restrict__ bitsets,
    const int *__restrict__ c_row_ptr, int *c_row_idx, int *c_col_idx,
    int *count_mismatch)
{
    const int batch_row = blockIdx.x;
    if (batch_row >= batch_row_count)
        return;
    const int row = bitset_rows[batch_row_begin + batch_row];
    uint32_t *bitset = bitsets +
                       static_cast<std::size_t>(batch_row) * word_count;
    dmma_wide_clear_and_mark(a_row_ptr, a_col_idx, row, b_row_ptr,
                             b_col_idx, word_count, bitset);

    __shared__ int prefix[DMMA_WIDE_BLOCK_THREADS];
    __shared__ int output_cursor;
    if (threadIdx.x == 0)
        output_cursor = c_row_ptr[row];
    __syncthreads();

    for (int word_base = 0; word_base < word_count;
         word_base += blockDim.x)
    {
        const int word = word_base + threadIdx.x;
        const uint32_t mask = word < word_count ? bitset[word] : 0u;
        const int own_count = __popc(mask);
        prefix[threadIdx.x] = own_count;
        __syncthreads();
#pragma unroll
        for (int offset = 1; offset < DMMA_WIDE_BLOCK_THREADS; offset <<= 1)
        {
            const int add = threadIdx.x >= offset
                                ? prefix[threadIdx.x - offset]
                                : 0;
            __syncthreads();
            prefix[threadIdx.x] += add;
            __syncthreads();
        }

        const int output_base = output_cursor;
        const int exclusive = prefix[threadIdx.x] - own_count;
        int local_offset = 0;
#pragma unroll
        for (int bit = 0; bit < 32; ++bit)
        {
            if ((mask & (uint32_t(1) << bit)) != 0)
            {
                const int output = output_base + exclusive + local_offset;
                c_row_idx[output] = row;
                c_col_idx[output] = word * 32 + bit;
                ++local_offset;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0)
            output_cursor += prefix[DMMA_WIDE_BLOCK_THREADS - 1];
        __syncthreads();
    }

    if (threadIdx.x == 0 && output_cursor != c_row_ptr[row + 1])
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

/* Compute the exact 8x8 structural mask for one candidate C tile.  Keeping
 * this operation in a reusable warp primitive lets the oversized-symbolic
 * path fuse candidate enumeration with exact-mask construction instead of
 * ever materialising a multi-billion-entry candidate array. */
__device__ __forceinline__ unsigned long long dmma_candidate_exact_mask(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int tile_row, int tile_col,
    int lane)
{
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
    return dmma_warp_or(local_output);
}

__global__ void dmma_exact_mask_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int candidate_count,
    const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols, uint64_t *candidate_masks,
    int *candidate_nnz, int *candidate_keep)
{
    const std::size_t global_warp =
        dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (global_warp >= static_cast<std::size_t>(candidate_count))
        return;

    const int tile_row = candidate_rows[global_warp];
    const int tile_col = candidate_cols[global_warp];
    const unsigned long long output_mask =
        dmma_candidate_exact_mask(a, b, tile_row, tile_col, lane);
    if (lane == 0)
    {
        candidate_masks[global_warp] = output_mask;
        const int nnz = __popcll(output_mask);
        candidate_nnz[global_warp] = nnz;
        candidate_keep[global_warp] = nnz != 0;
    }
}

/* Oversized candidate streams are enumerated and exact-tested in one pass.
 * The sorted multiway union is bounded by one warp per output tile row and
 * therefore needs no storage proportional to the candidate count. */
__global__ void dmma_fused_exact_count_sparse_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b,
    const int *__restrict__ bitset_flags, int *exact_row_tiles,
    unsigned long long *exact_row_nnz)
{
    const std::size_t row = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (row >= static_cast<std::size_t>(a.tile_row_count) ||
        (bitset_flags != nullptr && bitset_flags[row] != 0))
        return;

    int previous = -1;
    int tile_count = 0;
    unsigned long long nnz_count = 0;
    while (true)
    {
        int lane_minimum = INT_MAX;
        for (int pa = a.tile_row_ptr[row] + lane;
             pa < a.tile_row_ptr[row + 1]; pa += WARP_SIZE)
        {
            const int k_tile = a.tile_col_idx[pa];
            const int begin = b.tile_row_ptr[k_tile];
            const int end = b.tile_row_ptr[k_tile + 1];
            const int position =
                dmma_first_col_after(b.tile_col_idx, begin, end, previous);
            if (position < end)
                lane_minimum = min(lane_minimum, b.tile_col_idx[position]);
        }
        const int next = dmma_warp_min(lane_minimum);
        if (next == INT_MAX)
            break;

        const unsigned long long mask =
            dmma_candidate_exact_mask(a, b, static_cast<int>(row), next,
                                      lane);
        if (lane == 0 && mask != 0)
        {
            ++tile_count;
            nnz_count += static_cast<unsigned long long>(__popcll(mask));
        }
        previous = next;
    }

    if (lane == 0)
    {
        exact_row_tiles[row] = tile_count;
        exact_row_nnz[row] = nnz_count;
    }
}

__global__ void dmma_fused_exact_fill_sparse_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b,
    const int *__restrict__ bitset_flags,
    const int *__restrict__ exact_row_ptr, int *output_rows,
    int *output_cols, uint64_t *output_masks, int *output_nnz,
    int *count_mismatch)
{
    const std::size_t row = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (row >= static_cast<std::size_t>(a.tile_row_count) ||
        (bitset_flags != nullptr && bitset_flags[row] != 0))
        return;

    int previous = -1;
    int output_cursor = exact_row_ptr[row];
    while (true)
    {
        int lane_minimum = INT_MAX;
        for (int pa = a.tile_row_ptr[row] + lane;
             pa < a.tile_row_ptr[row + 1]; pa += WARP_SIZE)
        {
            const int k_tile = a.tile_col_idx[pa];
            const int begin = b.tile_row_ptr[k_tile];
            const int end = b.tile_row_ptr[k_tile + 1];
            const int position =
                dmma_first_col_after(b.tile_col_idx, begin, end, previous);
            if (position < end)
                lane_minimum = min(lane_minimum, b.tile_col_idx[position]);
        }
        const int next = dmma_warp_min(lane_minimum);
        if (next == INT_MAX)
            break;

        const unsigned long long mask =
            dmma_candidate_exact_mask(a, b, static_cast<int>(row), next,
                                      lane);
        if (lane == 0 && mask != 0)
        {
            if (output_cursor >= exact_row_ptr[row + 1])
            {
                atomicExch(count_mismatch, 1);
            }
            else
            {
                output_rows[output_cursor] = static_cast<int>(row);
                output_cols[output_cursor] = next;
                output_masks[output_cursor] = mask;
                output_nnz[output_cursor] = __popcll(mask);
                ++output_cursor;
            }
        }
        previous = next;
    }
    if (lane == 0 && output_cursor != exact_row_ptr[row + 1])
        atomicExch(count_mismatch, 1);
}

/* High-work rows reuse the bounded global bitsets already allocated by the
 * wide symbolic classifier.  Exact testing is deliberately performed by a
 * single warp after the cooperative mark so column order stays deterministic
 * and no per-row output cursor atomics are required. */
__global__ void dmma_fused_exact_count_bitset_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b,
    const int *__restrict__ bitset_rows, int batch_row_begin,
    int batch_row_count, int word_count, uint32_t *__restrict__ bitsets,
    int *exact_row_tiles, unsigned long long *exact_row_nnz)
{
    const int batch_row = blockIdx.x;
    if (batch_row >= batch_row_count)
        return;
    const int row = bitset_rows[batch_row_begin + batch_row];
    uint32_t *bitset = bitsets +
                       static_cast<std::size_t>(batch_row) * word_count;
    dmma_wide_clear_and_mark(a.tile_row_ptr, a.tile_col_idx, row,
                             b.tile_row_ptr, b.tile_col_idx, word_count,
                             bitset);

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int block_warps = blockDim.x / WARP_SIZE;
    int tile_count = 0;
    unsigned long long nnz_count = 0;
    for (int word = warp; word < word_count; word += block_warps)
    {
        uint32_t remaining = bitset[word];
        while (remaining != 0)
        {
            const int bit = __ffs(remaining) - 1;
            const int col = word * 32 + bit;
            const unsigned long long mask =
                dmma_candidate_exact_mask(a, b, row, col, lane);
            if (lane == 0 && mask != 0)
            {
                ++tile_count;
                nnz_count +=
                    static_cast<unsigned long long>(__popcll(mask));
            }
            remaining &= remaining - 1;
        }
    }

    __shared__ int warp_tile_counts[DMMA_WIDE_BLOCK_THREADS / WARP_SIZE];
    __shared__ unsigned long long
        warp_nnz_counts[DMMA_WIDE_BLOCK_THREADS / WARP_SIZE];
    if (lane == 0)
    {
        warp_tile_counts[warp] = tile_count;
        warp_nnz_counts[warp] = nnz_count;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        int row_tiles = 0;
        unsigned long long row_nnz = 0;
#pragma unroll
        for (int w = 0; w < DMMA_WIDE_BLOCK_THREADS / WARP_SIZE; ++w)
        {
            row_tiles += warp_tile_counts[w];
            row_nnz += warp_nnz_counts[w];
        }
        exact_row_tiles[row] = row_tiles;
        exact_row_nnz[row] = row_nnz;
    }
}

__global__ void dmma_fused_exact_fill_bitset_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b,
    const int *__restrict__ bitset_rows, int batch_row_begin,
    int batch_row_count, int word_count, uint32_t *__restrict__ bitsets,
    const int *__restrict__ exact_row_ptr, int *output_rows,
    int *output_cols, uint64_t *output_masks, int *output_nnz,
    int *count_mismatch)
{
    const int batch_row = blockIdx.x;
    if (batch_row >= batch_row_count)
        return;
    const int row = bitset_rows[batch_row_begin + batch_row];
    uint32_t *bitset = bitsets +
                       static_cast<std::size_t>(batch_row) * word_count;
    dmma_wide_clear_and_mark(a.tile_row_ptr, a.tile_col_idx, row,
                             b.tile_row_ptr, b.tile_col_idx, word_count,
                             bitset);
    if (threadIdx.x >= WARP_SIZE)
        return;

    const int lane = threadIdx.x;
    int output_cursor = exact_row_ptr[row];
    for (int word = 0; word < word_count; ++word)
    {
        uint32_t remaining = bitset[word];
        while (remaining != 0)
        {
            const int bit = __ffs(remaining) - 1;
            const int col = word * 32 + bit;
            const unsigned long long mask =
                dmma_candidate_exact_mask(a, b, row, col, lane);
            if (lane == 0 && mask != 0)
            {
                if (output_cursor >= exact_row_ptr[row + 1])
                {
                    atomicExch(count_mismatch, 1);
                }
                else
                {
                    output_rows[output_cursor] = row;
                    output_cols[output_cursor] = col;
                    output_masks[output_cursor] = mask;
                    output_nnz[output_cursor] = __popcll(mask);
                    ++output_cursor;
                }
            }
            remaining &= remaining - 1;
        }
    }
    if (lane == 0 && output_cursor != exact_row_ptr[row + 1])
        atomicExch(count_mismatch, 1);
}

/* Form the exact Boolean product of one 8x4 A mask and one 4x8 B
 * mask. B's physical mask is column-major (four k bits per output column),
 * matching dmma_candidate_exact_mask above. */
__device__ __forceinline__ unsigned long long dmma_tile_pair_exact_mask(
    uint32_t mask_a, uint32_t mask_b)
{
    unsigned long long output = 0;
    if (__popc(mask_a) * __popc(mask_b) <= DMMA_OUTPUT_ELEMS)
    {
        uint32_t remaining_a = mask_a;
        while (remaining_a != 0)
        {
            const int position_a = __ffs(remaining_a) - 1;
            const int row = position_a / DMMA_TILE_K;
            const int k = position_a % DMMA_TILE_K;
            uint32_t matching_b =
                mask_b & (uint32_t(0x11111111u) << k);
            while (matching_b != 0)
            {
                const int position_b = __ffs(matching_b) - 1;
                const int col = position_b / DMMA_TILE_K;
                output |= 1ull << (row * DMMA_TILE_N + col);
                matching_b &= matching_b - 1;
            }
            remaining_a &= remaining_a - 1;
        }
        return output;
    }

#pragma unroll
    for (int row = 0; row < DMMA_TILE_M; ++row)
    {
        const uint32_t a_bits =
            (mask_a >> (row * DMMA_TILE_K)) & 0xfu;
        unsigned int output_columns = 0;
#pragma unroll
        for (int col = 0; col < DMMA_TILE_N; ++col)
        {
            const uint32_t b_bits =
                (mask_b >> (col * DMMA_TILE_K)) & 0xfu;
            output_columns |= (a_bits & b_bits) != 0 ? 1u << col : 0u;
        }
        output |= static_cast<unsigned long long>(output_columns)
                  << (row * DMMA_TILE_N);
    }
    return output;
}

/* High-work rows are faster when exact masks are accumulated in the forward
 * direction: visit each A-tile x B-row tile pair once and OR its exact 8x8
 * product into a dense mask SPA indexed by C tile column. This replaces a
 * candidate-by-candidate merge of two long tile lists while staying bounded
 * by the same fixed 128 MiB scratch policy. */
__device__ __forceinline__ void dmma_exact_mask_spa_clear_and_mark(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int row,
    int output_tile_columns, unsigned long long *__restrict__ masks)
{
    for (int col = threadIdx.x; col < output_tile_columns;
         col += blockDim.x)
        masks[col] = 0;
    __syncthreads();

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    const int block_warps = blockDim.x / WARP_SIZE;
    for (int pa = a.tile_row_ptr[row] + warp;
         pa < a.tile_row_ptr[row + 1]; pa += block_warps)
    {
        const int k_tile = a.tile_col_idx[pa];
        const uint32_t mask_a = a.masks[pa];
        for (int pb = b.tile_row_ptr[k_tile] + lane;
             pb < b.tile_row_ptr[k_tile + 1]; pb += WARP_SIZE)
        {
            const unsigned long long pair_mask =
                dmma_tile_pair_exact_mask(mask_a, b.masks[pb]);
            if (pair_mask != 0)
                atomicOr(masks + b.tile_col_idx[pb], pair_mask);
        }
    }
    __syncthreads();
}

__global__ void dmma_fused_exact_count_mask_spa_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b,
    const int *__restrict__ bitset_rows, int batch_row_begin,
    int batch_row_count, int output_tile_columns,
    unsigned long long *__restrict__ mask_spas, int *exact_row_tiles,
    unsigned long long *exact_row_nnz)
{
    const int batch_row = blockIdx.x;
    if (batch_row >= batch_row_count)
        return;
    const int row = bitset_rows[batch_row_begin + batch_row];
    unsigned long long *masks =
        mask_spas + static_cast<std::size_t>(batch_row) *
                        output_tile_columns;
    dmma_exact_mask_spa_clear_and_mark(a, b, row, output_tile_columns,
                                       masks);

    int local_tiles = 0;
    unsigned long long local_nnz = 0;
    for (int col = threadIdx.x; col < output_tile_columns;
         col += blockDim.x)
    {
        const unsigned long long mask = masks[col];
        local_tiles += mask != 0;
        local_nnz += static_cast<unsigned long long>(__popcll(mask));
    }
    __shared__ int tile_sums[DMMA_WIDE_BLOCK_THREADS];
    __shared__ unsigned long long nnz_sums[DMMA_WIDE_BLOCK_THREADS];
    tile_sums[threadIdx.x] = local_tiles;
    nnz_sums[threadIdx.x] = local_nnz;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            tile_sums[threadIdx.x] += tile_sums[threadIdx.x + stride];
            nnz_sums[threadIdx.x] += nnz_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        exact_row_tiles[row] = tile_sums[0];
        exact_row_nnz[row] = nnz_sums[0];
    }
}

__global__ void dmma_fused_exact_fill_mask_spa_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b,
    const int *__restrict__ bitset_rows, int batch_row_begin,
    int batch_row_count, int output_tile_columns,
    unsigned long long *__restrict__ mask_spas,
    const int *__restrict__ exact_row_ptr, int *output_rows,
    int *output_cols, uint64_t *output_masks, int *output_nnz,
    int *count_mismatch)
{
    const int batch_row = blockIdx.x;
    if (batch_row >= batch_row_count)
        return;
    const int row = bitset_rows[batch_row_begin + batch_row];
    unsigned long long *masks =
        mask_spas + static_cast<std::size_t>(batch_row) *
                        output_tile_columns;
    dmma_exact_mask_spa_clear_and_mark(a, b, row, output_tile_columns,
                                       masks);

    __shared__ int prefix[DMMA_WIDE_BLOCK_THREADS];
    __shared__ int output_cursor;
    if (threadIdx.x == 0)
        output_cursor = exact_row_ptr[row];
    __syncthreads();
    for (int col_base = 0; col_base < output_tile_columns;
         col_base += blockDim.x)
    {
        const int col = col_base + threadIdx.x;
        const unsigned long long mask =
            col < output_tile_columns ? masks[col] : 0;
        const int own_count = mask != 0;
        prefix[threadIdx.x] = own_count;
        __syncthreads();
#pragma unroll
        for (int offset = 1; offset < DMMA_WIDE_BLOCK_THREADS; offset <<= 1)
        {
            const int add = threadIdx.x >= offset
                                ? prefix[threadIdx.x - offset]
                                : 0;
            __syncthreads();
            prefix[threadIdx.x] += add;
            __syncthreads();
        }
        if (own_count != 0)
        {
            const int output = output_cursor + prefix[threadIdx.x] - 1;
            output_rows[output] = row;
            output_cols[output] = col;
            output_masks[output] = mask;
            output_nnz[output] = __popcll(mask);
        }
        __syncthreads();
        if (threadIdx.x == 0)
            output_cursor += prefix[DMMA_WIDE_BLOCK_THREADS - 1];
        __syncthreads();
    }
    if (threadIdx.x == 0 && output_cursor != exact_row_ptr[row + 1])
        atomicExch(count_mismatch, 1);
}

__global__ void dmma_compact_candidates_kernel(
    int candidate_count, const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols,
    const uint64_t *__restrict__ candidate_masks,
    const int *__restrict__ candidate_nnz,
    const int *__restrict__ candidate_positions, int *output_rows,
    int *output_cols, uint64_t *output_masks, int *output_nnz)
{
    const std::size_t candidate = dmma_global_thread_index();
    if (candidate >= static_cast<std::size_t>(candidate_count) ||
        candidate_masks[candidate] == 0)
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
    const std::size_t tile = dmma_global_thread_index();
    if (tile < static_cast<std::size_t>(tile_count))
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
    unsigned char *output_col_idx, MAT_VAL_TYPE *output_values
#ifdef DMMA_ENABLE_TIMELINE_TRACE
    , DmmaTimelineView timeline
#endif
    )
{
#if __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;
    const std::size_t global_warp =
        dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    if (global_warp >= static_cast<std::size_t>(output_tile_count))
        return;

#ifdef DMMA_ENABLE_TIMELINE_TRACE
    if (timeline.warp_start != nullptr && lane == 0)
    {
        const unsigned int stride = 1u << timeline.sample_shift;
        const unsigned int block = blockIdx.x;
        if ((block & (stride - 1u)) == timeline.sample_phase)
        {
            const std::size_t sampled_block =
                static_cast<std::size_t>(block - timeline.sample_phase) >>
                timeline.sample_shift;
            const std::size_t slot =
                sampled_block * DMMA_WARPS_PER_BLOCK + local_warp;
            timeline.sm_id[slot] = dmma_read_smid();
            timeline.warp_start[slot] = dmma_read_globaltimer();
        }
    }
#endif

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

#ifdef DMMA_ENABLE_TIMELINE_TRACE
    if (timeline.warp_end != nullptr && lane == 0)
    {
        const unsigned int stride = 1u << timeline.sample_shift;
        const unsigned int block = blockIdx.x;
        if ((block & (stride - 1u)) == timeline.sample_phase)
        {
            const std::size_t sampled_block =
                static_cast<std::size_t>(block - timeline.sample_phase) >>
                timeline.sample_shift;
            const std::size_t slot =
                sampled_block * DMMA_WARPS_PER_BLOCK + local_warp;
            timeline.warp_end[slot] = dmma_read_globaltimer();
        }
    }
#endif
#endif
}

/* Build exact output metadata without a candidate array.  This path is used
 * only when the global candidate stream cannot be represented by an int.
 * Per-row exact counts are still int (a row cannot contain more than the int
 * tile-column count); global totals are reduced in 64 bits before any output
 * allocation or 32-bit scan is attempted. */
static inline bool dmma_build_fused_oversized_output_metadata(
    const DmmaDeviceTiles &a, const DmmaDeviceTiles &b,
    unsigned long long candidate_count,
    const int *d_wide_bitset_flags, const int *d_wide_bitset_rows,
    int wide_bitset_row_count, int wide_batch_capacity,
    int wide_word_count, uint32_t *d_wide_bitsets,
    int *d_exact_row_ptr, int **d_output_rows_result,
    int **d_output_cols_result, uint64_t **d_output_masks_result,
    int **d_output_nnz_result, int *output_tile_count_result,
    int *output_nnz_result, bool *wide_output_unrepresentable_result,
    unsigned long long *wide_output_tiles_result,
    unsigned long long *wide_output_nnz_result)
{
    if (d_exact_row_ptr == nullptr || d_output_rows_result == nullptr ||
        d_output_cols_result == nullptr || d_output_masks_result == nullptr ||
        d_output_nnz_result == nullptr ||
        output_tile_count_result == nullptr || output_nnz_result == nullptr ||
        wide_output_unrepresentable_result == nullptr ||
        wide_output_tiles_result == nullptr ||
        wide_output_nnz_result == nullptr)
        return false;

    *wide_output_unrepresentable_result = false;
    *wide_output_tiles_result = 0;
    *wide_output_nnz_result = 0;
    *d_output_rows_result = nullptr;
    *d_output_cols_result = nullptr;
    *d_output_masks_result = nullptr;
    *d_output_nnz_result = nullptr;
    *output_tile_count_result = 0;
    *output_nnz_result = 0;

    unsigned long long *d_exact_row_nnz = nullptr;
    unsigned long long *d_exact_mask_spas = nullptr;
    int *d_count_mismatch = nullptr;
    int *d_output_rows = nullptr;
    int *d_output_cols = nullptr;
    uint64_t *d_output_masks = nullptr;
    int *d_output_nnz = nullptr;
    long long exact_tile_count_wide = 0;
    unsigned long long exact_nnz_wide = 0;
    int exact_mask_spa_capacity = 0;
    int exact_count_batch_capacity = 0;
    int exact_fill_batch_capacity = 0;
    const std::size_t row_array_count =
        static_cast<std::size_t>(a.tile_row_count) + 1;

    if (!dmma_cuda_ok(cudaMemset(d_exact_row_ptr, 0,
                                 row_array_count * sizeof(int)),
                      "clear fused exact row counts") ||
        !dmma_cuda_ok(
            cudaMalloc(reinterpret_cast<void **>(&d_exact_row_nnz),
                       row_array_count * sizeof(unsigned long long)),
            "allocate fused exact row nnz") ||
        !dmma_cuda_ok(cudaMemset(d_exact_row_nnz, 0,
                                 row_array_count *
                                     sizeof(unsigned long long)),
                      "clear fused exact row nnz"))
        goto failure;

    if (wide_bitset_row_count > 0)
    {
        const std::size_t exact_mask_row_bytes =
            static_cast<std::size_t>(b.tile_col_count) *
            sizeof(unsigned long long);
        if (exact_mask_row_bytes > 0 &&
            exact_mask_row_bytes <= DMMA_WIDE_BITSET_SCRATCH_BYTES)
        {
            const std::size_t capacity =
                DMMA_WIDE_BITSET_SCRATCH_BYTES / exact_mask_row_bytes;
            exact_mask_spa_capacity = static_cast<int>(
                capacity < static_cast<std::size_t>(wide_bitset_row_count)
                    ? capacity
                    : static_cast<std::size_t>(wide_bitset_row_count));
            const std::size_t allocation_bytes =
                static_cast<std::size_t>(exact_mask_spa_capacity) *
                exact_mask_row_bytes;
            if (exact_mask_spa_capacity < 1 ||
                !dmma_cuda_ok(
                    cudaMalloc(
                        reinterpret_cast<void **>(&d_exact_mask_spas),
                        allocation_bytes),
                    "allocate bounded fused exact-mask SPAs"))
                goto failure;
        }
    }

    {
        unsigned int blocks = 0;
        if (!dmma_launch_blocks(
                static_cast<std::size_t>(a.tile_row_count),
                DMMA_WARPS_PER_BLOCK, &blocks,
                "fused oversized sparse exact count"))
            goto failure;
        dmma_fused_exact_count_sparse_kernel
            <<<blocks, DMMA_THREADS_PER_BLOCK>>>(
                a, b, d_wide_bitset_flags, d_exact_row_ptr,
                d_exact_row_nnz);
        if (!dmma_cuda_ok(cudaGetLastError(),
                          "launch fused oversized sparse exact count"))
            goto failure;
    }
    exact_count_batch_capacity =
        d_exact_mask_spas != nullptr ? exact_mask_spa_capacity
                                     : wide_batch_capacity;
    for (int row_begin = 0; row_begin < wide_bitset_row_count;
         row_begin += exact_count_batch_capacity)
    {
        const int remaining_rows = wide_bitset_row_count - row_begin;
        const int batch_rows = exact_count_batch_capacity < remaining_rows
                                   ? exact_count_batch_capacity
                                   : remaining_rows;
        if (d_exact_mask_spas != nullptr)
        {
            dmma_fused_exact_count_mask_spa_kernel
                <<<batch_rows, DMMA_WIDE_BLOCK_THREADS>>>(
                    a, b, d_wide_bitset_rows, row_begin, batch_rows,
                    b.tile_col_count, d_exact_mask_spas,
                    d_exact_row_ptr, d_exact_row_nnz);
        }
        else
        {
            dmma_fused_exact_count_bitset_kernel
                <<<batch_rows, DMMA_WIDE_BLOCK_THREADS>>>(
                    a, b, d_wide_bitset_rows, row_begin, batch_rows,
                    wide_word_count, d_wide_bitsets, d_exact_row_ptr,
                    d_exact_row_nnz);
        }
        if (!dmma_cuda_ok(cudaGetLastError(),
                          "launch fused oversized high-work exact count"))
            goto failure;
    }
    if (!dmma_cuda_ok(cudaDeviceSynchronize(),
                      "fused oversized exact count"))
        goto failure;

    if (!dmma_reduce_int64(d_exact_row_ptr,
                           static_cast<std::size_t>(a.tile_row_count),
                           &exact_tile_count_wide,
                           "reduce fused exact C tile count") ||
        !dmma_reduce_uint64(d_exact_row_nnz,
                            static_cast<std::size_t>(a.tile_row_count),
                            &exact_nnz_wide,
                            "reduce fused exact C nnz"))
        goto failure;

    *wide_output_tiles_result =
        static_cast<unsigned long long>(exact_tile_count_wide);
    *wide_output_nnz_result = exact_nnz_wide;

    if (exact_tile_count_wide < 0 || exact_tile_count_wide > INT_MAX ||
        exact_nnz_wide > static_cast<unsigned long long>(INT_MAX))
    {
        *wide_output_unrepresentable_result = true;
        std::fprintf(
            stderr,
            "Oversized symbolic stream handled without candidate "
            "materialization: candidates=%llu, exact C tiles=%lld, "
            "nnzC=%llu. The legacy SMatrix output uses 32-bit counts and "
            "cannot represent this exact result.\n",
            candidate_count, exact_tile_count_wide, exact_nnz_wide);
        goto failure;
    }

    *output_tile_count_result = static_cast<int>(exact_tile_count_wide);
    *output_nnz_result = static_cast<int>(exact_nnz_wide);
    if (!dmma_exclusive_scan_int(
            d_exact_row_ptr, row_array_count,
            "scan fused exact C row pointer"))
        goto failure;

    if (*output_tile_count_result == 0)
    {
        /* dmma_copy_output_to_host always copies the tile-nnz sentinel,
         * including for an empty output.  Preserve the same one-zero-element
         * contract as the ordinary candidate path. */
        if (!dmma_cuda_ok(
                cudaMalloc(reinterpret_cast<void **>(&d_output_nnz),
                           sizeof(int)),
                "allocate empty fused output nnz") ||
            !dmma_cuda_ok(cudaMemset(d_output_nnz, 0, sizeof(int)),
                          "clear empty fused output nnz"))
            goto failure;
        cudaFree(d_exact_row_nnz);
        cudaFree(d_exact_mask_spas);
        *d_output_nnz_result = d_output_nnz;
        return true;
    }

    if (!dmma_cuda_ok(
            cudaMalloc(reinterpret_cast<void **>(&d_output_rows),
                       static_cast<std::size_t>(*output_tile_count_result) *
                           sizeof(int)),
            "allocate fused output rows") ||
        !dmma_cuda_ok(
            cudaMalloc(reinterpret_cast<void **>(&d_output_cols),
                       static_cast<std::size_t>(*output_tile_count_result) *
                           sizeof(int)),
            "allocate fused output columns") ||
        !dmma_cuda_ok(
            cudaMalloc(reinterpret_cast<void **>(&d_output_masks),
                       static_cast<std::size_t>(*output_tile_count_result) *
                           sizeof(uint64_t)),
            "allocate fused output masks") ||
        !dmma_cuda_ok(
            cudaMalloc(reinterpret_cast<void **>(&d_output_nnz),
                       (static_cast<std::size_t>(*output_tile_count_result) +
                        1) * sizeof(int)),
            "allocate fused output nnz") ||
        !dmma_cuda_ok(
            cudaMemset(d_output_nnz, 0,
                       (static_cast<std::size_t>(*output_tile_count_result) +
                        1) * sizeof(int)),
            "clear fused output nnz") ||
        !dmma_cuda_ok(cudaMalloc(
                          reinterpret_cast<void **>(&d_count_mismatch),
                          sizeof(int)),
                      "allocate fused exact count check") ||
        !dmma_cuda_ok(cudaMemset(d_count_mismatch, 0, sizeof(int)),
                      "clear fused exact count check"))
        goto failure;

    {
        unsigned int blocks = 0;
        if (!dmma_launch_blocks(
                static_cast<std::size_t>(a.tile_row_count),
                DMMA_WARPS_PER_BLOCK, &blocks,
                "fused oversized sparse exact fill"))
            goto failure;
        dmma_fused_exact_fill_sparse_kernel
            <<<blocks, DMMA_THREADS_PER_BLOCK>>>(
                a, b, d_wide_bitset_flags, d_exact_row_ptr, d_output_rows,
                d_output_cols, d_output_masks, d_output_nnz,
                d_count_mismatch);
        if (!dmma_cuda_ok(cudaGetLastError(),
                          "launch fused oversized sparse exact fill"))
            goto failure;
    }
    exact_fill_batch_capacity =
        d_exact_mask_spas != nullptr ? exact_mask_spa_capacity
                                     : wide_batch_capacity;
    for (int row_begin = 0; row_begin < wide_bitset_row_count;
         row_begin += exact_fill_batch_capacity)
    {
        const int remaining_rows = wide_bitset_row_count - row_begin;
        const int batch_rows = exact_fill_batch_capacity < remaining_rows
                                   ? exact_fill_batch_capacity
                                   : remaining_rows;
        if (d_exact_mask_spas != nullptr)
        {
            dmma_fused_exact_fill_mask_spa_kernel
                <<<batch_rows, DMMA_WIDE_BLOCK_THREADS>>>(
                    a, b, d_wide_bitset_rows, row_begin, batch_rows,
                    b.tile_col_count, d_exact_mask_spas, d_exact_row_ptr,
                    d_output_rows, d_output_cols, d_output_masks,
                    d_output_nnz, d_count_mismatch);
        }
        else
        {
            dmma_fused_exact_fill_bitset_kernel
                <<<batch_rows, DMMA_WIDE_BLOCK_THREADS>>>(
                    a, b, d_wide_bitset_rows, row_begin, batch_rows,
                    wide_word_count, d_wide_bitsets, d_exact_row_ptr,
                    d_output_rows, d_output_cols, d_output_masks,
                    d_output_nnz, d_count_mismatch);
        }
        if (!dmma_cuda_ok(cudaGetLastError(),
                          "launch fused oversized high-work exact fill"))
            goto failure;
    }
    if (!dmma_cuda_ok(cudaDeviceSynchronize(),
                      "fused oversized exact fill"))
        goto failure;
    {
        int count_mismatch = 0;
        if (!dmma_cuda_ok(cudaMemcpy(&count_mismatch, d_count_mismatch,
                                     sizeof(int), cudaMemcpyDeviceToHost),
                          "read fused exact count check") ||
            count_mismatch != 0)
        {
            if (count_mismatch != 0)
                std::fprintf(stderr,
                             "Fused oversized exact count/fill mismatch.\n");
            goto failure;
        }
    }
    if (!dmma_exclusive_scan_int(
            d_output_nnz,
            static_cast<std::size_t>(*output_tile_count_result) + 1,
            "scan fused C nnz offsets"))
        goto failure;

    cudaFree(d_count_mismatch);
    cudaFree(d_exact_row_nnz);
    cudaFree(d_exact_mask_spas);
    *d_output_rows_result = d_output_rows;
    *d_output_cols_result = d_output_cols;
    *d_output_masks_result = d_output_masks;
    *d_output_nnz_result = d_output_nnz;
    return true;

failure:
    cudaFree(d_count_mismatch);
    cudaFree(d_exact_row_nnz);
    cudaFree(d_exact_mask_spas);
    cudaFree(d_output_rows);
    cudaFree(d_output_cols);
    cudaFree(d_output_masks);
    cudaFree(d_output_nnz);
    return false;
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
 * 16K output tile columns, while a bounded-scratch sorted union handles wider
 * matrices.  Numeric work is always one uniform DMMA kernel.
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
    uint32_t *d_wide_bitsets = nullptr;
    int *d_wide_bitset_flags = nullptr;
    int *d_wide_bitset_positions = nullptr;
    int *d_wide_bitset_rows = nullptr;
    int wide_word_count = 0;
    int wide_batch_capacity = 0;
    int wide_bitset_row_count = 0;
    int candidate_count = 0;
    unsigned long long candidate_count_total = 0;
    bool oversized_candidate_stream = false;
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
    unsigned int numeric_blocks = 0;
#ifdef DMMA_ENABLE_TIMELINE_TRACE
    DmmaTimelineView timeline{};
    std::size_t timeline_slots = 0;
    const char *timeline_path = nullptr;
#endif
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
        unsigned int blocks = 0;
        if (!dmma_launch_blocks(
                static_cast<std::size_t>(a.tile_row_count),
                DMMA_WARPS_PER_BLOCK, &blocks, "candidate count"))
            goto failure;
        dmma_candidate_count_spa_kernel<<<blocks, DMMA_THREADS_PER_BLOCK>>>(
            a.tile_row_ptr, a.tile_col_idx, a.tile_row_count,
            b.tile_row_ptr, b.tile_col_idx, b.tile_col_count,
            d_candidate_row_ptr);
        if (!dmma_cuda_ok(cudaGetLastError(), "launch candidate count") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(), "candidate count"))
            goto failure;
        long long candidate_count_shared = 0;
        if (!dmma_reduce_int64(
                d_candidate_row_ptr,
                static_cast<std::size_t>(a.tile_row_count),
                &candidate_count_shared,
                "reduce shared-SPA symbolic candidate count"))
            goto failure;
        if (candidate_count_shared < 0)
        {
            std::fprintf(stderr,
                         "Invalid negative symbolic candidate count "
                         "(%lld).\n",
                         candidate_count_shared);
            goto failure;
        }
        candidate_count_total =
            static_cast<unsigned long long>(candidate_count_shared);
        oversized_candidate_stream =
            candidate_count_shared > INT_MAX ||
            static_cast<unsigned long long>(candidate_count_shared) >
                DMMA_CANDIDATE_MATERIALIZE_LIMIT;
        if (!oversized_candidate_stream)
        {
            if (!dmma_exclusive_scan_int(
                    d_candidate_row_ptr,
                    static_cast<std::size_t>(a.tile_row_count) + 1,
                    "scan candidate row pointer"))
                goto failure;
            candidate_count = static_cast<int>(candidate_count_shared);
        }
    }
    else
    {
        const std::size_t word_count =
            (static_cast<std::size_t>(b.tile_col_count) + 31) / 32;
        const std::size_t row_bytes = word_count * sizeof(uint32_t);
        if (row_bytes == 0)
            goto failure;
        wide_word_count = static_cast<int>(word_count);

        const std::size_t row_array_count =
            static_cast<std::size_t>(a.tile_row_count) + 1;
        if (!dmma_cuda_ok(
                cudaMalloc(reinterpret_cast<void **>(&d_wide_bitset_flags),
                           row_array_count * sizeof(int)),
                "allocate wide symbolic row flags") ||
            !dmma_cuda_ok(
                cudaMalloc(
                    reinterpret_cast<void **>(&d_wide_bitset_positions),
                    row_array_count * sizeof(int)),
                "allocate wide symbolic row positions") ||
            !dmma_cuda_ok(cudaMemset(d_wide_bitset_flags, 0,
                                     row_array_count * sizeof(int)),
                          "clear wide symbolic row flags") ||
            !dmma_cuda_ok(cudaMemset(d_wide_bitset_positions, 0,
                                     row_array_count * sizeof(int)),
                          "clear wide symbolic row positions"))
            goto failure;

        /* If even one row cannot fit under the scratch cap, every row stays
         * on the sparse path. This remains correct and strictly bounded. */
        if (row_bytes <= DMMA_WIDE_BITSET_SCRATCH_BYTES)
        {
            unsigned int classify_blocks = 0;
            if (!dmma_launch_blocks(
                    static_cast<std::size_t>(a.tile_row_count),
                    DMMA_WARPS_PER_BLOCK, &classify_blocks,
                    "classify wide symbolic rows"))
                goto failure;
            dmma_classify_wide_rows_kernel
                <<<classify_blocks, DMMA_THREADS_PER_BLOCK>>>(
                    a.tile_row_ptr, a.tile_col_idx, a.tile_row_count,
                    b.tile_row_ptr, wide_word_count,
                    d_wide_bitset_flags, d_wide_bitset_positions);
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "classify wide symbolic rows") ||
                !dmma_exclusive_scan_int(
                    d_wide_bitset_positions, row_array_count,
                    "scan wide symbolic row flags") ||
                !dmma_cuda_ok(
                    cudaMemcpy(&wide_bitset_row_count,
                               d_wide_bitset_positions + a.tile_row_count,
                               sizeof(int), cudaMemcpyDeviceToHost),
                    "read wide bitset row count"))
                goto failure;

            if (wide_bitset_row_count > 0)
            {
                if (!dmma_cuda_ok(
                        cudaMalloc(
                            reinterpret_cast<void **>(&d_wide_bitset_rows),
                            static_cast<std::size_t>(wide_bitset_row_count) *
                                sizeof(int)),
                        "allocate compact wide symbolic rows"))
                    goto failure;
                const int threads = 256;
                unsigned int blocks = 0;
                if (!dmma_launch_blocks(
                        static_cast<std::size_t>(a.tile_row_count), threads,
                        &blocks, "compact wide symbolic rows"))
                    goto failure;
                dmma_compact_wide_rows_kernel<<<blocks, threads>>>(
                    a.tile_row_count, d_wide_bitset_flags,
                    d_wide_bitset_positions, d_wide_bitset_rows);
                if (!dmma_cuda_ok(cudaGetLastError(),
                                  "compact wide symbolic rows"))
                    goto failure;

                wide_batch_capacity = static_cast<int>(
                    DMMA_WIDE_BITSET_SCRATCH_BYTES / row_bytes);
                if (wide_batch_capacity > wide_bitset_row_count)
                    wide_batch_capacity = wide_bitset_row_count;
                const std::size_t scratch_words =
                    static_cast<std::size_t>(wide_batch_capacity) *
                    word_count;
                if (wide_batch_capacity < 1 ||
                    !dmma_cuda_ok(
                        cudaMalloc(
                            reinterpret_cast<void **>(&d_wide_bitsets),
                            scratch_words * sizeof(uint32_t)),
                        "allocate bounded wide symbolic bitsets"))
                    goto failure;
            }
        }

        unsigned int sparse_blocks = 0;
        if (!dmma_launch_blocks(
                static_cast<std::size_t>(a.tile_row_count),
                DMMA_WARPS_PER_BLOCK, &sparse_blocks,
                "sparse wide candidate count"))
            goto failure;
        dmma_candidate_count_wide_sparse_kernel
            <<<sparse_blocks, DMMA_THREADS_PER_BLOCK>>>(
                a.tile_row_ptr, a.tile_col_idx, a.tile_row_count,
                b.tile_row_ptr, b.tile_col_idx, d_wide_bitset_flags,
                d_candidate_row_ptr);
        if (!dmma_cuda_ok(cudaGetLastError(),
                          "launch sparse wide candidate count"))
            goto failure;
        for (int row_begin = 0; row_begin < wide_bitset_row_count;
             row_begin += wide_batch_capacity)
        {
            const int remaining_rows = wide_bitset_row_count - row_begin;
            const int batch_rows = wide_batch_capacity < remaining_rows
                                       ? wide_batch_capacity
                                       : remaining_rows;
            dmma_candidate_count_wide_bitset_kernel
                <<<batch_rows, DMMA_WIDE_BLOCK_THREADS>>>(
                    a.tile_row_ptr, a.tile_col_idx, d_wide_bitset_rows,
                    row_begin, batch_rows, b.tile_row_ptr, b.tile_col_idx,
                    wide_word_count, d_wide_bitsets,
                    d_candidate_row_ptr);
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "launch bitset wide candidate count"))
                goto failure;
        }
        if (!dmma_cuda_ok(cudaDeviceSynchronize(),
                          "hybrid wide candidate count"))
            goto failure;

        long long candidate_count_wide = 0;
        if (!dmma_reduce_int64(
                d_candidate_row_ptr,
                static_cast<std::size_t>(a.tile_row_count),
                &candidate_count_wide,
                "reduce wide symbolic candidate count"))
            goto failure;
        if (candidate_count_wide < 0)
        {
            std::fprintf(stderr,
                         "Invalid negative wide symbolic candidate count.\n");
            goto failure;
        }
        candidate_count_total =
            static_cast<unsigned long long>(candidate_count_wide);
        oversized_candidate_stream =
            candidate_count_wide > INT_MAX ||
            static_cast<unsigned long long>(candidate_count_wide) >
                DMMA_CANDIDATE_MATERIALIZE_LIMIT;
        if (!oversized_candidate_stream)
        {
            if (!dmma_exclusive_scan_int(
                    d_candidate_row_ptr,
                    static_cast<std::size_t>(a.tile_row_count) + 1,
                    "scan wide candidate row pointer"))
                goto failure;
            candidate_count = static_cast<int>(candidate_count_wide);
        }
    }

    if (!oversized_candidate_stream)
        candidate_count_total = static_cast<unsigned long long>(candidate_count);

    if (oversized_candidate_stream)
    {
        gettimeofday(&end, nullptr);
        stats->candidate_ms = dmma_elapsed_ms(begin, end);
        stats->candidate_tiles = candidate_count_total;
        std::printf("DMMA candidate stream=%llu uses bounded fused exact "
                    "symbolic (no candidate array).\n",
                    candidate_count_total);
        gettimeofday(&begin, nullptr);

        const bool fused_symbolic_ok =
            dmma_build_fused_oversized_output_metadata(
                a, b, candidate_count_total, d_wide_bitset_flags,
                d_wide_bitset_rows, wide_bitset_row_count,
                wide_batch_capacity, wide_word_count, d_wide_bitsets,
                d_candidate_row_ptr, &d_output_rows, &d_output_cols,
                &d_output_masks, &d_output_nnz, &output_tile_count,
                &output_nnz, &stats->wide_output_unrepresentable,
                &stats->wide_output_tiles, &stats->wide_output_nnz);
        if (!fused_symbolic_ok)
        {
            gettimeofday(&end, nullptr);
            stats->symbolic_ms = dmma_elapsed_ms(begin, end);
            gettimeofday(&total_end, nullptr);
            stats->total_ms = dmma_elapsed_ms(total_begin, total_end);
            goto failure;
        }

        d_output_row_ptr = d_candidate_row_ptr;
        d_candidate_row_ptr = nullptr;
        cudaFree(d_wide_bitsets);
        d_wide_bitsets = nullptr;
        cudaFree(d_wide_bitset_flags);
        d_wide_bitset_flags = nullptr;
        cudaFree(d_wide_bitset_positions);
        d_wide_bitset_positions = nullptr;
        cudaFree(d_wide_bitset_rows);
        d_wide_bitset_rows = nullptr;

        gettimeofday(&end, nullptr);
        stats->symbolic_ms = dmma_elapsed_ms(begin, end);
        stats->output_tiles = output_tile_count;
        stats->output_nnz = output_nnz;

        if (output_tile_count == 0)
        {
            gettimeofday(&total_end, nullptr);
            stats->total_ms = dmma_elapsed_ms(total_begin, total_end);
            gettimeofday(&begin, nullptr);
            if (!dmma_copy_output_to_host(
                    a.rows, b.cols, a.tile_row_count, b.tile_col_count,
                    0, 0, d_output_row_ptr, nullptr, d_output_nnz,
                    nullptr, nullptr, nullptr, output))
                goto symbolic_failure;
            gettimeofday(&end, nullptr);
            stats->output_copy_ms = dmma_elapsed_ms(begin, end);
            cudaFree(d_output_nnz);
            cudaFree(d_output_row_ptr);
            return true;
        }
        goto numeric_phase;
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
        unsigned int blocks = 0;
        if (!dmma_launch_blocks(
                static_cast<std::size_t>(a.tile_row_count),
                DMMA_WARPS_PER_BLOCK, &blocks, "candidate fill"))
            goto failure;
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
        unsigned int sparse_blocks = 0;
        if (!dmma_launch_blocks(
                static_cast<std::size_t>(a.tile_row_count),
                DMMA_WARPS_PER_BLOCK, &sparse_blocks,
                "sparse wide candidate fill"))
            goto failure;
        dmma_candidate_fill_wide_sparse_kernel
            <<<sparse_blocks, DMMA_THREADS_PER_BLOCK>>>(
                a.tile_row_ptr, a.tile_col_idx, a.tile_row_count,
                b.tile_row_ptr, b.tile_col_idx, d_wide_bitset_flags,
                d_candidate_row_ptr, d_candidate_rows, d_candidate_cols,
                d_candidate_count_mismatch);
        if (!dmma_cuda_ok(cudaGetLastError(),
                          "launch sparse wide candidate fill"))
            goto failure;
        for (int row_begin = 0; row_begin < wide_bitset_row_count;
             row_begin += wide_batch_capacity)
        {
            const int remaining_rows = wide_bitset_row_count - row_begin;
            const int batch_rows = wide_batch_capacity < remaining_rows
                                       ? wide_batch_capacity
                                       : remaining_rows;
            dmma_candidate_fill_wide_bitset_kernel
                <<<batch_rows, DMMA_WIDE_BLOCK_THREADS>>>(
                    a.tile_row_ptr, a.tile_col_idx, d_wide_bitset_rows,
                    row_begin, batch_rows, b.tile_row_ptr, b.tile_col_idx,
                    wide_word_count, d_wide_bitsets, d_candidate_row_ptr,
                    d_candidate_rows, d_candidate_cols,
                    d_candidate_count_mismatch);
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "launch bitset wide candidate fill"))
                goto failure;
        }
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
    cudaFree(d_wide_bitsets);
    d_wide_bitsets = nullptr;
    cudaFree(d_wide_bitset_flags);
    d_wide_bitset_flags = nullptr;
    cudaFree(d_wide_bitset_positions);
    d_wide_bitset_positions = nullptr;
    cudaFree(d_wide_bitset_rows);
    d_wide_bitset_rows = nullptr;
    gettimeofday(&end, nullptr);
    stats->candidate_ms = dmma_elapsed_ms(begin, end);
    stats->candidate_tiles = candidate_count_total;

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
        unsigned int blocks = 0;
        if (!dmma_launch_blocks(
                static_cast<std::size_t>(candidate_count),
                DMMA_WARPS_PER_BLOCK, &blocks, "exact C masks"))
            goto symbolic_failure;
        dmma_exact_mask_kernel<<<blocks, DMMA_THREADS_PER_BLOCK>>>(
            a, b, candidate_count, d_candidate_rows,
            d_candidate_cols, d_candidate_masks, d_candidate_nnz,
            d_candidate_keep);
        if (!dmma_cuda_ok(cudaGetLastError(), "launch exact C masks") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(), "exact C masks"))
            goto symbolic_failure;
    }
    {
        if (!dmma_exclusive_scan_int(
                d_candidate_keep,
                static_cast<std::size_t>(candidate_count) + 1,
                "scan exact C tile flags"))
            goto symbolic_failure;
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
        unsigned int blocks = 0;
        unsigned int output_blocks = 0;
        if (!dmma_launch_blocks(
                static_cast<std::size_t>(candidate_count), threads,
                &blocks, "compact exact C tiles") ||
            !dmma_launch_blocks(
                static_cast<std::size_t>(output_tile_count), threads,
                &output_blocks, "count exact C tile rows"))
            goto symbolic_failure;
        dmma_compact_candidates_kernel<<<blocks, threads>>>(
            candidate_count, d_candidate_rows, d_candidate_cols,
            d_candidate_masks, d_candidate_nnz, d_candidate_keep,
            d_output_rows, d_output_cols, d_output_masks, d_output_nnz);
        dmma_count_output_rows_kernel<<<output_blocks, threads>>>(
            output_tile_count, d_output_rows, d_output_row_ptr);
        if (!dmma_cuda_ok(cudaGetLastError(), "compact exact C tiles") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(), "compact exact C tiles"))
            goto symbolic_failure;
        if (!dmma_exclusive_scan_int(
                d_output_row_ptr,
                static_cast<std::size_t>(a.tile_row_count) + 1,
                "scan exact C row pointer"))
            goto symbolic_failure;
    }
    {
        long long output_nnz_wide = 0;
        if (!dmma_reduce_int64(
                d_output_nnz,
                static_cast<std::size_t>(output_tile_count),
                &output_nnz_wide, "reduce C nnz before offset scan"))
            goto symbolic_failure;
        if (output_nnz_wide < 0 || output_nnz_wide > INT_MAX)
        {
            /* The ordinary candidate path can still produce a structurally
             * valid C whose scalar nnz exceeds the legacy SMatrix32 output
             * boundary (mip1 is one such case).  Preserve the exact symbolic
             * totals and timings so callers report this as a wide-interface
             * limitation rather than a generic implementation failure. */
            if (output_nnz_wide > INT_MAX)
            {
                stats->wide_output_unrepresentable = true;
                stats->wide_output_tiles =
                    static_cast<unsigned long long>(output_tile_count);
                stats->wide_output_nnz =
                    static_cast<unsigned long long>(output_nnz_wide);
                gettimeofday(&end, nullptr);
                stats->symbolic_ms = dmma_elapsed_ms(begin, end);
                gettimeofday(&total_end, nullptr);
                stats->total_ms =
                    dmma_elapsed_ms(total_begin, total_end);
            }
            std::fprintf(stderr,
                         "C nnz count (%lld) exceeds 32-bit storage.\n",
                         output_nnz_wide);
            goto symbolic_failure;
        }
        if (!dmma_exclusive_scan_int(
                d_output_nnz,
                static_cast<std::size_t>(output_tile_count) + 1,
                "scan C nnz offsets"))
            goto symbolic_failure;
        output_nnz = static_cast<int>(output_nnz_wide);
    }

    /* Candidate storage is dead once the exact output metadata and both
     * output scans have been built.  Releasing it before the numeric output
     * allocation is important for high-expansion products: keeping the
     * candidate rows/columns/masks/counts/scan alive costs roughly 24 bytes
     * per symbolic candidate and used to make otherwise representable kmer
     * products fail during the numeric allocation. */
    cudaFree(d_candidate_masks);
    d_candidate_masks = nullptr;
    cudaFree(d_candidate_nnz);
    d_candidate_nnz = nullptr;
    cudaFree(d_candidate_keep);
    d_candidate_keep = nullptr;
    cudaFree(d_candidate_row_ptr);
    d_candidate_row_ptr = nullptr;
    cudaFree(d_candidate_rows);
    d_candidate_rows = nullptr;
    cudaFree(d_candidate_cols);
    d_candidate_cols = nullptr;

    gettimeofday(&end, nullptr);
    stats->symbolic_ms = dmma_elapsed_ms(begin, end);
    stats->output_tiles = output_tile_count;
    stats->output_nnz = output_nnz;

numeric_phase:
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
#ifdef DMMA_ENABLE_TIMELINE_TRACE
    timeline_path = std::getenv("DMMA_TRACE_FILE");
    if (timeline_path != nullptr && *timeline_path != '\0')
    {
        timeline.sample_shift =
            dmma_timeline_env_uint("DMMA_TRACE_SHIFT", 4, 10);
        const unsigned int stride = 1u << timeline.sample_shift;
        const unsigned int default_phase = stride > 7u ? 7u : stride - 1u;
        timeline.sample_phase =
            dmma_timeline_env_uint("DMMA_TRACE_PHASE", default_phase,
                                   stride - 1u);
        if (!dmma_launch_blocks(
                static_cast<std::size_t>(output_tile_count),
                DMMA_WARPS_PER_BLOCK, &numeric_blocks,
                "timeline uniform DMMA kernel"))
            goto numeric_failure;
        const std::size_t sampled_blocks =
            numeric_blocks <= timeline.sample_phase
                ? 0
                : (static_cast<std::size_t>(numeric_blocks - 1u -
                                            timeline.sample_phase) /
                       stride +
                   1u);
        timeline_slots = sampled_blocks * DMMA_WARPS_PER_BLOCK;
        if (timeline_slots != 0 &&
            (!dmma_cuda_ok(cudaMalloc(
                               reinterpret_cast<void **>(&timeline.warp_start),
                               timeline_slots * sizeof(*timeline.warp_start)),
                           "allocate timeline warp starts") ||
             !dmma_cuda_ok(cudaMalloc(
                               reinterpret_cast<void **>(&timeline.warp_end),
                               timeline_slots * sizeof(*timeline.warp_end)),
                           "allocate timeline warp ends") ||
             !dmma_cuda_ok(cudaMalloc(
                               reinterpret_cast<void **>(&timeline.sm_id),
                               timeline_slots * sizeof(*timeline.sm_id)),
                           "allocate timeline SM IDs") ||
             !dmma_cuda_ok(cudaMemset(
                               timeline.warp_start, 0,
                               timeline_slots * sizeof(*timeline.warp_start)),
                           "clear timeline warp starts") ||
             !dmma_cuda_ok(cudaMemset(
                               timeline.warp_end, 0,
                               timeline_slots * sizeof(*timeline.warp_end)),
                           "clear timeline warp ends") ||
             !dmma_cuda_ok(cudaMemset(
                               timeline.sm_id, 0xff,
                               timeline_slots * sizeof(*timeline.sm_id)),
                           "clear timeline SM IDs")))
            goto numeric_failure;
    }
#endif
    gettimeofday(&begin, nullptr);
    {
        if (numeric_blocks == 0 && !dmma_launch_blocks(
                static_cast<std::size_t>(output_tile_count),
                DMMA_WARPS_PER_BLOCK, &numeric_blocks,
                "uniform DMMA kernel"))
            goto numeric_failure;
        dmma_numeric_kernel<<<numeric_blocks, DMMA_THREADS_PER_BLOCK>>>(
            a, b, output_tile_count, d_output_rows, d_output_cols,
            d_output_masks, d_output_nnz, d_output_tile_row_ptr,
            d_output_value_cols, d_output_values
#ifdef DMMA_ENABLE_TIMELINE_TRACE
            , timeline
#endif
            );
        if (!dmma_cuda_ok(cudaGetLastError(), "launch uniform DMMA kernel") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(), "uniform DMMA kernel"))
            goto numeric_failure;
    }
    gettimeofday(&end, nullptr);
    stats->numeric_ms = dmma_elapsed_ms(begin, end);

#ifdef DMMA_ENABLE_TIMELINE_TRACE
    if (timeline_path != nullptr && *timeline_path != '\0' &&
        !dmma_write_timeline_trace(
            timeline_path, timeline, timeline_slots, numeric_blocks,
            output_tile_count))
        std::fprintf(stderr, "DMMA timeline export failed; continuing.\n");
    cudaFree(timeline.warp_start);
    cudaFree(timeline.warp_end);
    cudaFree(timeline.sm_id);
    timeline = DmmaTimelineView();
#endif

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
    cudaFree(d_wide_bitsets);
    cudaFree(d_wide_bitset_flags);
    cudaFree(d_wide_bitset_positions);
    cudaFree(d_wide_bitset_rows);
    return true;

numeric_failure:
#ifdef DMMA_ENABLE_TIMELINE_TRACE
    cudaFree(timeline.warp_start);
    cudaFree(timeline.warp_end);
    cudaFree(timeline.sm_id);
#endif
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
    cudaFree(d_candidate_row_ptr);
    cudaFree(d_candidate_rows);
    cudaFree(d_candidate_cols);
    cudaFree(d_candidate_count_mismatch);
    cudaFree(d_wide_bitsets);
    cudaFree(d_wide_bitset_flags);
    cudaFree(d_wide_bitset_positions);
    cudaFree(d_wide_bitset_rows);
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
