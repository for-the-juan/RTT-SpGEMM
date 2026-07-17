#ifndef RTT_SPGEMM_GPU_DMMA_TILES_H_
#define RTT_SPGEMM_GPU_DMMA_TILES_H_

#include "dmma_spgemm.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <new>
#include <type_traits>

static_assert(std::is_same<MAT_PTR_TYPE, int>::value,
              "GPU DMMA preprocessing currently requires 32-bit CSR offsets");

struct DmmaOwnedDeviceCsr
{
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    int *row_ptr = nullptr;
    int *col_idx = nullptr;
    MAT_VAL_TYPE *values = nullptr;
};

struct DmmaPreprocessStats
{
    double h2d_ms = 0.0;
    double validation_ms = 0.0;
    double a_key_sort_reduce_ms = 0.0;
    double a_tile_build_ms = 0.0;
    double b_key_sort_reduce_ms = 0.0;
    double b_tile_build_ms = 0.0;
    double b_csc_ms = 0.0;
    double nnz_cub_ms = 0.0;
    double total_ms = 0.0;
    std::size_t peak_workspace_bytes = 0;
};

struct DmmaPreparedOperands
{
    DmmaOwnedDeviceCsr csr;
    DmmaOwnedDeviceTiles a;
    DmmaOwnedDeviceTiles b;
    unsigned long long nnz_cub = 0;
    bool aat = false;
};

static inline void destroy_device_csr(DmmaOwnedDeviceCsr *csr)
{
    if (csr == nullptr)
        return;
    cudaFree(csr->row_ptr);
    cudaFree(csr->col_idx);
    cudaFree(csr->values);
    *csr = DmmaOwnedDeviceCsr();
}

static inline void destroy_prepared_operands(DmmaPreparedOperands *prepared)
{
    if (prepared == nullptr)
        return;
    destroy_device_tiles(&prepared->a);
    destroy_device_tiles(&prepared->b);
    destroy_device_csr(&prepared->csr);
    *prepared = DmmaPreparedOperands();
}

namespace gpu_dmma_detail
{

using Clock = std::chrono::steady_clock;
using EntryKey = unsigned long long;

static constexpr int kThreads = 256;
static constexpr int kTileThreads = 64;

static inline unsigned int block_count(std::size_t count,
                                       unsigned int threads = kThreads)
{
    return static_cast<unsigned int>(
        count / threads + (count % threads != 0 ? 1 : 0));
}

static inline double milliseconds(Clock::time_point begin,
                                  Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

template <typename T>
static inline bool device_allocate(T **pointer, std::size_t count,
                                   const char *label)
{
    *pointer = nullptr;
    if (count == 0)
        return true;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    {
        std::fprintf(stderr, "GPU preprocessing size overflow in %s.\n", label);
        return false;
    }
    return dmma_cuda_ok(
        cudaMalloc(reinterpret_cast<void **>(pointer), count * sizeof(T)),
        label);
}

__global__ void validate_csr_kernel(const int *row_ptr, const int *col_idx,
                                    int rows, int cols, int nnz, int *error)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index <= rows)
    {
        const int value = row_ptr[index];
        if (value < 0 || value > nnz ||
            (index == 0 && value != 0) ||
            (index == rows && value != nnz) ||
            (index < rows && value > row_ptr[index + 1]))
            atomicExch(error, 1);
    }
    if (index < nnz)
    {
        const int column = col_idx[index];
        if (column < 0 || column >= cols)
            atomicExch(error, 1);
    }
}

__global__ void fill_entry_keys_kernel(
    const int *row_ptr, const int *col_idx, const MAT_VAL_TYPE *values,
    int source_rows, int logical_cols, int tile_rows, int tile_cols,
    int tile_col_count, bool payload_col_major, bool logical_transpose,
    EntryKey *keys, MAT_VAL_TYPE *key_values)
{
    const int source_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (source_row >= source_rows)
        return;
    for (int entry = row_ptr[source_row]; entry < row_ptr[source_row + 1];
         ++entry)
    {
        const int source_col = col_idx[entry];
        const int logical_row = logical_transpose ? source_col : source_row;
        const int logical_col = logical_transpose ? source_row : source_col;
        if (logical_col < 0 || logical_col >= logical_cols)
            continue;
        const int tile_row = logical_row / tile_rows;
        const int tile_col = logical_col / tile_cols;
        const int local_row = logical_row % tile_rows;
        const int local_col = logical_col % tile_cols;
        const int physical = payload_col_major
                                 ? local_col * tile_rows + local_row
                                 : local_row * tile_cols + local_col;
        const EntryKey tile_key =
            static_cast<EntryKey>(tile_row) *
                static_cast<EntryKey>(tile_col_count) +
            static_cast<EntryKey>(tile_col);
        keys[entry] = tile_key * DMMA_INPUT_ELEMS + physical;
        key_values[entry] = values[entry];
    }
}

struct EntryToTileKey
{
    __host__ __device__ EntryKey operator()(EntryKey key) const
    {
        return key / DMMA_INPUT_ELEMS;
    }
};

__global__ void build_tile_metadata_kernel(
    const EntryKey *tile_keys, const int *tile_counts, int tile_count,
    int tile_col_count, int dense_threshold, int *tile_row_ptr,
    int *tile_col_idx, int *payload_spans, int *dense_flags)
{
    const int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= tile_count)
        return;
    const EntryKey key = tile_keys[tile];
    const int tile_row = static_cast<int>(key / tile_col_count);
    const int tile_col = static_cast<int>(key % tile_col_count);
    const int count = tile_counts[tile];
    tile_col_idx[tile] = tile_col;
    atomicAdd(tile_row_ptr + tile_row + 1, 1);
    const int dense = count >= dense_threshold ? 1 : 0;
    dense_flags[tile] = dense;
    payload_spans[tile] = dense ? DMMA_INPUT_ELEMS : count;
}

__global__ void pack_tiles_kernel(
    const EntryKey *entry_keys, const MAT_VAL_TYPE *entry_values,
    const int *entry_offsets, const int *entry_counts,
    const int *value_offsets, int tile_count, int dense_threshold,
    uint32_t *masks, MAT_VAL_TYPE *payload)
{
    const int tile = blockIdx.x;
    const int lane = threadIdx.x;
    if (tile >= tile_count)
        return;
    const int count = entry_counts[tile];
    const int begin = entry_offsets[tile];
    const int output = value_offsets[tile];
    const bool dense = count >= dense_threshold;
    if (lane == 0)
        masks[tile] = 0;
    if (dense && lane < DMMA_INPUT_ELEMS)
        payload[output + lane] = MAT_VAL_TYPE(0);
    __syncthreads();
    if (lane < count)
    {
        const int physical =
            static_cast<int>(entry_keys[begin + lane] % DMMA_INPUT_ELEMS);
        atomicOr(masks + tile, uint32_t(1) << physical);
        payload[output + (dense ? physical : lane)] =
            entry_values[begin + lane];
    }
}

__global__ void make_csc_keys_kernel(const EntryKey *csr_tile_keys,
                                     int tile_count, int tile_row_count,
                                     int tile_col_count, EntryKey *csc_keys,
                                     int *csc_ids)
{
    const int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= tile_count)
        return;
    const EntryKey key = csr_tile_keys[tile];
    const int row = static_cast<int>(key / tile_col_count);
    const int col = static_cast<int>(key % tile_col_count);
    csc_keys[tile] = static_cast<EntryKey>(col) * tile_row_count + row;
    csc_ids[tile] = tile;
}

__global__ void finish_csc_kernel(const EntryKey *csc_keys, int tile_count,
                                  int tile_row_count, int *tile_col_ptr,
                                  int *tile_row_idx)
{
    const int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= tile_count)
        return;
    const EntryKey key = csc_keys[tile];
    const int col = static_cast<int>(key / tile_row_count);
    const int row = static_cast<int>(key % tile_row_count);
    tile_row_idx[tile] = row;
    atomicAdd(tile_col_ptr + col + 1, 1);
}

__global__ void count_column_degrees_kernel(const int *col_idx, int nnz,
                                            int *degrees)
{
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry < nnz)
        atomicAdd(degrees + col_idx[entry], 1);
}

__global__ void fill_nnz_cub_terms_kernel(
    const int *row_ptr, const int *col_idx, const int *column_degrees,
    int nnz, bool aat, unsigned long long *terms)
{
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= nnz)
        return;
    const int column = col_idx[entry];
    terms[entry] = static_cast<unsigned long long>(
        aat ? column_degrees[column] : row_ptr[column + 1] - row_ptr[column]);
}

static inline bool upload_csr(const SMatrix &host, cudaStream_t stream,
                              DmmaOwnedDeviceCsr *out,
                              DmmaPreprocessStats *stats)
{
    const bool implicit_empty_row_ptr =
        host.m == 0 && host.nnz == 0 && host.rowpointer == nullptr;
    if (out == nullptr || host.m < 0 || host.n < 0 || host.nnz < 0 ||
        (host.rowpointer == nullptr && !implicit_empty_row_ptr) ||
        (host.nnz > 0 &&
         (host.columnindex == nullptr || host.value == nullptr)))
        return false;

    DmmaOwnedDeviceCsr result;
    result.rows = host.m;
    result.cols = host.n;
    result.nnz = host.nnz;
    const auto begin = Clock::now();
    if (!device_allocate(&result.row_ptr,
                         static_cast<std::size_t>(host.m) + 1,
                         "allocate device CSR row pointer") ||
        !device_allocate(&result.col_idx, host.nnz,
                         "allocate device CSR columns") ||
        !device_allocate(&result.values, host.nnz,
                         "allocate device CSR values") ||
        !(implicit_empty_row_ptr
              ? dmma_cuda_ok(cudaMemsetAsync(result.row_ptr, 0, sizeof(int),
                                              stream),
                             "initialize empty CSR row pointer")
              : dmma_cuda_ok(cudaMemcpyAsync(
                                 result.row_ptr, host.rowpointer,
                                 (static_cast<std::size_t>(host.m) + 1) *
                                     sizeof(int),
                                 cudaMemcpyHostToDevice, stream),
                             "upload CSR row pointer")) ||
        (host.nnz > 0 &&
         (!dmma_cuda_ok(cudaMemcpyAsync(
                            result.col_idx, host.columnindex,
                            static_cast<std::size_t>(host.nnz) * sizeof(int),
                            cudaMemcpyHostToDevice, stream),
                        "upload CSR columns") ||
          !dmma_cuda_ok(cudaMemcpyAsync(
                            result.values, host.value,
                            static_cast<std::size_t>(host.nnz) *
                                sizeof(MAT_VAL_TYPE),
                            cudaMemcpyHostToDevice, stream),
                        "upload CSR values"))) ||
        !dmma_cuda_ok(cudaStreamSynchronize(stream), "complete CSR upload"))
    {
        destroy_device_csr(&result);
        return false;
    }
    stats->h2d_ms = milliseconds(begin, Clock::now());

    int *d_error = nullptr;
    int h_error = 0;
    const auto validation_begin = Clock::now();
    if (!device_allocate(&d_error, 1, "allocate CSR validation flag") ||
        !dmma_cuda_ok(cudaMemsetAsync(d_error, 0, sizeof(int), stream),
                      "clear CSR validation flag"))
    {
        cudaFree(d_error);
        destroy_device_csr(&result);
        return false;
    }
    const std::size_t work = std::max(
        static_cast<std::size_t>(host.m) + 1,
        static_cast<std::size_t>(host.nnz));
    if (work > 0)
    {
        validate_csr_kernel<<<block_count(work), kThreads, 0, stream>>>(
            result.row_ptr, result.col_idx, host.m, host.n, host.nnz, d_error);
    }
    const bool validation_ok =
        dmma_cuda_ok(cudaGetLastError(), "launch CSR validation") &&
        dmma_cuda_ok(cudaMemcpyAsync(&h_error, d_error, sizeof(int),
                                     cudaMemcpyDeviceToHost, stream),
                     "read CSR validation flag") &&
        dmma_cuda_ok(cudaStreamSynchronize(stream), "complete CSR validation");
    cudaFree(d_error);
    stats->validation_ms = milliseconds(validation_begin, Clock::now());
    if (!validation_ok || h_error != 0)
    {
        if (h_error != 0)
            std::fprintf(stderr, "Invalid CSR input detected on the GPU.\n");
        destroy_device_csr(&result);
        return false;
    }

    destroy_device_csr(out);
    *out = result;
    return true;
}

static inline bool build_csc(
    const thrust::device_vector<EntryKey> &csr_tile_keys,
    cudaStream_t stream, DmmaOwnedDeviceTiles *tiles, double *elapsed_ms,
    std::size_t *workspace_bytes)
{
    const auto begin = Clock::now();
    const int count = tiles->view.num_tiles;
    if (!device_allocate(&tiles->tile_col_ptr,
                         static_cast<std::size_t>(tiles->view.tile_col_count) + 1,
                         "allocate B tile CSC pointer") ||
        !device_allocate(&tiles->tile_row_idx, count,
                         "allocate B tile CSC rows") ||
        !device_allocate(&tiles->csc_tile_ids, count,
                         "allocate B tile CSC IDs") ||
        !dmma_cuda_ok(cudaMemsetAsync(
                          tiles->tile_col_ptr, 0,
                          (static_cast<std::size_t>(tiles->view.tile_col_count) +
                           1) * sizeof(int),
                          stream),
                      "clear B tile CSC pointer"))
        return false;

    if (count > 0)
    {
        thrust::device_vector<EntryKey> csc_keys(count);
        thrust::device_vector<int> csc_ids(count);
        *workspace_bytes = std::max(
            *workspace_bytes,
            static_cast<std::size_t>(count) *
                (sizeof(EntryKey) + sizeof(int)));
        make_csc_keys_kernel<<<block_count(count), kThreads, 0, stream>>>(
            thrust::raw_pointer_cast(csr_tile_keys.data()), count,
            tiles->view.tile_row_count, tiles->view.tile_col_count,
            thrust::raw_pointer_cast(csc_keys.data()),
            thrust::raw_pointer_cast(csc_ids.data()));
        if (!dmma_cuda_ok(cudaGetLastError(), "launch B tile CSC keys"))
            return false;
        auto policy = thrust::cuda::par.on(stream);
        thrust::sort_by_key(policy, csc_keys.begin(), csc_keys.end(),
                            csc_ids.begin());
        if (!dmma_cuda_ok(cudaMemcpyAsync(
                              tiles->csc_tile_ids,
                              thrust::raw_pointer_cast(csc_ids.data()),
                              static_cast<std::size_t>(count) * sizeof(int),
                              cudaMemcpyDeviceToDevice, stream),
                          "store B tile CSC IDs"))
            return false;
        finish_csc_kernel<<<block_count(count), kThreads, 0, stream>>>(
            thrust::raw_pointer_cast(csc_keys.data()), count,
            tiles->view.tile_row_count, tiles->tile_col_ptr,
            tiles->tile_row_idx);
        if (!dmma_cuda_ok(cudaGetLastError(), "launch B tile CSC metadata"))
            return false;
    }
    auto col_ptr = thrust::device_pointer_cast(tiles->tile_col_ptr);
    thrust::inclusive_scan(
        thrust::cuda::par.on(stream), col_ptr,
        col_ptr + static_cast<std::size_t>(tiles->view.tile_col_count) + 1,
        col_ptr);
    if (!dmma_cuda_ok(cudaStreamSynchronize(stream),
                      "complete B tile CSC construction"))
        return false;
    tiles->view.tile_col_ptr = tiles->tile_col_ptr;
    tiles->view.tile_row_idx = tiles->tile_row_idx;
    tiles->view.csc_tile_ids = tiles->csc_tile_ids;
    *elapsed_ms = milliseconds(begin, Clock::now());
    return true;
}

static inline bool build_tiles(const DmmaOwnedDeviceCsr &csr,
                               bool logical_transpose, int tile_rows,
                               int tile_cols, bool payload_col_major,
                               int dense_threshold, bool build_tile_csc,
                               cudaStream_t stream,
                               DmmaOwnedDeviceTiles *out,
                               double *key_sort_reduce_ms,
                               double *tile_build_ms, double *csc_ms,
                               std::size_t *peak_workspace_bytes)
{
    if (out == nullptr || dense_threshold < 1 ||
        dense_threshold > DMMA_INPUT_ELEMS ||
        tile_rows * tile_cols != DMMA_INPUT_ELEMS)
        return false;
    const int rows = logical_transpose ? csr.cols : csr.rows;
    const int cols = logical_transpose ? csr.rows : csr.cols;
    const int tile_row_count =
        rows / tile_rows + (rows % tile_rows != 0 ? 1 : 0);
    const int tile_col_count =
        cols / tile_cols + (cols % tile_cols != 0 ? 1 : 0);
    if (static_cast<unsigned long long>(tile_row_count) *
            static_cast<unsigned long long>(tile_col_count) >
        std::numeric_limits<EntryKey>::max() / DMMA_INPUT_ELEMS)
    {
        std::fprintf(stderr, "GPU tile-key range overflow.\n");
        return false;
    }

    DmmaOwnedDeviceTiles result;
    result.view.rows = rows;
    result.view.cols = cols;
    result.view.tile_rows = tile_rows;
    result.view.tile_cols = tile_cols;
    result.view.tile_row_count = tile_row_count;
    result.view.tile_col_count = tile_col_count;

    try
    {
        const auto key_begin = Clock::now();
        thrust::device_vector<EntryKey> entry_keys(csr.nnz);
        thrust::device_vector<MAT_VAL_TYPE> entry_values(csr.nnz);
        thrust::device_vector<EntryKey> unique_entry_keys(csr.nnz);
        thrust::device_vector<MAT_VAL_TYPE> unique_entry_values(csr.nnz);
        *peak_workspace_bytes = std::max(
            *peak_workspace_bytes,
            static_cast<std::size_t>(csr.nnz) *
                2 * (sizeof(EntryKey) + sizeof(MAT_VAL_TYPE)));
        if (csr.rows > 0 && csr.nnz > 0)
        {
            fill_entry_keys_kernel<<<block_count(csr.rows), kThreads, 0,
                                     stream>>>(
                csr.row_ptr, csr.col_idx, csr.values, csr.rows, cols,
                tile_rows, tile_cols, tile_col_count, payload_col_major,
                logical_transpose,
                thrust::raw_pointer_cast(entry_keys.data()),
                thrust::raw_pointer_cast(entry_values.data()));
            if (!dmma_cuda_ok(cudaGetLastError(), "launch tile entry keys"))
                return false;
        }
        auto policy = thrust::cuda::par.on(stream);
        thrust::sort_by_key(policy, entry_keys.begin(), entry_keys.end(),
                            entry_values.begin());
        auto unique_end = thrust::reduce_by_key(
            policy, entry_keys.begin(), entry_keys.end(), entry_values.begin(),
            unique_entry_keys.begin(), unique_entry_values.begin(),
            thrust::equal_to<EntryKey>(), thrust::plus<MAT_VAL_TYPE>());
        const int unique_entry_count = static_cast<int>(
            unique_end.first - unique_entry_keys.begin());
        unique_entry_keys.resize(unique_entry_count);
        unique_entry_values.resize(unique_entry_count);
        if (!dmma_cuda_ok(cudaStreamSynchronize(stream),
                          "complete tile key sort/reduce"))
            return false;
        *key_sort_reduce_ms = milliseconds(key_begin, Clock::now());

        const auto tile_begin = Clock::now();
        thrust::device_vector<EntryKey> entry_tile_keys(unique_entry_count);
        thrust::transform(policy, unique_entry_keys.begin(),
                          unique_entry_keys.end(), entry_tile_keys.begin(),
                          EntryToTileKey());
        thrust::device_vector<EntryKey> unique_tile_keys(unique_entry_count);
        thrust::device_vector<int> tile_counts(unique_entry_count);
        auto tile_end = thrust::reduce_by_key(
            policy, entry_tile_keys.begin(), entry_tile_keys.end(),
            thrust::make_constant_iterator(1), unique_tile_keys.begin(),
            tile_counts.begin());
        const int tile_count =
            static_cast<int>(tile_end.first - unique_tile_keys.begin());
        unique_tile_keys.resize(tile_count);
        tile_counts.resize(tile_count);
        thrust::device_vector<int> entry_offsets(
            static_cast<std::size_t>(tile_count) + 1, 0);
        if (tile_count > 0)
            thrust::inclusive_scan(policy, tile_counts.begin(),
                                   tile_counts.end(),
                                   entry_offsets.begin() + 1);
        thrust::device_vector<int> payload_spans(tile_count);
        thrust::device_vector<int> dense_flags(tile_count);
        *peak_workspace_bytes = std::max(
            *peak_workspace_bytes,
            static_cast<std::size_t>(csr.nnz) *
                    2 * (sizeof(EntryKey) + sizeof(MAT_VAL_TYPE)) +
                static_cast<std::size_t>(unique_entry_count) *
                    (sizeof(EntryKey) * 2 + sizeof(int)) +
                static_cast<std::size_t>(tile_count) *
                    (sizeof(int) * 6 + sizeof(EntryKey)));

        if (!device_allocate(&result.tile_row_ptr,
                             static_cast<std::size_t>(tile_row_count) + 1,
                             "allocate tile row pointer") ||
            !device_allocate(&result.tile_col_idx, tile_count,
                             "allocate tile columns") ||
            !device_allocate(&result.value_offsets,
                             static_cast<std::size_t>(tile_count) + 1,
                             "allocate tile value offsets") ||
            !device_allocate(&result.masks, tile_count,
                             "allocate tile masks") ||
            !dmma_cuda_ok(cudaMemsetAsync(
                              result.tile_row_ptr, 0,
                              (static_cast<std::size_t>(tile_row_count) + 1) *
                                  sizeof(int),
                              stream),
                          "clear tile row pointer"))
        {
            destroy_device_tiles(&result);
            return false;
        }
        if (tile_count > 0)
        {
            build_tile_metadata_kernel<<<block_count(tile_count), kThreads, 0,
                                         stream>>>(
                thrust::raw_pointer_cast(unique_tile_keys.data()),
                thrust::raw_pointer_cast(tile_counts.data()), tile_count,
                tile_col_count, dense_threshold, result.tile_row_ptr,
                result.tile_col_idx,
                thrust::raw_pointer_cast(payload_spans.data()),
                thrust::raw_pointer_cast(dense_flags.data()));
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "launch tile metadata construction"))
            {
                destroy_device_tiles(&result);
                return false;
            }
        }
        auto row_ptr = thrust::device_pointer_cast(result.tile_row_ptr);
        thrust::inclusive_scan(policy, row_ptr,
                               row_ptr +
                                   static_cast<std::size_t>(tile_row_count) + 1,
                               row_ptr);
        const long long payload_size_wide = thrust::reduce(
            policy, payload_spans.begin(), payload_spans.end(), 0ll,
            thrust::plus<long long>());
        if (payload_size_wide > std::numeric_limits<int>::max())
        {
            std::fprintf(stderr,
                         "GPU tile payload exceeds the 32-bit offset range.\n");
            destroy_device_tiles(&result);
            return false;
        }
        const int payload_size = static_cast<int>(payload_size_wide);
        auto value_offsets =
            thrust::device_pointer_cast(result.value_offsets);
        if (tile_count > 0)
            thrust::exclusive_scan(policy, payload_spans.begin(),
                                   payload_spans.end(), value_offsets);
        const int dense_tiles = thrust::reduce(
            policy, dense_flags.begin(), dense_flags.end(), 0,
            thrust::plus<int>());
        if (!dmma_cuda_ok(cudaMemcpyAsync(
                              result.value_offsets + tile_count, &payload_size,
                              sizeof(int), cudaMemcpyHostToDevice, stream),
                          "store tile payload size") ||
            !device_allocate(&result.values, payload_size,
                             "allocate tile payload"))
        {
            destroy_device_tiles(&result);
            return false;
        }
        if (tile_count > 0)
        {
            pack_tiles_kernel<<<tile_count, kTileThreads, 0, stream>>>(
                thrust::raw_pointer_cast(unique_entry_keys.data()),
                thrust::raw_pointer_cast(unique_entry_values.data()),
                thrust::raw_pointer_cast(entry_offsets.data()),
                thrust::raw_pointer_cast(tile_counts.data()),
                result.value_offsets, tile_count, dense_threshold,
                result.masks, result.values);
            if (!dmma_cuda_ok(cudaGetLastError(), "launch tile payload pack"))
            {
                destroy_device_tiles(&result);
                return false;
            }
        }
        if (!dmma_cuda_ok(cudaStreamSynchronize(stream),
                          "complete tile construction"))
        {
            destroy_device_tiles(&result);
            return false;
        }

        result.view.num_tiles = tile_count;
        result.view.payload_size = payload_size;
        result.view.dense_tiles = dense_tiles;
        result.view.sparse_tiles = tile_count - dense_tiles;
        result.view.tile_row_ptr = result.tile_row_ptr;
        result.view.tile_col_idx = result.tile_col_idx;
        result.view.value_offsets = result.value_offsets;
        result.view.masks = result.masks;
        result.view.values = result.values;
        result.view.tile_col_ptr = nullptr;
        result.view.tile_row_idx = nullptr;
        result.view.csc_tile_ids = nullptr;
        *tile_build_ms = milliseconds(tile_begin, Clock::now());

        if (build_tile_csc &&
            !build_csc(unique_tile_keys, stream, &result, csc_ms,
                       peak_workspace_bytes))
        {
            destroy_device_tiles(&result);
            return false;
        }
    }
    catch (const std::bad_alloc &)
    {
        std::fprintf(stderr, "GPU preprocessing host allocation failed.\n");
        destroy_device_tiles(&result);
        return false;
    }
    catch (const thrust::system_error &error)
    {
        std::fprintf(stderr, "GPU preprocessing Thrust failure: %s\n",
                     error.what());
        destroy_device_tiles(&result);
        return false;
    }

    destroy_device_tiles(out);
    *out = result;
    return true;
}

static inline bool compute_nnz_cub(const DmmaOwnedDeviceCsr &csr, bool aat,
                                   cudaStream_t stream,
                                   unsigned long long *nnz_cub,
                                   DmmaPreprocessStats *stats)
{
    const auto begin = Clock::now();
    try
    {
        thrust::device_vector<int> column_degrees;
        if (aat)
        {
            column_degrees.assign(csr.cols, 0);
            if (csr.nnz > 0)
            {
                count_column_degrees_kernel<<<block_count(csr.nnz), kThreads,
                                               0, stream>>>(
                    csr.col_idx, csr.nnz,
                    thrust::raw_pointer_cast(column_degrees.data()));
                if (!dmma_cuda_ok(cudaGetLastError(),
                                  "launch AAT column degrees"))
                    return false;
            }
        }
        thrust::device_vector<unsigned long long> terms(csr.nnz);
        if (csr.nnz > 0)
        {
            fill_nnz_cub_terms_kernel<<<block_count(csr.nnz), kThreads, 0,
                                        stream>>>(
                csr.row_ptr, csr.col_idx,
                aat ? thrust::raw_pointer_cast(column_degrees.data()) : nullptr,
                csr.nnz, aat, thrust::raw_pointer_cast(terms.data()));
            if (!dmma_cuda_ok(cudaGetLastError(), "launch nnzCub terms"))
                return false;
        }
        *nnz_cub = thrust::reduce(thrust::cuda::par.on(stream), terms.begin(),
                                  terms.end(), 0ull,
                                  thrust::plus<unsigned long long>());
        if (!dmma_cuda_ok(cudaStreamSynchronize(stream),
                          "complete GPU nnzCub"))
            return false;
    }
    catch (const std::bad_alloc &)
    {
        std::fprintf(stderr, "GPU nnzCub allocation failed.\n");
        return false;
    }
    catch (const thrust::system_error &error)
    {
        std::fprintf(stderr, "GPU nnzCub failure: %s\n", error.what());
        return false;
    }
    stats->nnz_cub_ms = milliseconds(begin, Clock::now());
    return true;
}

} // namespace gpu_dmma_detail

static inline bool gpu_prepare_dmma_operands(
    const SMatrix &host, bool aat, int dense_threshold,
    DmmaPreparedOperands *out, DmmaPreprocessStats *stats,
    cudaStream_t stream = 0)
{
    using namespace gpu_dmma_detail;
    if (out == nullptr || stats == nullptr || dense_threshold < 1 ||
        dense_threshold > DMMA_INPUT_ELEMS || (!aat && host.m != host.n))
        return false;
    *stats = DmmaPreprocessStats();
    DmmaPreparedOperands result;
    result.aat = aat;
    const auto total_begin = Clock::now();
    if (!upload_csr(host, stream, &result.csr, stats) ||
        !build_tiles(result.csr, false, DMMA_TILE_M, DMMA_TILE_K, false,
                     dense_threshold, false, stream, &result.a,
                     &stats->a_key_sort_reduce_ms, &stats->a_tile_build_ms,
                     &stats->b_csc_ms, &stats->peak_workspace_bytes) ||
        !build_tiles(result.csr, aat, DMMA_TILE_K, DMMA_TILE_N, true,
                     dense_threshold, true, stream, &result.b,
                     &stats->b_key_sort_reduce_ms, &stats->b_tile_build_ms,
                     &stats->b_csc_ms, &stats->peak_workspace_bytes) ||
        !compute_nnz_cub(result.csr, aat, stream, &result.nnz_cub, stats) ||
        !dmma_cuda_ok(cudaStreamSynchronize(stream),
                      "complete GPU DMMA preprocessing"))
    {
        destroy_prepared_operands(&result);
        return false;
    }
    stats->total_ms = milliseconds(total_begin, Clock::now());
    destroy_prepared_operands(out);
    *out = result;
    return true;
}

#endif // RTT_SPGEMM_GPU_DMMA_TILES_H_
