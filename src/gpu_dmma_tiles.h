#ifndef RTT_SPGEMM_GPU_DMMA_TILES_H_
#define RTT_SPGEMM_GPU_DMMA_TILES_H_

#include "dmma_spgemm.h"
#include "dmma_reorder.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
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

struct DmmaDeviceCsrView
{
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    const int *row_ptr = nullptr;
    const int *col_idx = nullptr;
    const MAT_VAL_TYPE *values = nullptr;
};

static inline DmmaDeviceCsrView device_csr_view(
    const DmmaOwnedDeviceCsr &csr)
{
    DmmaDeviceCsrView view;
    view.rows = csr.rows;
    view.cols = csr.cols;
    view.nnz = csr.nnz;
    view.row_ptr = csr.row_ptr;
    view.col_idx = csr.col_idx;
    view.values = csr.values;
    return view;
}

struct DmmaBUpdateStats
{
    double csr_row_permute_ms = 0.0;
    double validation_ms = 0.0;
    double key_sort_reduce_ms = 0.0;
    double tile_build_ms = 0.0;
    double ordered_count_ms = 0.0;
    double ordered_pack_ms = 0.0;
    double ordered_layout_ms = 0.0;
    double ordered_payload_alloc_ms = 0.0;
    double ordered_payload_fill_ms = 0.0;
    double csc_ms = 0.0;
    double mapping_ms = 0.0;
    double low_fill_metadata_ms = 0.0;
    double super16_build_ms = 0.0;
    double total_ms = 0.0;
    std::size_t peak_workspace_bytes = 0;
    int source_entries = 0;
    int active_entries = 0;
    int unique_entries = 0;
    bool structure_rebuilt = false;
    bool ordered_csr_fallback = false;
    bool fused_csr_row_permute = false;
};

/* Grow-only scratch used by repeated B format rebuilds.  Resizing a
 * device_vector within its capacity does not allocate. */
struct DmmaBWorkspace
{
    thrust::device_vector<unsigned long long> entry_keys;
    thrust::device_vector<MAT_VAL_TYPE> entry_values;
};

struct DmmaDynamicB
{
    DmmaOwnedDeviceTiles tiles;
    rtt::super16::OwnedDeviceIndex super16;
    int source_nnz = 0;
    int source_rows = 0;
    int source_cols = 0;
    int active_rows = 0;
    int active_entries = 0;
    bool has_duplicates = false;
    bool payload_fully_initialized = false;
    bool fused_csr_row_permute = false;
    bool valid = false;
    DmmaBWorkspace workspace;
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

struct DmmaOfflineAStats
{
    double h2d_ms = 0.0;
    double validation_ms = 0.0;
    double reorder_ms = 0.0;
    double key_sort_reduce_ms = 0.0;
    double tile_build_ms = 0.0;
    double super16_build_ms = 0.0;
    double total_ms = 0.0;
    std::size_t peak_workspace_bytes = 0;
};

struct DmmaPreparedA
{
    DmmaOwnedDeviceCsr csr;
    DmmaReorderPlan reorder;
    DmmaOwnedDeviceTiles tiles;
    rtt::super16::OwnedDeviceIndex super16;
    int dense_threshold = 24;
    bool valid = false;
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

static inline void destroy_dynamic_b(DmmaDynamicB *matrix)
{
    if (matrix == nullptr)
        return;
    destroy_device_tiles(&matrix->tiles);
    *matrix = DmmaDynamicB();
}

static inline void destroy_prepared_a(DmmaPreparedA *prepared)
{
    if (prepared == nullptr)
        return;
    destroy_device_tiles(&prepared->tiles);
    destroy_dmma_reorder_plan(&prepared->reorder);
    destroy_device_csr(&prepared->csr);
    *prepared = DmmaPreparedA();
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

__global__ void permute_csr_row_lengths_kernel(
    const int *source_row_ptr, int rows, const int *row_new_to_old,
    int *permuted_row_ptr, int *error)
{
    const int new_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (new_row >= rows)
        return;
    const int old_row =
        row_new_to_old != nullptr ? row_new_to_old[new_row] : new_row;
    if (old_row < 0 || old_row >= rows)
    {
        atomicExch(error, 1);
        return;
    }
    permuted_row_ptr[new_row + 1] =
        source_row_ptr[old_row + 1] - source_row_ptr[old_row];
}

__global__ void copy_permuted_csr_rows_kernel(
    const int *source_row_ptr, const int *source_col_idx,
    const MAT_VAL_TYPE *source_values, int rows,
    const int *row_new_to_old, const int *permuted_row_ptr,
    int *permuted_col_idx, MAT_VAL_TYPE *permuted_values, int *error)
{
    const int new_row = blockIdx.x;
    if (new_row >= rows)
        return;
    const int old_row =
        row_new_to_old != nullptr ? row_new_to_old[new_row] : new_row;
    if (old_row < 0 || old_row >= rows)
    {
        if (threadIdx.x == 0)
            atomicExch(error, 1);
        return;
    }
    const int source_begin = source_row_ptr[old_row];
    const int source_end = source_row_ptr[old_row + 1];
    const int destination_begin = permuted_row_ptr[new_row];
    for (int offset = threadIdx.x; offset < source_end - source_begin;
         offset += blockDim.x)
    {
        permuted_col_idx[destination_begin + offset] =
            source_col_idx[source_begin + offset];
        permuted_values[destination_begin + offset] =
            source_values[source_begin + offset];
    }
}

__device__ __forceinline__ int dmma_warp_min_int(int value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = min(value, __shfl_down_sync(0xffffffffu, value, offset));
    return __shfl_sync(0xffffffffu, value, 0);
}

__global__ void count_ordered_b_tile_rows_warp_kernel(
    const int *row_ptr, const int *col_idx, int source_rows,
    const int *row_new_to_old, int active_rows, int *tile_row_ptr,
    unsigned long long *active_entry_count, int *error)
{
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int tile_row = blockIdx.x * warps_per_block + warp_in_block;
    const int tile_row_count =
        (active_rows + DMMA_TILE_K - 1) / DMMA_TILE_K;
    if (tile_row >= tile_row_count)
        return;

    const int new_row = tile_row * DMMA_TILE_K + lane;
    bool owns_row = lane < DMMA_TILE_K && new_row < active_rows;
    int row = owns_row && row_new_to_old != nullptr
                  ? row_new_to_old[new_row]
                  : new_row;
    if (owns_row && (row < 0 || row >= source_rows))
    {
        atomicExch(error, 2);
        owns_row = false;
    }
    int cursor = owns_row ? row_ptr[row] : 0;
    const int end = owns_row ? row_ptr[row + 1] : 0;
    if (owns_row)
        atomicAdd(active_entry_count,
                  static_cast<unsigned long long>(end - cursor));
    int current_tile =
        cursor < end ? col_idx[cursor] / DMMA_TILE_N : 0x7fffffff;
    int tile_count = 0;
    while (true)
    {
        const int next_tile = dmma_warp_min_int(current_tile);
        if (next_tile == 0x7fffffff)
            break;
        if (owns_row && current_tile == next_tile)
        {
            do
            {
                if (cursor > row_ptr[row] &&
                    col_idx[cursor] < col_idx[cursor - 1])
                    atomicExch(error, 1);
                ++cursor;
            } while (cursor < end &&
                     col_idx[cursor] / DMMA_TILE_N == next_tile);
            current_tile =
                cursor < end ? col_idx[cursor] / DMMA_TILE_N : 0x7fffffff;
        }
        if (lane == 0)
            ++tile_count;
    }
    if (lane == 0)
        tile_row_ptr[tile_row + 1] = tile_count;
}

__global__ void fill_ordered_b_tile_metadata_warp_kernel(
    const int *row_ptr, const int *col_idx, int active_rows,
    const int *row_new_to_old, int tile_col_count,
    const int *tile_row_ptr, int *tile_col_idx, int *tile_entry_counts,
    EntryKey *tile_keys, uint32_t *masks, int *error)
{
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int tile_row = blockIdx.x * warps_per_block + warp_in_block;
    const int tile_row_count =
        (active_rows + DMMA_TILE_K - 1) / DMMA_TILE_K;
    if (tile_row >= tile_row_count)
        return;

    const int new_row = tile_row * DMMA_TILE_K + lane;
    const bool owns_row = lane < DMMA_TILE_K && new_row < active_rows;
    const int row = owns_row && row_new_to_old != nullptr
                        ? row_new_to_old[new_row]
                        : new_row;
    int cursor = owns_row ? row_ptr[row] : 0;
    const int end = owns_row ? row_ptr[row + 1] : 0;
    int current_tile =
        cursor < end ? col_idx[cursor] / DMMA_TILE_N : 0x7fffffff;
    int output_tile = tile_row_ptr[tile_row];
    while (true)
    {
        const int next_tile = dmma_warp_min_int(current_tile);
        if (next_tile == 0x7fffffff)
            break;

        uint32_t lane_mask = 0;
        if (owns_row && current_tile == next_tile)
        {
            do
            {
                const int local_col = col_idx[cursor] % DMMA_TILE_N;
                lane_mask |= uint32_t(1) <<
                             (local_col * DMMA_TILE_K + lane);
                ++cursor;
            } while (cursor < end &&
                     col_idx[cursor] / DMMA_TILE_N == next_tile);
            current_tile =
                cursor < end ? col_idx[cursor] / DMMA_TILE_N : 0x7fffffff;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            lane_mask |=
                __shfl_down_sync(0xffffffffu, lane_mask, offset);
        if (lane == 0)
        {
            if (output_tile >= tile_row_ptr[tile_row + 1] ||
                next_tile < 0 || next_tile >= tile_col_count)
            {
                atomicExch(error, 1);
                return;
            }
            tile_col_idx[output_tile] = next_tile;
            tile_entry_counts[output_tile] = __popc(lane_mask);
            tile_keys[output_tile] =
                static_cast<EntryKey>(tile_row) * tile_col_count +
                next_tile;
            masks[output_tile] = lane_mask;
            ++output_tile;
        }
    }
    if (lane == 0 && output_tile != tile_row_ptr[tile_row + 1])
        atomicExch(error, 1);
}

__global__ void prepare_ordered_b_payload_layout_kernel(
    const int *tile_entry_counts, int tile_count, int dense_threshold,
    int *payload_spans, int *dense_flags)
{
    const int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= tile_count)
        return;
    const bool dense = tile_entry_counts[tile] >= dense_threshold;
    payload_spans[tile] =
        dense ? DMMA_INPUT_ELEMS : tile_entry_counts[tile];
    dense_flags[tile] = dense ? 1 : 0;
}

__global__ void reduce_ordered_b_tile_stats_kernel(
    const int *tile_entry_counts, const int *dense_flags,
    const int *payload_spans, int tile_count, unsigned long long *totals)
{
    unsigned long long unique = 0;
    unsigned long long dense = 0;
    unsigned long long payload = 0;
    for (int tile = blockIdx.x * blockDim.x + threadIdx.x;
         tile < tile_count; tile += blockDim.x * gridDim.x)
    {
        unique += static_cast<unsigned long long>(tile_entry_counts[tile]);
        dense += static_cast<unsigned long long>(dense_flags[tile]);
        payload += static_cast<unsigned long long>(payload_spans[tile]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        unique += __shfl_down_sync(0xffffffffu, unique, offset);
        dense += __shfl_down_sync(0xffffffffu, dense, offset);
        payload += __shfl_down_sync(0xffffffffu, payload, offset);
    }
    __shared__ unsigned long long warp_unique[8];
    __shared__ unsigned long long warp_dense[8];
    __shared__ unsigned long long warp_payload[8];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0)
    {
        warp_unique[warp] = unique;
        warp_dense[warp] = dense;
        warp_payload[warp] = payload;
    }
    __syncthreads();
    if (warp == 0)
    {
        unique = lane < blockDim.x / 32 ? warp_unique[lane] : 0;
        dense = lane < blockDim.x / 32 ? warp_dense[lane] : 0;
        payload = lane < blockDim.x / 32 ? warp_payload[lane] : 0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            unique += __shfl_down_sync(0xffffffffu, unique, offset);
            dense += __shfl_down_sync(0xffffffffu, dense, offset);
            payload += __shfl_down_sync(0xffffffffu, payload, offset);
        }
        if (lane == 0)
        {
            atomicAdd(totals, unique);
            atomicAdd(totals + 1, dense);
            atomicAdd(totals + 2, payload);
        }
    }
}

__global__ void fill_ordered_b_payload_warp_kernel(
    const int *row_ptr, const int *col_idx,
    const MAT_VAL_TYPE *source_values, int active_rows,
    const int *row_new_to_old, int dense_threshold, const int *tile_row_ptr,
    const int *tile_entry_counts, const int *value_offsets,
    const uint32_t *masks, MAT_VAL_TYPE *payload, int *error)
{
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int tile_row = blockIdx.x * warps_per_block + warp_in_block;
    const int tile_row_count =
        (active_rows + DMMA_TILE_K - 1) / DMMA_TILE_K;
    if (tile_row >= tile_row_count)
        return;

    const int new_row = tile_row * DMMA_TILE_K + lane;
    const bool owns_row = lane < DMMA_TILE_K && new_row < active_rows;
    const int row = owns_row && row_new_to_old != nullptr
                        ? row_new_to_old[new_row]
                        : new_row;
    int cursor = owns_row ? row_ptr[row] : 0;
    const int end = owns_row ? row_ptr[row + 1] : 0;
    int current_tile =
        cursor < end ? col_idx[cursor] / DMMA_TILE_N : 0x7fffffff;
    int output_tile = tile_row_ptr[tile_row];
    while (true)
    {
        const int next_tile = dmma_warp_min_int(current_tile);
        if (next_tile == 0x7fffffff)
            break;
        if (output_tile >= tile_row_ptr[tile_row + 1])
        {
            if (lane == 0)
                atomicExch(error, 1);
            return;
        }
        const bool dense =
            tile_entry_counts[output_tile] >= dense_threshold;
        const int output = value_offsets[output_tile];
        if (dense)
            payload[output + lane] = MAT_VAL_TYPE(0);
        __syncwarp();

        if (owns_row && current_tile == next_tile)
        {
            const uint32_t mask = masks[output_tile];
            do
            {
                const int column = col_idx[cursor];
                MAT_VAL_TYPE sum = source_values[cursor++];
                while (cursor < end && col_idx[cursor] == column)
                    sum += source_values[cursor++];
                const int physical =
                    (column % DMMA_TILE_N) * DMMA_TILE_K + lane;
                const uint32_t lower =
                    physical == 0
                        ? 0
                        : mask & ((uint32_t(1) << physical) - 1);
                const int destination =
                    output + (dense ? physical : __popc(lower));
                payload[destination] = sum;
            } while (cursor < end &&
                     col_idx[cursor] / DMMA_TILE_N == next_tile);
            current_tile =
                cursor < end ? col_idx[cursor] / DMMA_TILE_N : 0x7fffffff;
        }
        __syncwarp();
        ++output_tile;
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

__global__ void fill_mapped_entry_keys_kernel(
    const int *row_ptr, const int *col_idx, const MAT_VAL_TYPE *values,
    int source_rows, int logical_rows, int logical_cols, int tile_rows,
    int tile_cols, int tile_col_count, bool payload_col_major,
    bool logical_transpose, const int *row_old_to_new,
    const int *col_old_to_new, int active_logical_rows,
    int active_logical_cols, EntryKey invalid_key, EntryKey *keys,
    MAT_VAL_TYPE *key_values)
{
    const int source_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (source_row >= source_rows)
        return;
    for (int entry = row_ptr[source_row]; entry < row_ptr[source_row + 1];
         ++entry)
    {
        const int source_col = col_idx[entry];
        const int old_row = logical_transpose ? source_col : source_row;
        const int old_col = logical_transpose ? source_row : source_col;
        int logical_row = row_old_to_new != nullptr
                              ? row_old_to_new[old_row]
                              : old_row;
        int logical_col = col_old_to_new != nullptr
                              ? col_old_to_new[old_col]
                              : old_col;
        if (old_row < 0 || old_row >= logical_rows || old_col < 0 ||
            old_col >= logical_cols || logical_row < 0 || logical_col < 0 ||
            logical_row >= active_logical_rows ||
            logical_col >= active_logical_cols)
        {
            keys[entry] = invalid_key;
            key_values[entry] = MAT_VAL_TYPE(0);
            continue;
        }
        const int tile_row = logical_row / tile_rows;
        const int tile_col = logical_col / tile_cols;
        const int local_row = logical_row % tile_rows;
        const int local_col = logical_col % tile_cols;
        const int physical = payload_col_major
                                 ? local_col * tile_rows + local_row
                                 : local_row * tile_cols + local_col;
        const EntryKey tile_key =
            static_cast<EntryKey>(tile_row) * tile_col_count + tile_col;
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

__global__ void fill_nnz_cub_ab_terms_kernel(
    const int *a_col_idx, int a_nnz, const int *b_row_ptr,
    unsigned long long *terms)
{
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry < a_nnz)
    {
        const int k = a_col_idx[entry];
        terms[entry] = static_cast<unsigned long long>(
            b_row_ptr[k + 1] - b_row_ptr[k]);
    }
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
    *out = std::move(result);
    return true;
}

static inline bool validate_device_csr(const DmmaDeviceCsrView &view,
                                       cudaStream_t stream,
                                       double *elapsed_ms)
{
    const bool implicit_empty = view.rows == 0 && view.nnz == 0;
    if (view.rows < 0 || view.cols < 0 || view.nnz < 0 ||
        (view.row_ptr == nullptr && !implicit_empty) ||
        (view.nnz > 0 &&
         (view.col_idx == nullptr || view.values == nullptr)))
        return false;

    const auto begin = Clock::now();
    if (implicit_empty && view.row_ptr == nullptr)
    {
        if (elapsed_ms != nullptr)
            *elapsed_ms = milliseconds(begin, Clock::now());
        return true;
    }
    int *error = nullptr;
    int host_error = 0;
    if (!device_allocate(&error, 1, "allocate device CSR validation flag") ||
        !dmma_cuda_ok(cudaMemsetAsync(error, 0, sizeof(int), stream),
                      "clear device CSR validation flag"))
    {
        cudaFree(error);
        return false;
    }
    const std::size_t work = std::max(
        static_cast<std::size_t>(view.rows) + 1,
        static_cast<std::size_t>(view.nnz));
    if (work > 0)
    {
        validate_csr_kernel<<<block_count(work), kThreads, 0, stream>>>(
            view.row_ptr, view.col_idx, view.rows, view.cols, view.nnz,
            error);
    }
    const bool ok =
        dmma_cuda_ok(cudaGetLastError(), "launch device CSR validation") &&
        dmma_cuda_ok(cudaMemcpyAsync(&host_error, error, sizeof(int),
                                     cudaMemcpyDeviceToHost, stream),
                     "read device CSR validation flag") &&
        dmma_cuda_ok(cudaStreamSynchronize(stream),
                     "complete device CSR validation");
    cudaFree(error);
    if (elapsed_ms != nullptr)
        *elapsed_ms = milliseconds(begin, Clock::now());
    return ok && host_error == 0;
}

static inline bool permute_device_csr_rows(
    const DmmaDeviceCsrView &source, const int *row_new_to_old,
    cudaStream_t stream, DmmaOwnedDeviceCsr *out, double *elapsed_ms)
{
    if (out == nullptr || source.rows < 0 || source.cols < 0 ||
        source.nnz < 0 ||
        (source.rows > 0 && source.row_ptr == nullptr) ||
        (source.nnz > 0 &&
         (source.col_idx == nullptr || source.values == nullptr)))
        return false;

    const auto begin = Clock::now();
    DmmaOwnedDeviceCsr result;
    result.rows = source.rows;
    result.cols = source.cols;
    result.nnz = source.nnz;
    int *d_error = nullptr;
    int h_error = 0;
    if (!device_allocate(
            &result.row_ptr, static_cast<std::size_t>(source.rows) + 1,
            "allocate permuted B CSR row pointer") ||
        !device_allocate(&result.col_idx, source.nnz,
                         "allocate permuted B CSR columns") ||
        !device_allocate(&result.values, source.nnz,
                         "allocate permuted B CSR values") ||
        !device_allocate(&d_error, 1,
                         "allocate permuted B CSR error flag") ||
        !dmma_cuda_ok(cudaMemsetAsync(
            result.row_ptr, 0,
            (static_cast<std::size_t>(source.rows) + 1) * sizeof(int),
            stream),
            "clear permuted B CSR row pointer") ||
        !dmma_cuda_ok(cudaMemsetAsync(d_error, 0, sizeof(int), stream),
                      "clear permuted B CSR error flag"))
        goto failure;

    if (source.rows > 0)
    {
        permute_csr_row_lengths_kernel<<<block_count(source.rows), kThreads,
                                         0, stream>>>(
            source.row_ptr, source.rows, row_new_to_old, result.row_ptr,
            d_error);
        if (!dmma_cuda_ok(cudaGetLastError(),
                          "build permuted B CSR row lengths"))
            goto failure;
    }
    {
        auto row_ptr = thrust::device_pointer_cast(result.row_ptr);
        thrust::inclusive_scan(
            thrust::cuda::par.on(stream), row_ptr,
            row_ptr + static_cast<std::size_t>(source.rows) + 1, row_ptr);
    }
    if (source.rows > 0)
    {
        copy_permuted_csr_rows_kernel<<<source.rows, kThreads, 0, stream>>>(
            source.row_ptr, source.col_idx, source.values, source.rows,
            row_new_to_old, result.row_ptr, result.col_idx, result.values,
            d_error);
        if (!dmma_cuda_ok(cudaGetLastError(),
                          "copy permuted B CSR rows"))
            goto failure;
    }
    if (!dmma_cuda_ok(cudaMemcpyAsync(
            &h_error, d_error, sizeof(int), cudaMemcpyDeviceToHost, stream),
            "read permuted B CSR error flag") ||
        !dmma_cuda_ok(cudaStreamSynchronize(stream),
                      "complete B CSR row permutation") ||
        h_error != 0)
        goto failure;

    cudaFree(d_error);
    destroy_device_csr(out);
    *out = result;
    result = DmmaOwnedDeviceCsr();
    if (elapsed_ms != nullptr)
        *elapsed_ms = milliseconds(begin, Clock::now());
    return true;

failure:
    cudaFree(d_error);
    destroy_device_csr(&result);
    return false;
}

static inline bool build_csc(
    const thrust::device_vector<EntryKey> &csr_tile_keys,
    cudaStream_t stream, DmmaOwnedDeviceTiles *tiles, double *elapsed_ms,
    std::size_t *workspace_bytes, EntryKey *csc_keys_scratch = nullptr,
    int *csc_ids_scratch = nullptr)
{
    const auto begin = Clock::now();
    const int count = tiles->view.num_tiles;
    if (!device_allocate(&tiles->tile_col_ptr,
                         static_cast<std::size_t>(
                             tiles->view.tile_col_count) + 1,
                         "allocate B tile CSC pointer") ||
        !device_allocate(&tiles->tile_row_idx, count,
                         "allocate B tile CSC rows") ||
        !device_allocate(&tiles->csc_tile_ids, count,
                         "allocate B tile CSC IDs") ||
        !dmma_cuda_ok(cudaMemsetAsync(
                          tiles->tile_col_ptr, 0,
                          (static_cast<std::size_t>(
                               tiles->view.tile_col_count) +
                           1) * sizeof(int),
                          stream),
                      "clear B tile CSC pointer"))
        return false;

    if (count > 0)
    {
        thrust::device_vector<EntryKey> owned_csc_keys;
        thrust::device_vector<int> owned_csc_ids;
        if (csc_keys_scratch == nullptr || csc_ids_scratch == nullptr)
        {
            owned_csc_keys.resize(count);
            owned_csc_ids.resize(count);
            csc_keys_scratch =
                thrust::raw_pointer_cast(owned_csc_keys.data());
            csc_ids_scratch =
                thrust::raw_pointer_cast(owned_csc_ids.data());
        }
        *workspace_bytes = std::max(
            *workspace_bytes,
            static_cast<std::size_t>(count) *
                (sizeof(EntryKey) + sizeof(int)));
        make_csc_keys_kernel<<<block_count(count), kThreads, 0, stream>>>(
            thrust::raw_pointer_cast(csr_tile_keys.data()), count,
            tiles->view.tile_row_count, tiles->view.tile_col_count,
            csc_keys_scratch, csc_ids_scratch);
        if (!dmma_cuda_ok(cudaGetLastError(), "launch B tile CSC keys"))
            return false;
        auto policy = thrust::cuda::par.on(stream);
        auto csc_keys =
            thrust::device_pointer_cast(csc_keys_scratch);
        auto csc_ids =
            thrust::device_pointer_cast(csc_ids_scratch);
        thrust::sort_by_key(policy, csc_keys, csc_keys + count, csc_ids);
        if (!dmma_cuda_ok(cudaMemcpyAsync(
                              tiles->csc_tile_ids,
                              csc_ids_scratch,
                              static_cast<std::size_t>(count) * sizeof(int),
                              cudaMemcpyDeviceToDevice, stream),
                          "store B tile CSC IDs"))
            return false;
        finish_csc_kernel<<<block_count(count), kThreads, 0, stream>>>(
            csc_keys_scratch, count,
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

/* One thread owns one A tile row or B tile column.  This is static structural
 * preprocessing, not a timed task counter.  uint64 accumulation prevents a
 * wrap from making a dense row/column look sparse; overflow invalidates the
 * whole optional component before either low-fill kernel can launch. */
__global__ void build_low_fill_tile_nnz_sum_kernel(
    DmmaDeviceTiles tiles, int use_csc, int entry_count,
    uint32_t *__restrict__ sums, int *__restrict__ overflow)
{
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry >= entry_count)
        return;
    unsigned long long sum = 0;
    if (use_csc != 0)
    {
        for (int p = tiles.tile_col_ptr[entry];
             p < tiles.tile_col_ptr[entry + 1]; ++p)
            sum += static_cast<unsigned long long>(
                __popc(tiles.masks[tiles.csc_tile_ids[p]]));
    }
    else
    {
        for (int p = tiles.tile_row_ptr[entry];
             p < tiles.tile_row_ptr[entry + 1]; ++p)
            sum += static_cast<unsigned long long>(__popc(tiles.masks[p]));
    }
    if (sum > UINT_MAX)
        atomicExch(overflow, 1);
    else
        sums[entry] = static_cast<uint32_t>(sum);
}

static inline bool build_tiles(const DmmaOwnedDeviceCsr &csr,
                               bool logical_transpose, int tile_rows,
                               int tile_cols, bool payload_col_major,
                               int dense_threshold, bool build_tile_csc,
                               cudaStream_t stream,
                               DmmaOwnedDeviceTiles *out,
                               double *key_sort_reduce_ms,
                               double *tile_build_ms, double *csc_ms,
                               std::size_t *peak_workspace_bytes,
                               const int *row_old_to_new = nullptr,
                               const int *col_old_to_new = nullptr,
                               int active_rows = -1,
                               int active_cols = -1)
{
    if (out == nullptr || dense_threshold < 1 ||
        dense_threshold > DMMA_INPUT_ELEMS ||
        tile_rows * tile_cols != DMMA_INPUT_ELEMS)
        return false;
    const int rows = logical_transpose ? csr.cols : csr.rows;
    const int cols = logical_transpose ? csr.rows : csr.cols;
    if (active_rows < 0)
        active_rows = rows;
    if (active_cols < 0)
        active_cols = cols;
    if (active_rows < 0 || active_rows > rows || active_cols < 0 ||
        active_cols > cols)
        return false;
    const int tile_row_count =
        active_rows / tile_rows + (active_rows % tile_rows != 0 ? 1 : 0);
    const int tile_col_count =
        active_cols / tile_cols + (active_cols % tile_cols != 0 ? 1 : 0);
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
        const bool mapped = row_old_to_new != nullptr ||
                            col_old_to_new != nullptr ||
                            active_rows != rows || active_cols != cols;
        if (csr.rows > 0 && csr.nnz > 0 && mapped)
        {
            fill_mapped_entry_keys_kernel<<<block_count(csr.rows), kThreads,
                                            0, stream>>>(
                csr.row_ptr, csr.col_idx, csr.values, csr.rows, rows, cols,
                tile_rows, tile_cols, tile_col_count, payload_col_major,
                logical_transpose, row_old_to_new, col_old_to_new,
                active_rows, active_cols,
                std::numeric_limits<EntryKey>::max(),
                thrust::raw_pointer_cast(entry_keys.data()),
                thrust::raw_pointer_cast(entry_values.data()));
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "launch mapped tile entry keys"))
                return false;
        }
        else if (csr.rows > 0 && csr.nnz > 0)
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
        const int input_entry_count = mapped
            ? static_cast<int>(
                  thrust::lower_bound(
                      policy, entry_keys.begin(), entry_keys.end(),
                      std::numeric_limits<EntryKey>::max()) -
                  entry_keys.begin())
            : csr.nnz;
        auto unique_end = thrust::reduce_by_key(
            policy, entry_keys.begin(),
            entry_keys.begin() + input_entry_count, entry_values.begin(),
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
        result.view.structural_nnz =
            static_cast<unsigned long long>(unique_entry_count);
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
    *out = std::move(result);
    return true;
}

static inline bool rebuild_dynamic_b(
    const DmmaDeviceCsrView &csr, const int *inner_old_to_new,
    int active_inner_rows, int dense_threshold, DmmaDynamicB *out,
    DmmaBUpdateStats *stats, cudaStream_t stream = 0,
    bool logical_transpose = false)
{
    const int logical_rows = logical_transpose ? csr.cols : csr.rows;
    const int logical_cols = logical_transpose ? csr.rows : csr.cols;
    if (out == nullptr || stats == nullptr || csr.rows < 0 || csr.cols < 0 ||
        csr.nnz < 0 || active_inner_rows < 0 ||
        active_inner_rows > logical_rows || dense_threshold < 1 ||
        dense_threshold > DMMA_INPUT_ELEMS)
        return false;

    *stats = DmmaBUpdateStats();
    stats->source_entries = csr.nnz;
    stats->structure_rebuilt = true;
    out->valid = false;
    const auto total_begin = Clock::now();
    if (!validate_device_csr(csr, stream, &stats->validation_ms))
        return false;

    const int tile_row_count =
        active_inner_rows / DMMA_TILE_K +
        (active_inner_rows % DMMA_TILE_K != 0 ? 1 : 0);
    const int tile_col_count =
        logical_cols / DMMA_TILE_N +
        (logical_cols % DMMA_TILE_N != 0 ? 1 : 0);
    if (static_cast<unsigned long long>(tile_row_count) *
            static_cast<unsigned long long>(tile_col_count) >
        std::numeric_limits<EntryKey>::max() / DMMA_INPUT_ELEMS)
    {
        std::fprintf(stderr, "Dynamic B tile-key range overflow.\n");
        return false;
    }

    DmmaOwnedDeviceTiles result;
    result.view.rows = logical_rows;
    result.view.cols = logical_cols;
    result.view.tile_rows = DMMA_TILE_K;
    result.view.tile_cols = DMMA_TILE_N;
    result.view.tile_row_count = tile_row_count;
    result.view.tile_col_count = tile_col_count;

    try
    {
        const auto key_begin = Clock::now();
        DmmaBWorkspace &workspace = out->workspace;
        workspace.entry_keys.resize(csr.nnz);
        workspace.entry_values.resize(csr.nnz);
        const EntryKey invalid_key = std::numeric_limits<EntryKey>::max();
        if (csr.rows > 0 && csr.nnz > 0)
        {
            fill_mapped_entry_keys_kernel<<<block_count(csr.rows), kThreads,
                                            0, stream>>>(
                csr.row_ptr, csr.col_idx, csr.values, csr.rows, logical_rows,
                logical_cols, DMMA_TILE_K, DMMA_TILE_N, tile_col_count, true,
                logical_transpose, inner_old_to_new, nullptr,
                active_inner_rows, logical_cols, invalid_key,
                thrust::raw_pointer_cast(workspace.entry_keys.data()),
                thrust::raw_pointer_cast(workspace.entry_values.data()));
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "launch dynamic B mapped keys"))
                goto failure;
        }

        auto policy = thrust::cuda::par.on(stream);
        thrust::sort_by_key(policy, workspace.entry_keys.begin(),
                            workspace.entry_keys.end(),
                            workspace.entry_values.begin());
        const int active_entry_count = static_cast<int>(
            thrust::lower_bound(policy, workspace.entry_keys.begin(),
                                workspace.entry_keys.end(), invalid_key) -
            workspace.entry_keys.begin());
        stats->active_entries = active_entry_count;

        thrust::device_vector<EntryKey> unique_entry_keys(
            active_entry_count);
        thrust::device_vector<MAT_VAL_TYPE> unique_entry_values(
            active_entry_count);
        auto unique_end = thrust::reduce_by_key(
            policy, workspace.entry_keys.begin(),
            workspace.entry_keys.begin() + active_entry_count,
            workspace.entry_values.begin(), unique_entry_keys.begin(),
            unique_entry_values.begin(), thrust::equal_to<EntryKey>(),
            thrust::plus<MAT_VAL_TYPE>());
        const int unique_entry_count = static_cast<int>(
            unique_end.first - unique_entry_keys.begin());
        if (!dmma_cuda_ok(cudaStreamSynchronize(stream),
                          "complete dynamic B key sort/reduce"))
            goto failure;
        unique_entry_keys.resize(unique_entry_count);
        unique_entry_values.resize(unique_entry_count);
        stats->unique_entries = unique_entry_count;
        stats->key_sort_reduce_ms = milliseconds(key_begin, Clock::now());

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

        if (!device_allocate(&result.tile_row_ptr,
                             static_cast<std::size_t>(tile_row_count) + 1,
                             "allocate dynamic B tile row pointer") ||
            !device_allocate(&result.tile_col_idx, tile_count,
                             "allocate dynamic B tile columns") ||
            !device_allocate(&result.value_offsets,
                             static_cast<std::size_t>(tile_count) + 1,
                             "allocate dynamic B value offsets") ||
            !device_allocate(&result.masks, tile_count,
                             "allocate dynamic B masks") ||
            !dmma_cuda_ok(cudaMemsetAsync(
                              result.tile_row_ptr, 0,
                              (static_cast<std::size_t>(tile_row_count) + 1) *
                                  sizeof(int),
                              stream),
                          "clear dynamic B tile row pointer"))
            goto failure;

        if (tile_count > 0)
        {
            build_tile_metadata_kernel<<<block_count(tile_count), kThreads,
                                         0, stream>>>(
                thrust::raw_pointer_cast(unique_tile_keys.data()),
                thrust::raw_pointer_cast(tile_counts.data()), tile_count,
                tile_col_count, dense_threshold, result.tile_row_ptr,
                result.tile_col_idx,
                thrust::raw_pointer_cast(payload_spans.data()),
                thrust::raw_pointer_cast(dense_flags.data()));
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "build dynamic B tile metadata"))
                goto failure;
        }
        auto row_ptr = thrust::device_pointer_cast(result.tile_row_ptr);
        thrust::inclusive_scan(
            policy, row_ptr,
            row_ptr + static_cast<std::size_t>(tile_row_count) + 1,
            row_ptr);
        const long long payload_size_wide = thrust::reduce(
            policy, payload_spans.begin(), payload_spans.end(), 0ll,
            thrust::plus<long long>());
        if (payload_size_wide > std::numeric_limits<int>::max())
        {
            std::fprintf(stderr,
                         "Dynamic B payload exceeds 32-bit offsets.\n");
            goto failure;
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
                              result.value_offsets + tile_count,
                              &payload_size, sizeof(int),
                              cudaMemcpyHostToDevice, stream),
                          "store dynamic B payload size") ||
            !device_allocate(&result.values, payload_size,
                             "allocate dynamic B payload"))
            goto failure;
        if (tile_count > 0)
        {
            pack_tiles_kernel<<<tile_count, kTileThreads, 0, stream>>>(
                thrust::raw_pointer_cast(unique_entry_keys.data()),
                thrust::raw_pointer_cast(unique_entry_values.data()),
                thrust::raw_pointer_cast(entry_offsets.data()),
                thrust::raw_pointer_cast(tile_counts.data()),
                result.value_offsets, tile_count, dense_threshold,
                result.masks, result.values);
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "pack dynamic B tiles"))
                goto failure;
        }
        if (!dmma_cuda_ok(cudaStreamSynchronize(stream),
                          "complete dynamic B tile packing"))
            goto failure;

        result.view.num_tiles = tile_count;
        result.view.payload_size = payload_size;
        result.view.dense_tiles = dense_tiles;
        result.view.sparse_tiles = tile_count - dense_tiles;
        result.view.structural_nnz =
            static_cast<unsigned long long>(unique_entry_count);
        result.view.tile_row_ptr = result.tile_row_ptr;
        result.view.tile_col_idx = result.tile_col_idx;
        result.view.value_offsets = result.value_offsets;
        result.view.masks = result.masks;
        result.view.values = result.values;
        stats->tile_build_ms = milliseconds(tile_begin, Clock::now());

        double csc_ms = 0.0;
        if (!build_csc(unique_tile_keys, stream, &result, &csc_ms,
                       &stats->peak_workspace_bytes))
            goto failure;
        stats->csc_ms = csc_ms;

        if (!dmma_cuda_ok(cudaStreamSynchronize(stream),
                          "complete dynamic B rebuild"))
            goto failure;
        stats->peak_workspace_bytes = std::max(
            stats->peak_workspace_bytes,
            static_cast<std::size_t>(csr.nnz) *
                (sizeof(EntryKey) + sizeof(MAT_VAL_TYPE)));

        destroy_device_tiles(&out->tiles);
        out->tiles = result;
        result = DmmaOwnedDeviceTiles();
        out->source_nnz = csr.nnz;
        out->source_rows = logical_rows;
        out->source_cols = logical_cols;
        out->active_rows = active_inner_rows;
        out->active_entries = active_entry_count;
        out->has_duplicates = active_entry_count != unique_entry_count;
        /* pack_tiles_kernel initializes every sparse slot and all 32 lanes
         * of every dense tile before this synchronized rebuild commits. */
        out->payload_fully_initialized = true;
        out->fused_csr_row_permute = false;
        out->valid = true;
        stats->total_ms = milliseconds(total_begin, Clock::now());
        return true;
    }
    catch (const std::bad_alloc &)
    {
        std::fprintf(stderr,
                     "Dynamic B rebuild host allocation failed.\n");
    }
    catch (const thrust::system_error &error)
    {
        std::fprintf(stderr, "Dynamic B rebuild Thrust failure: %s\n",
                     error.what());
    }

failure:
    destroy_device_tiles(&result);
    out->valid = false;
    return false;
}

static inline bool rebuild_dynamic_b_from_ordered_csr(
    const DmmaDeviceCsrView &csr, const int *row_new_to_old,
    int active_rows, int dense_threshold, DmmaDynamicB *out,
    DmmaBUpdateStats *stats, cudaStream_t stream = 0)
{
    static_assert(DMMA_TILE_K == 4 && DMMA_TILE_N == 8 &&
                      DMMA_INPUT_ELEMS == 32,
                  "ordered B converter requires the 4x8 B tile layout");
    if (out == nullptr || stats == nullptr || csr.rows < 0 || csr.cols < 0 ||
        csr.nnz < 0 || active_rows < 0 || active_rows > csr.rows ||
        dense_threshold < 1 || dense_threshold > DMMA_INPUT_ELEMS)
        return false;

    *stats = DmmaBUpdateStats();
    stats->source_entries = csr.nnz;
    stats->structure_rebuilt = true;
    out->valid = false;
    const auto total_begin = Clock::now();
    if (!validate_device_csr(csr, stream, &stats->validation_ms))
        return false;

    const int tile_row_count =
        (active_rows + DMMA_TILE_K - 1) / DMMA_TILE_K;
    const int tile_col_count =
        (csr.cols + DMMA_TILE_N - 1) / DMMA_TILE_N;
    if (static_cast<unsigned long long>(tile_row_count) *
            static_cast<unsigned long long>(tile_col_count) >
        std::numeric_limits<EntryKey>::max())
    {
        std::fprintf(stderr, "Ordered B tile-key range overflow.\n");
        return false;
    }

    DmmaOwnedDeviceTiles result;
    result.view.rows = csr.rows;
    result.view.cols = csr.cols;
    result.view.tile_rows = DMMA_TILE_K;
    result.view.tile_cols = DMMA_TILE_N;
    result.view.tile_row_count = tile_row_count;
    result.view.tile_col_count = tile_col_count;
    unsigned long long *d_build_state = nullptr;
    int *d_error = nullptr;
    unsigned long long *d_active_entry_count = nullptr;
    unsigned char *d_layout_workspace = nullptr;
    int h_error = 0;
    const auto tile_begin = Clock::now();

    try
    {
        if (!device_allocate(
                &result.tile_row_ptr,
                static_cast<std::size_t>(tile_row_count) + 1,
                "allocate ordered B tile row pointer") ||
            !device_allocate(&d_build_state, 2,
                             "allocate ordered B build state") ||
            !dmma_cuda_ok(cudaMemsetAsync(
                              result.tile_row_ptr, 0,
                              (static_cast<std::size_t>(tile_row_count) + 1) *
                                  sizeof(int),
                              stream),
                          "clear ordered B tile row pointer") ||
            !dmma_cuda_ok(cudaMemsetAsync(
                              d_build_state, 0,
                              2 * sizeof(unsigned long long), stream),
                          "clear ordered B build state"))
            goto failure_ordered;
        d_error = reinterpret_cast<int *>(d_build_state);
        d_active_entry_count = d_build_state + 1;

        constexpr int kWarpsPerBlock = kThreads / 32;
        const int warp_blocks =
            (tile_row_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
        if (tile_row_count > 0)
        {
            count_ordered_b_tile_rows_warp_kernel<<<warp_blocks, kThreads, 0,
                                                    stream>>>(
                csr.row_ptr, csr.col_idx, csr.rows, row_new_to_old,
                active_rows, result.tile_row_ptr, d_active_entry_count,
                d_error);
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "count ordered B tile rows"))
                goto failure_ordered;
        }
        auto policy = thrust::cuda::par.on(stream);
        auto tile_row_ptr =
            thrust::device_pointer_cast(result.tile_row_ptr);
        thrust::inclusive_scan(
            policy, tile_row_ptr,
            tile_row_ptr + static_cast<std::size_t>(tile_row_count) + 1,
            tile_row_ptr);

        int tile_count = 0;
        unsigned long long active_entry_count_wide = 0;
        if (!dmma_cuda_ok(cudaMemcpyAsync(
                              &tile_count,
                              result.tile_row_ptr + tile_row_count,
                              sizeof(int), cudaMemcpyDeviceToHost, stream),
                          "read ordered B tile count") ||
            !dmma_cuda_ok(cudaMemcpyAsync(
                              &active_entry_count_wide,
                              d_active_entry_count,
                              sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost, stream),
                          "read ordered B active entry count") ||
            !dmma_cuda_ok(cudaMemcpyAsync(
                              &h_error, d_error, sizeof(int),
                              cudaMemcpyDeviceToHost, stream),
                          "read ordered B count error") ||
            !dmma_cuda_ok(cudaStreamSynchronize(stream),
                          "complete ordered B tile count"))
            goto failure_ordered;
        if (h_error == 1)
        {
            cudaFree(d_build_state);
            d_build_state = nullptr;
            d_error = nullptr;
            d_active_entry_count = nullptr;
            destroy_device_tiles(&result);
            DmmaOwnedDeviceCsr permuted;
            double row_permute_ms = 0.0;
            if (!permute_device_csr_rows(
                    csr, row_new_to_old, stream, &permuted,
                    &row_permute_ms))
                return false;
            DmmaBUpdateStats fallback_stats;
            const bool rebuilt = rebuild_dynamic_b(
                device_csr_view(permuted), nullptr, active_rows,
                dense_threshold, out,
                &fallback_stats, stream, false);
            const std::size_t permuted_csr_bytes =
                (static_cast<std::size_t>(permuted.rows) + 1) *
                    sizeof(int) +
                static_cast<std::size_t>(permuted.nnz) *
                    (sizeof(int) + sizeof(MAT_VAL_TYPE));
            destroy_device_csr(&permuted);
            if (!rebuilt)
                return false;
            *stats = fallback_stats;
            stats->csr_row_permute_ms = row_permute_ms;
            stats->peak_workspace_bytes += permuted_csr_bytes;
            stats->ordered_csr_fallback = true;
            stats->total_ms = milliseconds(total_begin, Clock::now());
            return true;
        }
        if (h_error != 0)
            goto failure_ordered;
        if (active_entry_count_wide >
            static_cast<unsigned long long>(
                std::numeric_limits<int>::max()))
        {
            std::fprintf(stderr,
                         "Ordered B active entry count exceeds 32-bit range.\n");
            goto failure_ordered;
        }
        const int active_entry_count =
            static_cast<int>(active_entry_count_wide);
        const auto count_end = Clock::now();
        stats->ordered_count_ms =
            milliseconds(tile_begin, count_end);

        thrust::device_vector<EntryKey> tile_keys(tile_count);
        const std::size_t tile_count_size =
            static_cast<std::size_t>(tile_count);
        const std::size_t metadata_words =
            3 * tile_count_size + 1;
        int *metadata_storage = nullptr;
        const std::size_t layout_int_bytes =
            3 * tile_count_size * sizeof(int);
        const std::size_t totals_offset =
            (layout_int_bytes + alignof(unsigned long long) - 1) &
            ~(alignof(unsigned long long) - 1);
        const std::size_t layout_workspace_bytes =
            totals_offset + 3 * sizeof(unsigned long long);
        if (!device_allocate(&metadata_storage, metadata_words,
                             "allocate ordered B metadata slab"))
            goto failure_ordered;
        result.metadata_storage = metadata_storage;
        result.tile_col_idx = metadata_storage;
        result.value_offsets = metadata_storage + tile_count_size;
        result.masks = reinterpret_cast<uint32_t *>(
            metadata_storage + 2 * tile_count_size + 1);
        if (!device_allocate(&d_layout_workspace, layout_workspace_bytes,
                             "allocate ordered B layout workspace"))
            goto failure_ordered;
        int *tile_entry_counts =
            reinterpret_cast<int *>(d_layout_workspace);
        int *payload_spans = tile_entry_counts + tile_count_size;
        int *dense_flags = payload_spans + tile_count_size;
        auto *d_totals = reinterpret_cast<unsigned long long *>(
            d_layout_workspace + totals_offset);
        if (!dmma_cuda_ok(cudaMemsetAsync(
                              d_totals, 0,
                              3 * sizeof(unsigned long long), stream),
                          "clear ordered B tile totals") ||
            !dmma_cuda_ok(cudaMemsetAsync(
                              result.value_offsets, 0, sizeof(int), stream),
                          "initialize ordered B value offsets"))
            goto failure_ordered;

        if (tile_row_count > 0)
        {
            fill_ordered_b_tile_metadata_warp_kernel<<<
                warp_blocks, kThreads, 0, stream>>>(
                csr.row_ptr, csr.col_idx, active_rows, row_new_to_old,
                tile_col_count, result.tile_row_ptr, result.tile_col_idx,
                tile_entry_counts,
                thrust::raw_pointer_cast(tile_keys.data()), result.masks,
                d_error);
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "build ordered B tile metadata"))
                goto failure_ordered;
        }
        if (tile_count > 0)
        {
            prepare_ordered_b_payload_layout_kernel<<<
                block_count(tile_count), kThreads, 0, stream>>>(
                tile_entry_counts, tile_count, dense_threshold,
                payload_spans, dense_flags);
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "prepare ordered B payload layout"))
                goto failure_ordered;
        }

        auto value_offsets =
            thrust::device_pointer_cast(result.value_offsets);
        if (tile_count > 0)
        {
            auto payload_span_ptr =
                thrust::device_pointer_cast(payload_spans);
            thrust::inclusive_scan(
                policy, payload_span_ptr,
                payload_span_ptr + tile_count,
                value_offsets + 1);
            reduce_ordered_b_tile_stats_kernel<<<
                block_count(tile_count), kThreads, 0, stream>>>(
                tile_entry_counts, dense_flags, payload_spans, tile_count,
                d_totals);
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "reduce ordered B tile statistics"))
                goto failure_ordered;
        }
        unsigned long long h_totals[3] = {0, 0, 0};
        if (!dmma_cuda_ok(cudaMemcpyAsync(
                              h_totals, d_totals,
                              3 * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost, stream),
                          "read ordered B tile totals") ||
            !dmma_cuda_ok(cudaStreamSynchronize(stream),
                          "complete ordered B layout scan"))
            goto failure_ordered;
        const auto layout_end = Clock::now();
        stats->ordered_layout_ms =
            milliseconds(count_end, layout_end);
        const unsigned long long payload_size_wide = h_totals[2];
        if (payload_size_wide > std::numeric_limits<int>::max())
        {
            std::fprintf(stderr,
                         "Ordered B payload exceeds 32-bit offsets.\n");
            goto failure_ordered;
        }
        const int payload_size =
            static_cast<int>(payload_size_wide);
        if (h_totals[0] >
                static_cast<unsigned long long>(
                    std::numeric_limits<int>::max()) ||
            h_totals[1] >
                static_cast<unsigned long long>(
                    std::numeric_limits<int>::max()))
        {
            std::fprintf(stderr,
                         "Ordered B tile statistics exceed 32-bit range.\n");
            goto failure_ordered;
        }
        const int unique_entry_count =
            static_cast<int>(h_totals[0]);
        const int dense_tiles = static_cast<int>(h_totals[1]);
        if (!device_allocate(&result.values, payload_size,
                             "allocate ordered B payload"))
            goto failure_ordered;
        const auto payload_alloc_end = Clock::now();
        stats->ordered_payload_alloc_ms =
            milliseconds(layout_end, payload_alloc_end);

        if (tile_row_count > 0)
        {
            fill_ordered_b_payload_warp_kernel<<<warp_blocks, kThreads, 0,
                                                 stream>>>(
                csr.row_ptr, csr.col_idx, csr.values, active_rows,
                row_new_to_old, dense_threshold, result.tile_row_ptr,
                tile_entry_counts,
                result.value_offsets, result.masks, result.values, d_error);
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "pack ordered B tile payload"))
                goto failure_ordered;
        }
        if (!dmma_cuda_ok(cudaMemcpyAsync(
                              &h_error, d_error, sizeof(int),
                              cudaMemcpyDeviceToHost, stream),
                          "read ordered B build error") ||
            !dmma_cuda_ok(cudaStreamSynchronize(stream),
                          "complete ordered B tile packing") ||
            h_error != 0)
            goto failure_ordered;
        const auto payload_fill_end = Clock::now();
        stats->ordered_payload_fill_ms =
            milliseconds(payload_alloc_end, payload_fill_end);
        stats->ordered_pack_ms =
            milliseconds(count_end, payload_fill_end);

        result.view.num_tiles = tile_count;
        result.view.payload_size = payload_size;
        result.view.dense_tiles = dense_tiles;
        result.view.sparse_tiles = tile_count - dense_tiles;
        result.view.structural_nnz =
            static_cast<unsigned long long>(unique_entry_count);
        result.view.tile_row_ptr = result.tile_row_ptr;
        result.view.tile_col_idx = result.tile_col_idx;
        result.view.value_offsets = result.value_offsets;
        result.view.masks = result.masks;
        result.view.values = result.values;
        stats->active_entries = active_entry_count;
        stats->unique_entries = unique_entry_count;
        stats->tile_build_ms = milliseconds(tile_begin, Clock::now());
        stats->peak_workspace_bytes = std::max(
            stats->peak_workspace_bytes,
            static_cast<std::size_t>(tile_count) *
                (2 * sizeof(EntryKey) + 4 * sizeof(int)));

        double csc_ms = 0.0;
        auto *csc_keys_scratch =
            reinterpret_cast<EntryKey *>(d_layout_workspace);
        int *csc_ids_scratch = reinterpret_cast<int *>(
            d_layout_workspace +
            static_cast<std::size_t>(tile_count) * sizeof(EntryKey));
        if (!build_csc(tile_keys, stream, &result, &csc_ms,
                       &stats->peak_workspace_bytes, csc_keys_scratch,
                       csc_ids_scratch))
            goto failure_ordered;
        stats->csc_ms = csc_ms;
        cudaFree(d_layout_workspace);
        d_layout_workspace = nullptr;

        cudaFree(d_build_state);
        d_build_state = nullptr;
        d_error = nullptr;
        d_active_entry_count = nullptr;
        destroy_device_tiles(&out->tiles);
        out->tiles = result;
        result = DmmaOwnedDeviceTiles();
        out->source_nnz = csr.nnz;
        out->source_rows = csr.rows;
        out->source_cols = csr.cols;
        out->active_rows = active_rows;
        out->active_entries = active_entry_count;
        out->has_duplicates =
            active_entry_count != unique_entry_count;
        out->payload_fully_initialized = true;
        out->fused_csr_row_permute = row_new_to_old != nullptr;
        out->valid = true;
        stats->fused_csr_row_permute = row_new_to_old != nullptr;
        stats->total_ms = milliseconds(total_begin, Clock::now());
        return true;
    }
    catch (const std::bad_alloc &)
    {
        std::fprintf(stderr,
                     "Ordered B rebuild host allocation failed.\n");
    }
    catch (const thrust::system_error &error)
    {
        std::fprintf(stderr, "Ordered B rebuild Thrust failure: %s\n",
                     error.what());
    }

failure_ordered:
    cudaFree(d_layout_workspace);
    cudaFree(d_build_state);
    destroy_device_tiles(&result);
    out->valid = false;
    return false;
}

static inline bool rebuild_dynamic_b_via_permuted_csr(
    const DmmaDeviceCsrView &source, const int *row_new_to_old,
    int active_rows, int dense_threshold, DmmaDynamicB *out,
    DmmaBUpdateStats *stats, cudaStream_t stream = 0)
{
    if (out == nullptr || stats == nullptr)
        return false;
    return rebuild_dynamic_b_from_ordered_csr(
        source, row_new_to_old, active_rows, dense_threshold, out, stats,
        stream);
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

static inline bool compute_nnz_cub_ab(
    const DmmaDeviceCsrView &a, const DmmaDeviceCsrView &b,
    cudaStream_t stream, unsigned long long *nnz_cub, double *elapsed_ms)
{
    if (nnz_cub == nullptr || a.cols != b.rows || a.nnz < 0 ||
        (a.nnz > 0 && (a.col_idx == nullptr || b.row_ptr == nullptr)))
        return false;
    const auto begin = Clock::now();
    try
    {
        thrust::device_vector<unsigned long long> terms(a.nnz);
        if (a.nnz > 0)
        {
            fill_nnz_cub_ab_terms_kernel<<<block_count(a.nnz), kThreads, 0,
                                           stream>>>(
                a.col_idx, a.nnz, b.row_ptr,
                thrust::raw_pointer_cast(terms.data()));
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "launch AB nnzCub terms"))
                return false;
        }
        *nnz_cub = thrust::reduce(thrust::cuda::par.on(stream), terms.begin(),
                                  terms.end(), 0ull,
                                  thrust::plus<unsigned long long>());
        if (!dmma_cuda_ok(cudaStreamSynchronize(stream),
                          "complete AB nnzCub"))
            return false;
    }
    catch (const std::bad_alloc &)
    {
        std::fprintf(stderr, "AB nnzCub host allocation failed.\n");
        return false;
    }
    catch (const thrust::system_error &error)
    {
        std::fprintf(stderr, "AB nnzCub Thrust failure: %s\n",
                     error.what());
        return false;
    }
    if (elapsed_ms != nullptr)
        *elapsed_ms = milliseconds(begin, Clock::now());
    return true;
}

} // namespace gpu_dmma_detail

/* Best-effort preparation for ExactTile-Sparse v1.  Failure invalidates only
 * the optional metadata and leaves the base hybrid tiles intact, so the next
 * SpGEMM call can select the unchanged RTT path.  Main calls A-row setup in
 * offline preprocessing and B-column setup after every online B rebuild. */
static inline bool gpu_prepare_low_fill_exact_tile_metadata(
    DmmaOwnedDeviceTiles *tiles, bool build_row_sums, bool build_col_sums,
    cudaStream_t stream = 0)
{
    if (tiles == nullptr || build_row_sums == build_col_sums ||
        tiles->view.num_tiles < 0 ||
        (build_row_sums && tiles->view.tile_row_ptr == nullptr) ||
        (build_col_sums &&
         (tiles->view.tile_col_ptr == nullptr ||
          (tiles->view.num_tiles > 0 &&
           (tiles->view.csc_tile_ids == nullptr ||
            tiles->view.masks == nullptr)))) ||
        (tiles->view.num_tiles > 0 && tiles->view.masks == nullptr))
        return false;

    uint32_t **owned = build_row_sums ? &tiles->row_tile_nnz_sum
                                      : &tiles->col_tile_nnz_sum;
    const int entries = build_row_sums ? tiles->view.tile_row_count
                                       : tiles->view.tile_col_count;
    cudaFree(*owned);
    *owned = nullptr;
    if (build_row_sums)
    {
        tiles->view.row_tile_nnz_sum = nullptr;
        tiles->view.row_tile_nnz_sum_valid = false;
    }
    else
    {
        tiles->view.col_tile_nnz_sum = nullptr;
        tiles->view.col_tile_nnz_sum_valid = false;
    }
    tiles->view.low_fill_metadata_overflow = false;

    if (entries < 0 ||
        static_cast<std::size_t>(entries) >
            DMMA_LOW_FILL_METADATA_BUDGET_BYTES / sizeof(uint32_t))
        return false;
    if (entries == 0)
    {
        if (build_row_sums)
            tiles->view.row_tile_nnz_sum_valid = true;
        else
            tiles->view.col_tile_nnz_sum_valid = true;
        return true;
    }

    int *d_overflow = nullptr;
    int h_overflow = 0;
    if (!gpu_dmma_detail::device_allocate(
            owned, static_cast<std::size_t>(entries),
            build_row_sums ? "allocate low-fill A row sums"
                           : "allocate low-fill B column sums") ||
        !gpu_dmma_detail::device_allocate(
            &d_overflow, 1, "allocate low-fill metadata overflow flag") ||
        !dmma_cuda_ok(cudaMemsetAsync(d_overflow, 0, sizeof(int), stream),
                      "clear low-fill metadata overflow flag"))
        goto fallback;

    gpu_dmma_detail::build_low_fill_tile_nnz_sum_kernel
        <<<gpu_dmma_detail::block_count(
               static_cast<std::size_t>(entries)),
           gpu_dmma_detail::kThreads, 0, stream>>>(
            tiles->view, build_col_sums ? 1 : 0, entries, *owned,
            d_overflow);
    if (!dmma_cuda_ok(cudaGetLastError(),
                      "launch low-fill tile nnz sums") ||
        !dmma_cuda_ok(cudaMemcpyAsync(
                          &h_overflow, d_overflow, sizeof(int),
                          cudaMemcpyDeviceToHost, stream),
                      "read low-fill metadata overflow flag") ||
        !dmma_cuda_ok(cudaStreamSynchronize(stream),
                      "complete low-fill metadata construction"))
        goto fallback;

    cudaFree(d_overflow);
    if (h_overflow != 0)
    {
        cudaFree(*owned);
        *owned = nullptr;
        tiles->view.low_fill_metadata_overflow = true;
        return false;
    }
    if (build_row_sums)
    {
        tiles->view.row_tile_nnz_sum = *owned;
        tiles->view.row_tile_nnz_sum_valid = true;
    }
    else
    {
        tiles->view.col_tile_nnz_sum = *owned;
        tiles->view.col_tile_nnz_sum_valid = true;
    }
    return true;

fallback:
    cudaFree(d_overflow);
    cudaFree(*owned);
    *owned = nullptr;
    cudaGetLastError();
    return false;
}

static inline bool gpu_upload_csr(const SMatrix &host,
                                  DmmaOwnedDeviceCsr *out,
                                  double *h2d_ms = nullptr,
                                  double *validation_ms = nullptr,
                                  cudaStream_t stream = 0)
{
    DmmaPreprocessStats stats;
    if (!gpu_dmma_detail::upload_csr(host, stream, out, &stats))
        return false;
    if (h2d_ms != nullptr)
        *h2d_ms = stats.h2d_ms;
    if (validation_ms != nullptr)
        *validation_ms = stats.validation_ms;
    return true;
}

static inline bool gpu_rebuild_dynamic_b_via_permuted_csr(
    const DmmaDeviceCsrView &csr, const int *inner_new_to_old,
    int active_inner_rows, int dense_threshold, DmmaDynamicB *out,
    DmmaBUpdateStats *stats, cudaStream_t stream = 0)
{
    return gpu_dmma_detail::rebuild_dynamic_b_via_permuted_csr(
        csr, inner_new_to_old, active_inner_rows, dense_threshold, out, stats,
        stream);
}

static inline bool gpu_rebuild_dynamic_b_transpose(
    const DmmaDeviceCsrView &source, const int *inner_old_to_new,
    int active_inner_rows, int dense_threshold, DmmaDynamicB *out,
    DmmaBUpdateStats *stats, cudaStream_t stream = 0)
{
    return gpu_dmma_detail::rebuild_dynamic_b(
        source, inner_old_to_new, active_inner_rows, dense_threshold, out,
        stats, stream, true);
}

static inline bool gpu_compute_nnz_cub_ab(
    const DmmaDeviceCsrView &a, const DmmaDeviceCsrView &b,
    unsigned long long *nnz_cub, double *elapsed_ms,
    cudaStream_t stream = 0)
{
    return gpu_dmma_detail::compute_nnz_cub_ab(
        a, b, stream, nnz_cub, elapsed_ms);
}

static inline bool gpu_compute_nnz_cub_derived(
    const DmmaOwnedDeviceCsr &a, bool aat, unsigned long long *nnz_cub,
    double *elapsed_ms, cudaStream_t stream = 0)
{
    if (nnz_cub == nullptr)
        return false;
    DmmaPreprocessStats stats;
    if (!gpu_dmma_detail::compute_nnz_cub(a, aat, stream, nnz_cub,
                                           &stats))
        return false;
    if (elapsed_ms != nullptr)
        *elapsed_ms = stats.nnz_cub_ms;
    return true;
}

static inline bool gpu_build_reordered_a_tiles(
    const DmmaOwnedDeviceCsr &csr, const int *row_old_to_new,
    const int *inner_old_to_new, int active_rows, int active_inner,
    int dense_threshold, DmmaOwnedDeviceTiles *out,
    double *key_sort_reduce_ms, double *tile_build_ms,
    std::size_t *peak_workspace_bytes, cudaStream_t stream = 0)
{
    double unused_csc_ms = 0.0;
    return gpu_dmma_detail::build_tiles(
        csr, false, DMMA_TILE_M, DMMA_TILE_K, false, dense_threshold, false,
        stream, out, key_sort_reduce_ms, tile_build_ms, &unused_csc_ms,
        peak_workspace_bytes, row_old_to_new, inner_old_to_new, active_rows,
        active_inner);
}

static inline bool gpu_prepare_reordered_a(
    const SMatrix &host, int dense_threshold, DmmaPreparedA *out,
    DmmaOfflineAStats *stats, cudaStream_t stream = 0,
    const DmmaReorderConfig *reorder_config = nullptr)
{
    if (out == nullptr || stats == nullptr || dense_threshold < 1 ||
        dense_threshold > DMMA_INPUT_ELEMS)
        return false;
    *stats = DmmaOfflineAStats();
    DmmaPreparedA result;
    result.dense_threshold = dense_threshold;
    const auto total_begin = std::chrono::steady_clock::now();
    if (!gpu_upload_csr(host, &result.csr, &stats->h2d_ms,
                        &stats->validation_ms, stream))
        goto failure;
    {
        const auto begin = std::chrono::steady_clock::now();
        if (!build_dmma_reorder_plan(
                result.csr.rows, result.csr.cols, result.csr.nnz,
                result.csr.row_ptr, result.csr.col_idx, dense_threshold,
                &result.reorder, stream, reorder_config))
            goto failure;
        stats->reorder_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        stats->peak_workspace_bytes = std::max(
            stats->peak_workspace_bytes,
            result.reorder.reorder_peak_workspace_bytes);
    }
    if (!gpu_build_reordered_a_tiles(
            result.csr, result.reorder.d_row_old_to_new,
            result.reorder.d_inner_old_to_new, result.reorder.active_rows,
            result.reorder.active_inner, dense_threshold, &result.tiles,
            &stats->key_sort_reduce_ms, &stats->tile_build_ms,
            &stats->peak_workspace_bytes, stream) ||
        !dmma_cuda_ok(cudaStreamSynchronize(stream),
                      "complete reordered A preparation"))
        goto failure;
    result.reorder.num_tiles = result.tiles.view.num_tiles;
    result.reorder.active_row_tiles = result.tiles.view.tile_row_count;
    result.reorder.active_k_tiles = result.tiles.view.tile_col_count;
    result.reorder.sparse_tiles = result.tiles.view.sparse_tiles;
    result.reorder.dense_tiles = result.tiles.view.dense_tiles;
    result.reorder.payload = result.tiles.view.payload_size;
    result.valid = true;
    stats->total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_begin).count();
    destroy_prepared_a(out);
    *out = std::move(result);
    return true;

failure:
    destroy_prepared_a(&result);
    return false;
}

static inline bool gpu_prepare_external_a(
    const SMatrix &host, const char *row_order_path,
    const char *inner_order_path, const char *reorder_name,
    int dense_threshold, DmmaPreparedA *out, DmmaOfflineAStats *stats,
    cudaStream_t stream = 0)
{
    if (out == nullptr || stats == nullptr || row_order_path == nullptr ||
        inner_order_path == nullptr || reorder_name == nullptr ||
        reorder_name[0] == '\0' || dense_threshold < 1 ||
        dense_threshold > DMMA_INPUT_ELEMS)
        return false;
    *stats = DmmaOfflineAStats();
    DmmaPreparedA result;
    result.dense_threshold = dense_threshold;
    const auto total_begin = std::chrono::steady_clock::now();
    if (!gpu_upload_csr(host, &result.csr, &stats->h2d_ms,
                        &stats->validation_ms, stream))
        goto failure;
    {
        const auto begin = std::chrono::steady_clock::now();
        if (!build_external_dmma_reorder_plan(
                host, row_order_path, inner_order_path, reorder_name,
                &result.reorder, stream))
            goto failure;
        stats->reorder_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
    }
    if (!gpu_build_reordered_a_tiles(
            result.csr, result.reorder.d_row_old_to_new,
            result.reorder.d_inner_old_to_new, result.reorder.active_rows,
            result.reorder.active_inner, dense_threshold, &result.tiles,
            &stats->key_sort_reduce_ms, &stats->tile_build_ms,
            &stats->peak_workspace_bytes, stream) ||
        !dmma_cuda_ok(cudaStreamSynchronize(stream),
                      "complete external A preparation"))
        goto failure;
    result.reorder.num_tiles = result.tiles.view.num_tiles;
    result.reorder.active_row_tiles = result.tiles.view.tile_row_count;
    result.reorder.active_k_tiles = result.tiles.view.tile_col_count;
    result.reorder.sparse_tiles = result.tiles.view.sparse_tiles;
    result.reorder.dense_tiles = result.tiles.view.dense_tiles;
    result.reorder.payload = result.tiles.view.payload_size;
    result.valid = true;
    stats->total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_begin).count();
    destroy_prepared_a(out);
    *out = std::move(result);
    return true;

failure:
    destroy_prepared_a(&result);
    return false;
}

static inline bool gpu_prepare_identity_a(
    const SMatrix &host, int dense_threshold, DmmaPreparedA *out,
    DmmaOfflineAStats *stats, cudaStream_t stream = 0)
{
    if (out == nullptr || stats == nullptr || dense_threshold < 1 ||
        dense_threshold > DMMA_INPUT_ELEMS)
        return false;
    *stats = DmmaOfflineAStats();
    DmmaPreparedA result;
    result.dense_threshold = dense_threshold;
    result.reorder.rows = host.m;
    result.reorder.cols = host.n;
    result.reorder.nnz = host.nnz;
    /* Keep the identity permutation but trim only a provably empty suffix.
     * This makes --no-reorder a fair baseline for the adaptive plan, whose
     * selected identity candidate performs the same safe tail trimming. */
    result.reorder.active_rows = 0;
    result.reorder.active_inner = 0;
    for (int row = 0; row < host.m; ++row)
    {
        if (host.rowpointer[row] != host.rowpointer[row + 1])
            result.reorder.active_rows = row + 1;
        for (int entry = host.rowpointer[row];
             entry < host.rowpointer[row + 1]; ++entry)
            result.reorder.active_inner =
                std::max(result.reorder.active_inner,
                         host.columnindex[entry] + 1);
    }
    result.reorder.unified = false;
    result.reorder.kind = DMMA_REORDER_IDENTITY;
    std::snprintf(result.reorder.algorithm,
                  sizeof(result.reorder.algorithm), "identity-baseline");
    result.reorder.sweeps = 0;
    const auto total_begin = std::chrono::steady_clock::now();
    if (!gpu_upload_csr(host, &result.csr, &stats->h2d_ms,
                        &stats->validation_ms, stream) ||
        !gpu_build_reordered_a_tiles(
            result.csr, nullptr, nullptr, result.reorder.active_rows,
            result.reorder.active_inner, dense_threshold,
            &result.tiles, &stats->key_sort_reduce_ms,
            &stats->tile_build_ms, &stats->peak_workspace_bytes, stream) ||
        !dmma_cuda_ok(cudaStreamSynchronize(stream),
                      "complete identity A preparation"))
    {
        destroy_prepared_a(&result);
        return false;
    }
    result.reorder.num_tiles = result.tiles.view.num_tiles;
    result.reorder.active_row_tiles = result.tiles.view.tile_row_count;
    result.reorder.active_k_tiles = result.tiles.view.tile_col_count;
    result.reorder.sparse_tiles = result.tiles.view.sparse_tiles;
    result.reorder.dense_tiles = result.tiles.view.dense_tiles;
    result.reorder.payload = result.tiles.view.payload_size;
    result.valid = true;
    stats->total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_begin).count();
    destroy_prepared_a(out);
    *out = std::move(result);
    return true;
}

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
