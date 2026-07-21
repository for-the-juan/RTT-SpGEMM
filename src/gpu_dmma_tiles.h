#ifndef RTT_SPGEMM_GPU_DMMA_TILES_H_
#define RTT_SPGEMM_GPU_DMMA_TILES_H_

#include "dmma_spgemm.h"
#include "dmma_reorder.h"
#include "dmma_b_values_clear_policy.h"

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
    double validation_ms = 0.0;
    double key_sort_reduce_ms = 0.0;
    double tile_build_ms = 0.0;
    double csc_ms = 0.0;
    double mapping_ms = 0.0;
    double low_fill_metadata_ms = 0.0;
    double value_update_ms = 0.0;
    double total_ms = 0.0;
    std::size_t peak_workspace_bytes = 0;
    int source_entries = 0;
    int active_entries = 0;
    int unique_entries = 0;
    bool structure_rebuilt = false;
};

/* Grow-only scratch used by repeated B structure updates.  Resizing a
 * device_vector within its capacity does not allocate, so the common
 * fixed-or-shrinking topology case keeps the sort/mapping workspace alive. */
struct DmmaBWorkspace
{
    thrust::device_vector<unsigned long long> entry_keys;
    thrust::device_vector<MAT_VAL_TYPE> entry_values;
    thrust::device_vector<int> source_ids;
    thrust::device_vector<int> unique_ids;
    thrust::device_vector<int> head_flags;
};

struct DmmaDynamicB
{
    DmmaOwnedDeviceTiles tiles;
    int *source_to_payload = nullptr;
    int source_nnz = 0;
    int source_capacity = 0;
    int source_rows = 0;
    int source_cols = 0;
    int active_rows = 0;
    int active_entries = 0;
    bool has_duplicates = false;
    /* Hard eligibility facts for values-only no-clear updates.  A successful
     * rebuild sets both only after pack and source-map construction finish. */
    bool active_source_mapping_complete = false;
    bool active_source_to_payload_injective = false;
    bool payload_fully_initialized = false;
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
    double total_ms = 0.0;
    std::size_t peak_workspace_bytes = 0;
};

struct DmmaPreparedA
{
    DmmaOwnedDeviceCsr csr;
    DmmaReorderPlan reorder;
    DmmaOwnedDeviceTiles tiles;
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
    cudaFree(matrix->source_to_payload);
    *matrix = DmmaDynamicB();
}

/* Values-only iterations need only the persistent source-to-payload map.
 * Release the sort/reduce scratch after the initial structure build so very
 * large B inputs do not pin O(nnz) temporary storage throughout SpGEMM. */
static inline void release_dynamic_b_rebuild_workspace(DmmaDynamicB *matrix)
{
    if (matrix == nullptr)
        return;
    matrix->workspace = DmmaBWorkspace();
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
    MAT_VAL_TYPE *key_values, int *source_ids)
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
        if (source_ids != nullptr)
            source_ids[entry] = entry;
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

__global__ void mark_unique_entry_heads_kernel(const EntryKey *keys,
                                                int count, int *heads)
{
    const int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry < count)
        heads[entry] = entry == 0 || keys[entry] != keys[entry - 1];
}

__global__ void build_unique_payload_map_kernel(
    const EntryKey *unique_keys, const int *entry_offsets,
    const int *entry_counts, const int *value_offsets, int tile_count,
    int dense_threshold, int *unique_to_payload)
{
    const int tile = blockIdx.x;
    if (tile >= tile_count)
        return;
    const int begin = entry_offsets[tile];
    const int count = entry_counts[tile];
    const int output = value_offsets[tile];
    const bool dense = count >= dense_threshold;
    for (int local = threadIdx.x; local < count; local += blockDim.x)
    {
        const int unique_entry = begin + local;
        const int physical = static_cast<int>(
            unique_keys[unique_entry] % DMMA_INPUT_ELEMS);
        unique_to_payload[unique_entry] =
            output + (dense ? physical : local);
    }
}

__global__ void finish_source_payload_map_kernel(
    int count, const int *sorted_source_ids, const int *inclusive_unique_ids,
    const int *unique_to_payload, int *source_to_payload)
{
    const int sorted_entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_entry >= count)
        return;
    const int unique_entry = inclusive_unique_ids[sorted_entry] - 1;
    source_to_payload[sorted_source_ids[sorted_entry]] =
        unique_to_payload[unique_entry];
}

__global__ void update_payload_values_kernel(
    int count, const MAT_VAL_TYPE *source_values,
    const int *source_to_payload, bool use_atomics, MAT_VAL_TYPE *payload)
{
    const int source = blockIdx.x * blockDim.x + threadIdx.x;
    if (source >= count)
        return;
    const int destination = source_to_payload[source];
    if (destination < 0)
        return;
    if (use_atomics)
        atomicAdd(payload + destination, source_values[source]);
    else
        payload[destination] = source_values[source];
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
    *out = result;
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
                thrust::raw_pointer_cast(entry_values.data()), nullptr);
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
    *out = result;
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
        workspace.source_ids.resize(csr.nnz);
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
                thrust::raw_pointer_cast(workspace.entry_values.data()),
                thrust::raw_pointer_cast(workspace.source_ids.data()));
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "launch dynamic B mapped keys"))
                goto failure;
        }

        auto policy = thrust::cuda::par.on(stream);
        auto zipped_values = thrust::make_zip_iterator(thrust::make_tuple(
            workspace.entry_values.begin(), workspace.source_ids.begin()));
        thrust::sort_by_key(policy, workspace.entry_keys.begin(),
                            workspace.entry_keys.end(), zipped_values);
        const int active_entry_count = static_cast<int>(
            thrust::lower_bound(policy, workspace.entry_keys.begin(),
                                workspace.entry_keys.end(), invalid_key) -
            workspace.entry_keys.begin());
        stats->active_entries = active_entry_count;

        workspace.head_flags.resize(active_entry_count);
        workspace.unique_ids.resize(active_entry_count);
        if (active_entry_count > 0)
        {
            mark_unique_entry_heads_kernel<<<
                block_count(active_entry_count), kThreads, 0, stream>>>(
                thrust::raw_pointer_cast(workspace.entry_keys.data()),
                active_entry_count,
                thrust::raw_pointer_cast(workspace.head_flags.data()));
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "mark dynamic B unique entries"))
                goto failure;
            thrust::inclusive_scan(policy, workspace.head_flags.begin(),
                                   workspace.head_flags.end(),
                                   workspace.unique_ids.begin());
        }

        int unique_entry_count = 0;
        if (active_entry_count > 0 &&
            !dmma_cuda_ok(cudaMemcpyAsync(
                              &unique_entry_count,
                              thrust::raw_pointer_cast(
                                  workspace.unique_ids.data()) +
                                  active_entry_count - 1,
                              sizeof(int), cudaMemcpyDeviceToHost, stream),
                          "read dynamic B unique entry count"))
            goto failure;

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
        const int reduced_unique_count = static_cast<int>(
            unique_end.first - unique_entry_keys.begin());
        if (!dmma_cuda_ok(cudaStreamSynchronize(stream),
                          "complete dynamic B key sort/reduce") ||
            reduced_unique_count != unique_entry_count)
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

        const auto mapping_begin = Clock::now();
        if (csr.nnz > out->source_capacity)
        {
            cudaFree(out->source_to_payload);
            out->source_to_payload = nullptr;
            out->source_capacity = 0;
            if (!device_allocate(&out->source_to_payload, csr.nnz,
                                 "allocate dynamic B source mapping"))
                goto failure;
            out->source_capacity = csr.nnz;
        }
        if (csr.nnz > 0 &&
            !dmma_cuda_ok(cudaMemsetAsync(
                              out->source_to_payload, 0xff,
                              static_cast<std::size_t>(csr.nnz) * sizeof(int),
                              stream),
                          "clear dynamic B source mapping"))
            goto failure;
        if (unique_entry_count > 0)
        {
            thrust::device_vector<int> unique_to_payload(
                unique_entry_count);
            build_unique_payload_map_kernel<<<tile_count, kTileThreads, 0,
                                              stream>>>(
                thrust::raw_pointer_cast(unique_entry_keys.data()),
                thrust::raw_pointer_cast(entry_offsets.data()),
                thrust::raw_pointer_cast(tile_counts.data()),
                result.value_offsets, tile_count, dense_threshold,
                thrust::raw_pointer_cast(unique_to_payload.data()));
            finish_source_payload_map_kernel<<<
                block_count(active_entry_count), kThreads, 0, stream>>>(
                active_entry_count,
                thrust::raw_pointer_cast(workspace.source_ids.data()),
                thrust::raw_pointer_cast(workspace.unique_ids.data()),
                thrust::raw_pointer_cast(unique_to_payload.data()),
                out->source_to_payload);
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "build dynamic B source mapping"))
                goto failure;
        }
        if (!dmma_cuda_ok(cudaStreamSynchronize(stream),
                          "complete dynamic B rebuild"))
            goto failure;
        stats->mapping_ms = milliseconds(mapping_begin, Clock::now());
        stats->peak_workspace_bytes = std::max(
            stats->peak_workspace_bytes,
            static_cast<std::size_t>(csr.nnz) *
                (sizeof(EntryKey) + sizeof(MAT_VAL_TYPE) +
                 sizeof(int) * 3));

        destroy_device_tiles(&out->tiles);
        out->tiles = result;
        result = DmmaOwnedDeviceTiles();
        out->source_nnz = csr.nnz;
        out->source_rows = logical_rows;
        out->source_cols = logical_cols;
        out->active_rows = active_inner_rows;
        out->active_entries = active_entry_count;
        out->has_duplicates = active_entry_count != unique_entry_count;
        /* Every active source is assigned the inclusive ID of its sorted
         * structural key.  With no repeated keys, unique_to_payload maps
         * those IDs to disjoint tile offsets (dense: physical lane; sparse:
         * local rank), hence the active source map is injective. */
        out->active_source_mapping_complete = true;
        out->active_source_to_payload_injective =
            active_entry_count == unique_entry_count;
        /* pack_tiles_kernel initializes every sparse slot and all 32 lanes
         * of every dense tile before this synchronized rebuild commits. */
        out->payload_fully_initialized = true;
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

static inline bool update_dynamic_b_values(
    const MAT_VAL_TYPE *device_values, int nnz, DmmaDynamicB *matrix,
    DmmaBUpdateStats *stats, cudaStream_t stream = 0)
{
    if (matrix == nullptr || stats == nullptr || !matrix->valid || nnz < 0 ||
        nnz != matrix->source_nnz ||
        (nnz > 0 && device_values == nullptr))
        return false;
    *stats = DmmaBUpdateStats();
    stats->source_entries = nnz;
    stats->active_entries = matrix->active_entries;
    const auto begin = Clock::now();
    if (matrix->tiles.view.payload_size > 0 &&
        !dmma_cuda_ok(cudaMemsetAsync(
                          matrix->tiles.values, 0,
                          static_cast<std::size_t>(
                              matrix->tiles.view.payload_size) *
                              sizeof(MAT_VAL_TYPE),
                          stream),
                      "clear dynamic B payload values"))
    {
        matrix->valid = false;
        return false;
    }
    if (nnz > 0)
    {
        update_payload_values_kernel<<<block_count(nnz), kThreads, 0,
                                       stream>>>(
            nnz, device_values, matrix->source_to_payload,
            matrix->has_duplicates, matrix->tiles.values);
        if (!dmma_cuda_ok(cudaGetLastError(),
                          "launch dynamic B value update"))
        {
            matrix->valid = false;
            return false;
        }
    }
    if (!dmma_cuda_ok(cudaStreamSynchronize(stream),
                      "complete dynamic B value update"))
    {
        matrix->valid = false;
        return false;
    }
    stats->value_update_ms = milliseconds(begin, Clock::now());
    stats->total_ms = stats->value_update_ms;
    return true;
}

/* Optional fast path.  The legacy entry point above remains byte-for-byte in
 * charge of the default always-clear policy.  This separate function is used
 * only for an explicit non-default policy, so the default Core path gains no
 * policy-selection work. */
static inline bool update_dynamic_b_values_with_policy(
    const MAT_VAL_TYPE *device_values, int nnz, DmmaDynamicB *matrix,
    DmmaBValuesClearPolicy policy, DmmaBValuesClearDecision *decision_out,
    DmmaBUpdateStats *stats, cudaStream_t stream = 0)
{
    if (matrix == nullptr || stats == nullptr || decision_out == nullptr ||
        !matrix->valid || nnz < 0 || nnz != matrix->source_nnz ||
        (nnz > 0 && device_values == nullptr) ||
        policy == DMMA_B_VALUES_ALWAYS_CLEAR)
        return false;

    *stats = DmmaBUpdateStats();
    stats->source_entries = nnz;
    stats->active_entries = matrix->active_entries;
    const auto begin = Clock::now();
    const DmmaBValuesClearDecision decision = dmma_choose_b_values_clear(
        policy, true, matrix->valid, matrix->has_duplicates,
        matrix->active_source_mapping_complete,
        matrix->active_source_to_payload_injective,
        matrix->payload_fully_initialized,
        matrix->tiles.view.dense_tiles == 0, true);
    *decision_out = decision;

    if (decision.clear_payload && matrix->tiles.view.payload_size > 0 &&
        !dmma_cuda_ok(cudaMemsetAsync(
                          matrix->tiles.values, 0,
                          static_cast<std::size_t>(
                              matrix->tiles.view.payload_size) *
                              sizeof(MAT_VAL_TYPE),
                          stream),
                      "clear dynamic B payload values (safe fallback)"))
    {
        matrix->valid = false;
        return false;
    }
    if (nnz > 0)
    {
        update_payload_values_kernel<<<block_count(nnz), kThreads, 0,
                                       stream>>>(
            nnz, device_values, matrix->source_to_payload,
            matrix->has_duplicates, matrix->tiles.values);
        if (!dmma_cuda_ok(cudaGetLastError(),
                          "launch dynamic B value update"))
        {
            matrix->valid = false;
            return false;
        }
    }
    if (!dmma_cuda_ok(cudaStreamSynchronize(stream),
                      "complete dynamic B value update"))
    {
        matrix->valid = false;
        return false;
    }
    stats->value_update_ms = milliseconds(begin, Clock::now());
    stats->total_ms = stats->value_update_ms;
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
 * offline preprocessing and B-column setup after every structural rebuild;
 * values-only B updates reuse the sums because masks are unchanged. */
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

static inline bool gpu_rebuild_dynamic_b(
    const DmmaDeviceCsrView &csr, const int *inner_old_to_new,
    int active_inner_rows, int dense_threshold, DmmaDynamicB *out,
    DmmaBUpdateStats *stats, cudaStream_t stream = 0)
{
    return gpu_dmma_detail::rebuild_dynamic_b(
        csr, inner_old_to_new, active_inner_rows, dense_threshold, out, stats,
        stream, false);
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

static inline bool gpu_update_dynamic_b_values(
    const MAT_VAL_TYPE *device_values, int nnz, DmmaDynamicB *matrix,
    DmmaBUpdateStats *stats, cudaStream_t stream = 0)
{
    return gpu_dmma_detail::update_dynamic_b_values(
        device_values, nnz, matrix, stats, stream);
}

static inline bool gpu_update_dynamic_b_values_with_policy(
    const MAT_VAL_TYPE *device_values, int nnz, DmmaDynamicB *matrix,
    DmmaBValuesClearPolicy policy, DmmaBValuesClearDecision *decision,
    DmmaBUpdateStats *stats, cudaStream_t stream = 0)
{
    return gpu_dmma_detail::update_dynamic_b_values_with_policy(
        device_values, nnz, matrix, policy, decision, stats, stream);
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
    *out = result;
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
    *out = result;
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
    *out = result;
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
