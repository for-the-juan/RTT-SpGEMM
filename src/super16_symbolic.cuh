#ifndef RTT_SPGEMM_SUPER16_SYMBOLIC_CUH_
#define RTT_SPGEMM_SUPER16_SYMBOLIC_CUH_

/*
 * Experimental CUDA implementation of the single-path Super16 symbolic
 * design.  This file is intentionally not wired into dmma_spgemm.h yet: it
 * can be compiled and tested independently while the frozen 8x4/4x8/8x8
 * production path continues to run the registered campaigns.
 *
 * Super16 is a value-free index over the existing leaf tiles.  It never
 * changes DMMA_TILE_M/K/N and it never owns or copies a numerical value.
 */

#include "dmma_device_tiles.h"
#include "dmma_tiles.h"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace rtt
{
namespace super16
{

/* Candidate discovery and exact mask construction have deliberately
 * different structural units.  Candidate items cover flattened parent
 * products, while exact items cover short K-list merge diagonals.  Keeping
 * the old shared Q=256 here silently made exact tails eight times coarser
 * than the frozen Super16 contract. */
static constexpr std::uint64_t kCandidateWorkQuantum = 256;
static constexpr std::uint64_t kExactWorkQuantum = 32;
static constexpr int kQueueThreads = 128;
static constexpr int kBuildThreads = 256;
static constexpr unsigned int kMaxBuildBlocks = 65535;
static constexpr std::size_t kCandidateBitmapBudgetBytes =
    std::size_t{128} * 1024 * 1024;
static constexpr std::uint32_t kCandidatePageColumns = 1024;
static constexpr std::uint32_t kCandidatePageWords = 32;
static constexpr std::size_t kCandidatePageHashBudgetBytes =
    std::size_t{128} * 1024 * 1024;
static constexpr std::uint32_t kTileSpaWords = 512;
static constexpr std::uint32_t kTileSpaColumnLimit = kTileSpaWords * 32;

enum class OperandRole : std::uint32_t
{
    A8x4 = 1,
    B4x8 = 2,
};

struct WorkItem
{
    std::uint32_t owner;
    std::uint32_t chunk_ordinal;
};

static_assert(std::is_standard_layout<WorkItem>::value,
              "Super16 WorkItem must have a stable ABI");
static_assert(std::is_trivially_copyable<WorkItem>::value,
              "Super16 WorkItem must be trivially copyable");
static_assert(sizeof(WorkItem) == 8, "Super16 WorkItem must remain 8 bytes");

/* All structural offsets are 64 bit even though the current leaf format is
 * int indexed.  Child leaf IDs remain 32 bit because DmmaDeviceTiles itself
 * cannot contain more than INT_MAX leaves. */
struct DeviceIndexView
{
    OperandRole role = OperandRole::A8x4;
    int rows = 0;
    int cols = 0;
    std::uint32_t parent_row_count = 0;
    std::uint32_t parent_col_count = 0;
    std::uint32_t parent_count = 0;

    const std::uint64_t *row_ptr = nullptr;
    const std::uint32_t *col_idx = nullptr;
    const std::uint64_t *child_ptr = nullptr;
    const std::uint8_t *child_mask = nullptr;
    const std::uint32_t *child_leaf_ids = nullptr;
    /* Sixteen structural Boolean rows for the full 16x16 parent tile. */
    const std::uint16_t *boolean_row_masks = nullptr;

    /* Populated for B4x8 and null for A8x4. */
    const std::uint64_t *col_ptr = nullptr;
    const std::uint32_t *row_idx = nullptr;
    const std::uint32_t *csc_parent_ids = nullptr;
};

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept
        : pointer_(other.pointer_), count_(other.count_), capacity_(other.capacity_)
    {
        other.pointer_ = nullptr;
        other.count_ = 0;
        other.capacity_ = 0;
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this != &other)
        {
            reset();
            pointer_ = other.pointer_;
            count_ = other.count_;
            capacity_ = other.capacity_;
            other.pointer_ = nullptr;
            other.count_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    bool allocate(std::uint64_t count, const char *label)
    {
        reset();
        if (count == 0)
            return true;
        if (count > static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max() / sizeof(T)))
        {
            std::fprintf(stderr, "Super16 allocation overflow in %s.\n",
                         label);
            return false;
        }
        const std::size_t bytes = static_cast<std::size_t>(count) * sizeof(T);
        cudaError_t status =
            cudaMalloc(reinterpret_cast<void **>(&pointer_), bytes);
        if (status != cudaSuccess)
        {
            std::fprintf(stderr, "CUDA error in %s: %s\n", label,
                         cudaGetErrorString(status));
            pointer_ = nullptr;
            return false;
        }
        count_ = count;
        capacity_ = count;
        return true;
    }

    void reset() noexcept
    {
        if (pointer_ != nullptr)
            recycle(pointer_, capacity_);
        pointer_ = nullptr;
        count_ = 0;
        capacity_ = 0;
    }

    T *get() noexcept { return pointer_; }
    const T *get() const noexcept { return pointer_; }
    T *release() noexcept
    {
        T *result = pointer_;
        pointer_ = nullptr;
        count_ = 0;
        capacity_ = 0;
        return result;
    }
    std::uint64_t size() const noexcept { return count_; }
    std::size_t bytes() const noexcept
    {
        return static_cast<std::size_t>(count_) * sizeof(T);
    }

    static T *acquire(std::uint64_t count, const char *label)
    {
        DeviceBuffer buffer;
        return buffer.allocate(count, label) ? buffer.release() : nullptr;
    }

    static void recycle(T *pointer, std::uint64_t capacity) noexcept
    {
        (void)capacity;
        if (pointer != nullptr)
            cudaFree(pointer);
    }

private:
    T *pointer_ = nullptr;
    std::uint64_t count_ = 0;
    std::uint64_t capacity_ = 0;
};

class OwnedDeviceIndex
{
public:
    OwnedDeviceIndex() = default;
    ~OwnedDeviceIndex() = default;
    OwnedDeviceIndex(const OwnedDeviceIndex &) = delete;
    OwnedDeviceIndex &operator=(const OwnedDeviceIndex &) = delete;
    OwnedDeviceIndex(OwnedDeviceIndex &&) noexcept = default;
    OwnedDeviceIndex &operator=(OwnedDeviceIndex &&) noexcept = default;

    void reset() noexcept
    {
        valid_ = false;
        metadata_ = DeviceIndexView{};
        row_ptr_.reset();
        col_idx_.reset();
        child_ptr_.reset();
        child_mask_.reset();
        child_leaf_ids_.reset();
        boolean_row_masks_.reset();
        col_ptr_.reset();
        row_idx_.reset();
        csc_parent_ids_.reset();
    }

    bool valid() const noexcept { return valid_; }

    DeviceIndexView view() const noexcept
    {
        if (!valid_)
            return DeviceIndexView{};
        DeviceIndexView result = metadata_;
        result.row_ptr = row_ptr_.get();
        result.col_idx = col_idx_.get();
        result.child_ptr = child_ptr_.get();
        result.child_mask = child_mask_.get();
        result.child_leaf_ids = child_leaf_ids_.get();
        result.boolean_row_masks = boolean_row_masks_.get();
        result.col_ptr = col_ptr_.get();
        result.row_idx = row_idx_.get();
        result.csc_parent_ids = csc_parent_ids_.get();
        return result;
    }

    std::size_t persistent_bytes() const noexcept
    {
        return row_ptr_.bytes() + col_idx_.bytes() + child_ptr_.bytes() +
               child_mask_.bytes() + child_leaf_ids_.bytes() +
               boolean_row_masks_.bytes() +
               col_ptr_.bytes() + row_idx_.bytes() +
               csc_parent_ids_.bytes();
    }

private:
    friend bool build_device_index(const DmmaDeviceTiles &, OperandRole,
                                   cudaStream_t, OwnedDeviceIndex *);

    DeviceIndexView metadata_{};
    DeviceBuffer<std::uint64_t> row_ptr_;
    DeviceBuffer<std::uint32_t> col_idx_;
    DeviceBuffer<std::uint64_t> child_ptr_;
    DeviceBuffer<std::uint8_t> child_mask_;
    DeviceBuffer<std::uint32_t> child_leaf_ids_;
    DeviceBuffer<std::uint16_t> boolean_row_masks_;
    DeviceBuffer<std::uint64_t> col_ptr_;
    DeviceBuffer<std::uint32_t> row_idx_;
    DeviceBuffer<std::uint32_t> csc_parent_ids_;
    bool valid_ = false;
};

struct WorkMetrics
{
    std::uint64_t owner_count = 0;
    std::uint64_t nonzero_owner_count = 0;
    std::uint64_t item_count = 0;
    std::uint64_t total_structural_work = 0;
    std::uint64_t max_owner_work = 0;
    std::uint64_t p95_owner_work = 0;
    std::uint64_t p99_owner_work = 0;
    std::uint64_t max_chunks_per_owner = 0;
    double mean_owner_work = 0.0;
    double mean_chunks_per_nonzero_owner = 0.0;
};

struct SymbolicMetrics
{
    std::uint64_t parent_a = 0;
    std::uint64_t parent_b = 0;
    std::uint64_t raw_candidate_pairs = 0;
    std::uint64_t parent_c_candidates = 0;
    std::uint64_t final_leaf_c_tiles = 0;
    std::uint64_t nonempty_parent_c = 0;
    std::uint64_t empty_parent_c = 0;
    std::uint64_t step2_parent_invocations = 0;
    std::uint64_t candidate_bitmap_batch_count = 0;
    std::uint64_t candidate_bitmap_words_per_row = 0;
    std::size_t candidate_bitmap_bytes = 0;
    std::uint64_t candidate_touched_pages = 0;
    std::uint64_t candidate_touched_page_upper_bound = 0;
    std::uint64_t candidate_page_hash_capacity = 0;
    std::size_t candidate_page_hash_bytes = 0;
    std::size_t candidate_page_bitmap_bytes = 0;
    bool candidate_page_hash_overflow = false;

    WorkMetrics candidate;
    WorkMetrics exact;

    double work_index_ms = 0.0;
    double item_build_ms = 0.0;
    double candidate_queue_ms = 0.0;
    double sort_unique_ms = 0.0;
    double candidate_keygen_ms = 0.0;
    double candidate_sort_unique_ms = 0.0;
    double candidate_pass1_or_ms = 0.0;
    double candidate_popcount_ms = 0.0;
    double candidate_scan_ms = 0.0;
    double candidate_pass2_or_ms = 0.0;
    double candidate_emit_ms = 0.0;
    double candidate_page_hash_ms = 0.0;
    double candidate_page_compact_sort_ms = 0.0;
    double candidate_page_bitmap_or_ms = 0.0;
    double candidate_page_enumerate_ms = 0.0;
    double exact_queue_ms = 0.0;
    double quadrant_or_ms = 0.0;
    double exact_merge_quadrant_or_ms = 0.0;
    double finalize_ms = 0.0;
    double scan_ms = 0.0;
    double symbolic_ms = 0.0;

    std::size_t sidecar_bytes = 0;
    std::size_t work_item_bytes = 0;
    std::size_t scratch_bytes = 0;
    /* Thrust does not expose its allocator's instantaneous high-water mark.
     * The sort estimate conservatively reserves one additional key array. */
    std::size_t sort_temporary_bytes_estimate = 0;
    std::size_t peak_temporary_bytes = 0;
};

class WorkPlan
{
public:
    WorkPlan() = default;
    WorkPlan(const WorkPlan &) = delete;
    WorkPlan &operator=(const WorkPlan &) = delete;
    WorkPlan(WorkPlan &&) noexcept = default;
    WorkPlan &operator=(WorkPlan &&) noexcept = default;

    void reset() noexcept
    {
        owner_count = 0;
        item_count = 0;
        owner_work.reset();
        owner_item_ptr.reset();
        items.reset();
        ticket.reset();
    }

    std::uint32_t owner_count = 0;
    std::uint64_t item_count = 0;
    DeviceBuffer<std::uint64_t> owner_work;
    DeviceBuffer<std::uint64_t> owner_item_ptr;
    DeviceBuffer<WorkItem> items;
    DeviceBuffer<unsigned long long> ticket;
};

class CandidateState
{
public:
    CandidateState() = default;
    CandidateState(const CandidateState &) = delete;
    CandidateState &operator=(const CandidateState &) = delete;
    CandidateState(CandidateState &&) noexcept = default;
    CandidateState &operator=(CandidateState &&) noexcept = default;

    void reset() noexcept
    {
        raw_pair_count = 0;
        candidate_count = 0;
        a_parent_pair_prefix.reset();
        row_work_ptr.reset();
        plan.reset();
        keys.reset();
        candidate_row_ptr.reset();
    }

    std::uint64_t raw_pair_count = 0;
    std::uint64_t candidate_count = 0;
    DeviceBuffer<std::uint64_t> a_parent_pair_prefix;
    DeviceBuffer<std::uint64_t> row_work_ptr;
    WorkPlan plan;
    /* Exact-size, row-major parent C keys emitted by the bounded SPA path. */
    DeviceBuffer<std::uint64_t> keys;
    DeviceBuffer<std::uint64_t> candidate_row_ptr;
};

class ExactState
{
public:
    ExactState() = default;
    ExactState(const ExactState &) = delete;
    ExactState &operator=(const ExactState &) = delete;
    ExactState(ExactState &&) noexcept = default;
    ExactState &operator=(ExactState &&) noexcept = default;

    void reset() noexcept
    {
        candidate_count = 0;
        plan.reset();
        parent_quadrant_masks.reset();
        matched_k_parents.reset();
    }

    std::uint64_t candidate_count = 0;
    WorkPlan plan;
    /* Four native 8x8 masks per parent C candidate: q00,q01,q10,q11. */
    DeviceBuffer<unsigned long long> parent_quadrant_masks;
    /* Numeric/tail metadata: plan.owner_work is the structural merge-scan
     * count; matched_k_parents is the number of matched parent-K entries. */
    DeviceBuffer<unsigned long long> matched_k_parents;
};

class NativeOutput
{
public:
    NativeOutput() = default;
    NativeOutput(const NativeOutput &) = delete;
    NativeOutput &operator=(const NativeOutput &) = delete;
    NativeOutput(NativeOutput &&) noexcept = default;
    NativeOutput &operator=(NativeOutput &&) noexcept = default;

    void reset() noexcept
    {
        leaf_row_count = 0;
        leaf_col_count = 0;
        tile_count = 0;
        nnz = 0;
        row_ptr64.reset();
        row_ptr.reset();
        rows.reset();
        cols.reset();
        masks.reset();
        nnz_offsets.reset();
        numeric_work.reset();
        parent_task_ids.reset();
        child_local_offsets.reset();
    }

    int leaf_row_count = 0;
    int leaf_col_count = 0;
    int tile_count = 0;
    int nnz = 0;

    /* row_ptr64 is the canonical structural result.  row_ptr is a checked
     * int mirror for the existing numeric/tile2csr ABI. */
    DeviceBuffer<std::uint64_t> row_ptr64;
    DeviceBuffer<int> row_ptr;
    DeviceBuffer<int> rows;
    DeviceBuffer<int> cols;
    DeviceBuffer<std::uint64_t> masks;
    /* Per-tile popcounts before scan, numeric value offsets after scan. */
    DeviceBuffer<int> nnz_offsets;
    /* Dense/bitmask-aware scheduling estimate aligned one-to-one with the
     * final native C tile arrays.  It is metadata only; numeric continues to
     * decode the original hybrid A/B payloads. */
    DeviceBuffer<std::uint64_t> numeric_work;
    /* Four child-task IDs per C16 parent candidate.  Empty quadrants remain
     * -1.  This lets numeric consume the parent work item directly while the
     * legacy C8 arrays are temporarily retained for CSR export. */
    DeviceBuffer<int> parent_task_ids;
    /* Finalize already needs two row-local offsets per parent.  Tail
     * admission reuses this workspace to derive sparse child task IDs. */
    DeviceBuffer<std::uint64_t> child_local_offsets;
};

/* Parent-level symbolic result.  Native 8x8 leaf finalization is deliberately
 * not represented here yet, so callers cannot mistake quadrant masks for a
 * production DmmaOwnedDeviceTiles result. */
class SymbolicOutput
{
public:
    SymbolicOutput() = default;
    SymbolicOutput(const SymbolicOutput &) = delete;
    SymbolicOutput &operator=(const SymbolicOutput &) = delete;
    SymbolicOutput(SymbolicOutput &&) noexcept = default;
    SymbolicOutput &operator=(SymbolicOutput &&) noexcept = default;

    void reset() noexcept
    {
        candidates.reset();
        exact.reset();
        native.reset();
        metrics = SymbolicMetrics{};
        valid = false;
    }

    CandidateState candidates;
    ExactState exact;
    NativeOutput native;
    SymbolicMetrics metrics;
    bool valid = false;
};

namespace detail
{

struct LeafRecord
{
    std::uint64_t parent_key;
    std::uint32_t leaf_id;
    std::uint8_t slot;
    std::uint8_t padding[3];
};

struct ParentCscRecord
{
    std::uint64_t key;
    std::uint32_t parent_id;
    std::uint32_t padding;
};

struct LeafRecordLess
{
    __host__ __device__ bool operator()(const LeafRecord &left,
                                        const LeafRecord &right) const
    {
        return left.parent_key < right.parent_key ||
               (left.parent_key == right.parent_key &&
                (left.slot < right.slot ||
                 (left.slot == right.slot &&
                  left.leaf_id < right.leaf_id)));
    }
};

struct ParentCscRecordLess
{
    __host__ __device__ bool operator()(const ParentCscRecord &left,
                                        const ParentCscRecord &right) const
    {
        return left.key < right.key ||
               (left.key == right.key &&
                left.parent_id < right.parent_id);
    }
};

inline unsigned int build_blocks(std::uint64_t count)
{
    if (count == 0)
        return 0;
    std::uint64_t blocks =
        count / kBuildThreads + (count % kBuildThreads != 0 ? 1 : 0);
    blocks = std::min<std::uint64_t>(blocks, kMaxBuildBlocks);
    return static_cast<unsigned int>(blocks);
}

inline std::uint32_t floor_power_of_two(std::uint64_t value)
{
    if (value == 0)
        return 0;
    std::uint64_t result = 1;
    while (result <= value / 2 && result < (std::uint64_t{1} << 31))
        result <<= 1;
    return static_cast<std::uint32_t>(result);
}

inline std::uint32_t ceil_power_of_two(std::uint64_t value)
{
    if (value <= 1)
        return 1;
    const std::uint32_t floor = floor_power_of_two(value);
    if (floor >= value)
        return floor;
    return floor <= (std::uint32_t{1} << 30) ? floor << 1 : 0;
}

inline bool cuda_ok(cudaError_t status, const char *label)
{
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "CUDA error in %s: %s\n", label,
                 cudaGetErrorString(status));
    return false;
}

class EventInterval
{
public:
    EventInterval() = default;
    ~EventInterval()
    {
        if (begin_ != nullptr)
            cudaEventDestroy(begin_);
        if (end_ != nullptr)
            cudaEventDestroy(end_);
    }
    EventInterval(const EventInterval &) = delete;
    EventInterval &operator=(const EventInterval &) = delete;
    EventInterval(EventInterval &&other) noexcept
        : begin_(other.begin_), end_(other.end_)
    {
        other.begin_ = nullptr;
        other.end_ = nullptr;
    }
    EventInterval &operator=(EventInterval &&other) noexcept
    {
        if (this != &other)
        {
            if (begin_ != nullptr)
                cudaEventDestroy(begin_);
            if (end_ != nullptr)
                cudaEventDestroy(end_);
            begin_ = other.begin_;
            end_ = other.end_;
            other.begin_ = nullptr;
            other.end_ = nullptr;
        }
        return *this;
    }

    bool create()
    {
        return cuda_ok(cudaEventCreate(&begin_),
                       "create Super16 phase begin event") &&
               cuda_ok(cudaEventCreate(&end_),
                       "create Super16 phase end event");
    }
    bool record_begin(cudaStream_t stream)
    {
        return cuda_ok(cudaEventRecord(begin_, stream),
                       "record Super16 phase begin");
    }
    bool record_end(cudaStream_t stream)
    {
        return cuda_ok(cudaEventRecord(end_, stream),
                       "record Super16 phase end");
    }
    bool elapsed(double *milliseconds) const
    {
        float value = 0.0f;
        if (milliseconds == nullptr ||
            !cuda_ok(cudaEventElapsedTime(&value, begin_, end_),
                     "read Super16 phase time"))
            return false;
        *milliseconds = value;
        return true;
    }

private:
    cudaEvent_t begin_ = nullptr;
    cudaEvent_t end_ = nullptr;
};

inline bool sync_and_read_error(cudaStream_t stream, const int *error,
                                const char *label)
{
    int host_error = 0;
    if (!cuda_ok(cudaMemcpyAsync(&host_error, error, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream), label) ||
        !cuda_ok(cudaStreamSynchronize(stream), label))
        return false;
    if (host_error != 0)
    {
        std::fprintf(stderr, "Super16 validation failed in %s (code=%d).\n",
                     label, host_error);
        return false;
    }
    return true;
}

inline bool copy_scalar_to_host(cudaStream_t stream, const std::uint64_t *src,
                                std::uint64_t *dst, const char *label)
{
    return cuda_ok(cudaMemcpyAsync(dst, src, sizeof(*dst),
                                   cudaMemcpyDeviceToHost, stream), label) &&
           cuda_ok(cudaStreamSynchronize(stream), label);
}

/* Direct dual-view builder.  The production leaf CSR is already sorted by
 * leaf column within each row.  A 16x16 parent row is therefore only a
 * 2-way (A8x4) or 4-way (B4x8) merge; a global leaf-record radix sort is
 * unnecessary. */
__global__ void count_parent_columns_from_leaf_csr_kernel(
    DmmaDeviceTiles leaf, std::uint32_t row_group, std::uint32_t col_group,
    std::uint32_t parent_rows, std::uint64_t *row_counts)
{
    for (std::uint32_t parent_row = blockIdx.x * blockDim.x + threadIdx.x;
         parent_row < parent_rows;
         parent_row += static_cast<std::uint32_t>(gridDim.x * blockDim.x))
    {
        int cursor[4] = {};
        int end[4] = {};
#pragma unroll
        for (int child = 0; child < 4; ++child)
        {
            const std::uint32_t leaf_row = parent_row * row_group + child;
            if (child < static_cast<int>(row_group) &&
                leaf_row < static_cast<std::uint32_t>(leaf.tile_row_count))
            {
                cursor[child] = leaf.tile_row_ptr[leaf_row];
                end[child] = leaf.tile_row_ptr[leaf_row + 1];
            }
        }
        std::uint64_t count = 0;
        while (true)
        {
            int next_parent_col = INT_MAX;
#pragma unroll
            for (int child = 0; child < 4; ++child)
                if (child < static_cast<int>(row_group) &&
                    cursor[child] < end[child])
                    next_parent_col = min(
                        next_parent_col,
                        leaf.tile_col_idx[cursor[child]] /
                            static_cast<int>(col_group));
            if (next_parent_col == INT_MAX)
                break;
            ++count;
#pragma unroll
            for (int child = 0; child < 4; ++child)
                if (child < static_cast<int>(row_group))
                    while (cursor[child] < end[child] &&
                           leaf.tile_col_idx[cursor[child]] /
                                   static_cast<int>(col_group) ==
                               next_parent_col)
                        ++cursor[child];
        }
        row_counts[parent_row] = count;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
        row_counts[parent_rows] = 0;
}

__global__ void fill_parent_view_from_leaf_csr_kernel(
    DmmaDeviceTiles leaf, std::uint32_t row_group, std::uint32_t col_group,
    std::uint32_t parent_rows, const std::uint64_t *parent_row_ptr,
    std::uint32_t *parent_col_idx, std::uint64_t *child_ptr,
    std::uint8_t *child_mask, std::uint32_t *child_leaf_ids)
{
    for (std::uint32_t parent_row = blockIdx.x * blockDim.x + threadIdx.x;
         parent_row < parent_rows;
         parent_row += static_cast<std::uint32_t>(gridDim.x * blockDim.x))
    {
        int cursor[4] = {};
        int end[4] = {};
#pragma unroll
        for (int child = 0; child < 4; ++child)
        {
            const std::uint32_t leaf_row = parent_row * row_group + child;
            if (child < static_cast<int>(row_group) &&
                leaf_row < static_cast<std::uint32_t>(leaf.tile_row_count))
            {
                cursor[child] = leaf.tile_row_ptr[leaf_row];
                end[child] = leaf.tile_row_ptr[leaf_row + 1];
            }
        }
        std::uint64_t parent_id = parent_row_ptr[parent_row];
        std::uint64_t child_output =
            leaf.tile_row_ptr[min(parent_row * row_group,
                                  static_cast<std::uint32_t>(
                                      leaf.tile_row_count))];
        while (true)
        {
            int next_parent_col = INT_MAX;
#pragma unroll
            for (int child = 0; child < 4; ++child)
                if (child < static_cast<int>(row_group) &&
                    cursor[child] < end[child])
                    next_parent_col = min(
                        next_parent_col,
                        leaf.tile_col_idx[cursor[child]] /
                            static_cast<int>(col_group));
            if (next_parent_col == INT_MAX)
                break;
            parent_col_idx[parent_id] =
                static_cast<std::uint32_t>(next_parent_col);
            child_ptr[parent_id] = child_output;
            std::uint8_t occupied = 0;
#pragma unroll
            for (int child = 0; child < 4; ++child)
            {
                if (child >= static_cast<int>(row_group))
                    continue;
                while (cursor[child] < end[child] &&
                       leaf.tile_col_idx[cursor[child]] /
                               static_cast<int>(col_group) ==
                           next_parent_col)
                {
                    const int leaf_id = cursor[child]++;
                    const int leaf_col = leaf.tile_col_idx[leaf_id];
                    const int slot = child * static_cast<int>(col_group) +
                                     leaf_col % static_cast<int>(col_group);
                    occupied = static_cast<std::uint8_t>(
                        occupied | static_cast<std::uint8_t>(1u << slot));
                    child_leaf_ids[child_output++] =
                        static_cast<std::uint32_t>(leaf_id);
                }
            }
            child_mask[parent_id] = occupied;
            ++parent_id;
        }
        child_ptr[parent_id] = child_output;
    }
}

__global__ void build_parent_boolean_rows_kernel(
    DmmaDeviceTiles leaf, OperandRole role, std::uint32_t parent_count,
    const std::uint64_t *child_ptr, const std::uint8_t *child_mask,
    const std::uint32_t *child_leaf_ids, std::uint16_t *boolean_rows)
{
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t parent =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         parent < parent_count; parent += stride)
    {
        const std::uint8_t occupied = child_mask[parent];
#pragma unroll
        for (int logical_row = 0; logical_row < 16; ++logical_row)
        {
            std::uint16_t result = 0;
            if (role == OperandRole::A8x4)
            {
                const int child_row = logical_row / 8;
                const int leaf_row = logical_row % 8;
#pragma unroll
                for (int t = 0; t < 4; ++t)
                {
                    const int slot = child_row * 4 + t;
                    const std::uint8_t bit =
                        static_cast<std::uint8_t>(1u << slot);
                    if ((occupied & bit) == 0)
                        continue;
                    const std::uint8_t lower =
                        static_cast<std::uint8_t>(occupied & (bit - 1));
                    const std::uint32_t leaf_id = child_leaf_ids[
                        child_ptr[parent] + __popc(lower)];
                    const std::uint16_t four = static_cast<std::uint16_t>(
                        (leaf.masks[leaf_id] >> (leaf_row * 4)) & 0xfu);
                    result = static_cast<std::uint16_t>(result | (four << (t * 4)));
                }
            }
            else
            {
                const int t = logical_row / 4;
                const int leaf_k = logical_row % 4;
#pragma unroll
                for (int child_col = 0; child_col < 2; ++child_col)
                {
                    const int slot = t * 2 + child_col;
                    const std::uint8_t bit =
                        static_cast<std::uint8_t>(1u << slot);
                    if ((occupied & bit) == 0)
                        continue;
                    const std::uint8_t lower =
                        static_cast<std::uint8_t>(occupied & (bit - 1));
                    const std::uint32_t leaf_id = child_leaf_ids[
                        child_ptr[parent] + __popc(lower)];
                    const std::uint32_t mask = leaf.masks[leaf_id];
#pragma unroll
                    for (int leaf_col = 0; leaf_col < 8; ++leaf_col)
                    {
                        if ((mask & (1u << (leaf_col * 4 + leaf_k))) != 0)
                            result = static_cast<std::uint16_t>(
                                result |
                                (1u << (child_col * 8 + leaf_col)));
                    }
                }
            }
            boolean_rows[parent * 16 + logical_row] = result;
        }
    }
}

__global__ void make_parent_csc_records_kernel(
    std::uint32_t parent_row_count, const std::uint64_t *parent_row_ptr,
    const std::uint32_t *parent_col_idx, ParentCscRecord *records)
{
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t row =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < parent_row_count; row += stride)
    {
        for (std::uint64_t id = parent_row_ptr[row];
             id < parent_row_ptr[row + 1]; ++id)
        {
            ParentCscRecord record{};
            record.key =
                static_cast<std::uint64_t>(parent_col_idx[id]) *
                    parent_row_count +
                row;
            record.parent_id = static_cast<std::uint32_t>(id);
            records[id] = record;
        }
    }
}

__global__ void fill_parent_csc_kernel(
    const ParentCscRecord *records, std::uint32_t parent_count,
    std::uint32_t parent_row_count, std::uint64_t *parent_col_ptr,
    std::uint32_t *parent_row_idx, std::uint32_t *csc_parent_ids,
    int *error)
{
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t position =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         position < parent_count; position += stride)
    {
        const ParentCscRecord record = records[position];
        const std::uint32_t col =
            static_cast<std::uint32_t>(record.key / parent_row_count);
        const std::uint32_t row =
            static_cast<std::uint32_t>(record.key % parent_row_count);
        if (position > 0)
        {
            const ParentCscRecord prior = records[position - 1];
            const std::uint32_t prior_col =
                static_cast<std::uint32_t>(prior.key / parent_row_count);
            const std::uint32_t prior_row =
                static_cast<std::uint32_t>(prior.key % parent_row_count);
            if (prior_col == col && prior_row >= row)
                atomicCAS(error, 0, 6);
        }
        parent_row_idx[position] = row;
        csc_parent_ids[position] = record.parent_id;
        atomicAdd(reinterpret_cast<unsigned long long *>(parent_col_ptr + col + 1),
                  1ull);
    }
}

__global__ void parent_pair_degrees_kernel(
    DeviceIndexView a, DeviceIndexView b, std::uint64_t *pair_prefix,
    int *error)
{
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t a_id =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         a_id < a.parent_count; a_id += stride)
    {
        const std::uint32_t k = a.col_idx[a_id];
        if (k >= b.parent_row_count)
        {
            atomicCAS(error, 0, 20);
            pair_prefix[a_id] = 0;
        }
        else
            pair_prefix[a_id] = b.row_ptr[k + 1] - b.row_ptr[k];
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
        pair_prefix[a.parent_count] = 0;
}

__global__ void map_row_work_kernel(DeviceIndexView a,
                                    const std::uint64_t *pair_prefix,
                                    std::uint64_t *row_work_ptr)
{
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t row =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row <= a.parent_row_count; row += stride)
        row_work_ptr[row] = pair_prefix[a.row_ptr[row]];
}

__global__ void candidate_owner_work_kernel(
    std::uint32_t owner_count, const std::uint64_t *row_work_ptr,
    std::uint64_t *owner_work)
{
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t owner =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         owner < owner_count; owner += stride)
        owner_work[owner] = row_work_ptr[owner + 1] - row_work_ptr[owner];
}

/* TileSpGEMM/FlexSpGEMM step1: one warp owns one 16x16 parent row and
 * accumulates its candidate columns in a 512-word shared-memory SPA. */
__global__ void tile_step1_spa_count_kernel(
    DeviceIndexView a, DeviceIndexView b, std::uint64_t *candidate_row_counts,
    unsigned long long *raw_pair_count)
{
    const unsigned int global_warp =
        (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int local_warp = threadIdx.x >> 5;
    __shared__ std::uint32_t spa[4 * kTileSpaWords];
    if (global_warp >= a.parent_row_count)
        return;
    std::uint32_t *local = spa + local_warp * kTileSpaWords;
    const unsigned int words = (b.parent_col_count + 31u) >> 5;
    for (unsigned int word = lane; word < words; word += 32)
        local[word] = 0;
    __syncwarp();
    unsigned long long row_pairs = 0;
    for (std::uint64_t ai = a.row_ptr[global_warp];
         ai < a.row_ptr[global_warp + 1]; ++ai)
    {
        const std::uint32_t k = a.col_idx[ai];
        if (lane == 0)
            row_pairs += b.row_ptr[k + 1] - b.row_ptr[k];
        for (std::uint64_t bi = b.row_ptr[k] + lane;
             bi < b.row_ptr[k + 1]; bi += 32)
        {
            const std::uint32_t col = b.col_idx[bi];
            atomicOr(local + (col >> 5), 1u << (col & 31u));
        }
    }
    __syncwarp();
    unsigned int count = 0;
    for (unsigned int word = lane; word < words; word += 32)
        count += __popc(local[word]);
    count = __reduce_add_sync(0xffffffffu, count);
    if (lane == 0)
    {
        candidate_row_counts[global_warp] = count;
        atomicAdd(raw_pair_count, row_pairs);
    }
}

__global__ void tile_step1_spa_fill_kernel(
    DeviceIndexView a, DeviceIndexView b,
    const std::uint64_t *candidate_row_ptr, std::uint64_t *candidate_keys)
{
    const unsigned int global_warp =
        (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int local_warp = threadIdx.x >> 5;
    __shared__ std::uint32_t spa[4 * kTileSpaWords];
    if (global_warp >= a.parent_row_count)
        return;
    std::uint32_t *local = spa + local_warp * kTileSpaWords;
    const unsigned int words = (b.parent_col_count + 31u) >> 5;
    const unsigned int rounded_words = (words + 31u) & ~31u;
    for (unsigned int word = lane; word < rounded_words; word += 32)
        local[word] = 0;
    __syncwarp();
    for (std::uint64_t ai = a.row_ptr[global_warp];
         ai < a.row_ptr[global_warp + 1]; ++ai)
    {
        const std::uint32_t k = a.col_idx[ai];
        for (std::uint64_t bi = b.row_ptr[k] + lane;
             bi < b.row_ptr[k + 1]; bi += 32)
        {
            const std::uint32_t col = b.col_idx[bi];
            atomicOr(local + (col >> 5), 1u << (col & 31u));
        }
    }
    __syncwarp();
    std::uint64_t running = 0;
    for (unsigned int word = lane; word < rounded_words; word += 32)
    {
        const std::uint32_t mask = word < words ? local[word] : 0u;
        const unsigned int own = __popc(mask);
        unsigned int inclusive = own;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            const unsigned int prior =
                __shfl_up_sync(0xffffffffu, inclusive, offset);
            if (lane >= static_cast<unsigned int>(offset))
                inclusive += prior;
        }
        const unsigned int batch =
            __shfl_sync(0xffffffffu, inclusive, 31);
        std::uint64_t output = candidate_row_ptr[global_warp] + running +
                               inclusive - own;
        std::uint32_t remaining = mask;
        while (remaining != 0)
        {
            const int bit = __ffs(static_cast<int>(remaining)) - 1;
            const std::uint32_t col = word * 32u + bit;
            candidate_keys[output++] =
                (static_cast<std::uint64_t>(global_warp) << 32) | col;
            remaining &= remaining - 1;
        }
        running += batch;
    }
}

__global__ void make_chunk_counts_kernel(
    std::uint32_t owner_count, const std::uint64_t *owner_work,
    std::uint64_t quantum, std::uint64_t *owner_item_ptr, int *error)
{
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t owner =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         owner < owner_count; owner += stride)
    {
        const std::uint64_t work = owner_work[owner];
        if (quantum == 0)
        {
            atomicCAS(error, 0, 20);
            owner_item_ptr[owner] = 0;
            continue;
        }
        const std::uint64_t chunks =
            work == 0 ? 0 : 1 + (work - 1) / quantum;
        if (chunks > 0xffffffffull)
            atomicCAS(error, 0, 21);
        owner_item_ptr[owner] = chunks;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
        owner_item_ptr[owner_count] = 0;
}

__device__ __forceinline__ std::uint32_t locate_owner(
    const std::uint64_t *owner_item_ptr, std::uint32_t owner_count,
    std::uint64_t item_id)
{
    std::uint32_t low = 0;
    std::uint32_t high = owner_count;
    while (low < high)
    {
        const std::uint32_t middle = low + (high - low) / 2;
        if (owner_item_ptr[middle + 1] <= item_id)
            low = middle + 1;
        else
            high = middle;
    }
    return low;
}

__global__ void fill_work_items_kernel(
    std::uint32_t owner_count, std::uint64_t item_count,
    const std::uint64_t *owner_item_ptr, WorkItem *items, int *error)
{
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t item =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         item < item_count; item += stride)
    {
        const std::uint32_t owner =
            locate_owner(owner_item_ptr, owner_count, item);
        if (owner >= owner_count || owner_item_ptr[owner] > item)
        {
            atomicCAS(error, 0, 22);
            continue;
        }
        const std::uint64_t ordinal = item - owner_item_ptr[owner];
        if (ordinal > 0xffffffffull)
        {
            atomicCAS(error, 0, 23);
            continue;
        }
        items[item] = WorkItem{owner, static_cast<std::uint32_t>(ordinal)};
    }
}

__device__ __forceinline__ std::uint64_t queue_fetch(
    unsigned long long *ticket, int lane)
{
    unsigned long long value = 0;
    if (lane == 0)
        value = atomicAdd(ticket, 1ull);
    return __shfl_sync(0xffffffffu, value, 0);
}

template <typename Group>
__device__ __forceinline__ std::uint64_t queue_fetch_group(
    const Group &group, unsigned long long *ticket)
{
    unsigned long long value = 0;
    if (group.thread_rank() == 0)
        value = atomicAdd(ticket, 1ull);
    return group.shfl(value, 0);
}

template <typename Group>
__device__ __forceinline__ unsigned long long group_or(
    const Group &group, unsigned long long value)
{
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1)
        value |= group.shfl_down(value, offset);
    return value;
}

__device__ __forceinline__ std::uint64_t item_begin(
    const WorkItem &item, std::uint64_t quantum)
{
    return static_cast<std::uint64_t>(item.chunk_ordinal) * quantum;
}

__device__ __forceinline__ std::uint64_t item_end(
    const WorkItem &item, std::uint64_t owner_work, std::uint64_t quantum)
{
    const std::uint64_t begin = item_begin(item, quantum);
    return begin + min(quantum, owner_work - begin);
}

__device__ __forceinline__ std::uint64_t locate_a_parent(
    const std::uint64_t *pair_prefix, std::uint64_t first_a,
    std::uint64_t last_a, std::uint64_t flattened)
{
    std::uint64_t low = first_a;
    std::uint64_t high = last_a;
    while (low < high)
    {
        const std::uint64_t middle = low + (high - low) / 2;
        if (pair_prefix[middle + 1] <= flattened)
            low = middle + 1;
        else
            high = middle;
    }
    return low;
}

__global__ void reference_candidate_key_queue_kernel(
    DeviceIndexView a, DeviceIndexView b,
    const std::uint64_t *pair_prefix, const std::uint64_t *row_work_ptr,
    const WorkItem *items, std::uint64_t item_count,
    unsigned long long *ticket, std::uint64_t *keys, int *error)
{
    const int lane = threadIdx.x & 31;
    while (true)
    {
        const std::uint64_t item_id = queue_fetch(ticket, lane);
        if (item_id >= item_count)
            return;
        const WorkItem item = items[item_id];
        if (item.owner >= a.parent_row_count)
        {
            if (lane == 0)
                atomicCAS(error, 0, 30);
            continue;
        }
        const std::uint64_t owner_work =
            row_work_ptr[item.owner + 1] - row_work_ptr[item.owner];
        const std::uint64_t begin = item_begin(item, kCandidateWorkQuantum);
        if (begin >= owner_work)
        {
            if (lane == 0)
                atomicCAS(error, 0, 31);
            continue;
        }
        const std::uint64_t end =
            item_end(item, owner_work, kCandidateWorkQuantum);
        const std::uint64_t global_base = row_work_ptr[item.owner];
        const std::uint64_t first_a = a.row_ptr[item.owner];
        const std::uint64_t last_a = a.row_ptr[item.owner + 1];
        for (std::uint64_t local = begin + lane; local < end; local += 32)
        {
            const std::uint64_t flattened = global_base + local;
            const std::uint64_t a_id =
                locate_a_parent(pair_prefix, first_a, last_a, flattened);
            if (a_id >= last_a || pair_prefix[a_id] > flattened ||
                pair_prefix[a_id + 1] <= flattened)
            {
                atomicCAS(error, 0, 32);
                continue;
            }
            const std::uint32_t k = a.col_idx[a_id];
            const std::uint64_t b_id =
                b.row_ptr[k] + flattened - pair_prefix[a_id];
            if (b_id >= b.row_ptr[k + 1])
            {
                atomicCAS(error, 0, 33);
                continue;
            }
            keys[flattened] =
                (static_cast<std::uint64_t>(item.owner) << 32) |
                static_cast<std::uint64_t>(b.col_idx[b_id]);
        }
    }
}

/* Production candidate discovery uses one bounded row-batch bitmap path for
 * every matrix.  The flattened-key kernel above is retained only as an
 * independent correctness reference and is never called by build_candidates.
 * Lanes targeting the same bitmap word combine their bits before issuing one
 * atomicOr, which is essential for high-fanout rows. */
__device__ __forceinline__ std::uint64_t mix_hash64(std::uint64_t value)
{
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31;
    return value;
}

__device__ __forceinline__ bool page_hash_insert_key(
    unsigned long long *hash_keys, std::uint32_t capacity,
    std::uint64_t page_key)
{
    const unsigned long long stored = page_key + 1;
    std::uint32_t slot =
        static_cast<std::uint32_t>(mix_hash64(page_key)) & (capacity - 1);
    for (std::uint32_t probe = 0; probe < capacity; ++probe)
    {
        const unsigned long long prior =
            atomicCAS(hash_keys + slot, 0ull, stored);
        if (prior == 0ull || prior == stored)
            return true;
        slot = (slot + 1) & (capacity - 1);
    }
    return false;
}

__device__ __forceinline__ std::uint32_t page_hash_lookup(
    const unsigned long long *hash_keys, const std::uint32_t *hash_values,
    std::uint32_t capacity, std::uint64_t page_key)
{
    const unsigned long long stored = page_key + 1;
    std::uint32_t slot =
        static_cast<std::uint32_t>(mix_hash64(page_key)) & (capacity - 1);
    for (std::uint32_t probe = 0; probe < capacity; ++probe)
    {
        const unsigned long long found = hash_keys[slot];
        if (found == stored)
            return hash_values[slot];
        if (found == 0ull)
            return 0xffffffffu;
        slot = (slot + 1) & (capacity - 1);
    }
    return 0xffffffffu;
}

__global__ void count_candidate_rows_kernel(
    const std::uint64_t *keys, std::uint64_t count,
    std::uint32_t parent_row_count, std::uint64_t *row_ptr, int *error)
{
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t i =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride)
    {
        const std::uint32_t row = static_cast<std::uint32_t>(keys[i] >> 32);
        if (row >= parent_row_count)
            atomicCAS(error, 0, 34);
        else
            atomicAdd(reinterpret_cast<unsigned long long *>(row_ptr + row),
                      1ull);
    }
}

__device__ __forceinline__ std::uint64_t merge_co_rank(
    std::uint64_t diagonal, DeviceIndexView a, DeviceIndexView b,
    std::uint64_t a_begin, std::uint64_t a_size, std::uint64_t b_begin,
    std::uint64_t b_size)
{
    std::uint64_t low = diagonal > b_size ? diagonal - b_size : 0;
    std::uint64_t high = min(diagonal, a_size);
    while (low <= high)
    {
        const std::uint64_t a_count = low + (high - low) / 2;
        const std::uint64_t b_count = diagonal - a_count;
        if (a_count > 0 && b_count < b_size &&
            a.col_idx[a_begin + a_count - 1] >
                b.row_idx[b_begin + b_count])
        {
            high = a_count - 1;
            continue;
        }
        if (b_count > 0 && a_count < a_size &&
            b.row_idx[b_begin + b_count - 1] >=
                a.col_idx[a_begin + a_count])
        {
            low = a_count + 1;
            continue;
        }
        return (a_count << 32) | b_count;
    }
    return ~std::uint64_t{0};
}

__device__ __forceinline__ int child_leaf_for_slot(
    DeviceIndexView index, std::uint32_t parent_id, int slot)
{
    if (parent_id >= index.parent_count || slot < 0 || slot >= 8)
        return -1;
    const std::uint8_t bit = static_cast<std::uint8_t>(1u << slot);
    const std::uint8_t mask = index.child_mask[parent_id];
    if ((mask & bit) == 0)
        return -1;
    const std::uint8_t lower = static_cast<std::uint8_t>(mask & (bit - 1));
    return static_cast<int>(
        index.child_leaf_ids[index.child_ptr[parent_id] + __popc(lower)]);
}

/* Tile/Flex step2 mapping: one 16-thread group owns a complete parent C
 * candidate.  This avoids constructing and globally scheduling merge-path
 * WorkItems for the overwhelmingly short tile-list intersections. */
__device__ __forceinline__ std::uint64_t tile_binary_find(
    const std::uint32_t *values, std::uint64_t begin, std::uint64_t end,
    std::uint32_t target)
{
    std::uint64_t low = begin;
    std::uint64_t high = end;
    while (low < high)
    {
        const std::uint64_t middle = low + (high - low) / 2;
        if (values[middle] < target)
            low = middle + 1;
        else
            high = middle;
    }
    return low < end && values[low] == target ? low : ~std::uint64_t{0};
}

__global__ void tile_step2_exact_kernel(
    DeviceIndexView a, DeviceIndexView b,
    const std::uint64_t *candidate_keys, std::uint32_t candidate_count,
    int tiles_per_group,
    std::uint64_t *intersection_work,
    unsigned long long *quadrant_masks,
    unsigned long long *matched_k_parents, int *error)
{
    namespace cg = cooperative_groups;
    const cg::thread_block block = cg::this_thread_block();
    const cg::thread_block_tile<16> group = cg::tiled_partition<16>(block);
    const int lane = static_cast<int>(group.thread_rank());
    constexpr int kSpeculativeMatches = 32;
    constexpr int kGroupsPerBlock = 8;
    __shared__ int shared_a[kGroupsPerBlock * kSpeculativeMatches];
    __shared__ int shared_b[kGroupsPerBlock * kSpeculativeMatches];
    __shared__ int shared_count[kGroupsPerBlock];
    const int local_group = threadIdx.x / 16;
    int *matched_a = shared_a + local_group * kSpeculativeMatches;
    int *matched_b = shared_b + local_group * kSpeculativeMatches;
    const std::uint64_t first_group =
        (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
        16;
    const std::uint64_t group_stride =
        static_cast<std::uint64_t>(gridDim.x) * (blockDim.x / 16);
    for (std::uint64_t batch = first_group * tiles_per_group;
         batch < candidate_count; batch += group_stride * tiles_per_group)
    {
      for (int tile_in_group = 0; tile_in_group < tiles_per_group;
           ++tile_in_group)
      {
        const std::uint64_t candidate = batch + tile_in_group;
        if (candidate >= candidate_count)
            break;
        const std::uint64_t key = candidate_keys[candidate];
        const std::uint32_t row = static_cast<std::uint32_t>(key >> 32);
        const std::uint32_t col = static_cast<std::uint32_t>(key);
        if (row >= a.parent_row_count || col >= b.parent_col_count)
        {
            if (lane == 0)
                atomicCAS(error, 0, 71);
            continue;
        }
        const std::uint64_t a_begin = a.row_ptr[row];
        const std::uint64_t a_end = a.row_ptr[row + 1];
        const std::uint64_t b_begin = b.col_ptr[col];
        const std::uint64_t b_end = b.col_ptr[col + 1];
        if (lane == 0)
            intersection_work[candidate] =
                (a_end - a_begin) + (b_end - b_begin);
        std::uint16_t local_boolean_row = 0;
        unsigned long long matches = 0;
        if (lane == 0)
            shared_count[local_group] = 0;
        group.sync();
        const std::uint64_t a_size = a_end - a_begin;
        const std::uint64_t b_size = b_end - b_begin;
        const std::uint64_t inner_parent_count = a.parent_col_count;
        const bool a_full = a_size == inner_parent_count;
        const bool b_full = b_size == inner_parent_count;
        /* Preserve Tile/Flex's three dense-intersection fast paths.  The
         * previous port sent these cases through the 32-entry speculative
         * buffer and then replayed a lane-0 serial merge, which dominates
         * hugetric-like matrices.  A full side has implicit K positions, so
         * no search or merge is required. */
        if (a_full || b_full)
        {
            const std::uint64_t match_count = a_full ? b_size : a_size;
            matches = match_count;
            for (std::uint64_t match = 0; match < match_count; ++match)
            {
                std::uint64_t ai;
                std::uint32_t b_parent_id;
                if (a_full)
                {
                    const std::uint64_t bi = b_begin + match;
                    const std::uint32_t k = b.row_idx[bi];
                    ai = a_begin + k;
                    b_parent_id = b.csc_parent_ids[bi];
                }
                else
                {
                    ai = a_begin + match;
                    const std::uint32_t k = a.col_idx[ai];
                    b_parent_id = b.csc_parent_ids[b_begin + k];
                }
                std::uint16_t k_bits =
                    a.boolean_row_masks[ai * 16 + lane];
                std::uint16_t output_row = 0;
                while (k_bits != 0)
                {
                    const int k = __ffs(static_cast<int>(k_bits)) - 1;
                    output_row = static_cast<std::uint16_t>(
                        output_row |
                        b.boolean_row_masks[
                            static_cast<std::uint64_t>(b_parent_id) * 16 + k]);
                    k_bits = static_cast<std::uint16_t>(k_bits & (k_bits - 1));
                }
                local_boolean_row = static_cast<std::uint16_t>(
                    local_boolean_row | output_row);
            }
        }
        else
        {
          const bool use_speculative = a_size > 8 && b_size > 8;
          if (!use_speculative)
          {
              if (lane == 0)
                  shared_count[local_group] = kSpeculativeMatches + 1;
          }
          else if (a_size <= b_size)
          {
            for (std::uint64_t offset = lane; offset < a_size; offset += 16)
            {
                const std::uint64_t ai = a_begin + offset;
                const std::uint64_t bi = tile_binary_find(
                    b.row_idx, b_begin, b_end, a.col_idx[ai]);
                if (bi != ~std::uint64_t{0})
                {
                    const int slot = atomicAdd(shared_count + local_group, 1);
                    if (slot < kSpeculativeMatches)
                    {
                        matched_a[slot] = static_cast<int>(ai);
                        matched_b[slot] =
                            static_cast<int>(b.csc_parent_ids[bi]);
                    }
                }
            }
        }
          else
          {
            for (std::uint64_t offset = lane; offset < b_size; offset += 16)
            {
                const std::uint64_t bi = b_begin + offset;
                const std::uint64_t ai = tile_binary_find(
                    a.col_idx, a_begin, a_end, b.row_idx[bi]);
                if (ai != ~std::uint64_t{0})
                {
                    const int slot = atomicAdd(shared_count + local_group, 1);
                    if (slot < kSpeculativeMatches)
                    {
                        matched_a[slot] = static_cast<int>(ai);
                        matched_b[slot] =
                            static_cast<int>(b.csc_parent_ids[bi]);
                    }
                }
            }
          }
          group.sync();
          const int matched_count = shared_count[local_group];
          matches = matched_count;
          if (matched_count <= kSpeculativeMatches)
          {
            for (int match = 0; match < matched_count; ++match)
            {
                const int a_parent_id = matched_a[match];
                const int b_parent_id = matched_b[match];
                std::uint16_t k_bits =
                    a.boolean_row_masks[
                        static_cast<std::uint64_t>(a_parent_id) * 16 + lane];
                std::uint16_t output_row = 0;
                while (k_bits != 0)
                {
                    const int k = __ffs(static_cast<int>(k_bits)) - 1;
                    output_row = static_cast<std::uint16_t>(
                        output_row |
                        b.boolean_row_masks[
                            static_cast<std::uint64_t>(b_parent_id) * 16 + k]);
                    k_bits =
                        static_cast<std::uint16_t>(k_bits & (k_bits - 1));
                }
                local_boolean_row = static_cast<std::uint16_t>(
                    local_boolean_row | output_row);
            }
          }
          else
          {
            /* Rare high-intersection fallback: replay a stable merge without
             * storing an unbounded speculative list. */
            matches = 0;
            std::uint64_t ai = a_begin;
            std::uint64_t bi = b_begin;
            while (true)
            {
                int a_parent_id = -1;
                int b_parent_id = -1;
                int done = 0;
                if (lane == 0)
                {
                    while (ai < a_end && bi < b_end)
                    {
                        const std::uint32_t ak = a.col_idx[ai];
                        const std::uint32_t bk = b.row_idx[bi];
                        if (ak < bk) ++ai;
                        else if (ak > bk) ++bi;
                        else
                        {
                            a_parent_id = static_cast<int>(ai++);
                            b_parent_id =
                                static_cast<int>(b.csc_parent_ids[bi++]);
                            ++matches;
                            break;
                        }
                    }
                    done = a_parent_id < 0;
                }
                done = group.shfl(done, 0);
                if (done) break;
                a_parent_id = group.shfl(a_parent_id, 0);
                b_parent_id = group.shfl(b_parent_id, 0);
                std::uint16_t k_bits =
                    a.boolean_row_masks[
                        static_cast<std::uint64_t>(a_parent_id) * 16 + lane];
                std::uint16_t output_row = 0;
                while (k_bits != 0)
                {
                    const int k = __ffs(static_cast<int>(k_bits)) - 1;
                    output_row = static_cast<std::uint16_t>(
                        output_row |
                        b.boolean_row_masks[
                            static_cast<std::uint64_t>(b_parent_id) * 16 + k]);
                    k_bits =
                        static_cast<std::uint16_t>(k_bits & (k_bits - 1));
                }
                local_boolean_row = static_cast<std::uint16_t>(
                    local_boolean_row | output_row);
            }
          }
        }
        const int native_row = lane & 7;
        unsigned long long q00 =
            lane < 8 ? static_cast<unsigned long long>(local_boolean_row & 0xffu)
                           << (native_row * 8)
                     : 0ull;
        unsigned long long q01 =
            lane < 8 ? static_cast<unsigned long long>(local_boolean_row >> 8)
                           << (native_row * 8)
                     : 0ull;
        unsigned long long q10 =
            lane >= 8 ? static_cast<unsigned long long>(local_boolean_row & 0xffu)
                            << (native_row * 8)
                      : 0ull;
        unsigned long long q11 =
            lane >= 8 ? static_cast<unsigned long long>(local_boolean_row >> 8)
                            << (native_row * 8)
                      : 0ull;
        q00 = group_or(group, q00);
        q01 = group_or(group, q01);
        q10 = group_or(group, q10);
        q11 = group_or(group, q11);
        if (lane == 0)
        {
            quadrant_masks[candidate * 4] = q00;
            quadrant_masks[candidate * 4 + 1] = q01;
            quadrant_masks[candidate * 4 + 2] = q10;
            quadrant_masks[candidate * 4 + 3] = q11;
            matched_k_parents[candidate] = matches;
        }
      }
    }
}

/* Finalization is deterministic even though exact masks were produced by a
 * global queue.  One lightweight thread walks each parent C row to assign
 * stable local offsets for its two native 8-row children; the subsequent
 * fill is fully parallel over parent candidates. */
__global__ void finalize_row_counts_kernel(
    std::uint32_t parent_row_count, std::uint32_t parent_col_count,
    int leaf_row_count, int leaf_col_count,
    const std::uint64_t *candidate_row_ptr,
    const std::uint64_t *candidate_keys,
    const unsigned long long *quadrant_masks,
    std::uint64_t *candidate_child_local_offsets,
    std::uint64_t *native_row_counts, int *error)
{
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t parent_row =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         parent_row < parent_row_count; parent_row += stride)
    {
        std::uint64_t child_counts[2] = {0, 0};
        std::uint32_t previous_col = 0;
        bool have_previous = false;
        const std::uint64_t begin = candidate_row_ptr[parent_row];
        const std::uint64_t end = candidate_row_ptr[parent_row + 1];
        for (std::uint64_t candidate = begin; candidate < end; ++candidate)
        {
            const std::uint64_t key = candidate_keys[candidate];
            const std::uint32_t key_row =
                static_cast<std::uint32_t>(key >> 32);
            const std::uint32_t key_col = static_cast<std::uint32_t>(key);
            if (key_row != parent_row || key_col >= parent_col_count ||
                (have_previous && key_col <= previous_col))
            {
                atomicCAS(error, 0, 50);
                continue;
            }
            previous_col = key_col;
            have_previous = true;
#pragma unroll
            for (int child_row = 0; child_row < 2; ++child_row)
            {
                candidate_child_local_offsets[candidate * 2 + child_row] =
                    child_counts[child_row];
#pragma unroll
                for (int child_col = 0; child_col < 2; ++child_col)
                {
                    const int quadrant = child_row * 2 + child_col;
                    const unsigned long long mask =
                        quadrant_masks[candidate * 4 + quadrant];
                    if (mask == 0)
                        continue;
                    const std::uint64_t native_row = parent_row * 2 + child_row;
                    const std::uint64_t native_col =
                        static_cast<std::uint64_t>(key_col) * 2 + child_col;
                    if (native_row >= static_cast<std::uint64_t>(leaf_row_count) ||
                        native_col >= static_cast<std::uint64_t>(leaf_col_count))
                    {
                        atomicCAS(error, 0, 51);
                        continue;
                    }
                    ++child_counts[child_row];
                }
            }
        }
#pragma unroll
        for (int child_row = 0; child_row < 2; ++child_row)
        {
            const std::uint64_t native_row = parent_row * 2 + child_row;
            if (native_row < static_cast<std::uint64_t>(leaf_row_count))
                native_row_counts[native_row] = child_counts[child_row];
            else if (child_counts[child_row] != 0)
                atomicCAS(error, 0, 52);
        }
    }
}

/* Parallel stable replacement for finalize_row_counts_kernel.  One block
 * owns one parent row; each 128-candidate chunk is scanned in row order for
 * the top and bottom 8x8 child rows. */
__global__ void tile_finalize_row_counts_kernel(
    std::uint32_t parent_row_count, std::uint32_t parent_col_count,
    int leaf_row_count, int leaf_col_count,
    const std::uint64_t *candidate_row_ptr,
    const std::uint64_t *candidate_keys,
    const unsigned long long *quadrant_masks,
    std::uint64_t *candidate_child_local_offsets,
    std::uint64_t *native_row_counts,
    unsigned long long *nonempty_parent_count, int *error)
{
    const std::uint32_t parent_row = blockIdx.x;
    if (parent_row >= parent_row_count)
        return;
    constexpr int warps = 4;
    __shared__ unsigned int warp_top[warps];
    __shared__ unsigned int warp_bottom[warps];
    __shared__ unsigned long long row_top;
    __shared__ unsigned long long row_bottom;
    if (threadIdx.x == 0)
    {
        row_top = 0;
        row_bottom = 0;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const std::uint64_t begin = candidate_row_ptr[parent_row];
    const std::uint64_t end = candidate_row_ptr[parent_row + 1];
    for (std::uint64_t chunk = begin; chunk < end; chunk += blockDim.x)
    {
        const std::uint64_t candidate = chunk + threadIdx.x;
        unsigned int top = 0;
        unsigned int bottom = 0;
        if (candidate < end)
        {
            const std::uint64_t key = candidate_keys[candidate];
            const std::uint32_t key_row = static_cast<std::uint32_t>(key >> 32);
            const std::uint32_t key_col = static_cast<std::uint32_t>(key);
            if (key_row != parent_row || key_col >= parent_col_count ||
                (candidate > begin && candidate_keys[candidate - 1] >= key))
                atomicCAS(error, 0, 50);
            else
            {
                const std::uint64_t col0 =
                    static_cast<std::uint64_t>(key_col) * 2;
                const bool top_valid =
                    static_cast<std::uint64_t>(parent_row) * 2 <
                    static_cast<std::uint64_t>(leaf_row_count);
                const bool bottom_valid =
                    static_cast<std::uint64_t>(parent_row) * 2 + 1 <
                    static_cast<std::uint64_t>(leaf_row_count);
                top = top_valid && col0 < static_cast<std::uint64_t>(leaf_col_count) &&
                      quadrant_masks[candidate * 4] != 0;
                top += top_valid && col0 + 1 < static_cast<std::uint64_t>(leaf_col_count) &&
                       quadrant_masks[candidate * 4 + 1] != 0;
                bottom = bottom_valid && col0 < static_cast<std::uint64_t>(leaf_col_count) &&
                         quadrant_masks[candidate * 4 + 2] != 0;
                bottom += bottom_valid && col0 + 1 < static_cast<std::uint64_t>(leaf_col_count) &&
                          quadrant_masks[candidate * 4 + 3] != 0;
            }
            if ((top + bottom) != 0)
                atomicAdd(nonempty_parent_count, 1ull);
        }
        unsigned int top_scan = top;
        unsigned int bottom_scan = bottom;
#pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1)
        {
            const unsigned int other_top =
                __shfl_up_sync(0xffffffffu, top_scan, offset);
            const unsigned int other_bottom =
                __shfl_up_sync(0xffffffffu, bottom_scan, offset);
            if (lane >= offset)
            {
                top_scan += other_top;
                bottom_scan += other_bottom;
            }
        }
        if (lane == 31)
        {
            warp_top[warp] = top_scan;
            warp_bottom[warp] = bottom_scan;
        }
        __syncthreads();
        if (warp == 0)
        {
            unsigned int value_top = lane < warps ? warp_top[lane] : 0;
            unsigned int value_bottom = lane < warps ? warp_bottom[lane] : 0;
#pragma unroll
            for (int offset = 1; offset < 32; offset <<= 1)
            {
                const unsigned int other_top =
                    __shfl_up_sync(0xffffffffu, value_top, offset);
                const unsigned int other_bottom =
                    __shfl_up_sync(0xffffffffu, value_bottom, offset);
                if (lane >= offset)
                {
                    value_top += other_top;
                    value_bottom += other_bottom;
                }
            }
            if (lane < warps)
            {
                warp_top[lane] = value_top;
                warp_bottom[lane] = value_bottom;
            }
        }
        __syncthreads();
        if (candidate < end)
        {
            const unsigned int prior_top =
                top_scan - top + (warp == 0 ? 0 : warp_top[warp - 1]);
            const unsigned int prior_bottom =
                bottom_scan - bottom +
                (warp == 0 ? 0 : warp_bottom[warp - 1]);
            candidate_child_local_offsets[candidate * 2] =
                row_top + prior_top;
            candidate_child_local_offsets[candidate * 2 + 1] =
                row_bottom + prior_bottom;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            row_top += warp_top[warps - 1];
            row_bottom += warp_bottom[warps - 1];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        const std::uint64_t top_row = static_cast<std::uint64_t>(parent_row) * 2;
        if (top_row < static_cast<std::uint64_t>(leaf_row_count))
            native_row_counts[top_row] = row_top;
        if (top_row + 1 < static_cast<std::uint64_t>(leaf_row_count))
            native_row_counts[top_row + 1] = row_bottom;
    }
}

__global__ void cast_native_row_ptr_kernel(const std::uint64_t *row_ptr64,
                                           int leaf_row_count,
                                           int *row_ptr, int *error)
{
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t row =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row <= static_cast<std::uint64_t>(leaf_row_count); row += stride)
    {
        const std::uint64_t value = row_ptr64[row];
        if (value > static_cast<std::uint64_t>(INT_MAX))
            atomicCAS(error, 0, 53);
        else
            row_ptr[row] = static_cast<int>(value);
    }
}

__global__ void fill_native_output_kernel(
    std::uint64_t candidate_count, int leaf_row_count, int leaf_col_count,
    const std::uint64_t *candidate_keys,
    const unsigned long long *quadrant_masks,
    const std::uint64_t *candidate_child_local_offsets,
    const std::uint64_t *native_row_ptr, int *output_rows, int *output_cols,
    std::uint64_t *output_masks, int *output_nnz, int *parent_task_ids,
    int *error)
{
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t candidate =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         candidate < candidate_count; candidate += stride)
    {
        const std::uint64_t key = candidate_keys[candidate];
        const std::uint32_t parent_row =
            static_cast<std::uint32_t>(key >> 32);
        const std::uint32_t parent_col = static_cast<std::uint32_t>(key);
#pragma unroll
        for (int child_row = 0; child_row < 2; ++child_row)
        {
            const std::uint64_t native_row =
                static_cast<std::uint64_t>(parent_row) * 2 + child_row;
            std::uint64_t local =
                candidate_child_local_offsets[candidate * 2 + child_row];
#pragma unroll
            for (int child_col = 0; child_col < 2; ++child_col)
            {
                const int quadrant = child_row * 2 + child_col;
                const std::uint64_t mask =
                    quadrant_masks[candidate * 4 + quadrant];
                if (mask == 0)
                    continue;
                const std::uint64_t native_col =
                    static_cast<std::uint64_t>(parent_col) * 2 + child_col;
                if (native_row >= static_cast<std::uint64_t>(leaf_row_count) ||
                    native_col >= static_cast<std::uint64_t>(leaf_col_count))
                {
                    atomicCAS(error, 0, 54);
                    continue;
                }
                const std::uint64_t output = native_row_ptr[native_row] + local;
                if (output >= native_row_ptr[native_row + 1] ||
                    output > static_cast<std::uint64_t>(INT_MAX))
                {
                    atomicCAS(error, 0, 55);
                    continue;
                }
                if (output_rows != nullptr)
                    output_rows[output] = static_cast<int>(native_row);
                if (output_cols != nullptr)
                    output_cols[output] = static_cast<int>(native_col);
                if (output_masks != nullptr)
                    output_masks[output] = mask;
                output_nnz[output] = __popcll(mask);
                if (parent_task_ids != nullptr)
                    parent_task_ids[candidate * 4 + quadrant] =
                        static_cast<int>(output);
                ++local;
            }
        }
    }
}

/* Preserve the exact merge counts already needed by the production tail
 * scheduler.  High 32 bits are comparisons/scans and low 32 bits are
 * matches.  row_ptr only locates the A/B tile lists; neither operand payload
 * is converted away from its dense/bitmask representation. */
__device__ __forceinline__ std::uint32_t diagnostic_bin(
    std::uint32_t coordinate, std::uint32_t extent, std::uint32_t bins)
{
    if (extent == 0) return 0;
    const std::uint64_t scaled =
        static_cast<std::uint64_t>(coordinate) * bins;
    const std::uint32_t result = static_cast<std::uint32_t>(scaled / extent);
    return result < bins ? result : bins - 1;
}

__global__ void diagnostic_leaf_tile_histogram_kernel(
    DmmaDeviceTiles tiles, std::uint32_t parent_rows,
    std::uint32_t parent_cols, std::uint32_t bins,
    unsigned long long *histogram)
{
    for (int leaf_row = blockIdx.x * blockDim.x + threadIdx.x;
         leaf_row < tiles.tile_row_count;
         leaf_row += blockDim.x * gridDim.x)
    {
        const std::uint32_t parent_row =
            static_cast<std::uint32_t>(leaf_row) / 2;
        const std::uint32_t bin_row =
            diagnostic_bin(parent_row, parent_rows, bins);
        for (int tile = tiles.tile_row_ptr[leaf_row];
             tile < tiles.tile_row_ptr[leaf_row + 1]; ++tile)
        {
            const std::uint32_t parent_col =
                static_cast<std::uint32_t>(tiles.tile_col_idx[tile]) / 2;
            const std::uint32_t bin_col =
                diagnostic_bin(parent_col, parent_cols, bins);
            atomicAdd(histogram + static_cast<std::uint64_t>(bin_row) * bins +
                          bin_col,
                      1ull);
        }
    }
}

/* Layout: candidate, nonempty, empty, child-count=1,2,3,4. */
__global__ void diagnostic_candidate_histogram_kernel(
    const std::uint64_t *keys, const unsigned long long *quadrant_masks,
    std::uint64_t candidate_count, std::uint32_t parent_rows,
    std::uint32_t parent_cols, std::uint32_t bins,
    unsigned long long *histograms)
{
    const std::uint64_t cells = static_cast<std::uint64_t>(bins) * bins;
    for (std::uint64_t candidate =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         candidate < candidate_count;
         candidate += static_cast<std::uint64_t>(blockDim.x) * gridDim.x)
    {
        const std::uint64_t key = keys[candidate];
        const std::uint32_t row = static_cast<std::uint32_t>(key >> 32);
        const std::uint32_t col = static_cast<std::uint32_t>(key);
        const std::uint32_t br = diagnostic_bin(row, parent_rows, bins);
        const std::uint32_t bc = diagnostic_bin(col, parent_cols, bins);
        const std::uint64_t cell = static_cast<std::uint64_t>(br) * bins + bc;
        int children = 0;
#pragma unroll
        for (int quadrant = 0; quadrant < 4; ++quadrant)
            children += quadrant_masks[candidate * 4 + quadrant] != 0;
        atomicAdd(histograms + cell, 1ull);
        atomicAdd(histograms + (children == 0 ? 2 : 1) * cells + cell, 1ull);
        if (children > 0)
            atomicAdd(histograms + (2 + children) * cells + cell, 1ull);
    }
}

} // namespace detail

inline bool export_diagnostic_heatmaps(
    const DmmaDeviceTiles &a_leaf, const DmmaDeviceTiles &b_leaf,
    const DeviceIndexView &a, const DeviceIndexView &b,
    const CandidateState &candidates, const ExactState &exact,
    cudaStream_t stream, const char *prefix, int bins)
{
    if (prefix == nullptr || *prefix == '\0' || bins < 16 || bins > 1024 ||
        exact.candidate_count != candidates.candidate_count)
        return false;
    const std::uint64_t cells = static_cast<std::uint64_t>(bins) * bins;
    DeviceBuffer<unsigned long long> a_hist, b_hist, candidate_hists;
    if (!a_hist.allocate(cells, "allocate diagnostic A histogram") ||
        !b_hist.allocate(cells, "allocate diagnostic B histogram") ||
        !candidate_hists.allocate(cells * 7,
                                  "allocate diagnostic candidate histograms") ||
        !detail::cuda_ok(cudaMemsetAsync(a_hist.get(), 0, a_hist.bytes(), stream),
                         "clear diagnostic A histogram") ||
        !detail::cuda_ok(cudaMemsetAsync(b_hist.get(), 0, b_hist.bytes(), stream),
                         "clear diagnostic B histogram") ||
        !detail::cuda_ok(cudaMemsetAsync(candidate_hists.get(), 0,
                                         candidate_hists.bytes(), stream),
                         "clear diagnostic candidate histograms"))
        return false;
    detail::diagnostic_leaf_tile_histogram_kernel
        <<<detail::build_blocks(a_leaf.tile_row_count), kBuildThreads, 0, stream>>>(
            a_leaf, a.parent_row_count, a.parent_col_count,
            static_cast<std::uint32_t>(bins), a_hist.get());
    detail::diagnostic_leaf_tile_histogram_kernel
        <<<detail::build_blocks(b_leaf.tile_row_count), kBuildThreads, 0, stream>>>(
            b_leaf, b.parent_row_count, b.parent_col_count,
            static_cast<std::uint32_t>(bins), b_hist.get());
    detail::diagnostic_candidate_histogram_kernel
        <<<detail::build_blocks(candidates.candidate_count), kBuildThreads, 0,
           stream>>>(candidates.keys.get(), exact.parent_quadrant_masks.get(),
                     candidates.candidate_count, a.parent_row_count,
                     b.parent_col_count, static_cast<std::uint32_t>(bins),
                     candidate_hists.get());
    if (!detail::cuda_ok(cudaGetLastError(),
                         "launch Super16 diagnostic histograms"))
        return false;
    std::vector<unsigned long long> host(cells * 9);
    if (!detail::cuda_ok(cudaMemcpyAsync(host.data(), a_hist.get(),
                                         cells * sizeof(unsigned long long),
                                         cudaMemcpyDeviceToHost, stream),
                         "copy diagnostic A histogram") ||
        !detail::cuda_ok(cudaMemcpyAsync(host.data() + cells, b_hist.get(),
                                         cells * sizeof(unsigned long long),
                                         cudaMemcpyDeviceToHost, stream),
                         "copy diagnostic B histogram") ||
        !detail::cuda_ok(cudaMemcpyAsync(host.data() + cells * 2,
                                         candidate_hists.get(),
                                         cells * 7 * sizeof(unsigned long long),
                                         cudaMemcpyDeviceToHost, stream),
                         "copy diagnostic candidate histograms") ||
        !detail::cuda_ok(cudaStreamSynchronize(stream),
                         "complete Super16 diagnostic histograms"))
        return false;
    const std::string grid_path = std::string(prefix) + "_super16_heatmap.csv";
    FILE *file = std::fopen(grid_path.c_str(), "w");
    if (file == nullptr) return false;
    std::fprintf(file, "bin_row,bin_col,a_tiles,b_tiles,candidates,nonempty,empty,children1,children2,children3,children4\n");
    unsigned long long totals[9] = {};
    for (int row = 0; row < bins; ++row)
        for (int col = 0; col < bins; ++col)
        {
            const std::uint64_t cell = static_cast<std::uint64_t>(row) * bins + col;
            std::fprintf(file, "%d,%d", row, col);
            for (int field = 0; field < 9; ++field)
            {
                const unsigned long long value = host[field * cells + cell];
                totals[field] += value;
                std::fprintf(file, ",%llu", value);
            }
            std::fprintf(file, "\n");
        }
    if (std::fclose(file) != 0) return false;
    const std::string meta_path = std::string(prefix) + "_super16_meta.csv";
    file = std::fopen(meta_path.c_str(), "w");
    if (file == nullptr) return false;
    std::fprintf(file, "key,value\nparent_rows,%u\nparent_cols,%u\nbins,%d\n",
                 a.parent_row_count, b.parent_col_count, bins);
    const char *names[9] = {"a_tiles", "b_tiles", "candidates", "nonempty",
                            "empty", "children1", "children2", "children3",
                            "children4"};
    for (int field = 0; field < 9; ++field)
        std::fprintf(file, "%s,%llu\n", names[field], totals[field]);
    return std::fclose(file) == 0;
}

inline bool build_device_index(const DmmaDeviceTiles &leaf, OperandRole role,
                               cudaStream_t stream, OwnedDeviceIndex *output)
{
    if (output == nullptr)
        return false;
    output->reset();
    if ((role != OperandRole::A8x4 && role != OperandRole::B4x8) ||
        leaf.rows < 0 || leaf.cols < 0 || leaf.tile_row_count < 0 ||
        leaf.tile_col_count < 0 || leaf.num_tiles < 0 ||
        leaf.tile_row_ptr == nullptr ||
        (leaf.num_tiles > 0 &&
         (leaf.tile_col_idx == nullptr || leaf.masks == nullptr ||
          leaf.tile_row_count == 0 ||
          leaf.tile_col_count == 0)) ||
        (role == OperandRole::A8x4 &&
         (leaf.tile_rows != DMMA_TILE_M || leaf.tile_cols != DMMA_TILE_K)) ||
        (role == OperandRole::B4x8 &&
         (leaf.tile_rows != DMMA_TILE_K || leaf.tile_cols != DMMA_TILE_N)))
    {
        std::fprintf(stderr, "Invalid leaf geometry for Super16 sidecar.\n");
        return false;
    }

    const std::uint32_t row_group =
        role == OperandRole::A8x4 ? 2u : 4u;
    const std::uint32_t col_group =
        role == OperandRole::A8x4 ? 4u : 2u;
    const std::uint32_t parent_rows =
        leaf.tile_row_count == 0
            ? 0
            : 1u + (static_cast<std::uint32_t>(leaf.tile_row_count) - 1u) /
                       row_group;
    const std::uint32_t parent_cols =
        leaf.tile_col_count == 0
            ? 0
            : 1u + (static_cast<std::uint32_t>(leaf.tile_col_count) - 1u) /
                       col_group;

    OwnedDeviceIndex result;
    result.metadata_.role = role;
    result.metadata_.rows = leaf.rows;
    result.metadata_.cols = leaf.cols;
    result.metadata_.parent_row_count = parent_rows;
    result.metadata_.parent_col_count = parent_cols;

    DeviceBuffer<int> error;
    if (!error.allocate(1, "allocate Super16 error flag") ||
        !detail::cuda_ok(cudaMemsetAsync(error.get(), 0, sizeof(int), stream),
                         "clear Super16 error flag"))
        return false;
    const std::uint64_t parent_capacity =
        static_cast<std::uint64_t>(leaf.num_tiles);
    /* Allocate the direct builder's upper-bound arena before launching any
     * count/scan work.  cudaMalloc between count and fill is synchronizing
     * and previously serialized the whole stream several times. */
    if (!result.row_ptr_.allocate(static_cast<std::uint64_t>(parent_rows) + 1,
                                  "allocate Super16 parent row pointer") ||
        !result.col_idx_.allocate(parent_capacity,
                                  "allocate direct Super16 parent columns") ||
        !result.child_ptr_.allocate(parent_capacity + 1,
                                    "allocate direct Super16 child pointer") ||
        !result.child_mask_.allocate(parent_capacity,
                                     "allocate direct Super16 child masks") ||
        !result.child_leaf_ids_.allocate(
            static_cast<std::uint64_t>(leaf.num_tiles),
            "allocate direct Super16 child leaf IDs") ||
        !result.boolean_row_masks_.allocate(
            parent_capacity * 16,
            "allocate direct Super16 parent Boolean rows") ||
        !detail::cuda_ok(cudaMemsetAsync(
                             result.row_ptr_.get(), 0,
                             (static_cast<std::size_t>(parent_rows) + 1) *
                                 sizeof(std::uint64_t),
                             stream),
                         "clear Super16 parent row pointer"))
        return false;
    const unsigned int parent_row_blocks =
        detail::build_blocks(static_cast<std::uint64_t>(parent_rows));
    if (parent_row_blocks > 0)
        detail::count_parent_columns_from_leaf_csr_kernel
            <<<parent_row_blocks, kBuildThreads, 0, stream>>>(
                leaf, row_group, col_group, parent_rows,
                result.row_ptr_.get());
    else if (!detail::cuda_ok(cudaMemsetAsync(result.row_ptr_.get(), 0,
                                              sizeof(std::uint64_t), stream),
                              "initialize empty Super16 parent rows"))
        return false;
    try
    {
        auto policy = thrust::cuda::par.on(stream);
        thrust::device_ptr<std::uint64_t> ptr(result.row_ptr_.get());
        thrust::exclusive_scan(policy, ptr, ptr + parent_rows + 1, ptr);
    }
    catch (const std::exception &exception)
    {
        std::fprintf(stderr, "Thrust error scanning Super16 parent rows: %s\n",
                     exception.what());
        cudaGetLastError();
        return false;
    }

    std::uint64_t parent_count64 = 0;
    if (!detail::copy_scalar_to_host(
            stream, result.row_ptr_.get() + parent_rows, &parent_count64,
            "read direct Super16 parent count") ||
        parent_count64 > std::numeric_limits<std::uint32_t>::max())
        return false;
    const std::uint32_t parent_count =
        static_cast<std::uint32_t>(parent_count64);
    result.metadata_.parent_count = parent_count;
    if (parent_row_blocks > 0)
        detail::fill_parent_view_from_leaf_csr_kernel
            <<<parent_row_blocks, kBuildThreads, 0, stream>>>(
                leaf, row_group, col_group, parent_rows,
                result.row_ptr_.get(), result.col_idx_.get(),
                result.child_ptr_.get(), result.child_mask_.get(),
                result.child_leaf_ids_.get());
    else if (!detail::cuda_ok(cudaMemsetAsync(result.child_ptr_.get(), 0,
                                              sizeof(std::uint64_t), stream),
                              "initialize empty Super16 child pointer"))
        return false;
    if (parent_count > 0)
        detail::build_parent_boolean_rows_kernel
            <<<detail::build_blocks(parent_count), kBuildThreads, 0, stream>>>(
                leaf, role, parent_count, result.child_ptr_.get(),
                result.child_mask_.get(), result.child_leaf_ids_.get(),
                result.boolean_row_masks_.get());
    if (!detail::cuda_ok(cudaGetLastError(),
                         "launch direct Super16 parent materialization"))
        return false;

    if (role == OperandRole::B4x8)
    {
        DeviceBuffer<detail::ParentCscRecord> csc_records;
        if (!csc_records.allocate(parent_count,
                                  "allocate Super16 B CSC records") ||
            !result.col_ptr_.allocate(static_cast<std::uint64_t>(parent_cols) + 1,
                                      "allocate Super16 B parent col pointer") ||
            !result.row_idx_.allocate(parent_count,
                                      "allocate Super16 B parent rows") ||
            !result.csc_parent_ids_.allocate(
                parent_count, "allocate Super16 B CSC parent IDs") ||
            !detail::cuda_ok(cudaMemsetAsync(
                                 result.col_ptr_.get(), 0,
                                 (static_cast<std::size_t>(parent_cols) + 1) *
                                     sizeof(std::uint64_t),
                                 stream),
                             "clear Super16 B parent col pointer"))
            return false;
        if (parent_count > 0)
        {
            detail::make_parent_csc_records_kernel
                <<<detail::build_blocks(parent_rows), kBuildThreads, 0, stream>>>(
                    parent_rows, result.row_ptr_.get(), result.col_idx_.get(),
                    csc_records.get());
            try
            {
                auto policy = thrust::cuda::par.on(stream);
                thrust::device_ptr<detail::ParentCscRecord> ptr(
                    csc_records.get());
                thrust::sort(policy, ptr, ptr + parent_count,
                             detail::ParentCscRecordLess{});
            }
            catch (const std::exception &exception)
            {
                std::fprintf(stderr,
                             "Thrust error building Super16 B parent CSC: %s\n",
                             exception.what());
                cudaGetLastError();
                return false;
            }
            detail::fill_parent_csc_kernel
                <<<detail::build_blocks(parent_count), kBuildThreads, 0, stream>>>(
                    csc_records.get(), parent_count, parent_rows,
                    result.col_ptr_.get(), result.row_idx_.get(),
                    result.csc_parent_ids_.get(), error.get());
            if (!detail::cuda_ok(cudaGetLastError(),
                                 "launch Super16 B CSC materialization"))
                return false;
        }
        try
        {
            auto policy = thrust::cuda::par.on(stream);
            thrust::device_ptr<std::uint64_t> ptr(result.col_ptr_.get());
            thrust::inclusive_scan(policy, ptr, ptr + parent_cols + 1, ptr);
        }
        catch (const std::exception &exception)
        {
            std::fprintf(stderr, "Thrust error scanning Super16 B CSC: %s\n",
                         exception.what());
            cudaGetLastError();
            return false;
        }
    }

    if (!detail::sync_and_read_error(stream, error.get(),
                                     "build Super16 sidecar"))
        return false;
    result.valid_ = true;
    *output = std::move(result);
    return true;
}

inline bool finalize_work_plan(cudaStream_t stream, WorkPlan *plan,
                               std::uint64_t quantum, int *error)
{
    if (plan == nullptr || error == nullptr || quantum == 0)
        return false;
    if (!plan->owner_item_ptr.allocate(
            static_cast<std::uint64_t>(plan->owner_count) + 1,
            "allocate Super16 owner item pointer"))
        return false;
    const unsigned int owner_blocks = detail::build_blocks(plan->owner_count);
    if (owner_blocks != 0)
    {
        detail::make_chunk_counts_kernel
            <<<owner_blocks, kBuildThreads, 0, stream>>>(
                plan->owner_count, plan->owner_work.get(),
                quantum, plan->owner_item_ptr.get(), error);
        if (!detail::cuda_ok(cudaGetLastError(),
                             "launch Super16 chunk counting"))
            return false;
    }
    else if (!detail::cuda_ok(cudaMemsetAsync(plan->owner_item_ptr.get(), 0,
                                              sizeof(std::uint64_t), stream),
                              "initialize empty Super16 item pointer"))
        return false;
    if (!detail::sync_and_read_error(stream, error,
                                     "build Super16 chunk counts"))
        return false;

    try
    {
        auto policy = thrust::cuda::par.on(stream);
        thrust::device_ptr<std::uint64_t> ptr(plan->owner_item_ptr.get());
        thrust::exclusive_scan(policy, ptr, ptr + plan->owner_count + 1, ptr);
    }
    catch (const std::exception &exception)
    {
        std::fprintf(stderr, "Thrust error scanning Super16 work items: %s\n",
                     exception.what());
        cudaGetLastError();
        return false;
    }
    if (!detail::copy_scalar_to_host(
            stream, plan->owner_item_ptr.get() + plan->owner_count,
            &plan->item_count, "read Super16 work item count") ||
        !plan->items.allocate(plan->item_count,
                              "allocate Super16 work items") ||
        !plan->ticket.allocate(1, "allocate Super16 global ticket"))
        return false;
    if (plan->item_count != 0)
    {
        detail::fill_work_items_kernel
            <<<detail::build_blocks(plan->item_count), kBuildThreads, 0, stream>>>(
                plan->owner_count, plan->item_count,
                plan->owner_item_ptr.get(), plan->items.get(), error);
        if (!detail::cuda_ok(cudaGetLastError(),
                             "launch Super16 work item fill"))
            return false;
    }
    return detail::sync_and_read_error(stream, error,
                                       "build Super16 work items");
}

inline unsigned int persistent_blocks()
{
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
        properties.multiProcessorCount <= 0)
    {
        cudaGetLastError();
        return 1;
    }
    return static_cast<unsigned int>(properties.multiProcessorCount * 4);
}

/* Compile/reference-only bounded full-row bitmap implementation.  Production
 * build_candidates below uses sparse 1024-column pages so it never clears the
 * full logical row width. */
/* Historical row-batch symbolic reference removed: production uses Tile/Flex SPA. */

inline bool build_candidates_tile_spa(
    const DeviceIndexView &a, const DeviceIndexView &b, cudaStream_t stream,
    CandidateState *output, SymbolicMetrics *metrics = nullptr)
{
    if (output == nullptr || b.parent_col_count > kTileSpaColumnLimit)
        return false;
    output->reset();
    CandidateState result;
    DeviceBuffer<std::uint64_t> raw_pair_count;
    if (!result.candidate_row_ptr.allocate(
            static_cast<std::uint64_t>(a.parent_row_count) + 1,
            "allocate Tile-style candidate row pointer") ||
        !detail::cuda_ok(cudaMemsetAsync(
                             result.candidate_row_ptr.get(), 0,
                             (static_cast<std::size_t>(a.parent_row_count) + 1) *
                                 sizeof(std::uint64_t),
                             stream),
                         "clear Tile-style candidate row pointer") ||
        !raw_pair_count.allocate(1, "allocate Tile-style raw pair count") ||
        !detail::cuda_ok(cudaMemsetAsync(raw_pair_count.get(), 0,
                                         sizeof(unsigned long long), stream),
                         "clear Tile-style raw pair count"))
        return false;
    detail::EventInterval symbolic_event;
    if (metrics != nullptr &&
        (!symbolic_event.create() || !symbolic_event.record_begin(stream)))
        return false;
    constexpr unsigned int threads = 128;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>(
            (static_cast<std::uint64_t>(a.parent_row_count) + 3) / 4,
            kMaxBuildBlocks));
    if (blocks > 0)
        detail::tile_step1_spa_count_kernel<<<blocks, threads, 0, stream>>>(
            a, b, result.candidate_row_ptr.get(),
            reinterpret_cast<unsigned long long *>(raw_pair_count.get()));
    if (!detail::cuda_ok(cudaGetLastError(),
                         "launch Tile-style SPA candidate count"))
        return false;
    try
    {
        auto policy = thrust::cuda::par.on(stream);
        thrust::device_ptr<std::uint64_t> ptr(
            result.candidate_row_ptr.get());
        thrust::exclusive_scan(policy, ptr, ptr + a.parent_row_count + 1, ptr);
    }
    catch (const std::exception &exception)
    {
        std::fprintf(stderr, "Thrust error scanning Tile-style candidates: %s\n",
                     exception.what());
        cudaGetLastError();
        return false;
    }
    std::uint64_t raw_pairs = 0;
    if (!detail::copy_scalar_to_host(
            stream, result.candidate_row_ptr.get() + a.parent_row_count,
            &result.candidate_count, "read Tile-style candidate count") ||
        !detail::copy_scalar_to_host(stream, raw_pair_count.get(), &raw_pairs,
                                     "read Tile-style raw pair count") ||
        !result.keys.allocate(result.candidate_count,
                              "allocate Tile-style candidate keys"))
        return false;
    if (blocks > 0)
        detail::tile_step1_spa_fill_kernel<<<blocks, threads, 0, stream>>>(
            a, b, result.candidate_row_ptr.get(), result.keys.get());
    if (!detail::cuda_ok(cudaGetLastError(),
                         "launch Tile-style SPA candidate fill") ||
        (metrics != nullptr && !symbolic_event.record_end(stream)) ||
        !detail::cuda_ok(cudaStreamSynchronize(stream),
                         "complete Tile-style SPA candidates"))
        return false;
    if (metrics != nullptr)
    {
        double elapsed = 0.0;
        if (!symbolic_event.elapsed(&elapsed))
            return false;
        metrics->parent_a = a.parent_count;
        metrics->parent_b = b.parent_count;
        metrics->raw_candidate_pairs = raw_pairs;
        metrics->parent_c_candidates = result.candidate_count;
        metrics->candidate.owner_count = a.parent_row_count;
        metrics->candidate.total_structural_work = raw_pairs;
        metrics->candidate.mean_owner_work =
            a.parent_row_count == 0
                ? 0.0
                : static_cast<double>(raw_pairs) / a.parent_row_count;
        metrics->candidate_queue_ms = elapsed;
        metrics->candidate_keygen_ms = elapsed;
        metrics->scratch_bytes += result.keys.bytes() +
                                  result.candidate_row_ptr.bytes();
        metrics->peak_temporary_bytes =
            std::max(metrics->peak_temporary_bytes, metrics->scratch_bytes);
    }
    *output = std::move(result);
    return true;
}

/* Wide-column counterpart of FlexSpGEMM's row-hash/bin path.  It
 * materializes the structural products once, radix-sorts their packed
 * (row,col) keys and uniques them.  Unlike the retired touched-page hash its
 * capacity is derived exactly from structural work and cannot fill up. */
inline bool build_candidates_wide_sort(
    const DeviceIndexView &a, const DeviceIndexView &b, cudaStream_t stream,
    CandidateState *output, SymbolicMetrics *metrics = nullptr)
{
    const auto begin = std::chrono::steady_clock::now();
    if (output == nullptr)
        return false;
    output->reset();
    CandidateState result;
    DeviceBuffer<int> error;
    if (!error.allocate(1, "allocate wide candidate error") ||
        !detail::cuda_ok(cudaMemsetAsync(error.get(), 0, sizeof(int), stream),
                         "clear wide candidate error") ||
        !result.a_parent_pair_prefix.allocate(
            static_cast<std::uint64_t>(a.parent_count) + 1,
            "allocate wide pair prefix") ||
        !result.row_work_ptr.allocate(
            static_cast<std::uint64_t>(a.parent_row_count) + 1,
            "allocate wide row-work pointer"))
        return false;
    if (a.parent_count > 0)
        detail::parent_pair_degrees_kernel
            <<<detail::build_blocks(a.parent_count), kBuildThreads, 0, stream>>>(
                a, b, result.a_parent_pair_prefix.get(), error.get());
    else
        cudaMemsetAsync(result.a_parent_pair_prefix.get(), 0,
                        sizeof(std::uint64_t), stream);
    try
    {
        auto policy = thrust::cuda::par.on(stream);
        thrust::device_ptr<std::uint64_t> prefix(
            result.a_parent_pair_prefix.get());
        thrust::exclusive_scan(policy, prefix, prefix + a.parent_count + 1,
                               prefix);
    }
    catch (const std::exception &exception)
    {
        std::fprintf(stderr, "Thrust error scanning wide pairs: %s\n",
                     exception.what());
        cudaGetLastError();
        return false;
    }
    if (!detail::copy_scalar_to_host(
            stream, result.a_parent_pair_prefix.get() + a.parent_count,
            &result.raw_pair_count, "read wide raw pair count"))
        return false;
    detail::map_row_work_kernel
        <<<detail::build_blocks(static_cast<std::uint64_t>(a.parent_row_count) + 1),
           kBuildThreads, 0, stream>>>(
            a, result.a_parent_pair_prefix.get(), result.row_work_ptr.get());
    result.plan.owner_count = a.parent_row_count;
    if (!result.plan.owner_work.allocate(a.parent_row_count,
                                         "allocate wide owner work"))
        return false;
    if (a.parent_row_count > 0)
        detail::candidate_owner_work_kernel
            <<<detail::build_blocks(a.parent_row_count), kBuildThreads, 0,
               stream>>>(a.parent_row_count, result.row_work_ptr.get(),
                          result.plan.owner_work.get());
    if (!finalize_work_plan(stream, &result.plan, kCandidateWorkQuantum,
                            error.get()) ||
        !result.keys.allocate(result.raw_pair_count,
                              "allocate wide raw candidate keys") ||
        !detail::cuda_ok(cudaMemsetAsync(result.plan.ticket.get(), 0,
                                         sizeof(unsigned long long), stream),
                         "reset wide candidate ticket"))
        return false;
    if (result.plan.item_count > 0)
        detail::reference_candidate_key_queue_kernel
            <<<persistent_blocks(), kQueueThreads, 0, stream>>>(
                a, b, result.a_parent_pair_prefix.get(),
                result.row_work_ptr.get(), result.plan.items.get(),
                result.plan.item_count, result.plan.ticket.get(),
                result.keys.get(), error.get());
    if (!detail::sync_and_read_error(stream, error.get(),
                                     "generate wide candidate keys"))
        return false;
    try
    {
        auto policy = thrust::cuda::par.on(stream);
        thrust::device_ptr<std::uint64_t> keys(result.keys.get());
        thrust::sort(policy, keys, keys + result.raw_pair_count);
        const auto unique_end =
            thrust::unique(policy, keys, keys + result.raw_pair_count);
        result.candidate_count =
            static_cast<std::uint64_t>(unique_end - keys);
    }
    catch (const std::exception &exception)
    {
        std::fprintf(stderr, "Thrust error sorting wide candidates: %s\n",
                     exception.what());
        cudaGetLastError();
        return false;
    }
    if (!result.candidate_row_ptr.allocate(
            static_cast<std::uint64_t>(a.parent_row_count) + 1,
            "allocate wide candidate row pointer") ||
        !detail::cuda_ok(cudaMemsetAsync(
                             result.candidate_row_ptr.get(), 0,
                             (static_cast<std::size_t>(a.parent_row_count) + 1) *
                                 sizeof(std::uint64_t),
                             stream),
                         "clear wide candidate row pointer"))
        return false;
    if (result.candidate_count > 0)
        detail::count_candidate_rows_kernel
            <<<detail::build_blocks(result.candidate_count), kBuildThreads, 0,
               stream>>>(result.keys.get(), result.candidate_count,
                          a.parent_row_count,
                          result.candidate_row_ptr.get(), error.get());
    try
    {
        auto policy = thrust::cuda::par.on(stream);
        thrust::device_ptr<std::uint64_t> ptr(
            result.candidate_row_ptr.get());
        thrust::exclusive_scan(policy, ptr, ptr + a.parent_row_count + 1, ptr);
    }
    catch (const std::exception &exception)
    {
        std::fprintf(stderr, "Thrust error scanning wide candidate rows: %s\n",
                     exception.what());
        cudaGetLastError();
        return false;
    }
    if (!detail::sync_and_read_error(stream, error.get(),
                                     "complete wide candidates"))
        return false;
    if (metrics != nullptr)
    {
        const double elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - begin)
                                   .count();
        metrics->parent_a = a.parent_count;
        metrics->parent_b = b.parent_count;
        metrics->raw_candidate_pairs = result.raw_pair_count;
        metrics->parent_c_candidates = result.candidate_count;
        metrics->candidate.owner_count = a.parent_row_count;
        metrics->candidate.item_count = result.plan.item_count;
        metrics->candidate.total_structural_work = result.raw_pair_count;
        metrics->candidate_queue_ms = elapsed;
        metrics->candidate_sort_unique_ms = elapsed;
        metrics->scratch_bytes += result.keys.bytes() +
                                  result.candidate_row_ptr.bytes();
        metrics->peak_temporary_bytes =
            std::max(metrics->peak_temporary_bytes, metrics->scratch_bytes);
    }
    *output = std::move(result);
    return true;
}

/* The sole production candidate path: hash only unique (row, 1024-column
 * page) keys, materialize one 128-byte bitmap per touched page, then emit
 * sorted parent C keys.  Structural pairs are traversed twice but never
 * materialized, and logical empty width is never scanned. */
inline bool build_candidates(const DeviceIndexView &a,
                             const DeviceIndexView &b, cudaStream_t stream,
                             CandidateState *output,
                             SymbolicMetrics *metrics = nullptr)
{
    /* Match TileSpGEMM/FlexSpGEMM step1 for matrices covered by their
     * shared-memory SPA.  Keep the measured wide-sort baseline for wider
     * inputs until the row-binned wide kernel is ready: the row-batch bitmap
     * prototype repeatedly scanned the 16K-column batches and regressed
     * great-britain_osm by more than an order of magnitude. */
    if (b.parent_col_count <= kTileSpaColumnLimit)
        return build_candidates_tile_spa(a, b, stream, output, metrics);
    return build_candidates_wide_sort(a, b, stream, output, metrics);
}

inline bool build_exact_masks(const DmmaDeviceTiles &a_leaf,
                              const DmmaDeviceTiles &b_leaf,
                              const DeviceIndexView &a,
                              const DeviceIndexView &b,
                              const CandidateState &candidates,
                              cudaStream_t stream, ExactState *output,
                              SymbolicMetrics *metrics = nullptr)
{
    const auto function_begin = std::chrono::steady_clock::now();
    if (output == nullptr)
        return false;
    output->reset();
    if (a.role != OperandRole::A8x4 || b.role != OperandRole::B4x8 ||
        b.col_ptr == nullptr || a.boolean_row_masks == nullptr ||
        b.boolean_row_masks == nullptr || candidates.candidate_count >
                                    std::numeric_limits<std::uint32_t>::max() ||
        (candidates.candidate_count > 0 && candidates.keys.get() == nullptr))
    {
        std::fprintf(stderr, "Invalid Super16 exact-mask operands.\n");
        return false;
    }

    ExactState result;
    DeviceBuffer<int> error;
    result.candidate_count = candidates.candidate_count;
    result.plan.owner_count =
        static_cast<std::uint32_t>(candidates.candidate_count);
    if (!error.allocate(1, "allocate Super16 exact error") ||
        !detail::cuda_ok(cudaMemsetAsync(error.get(), 0, sizeof(int), stream),
                         "clear Super16 exact error") ||
        !result.plan.owner_work.allocate(
            candidates.candidate_count,
            "allocate Tile-style intersection-work metadata") ||
        !result.parent_quadrant_masks.allocate(
            candidates.candidate_count * 4,
            "allocate Super16 exact quadrant masks") ||
        !result.matched_k_parents.allocate(
            candidates.candidate_count,
            "allocate Super16 exact matched-K metadata"))
        return false;
    const auto queue_begin = std::chrono::steady_clock::now();
    if (candidates.candidate_count > 0)
    {
        constexpr unsigned int threads = 128;
        const int tiles_per_group =
            candidates.candidate_count >= 1000000 ? 8 : 1;
        const std::uint64_t candidates_per_block =
            static_cast<std::uint64_t>(8 * tiles_per_group);
        const std::uint64_t groups =
            (candidates.candidate_count + candidates_per_block - 1) /
            candidates_per_block;
        const unsigned int blocks = static_cast<unsigned int>(
            std::min<std::uint64_t>(groups, kMaxBuildBlocks));
        detail::tile_step2_exact_kernel<<<blocks, threads, 0, stream>>>(
                a, b, candidates.keys.get(), result.plan.owner_count,
                tiles_per_group,
                result.plan.owner_work.get(),
                result.parent_quadrant_masks.get(),
                result.matched_k_parents.get(), error.get());
        if (!detail::cuda_ok(cudaGetLastError(),
                             "launch Tile-style exact-mask kernel"))
            return false;
    }
    if (!detail::sync_and_read_error(stream, error.get(),
                                     "Super16 exact-mask generation"))
        return false;
    const auto queue_end = std::chrono::steady_clock::now();

    if (metrics != nullptr)
    {
        metrics->exact.owner_count = result.plan.owner_count;
        metrics->exact.item_count = result.plan.owner_count;
        metrics->item_build_ms +=
            std::chrono::duration<double, std::milli>(queue_begin -
                                                       function_begin)
                .count();
        metrics->exact_queue_ms +=
            std::chrono::duration<double, std::milli>(queue_end - queue_begin)
                .count();
        metrics->exact_merge_quadrant_or_ms = metrics->exact_queue_ms;
        const std::size_t exact_temporary_bytes =
            result.plan.owner_work.bytes() +
            result.parent_quadrant_masks.bytes() +
            result.matched_k_parents.bytes();
        metrics->scratch_bytes += exact_temporary_bytes;
        const std::size_t candidate_live_bytes =
            candidates.a_parent_pair_prefix.bytes() +
            candidates.row_work_ptr.bytes() +
            candidates.plan.owner_work.bytes() +
            candidates.plan.owner_item_ptr.bytes() +
            candidates.plan.items.bytes() + candidates.plan.ticket.bytes() +
            candidates.keys.bytes() + candidates.candidate_row_ptr.bytes();
        metrics->peak_temporary_bytes =
            std::max(metrics->peak_temporary_bytes,
                     candidate_live_bytes + exact_temporary_bytes);
    }
    *output = std::move(result);
    return true;
}

inline bool finalize_native_output(
    const DmmaDeviceTiles &a_leaf, const DmmaDeviceTiles &b_leaf,
    const DeviceIndexView &a, const DeviceIndexView &b,
    const CandidateState &candidates, const ExactState &exact,
    cudaStream_t stream, bool collect_numeric_work,
    bool materialize_child_graph, NativeOutput *output,
    SymbolicMetrics *metrics = nullptr)
{
    if (output == nullptr)
        return false;
    output->reset();
    if (a_leaf.tile_row_count < 0 || b_leaf.tile_col_count < 0 ||
        a.parent_row_count !=
            static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(a_leaf.tile_row_count) + 1) / 2) ||
        b.parent_col_count !=
            static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(b_leaf.tile_col_count) + 1) / 2) ||
        exact.candidate_count != candidates.candidate_count ||
        (candidates.candidate_count > 0 &&
         (candidates.keys.get() == nullptr ||
          exact.parent_quadrant_masks.get() == nullptr)) ||
        candidates.candidate_row_ptr.get() == nullptr)
    {
        std::fprintf(stderr, "Invalid Super16 native-finalize operands.\n");
        return false;
    }

    const auto begin_time = std::chrono::steady_clock::now();
    NativeOutput result;
    DeviceBuffer<int> error;
    DeviceBuffer<unsigned long long> nonempty_parent_count;
    result.leaf_row_count = a_leaf.tile_row_count;
    result.leaf_col_count = b_leaf.tile_col_count;
    if (!error.allocate(1, "allocate Super16 finalize error") ||
        !nonempty_parent_count.allocate(
            1, "allocate Super16 parent audit counter") ||
        !detail::cuda_ok(cudaMemsetAsync(
                             nonempty_parent_count.get(), 0,
                             sizeof(unsigned long long), stream),
                         "clear Super16 parent audit counter") ||
        !detail::cuda_ok(cudaMemsetAsync(error.get(), 0, sizeof(int), stream),
                         "clear Super16 finalize error") ||
        !result.child_local_offsets.allocate(
            candidates.candidate_count * 2,
            "allocate Super16 child local offsets") ||
        !result.row_ptr64.allocate(
            static_cast<std::uint64_t>(result.leaf_row_count) + 1,
            "allocate Super16 native 64-bit row pointer") ||
        !result.row_ptr.allocate(
            static_cast<std::uint64_t>(result.leaf_row_count) + 1,
            "allocate Super16 native row pointer") ||
        !detail::cuda_ok(cudaMemsetAsync(
                             result.row_ptr64.get(), 0,
                             (static_cast<std::size_t>(result.leaf_row_count) +
                              1) * sizeof(std::uint64_t),
                             stream),
                         "clear Super16 native row counts"))
        return false;

    if (a.parent_row_count > 0)
    {
        detail::tile_finalize_row_counts_kernel
            <<<a.parent_row_count, 128, 0, stream>>>(
                a.parent_row_count, b.parent_col_count,
                result.leaf_row_count, result.leaf_col_count,
                candidates.candidate_row_ptr.get(), candidates.keys.get(),
                exact.parent_quadrant_masks.get(), result.child_local_offsets.get(),
                result.row_ptr64.get(), nonempty_parent_count.get(),
                error.get());
        if (!detail::cuda_ok(cudaGetLastError(),
                             "launch Super16 native row counting"))
            return false;
    }
    if (!detail::sync_and_read_error(stream, error.get(),
                                     "count Super16 native rows"))
        return false;
    unsigned long long nonempty_parents = 0;
    if (!detail::cuda_ok(cudaMemcpyAsync(
                             &nonempty_parents,
                             nonempty_parent_count.get(),
                             sizeof(nonempty_parents),
                             cudaMemcpyDeviceToHost, stream),
                         "read Super16 nonempty parent count") ||
        !detail::cuda_ok(cudaStreamSynchronize(stream),
                         "complete Super16 parent audit"))
        return false;
    try
    {
        auto policy = thrust::cuda::par.on(stream);
        thrust::device_ptr<std::uint64_t> ptr(result.row_ptr64.get());
        thrust::exclusive_scan(policy, ptr,
                               ptr + result.leaf_row_count + 1, ptr);
    }
    catch (const std::exception &exception)
    {
        std::fprintf(stderr,
                     "Thrust error scanning Super16 native rows: %s\n",
                     exception.what());
        cudaGetLastError();
        return false;
    }
    std::uint64_t tile_count = 0;
    if (!detail::copy_scalar_to_host(
            stream, result.row_ptr64.get() + result.leaf_row_count,
            &tile_count, "read Super16 native tile count"))
        return false;
    if (tile_count > static_cast<std::uint64_t>(INT_MAX))
    {
        std::fprintf(stderr,
                     "Super16 native tile count exceeds the legacy int ABI.\n");
        return false;
    }
    result.tile_count = static_cast<int>(tile_count);
    if ((materialize_child_graph &&
         (!result.rows.allocate(tile_count,
                                "allocate Super16 native output rows") ||
          !result.cols.allocate(tile_count,
                                "allocate Super16 native output columns") ||
          !result.masks.allocate(tile_count,
                                 "allocate Super16 native output masks"))) ||
        !result.nnz_offsets.allocate(
            tile_count + 1, "allocate Super16 native nnz offsets") ||
        (!materialize_child_graph &&
         (!result.parent_task_ids.allocate(
              candidates.candidate_count * 4,
              "allocate Super16 parent child-task map") ||
          !detail::cuda_ok(cudaMemsetAsync(
                               result.parent_task_ids.get(), 0xff,
                               static_cast<std::size_t>(
                                   candidates.candidate_count * 4) *
                                   sizeof(int),
                               stream),
                           "clear Super16 parent child-task map"))) ||
        !detail::cuda_ok(cudaMemsetAsync(
                             result.nnz_offsets.get(), 0,
                             (static_cast<std::size_t>(tile_count) + 1) *
                                 sizeof(int),
                             stream),
                         "clear Super16 native nnz counts"))
        return false;

    detail::cast_native_row_ptr_kernel
        <<<detail::build_blocks(
               static_cast<std::uint64_t>(result.leaf_row_count) + 1),
           kBuildThreads, 0, stream>>>(
            result.row_ptr64.get(), result.leaf_row_count,
            result.row_ptr.get(), error.get());
    if (candidates.candidate_count > 0)
    {
        detail::fill_native_output_kernel
            <<<detail::build_blocks(candidates.candidate_count), kBuildThreads,
               0, stream>>>(
                candidates.candidate_count, result.leaf_row_count,
                result.leaf_col_count, candidates.keys.get(),
                exact.parent_quadrant_masks.get(), result.child_local_offsets.get(),
                result.row_ptr64.get(), result.rows.get(), result.cols.get(),
                result.masks.get(), result.nnz_offsets.get(),
                result.parent_task_ids.get(), error.get());
    }
    if (!detail::cuda_ok(cudaGetLastError(),
                         "launch Super16 native output fill") ||
        !detail::sync_and_read_error(stream, error.get(),
                                     "fill Super16 native output"))
        return false;

    /* Tail admission consumes the step2 parent work/matches and refines only
     * potential heavy parents.  Never rebuild an output-sized numeric_work
     * array here. */

    unsigned long long nnz_wide = 0;
    if (tile_count > 0)
    {
        try
        {
            auto policy = thrust::cuda::par.on(stream);
            thrust::device_ptr<int> ptr(result.nnz_offsets.get());
            nnz_wide = thrust::reduce(
                policy, ptr, ptr + tile_count, 0ull,
                thrust::plus<unsigned long long>());
        }
        catch (const std::exception &exception)
        {
            std::fprintf(stderr,
                         "Thrust error reducing Super16 native nnz: %s\n",
                         exception.what());
            cudaGetLastError();
            return false;
        }
    }
    if (nnz_wide > static_cast<unsigned long long>(INT_MAX))
    {
        std::fprintf(stderr,
                     "Super16 native nnz exceeds the legacy int ABI.\n");
        return false;
    }
    result.nnz = static_cast<int>(nnz_wide);
    try
    {
        auto policy = thrust::cuda::par.on(stream);
        thrust::device_ptr<int> ptr(result.nnz_offsets.get());
        thrust::exclusive_scan(policy, ptr, ptr + tile_count + 1, ptr);
    }
    catch (const std::exception &exception)
    {
        std::fprintf(stderr,
                     "Thrust error scanning Super16 native nnz: %s\n",
                     exception.what());
        cudaGetLastError();
        return false;
    }
    if (!detail::cuda_ok(cudaStreamSynchronize(stream),
                         "complete Super16 native finalize"))
        return false;

    if (metrics != nullptr)
    {
        metrics->final_leaf_c_tiles = tile_count;
        metrics->nonempty_parent_c = nonempty_parents;
        metrics->empty_parent_c =
            candidates.candidate_count - nonempty_parents;
        metrics->step2_parent_invocations = candidates.candidate_count;
        metrics->scratch_bytes += result.child_local_offsets.bytes() +
                                  result.numeric_work.bytes();
        metrics->finalize_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin_time)
                .count();
    }
    *output = std::move(result);
    return true;
}

/* Execute the one allowed parent-symbolic path: candidate queue followed by
 * exact merge-path queue.  Failure leaves output invalid and never falls back
 * to the legacy symbolic implementation. */
inline bool run_parent_symbolic(const DmmaDeviceTiles &a_leaf,
                                const DmmaDeviceTiles &b_leaf,
                                const DeviceIndexView &a,
                                const DeviceIndexView &b,
                                cudaStream_t stream, SymbolicOutput *output,
                                bool collect_numeric_work = false,
                                bool materialize_child_graph = true)
{
    const auto begin_time = std::chrono::steady_clock::now();
    if (output == nullptr)
        return false;
    output->reset();
    SymbolicOutput result;
    if (!build_candidates(a, b, stream, &result.candidates,
                          &result.metrics) ||
        !build_exact_masks(a_leaf, b_leaf, a, b, result.candidates, stream,
                           &result.exact, &result.metrics))
        return false;
    if (const char *prefix = std::getenv("RTT_SUPER16_DIAG_PREFIX"))
    {
        int bins = 256;
        if (const char *text = std::getenv("RTT_SUPER16_DIAG_BINS"))
            bins = std::atoi(text);
        if (!export_diagnostic_heatmaps(
                a_leaf, b_leaf, a, b, result.candidates, result.exact,
                stream, prefix, bins))
        {
            std::fprintf(stderr, "Unable to export Super16 diagnostic heatmaps.\n");
            return false;
        }
    }
    if (!finalize_native_output(a_leaf, b_leaf, a, b, result.candidates,
                                result.exact, stream, collect_numeric_work,
                                materialize_child_graph,
                                &result.native,
                                &result.metrics))
        return false;
    result.metrics.symbolic_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin_time)
            .count();
    result.valid = true;
    *output = std::move(result);
    return true;
}

} // namespace super16
} // namespace rtt

#endif // RTT_SPGEMM_SUPER16_SYMBOLIC_CUH_
