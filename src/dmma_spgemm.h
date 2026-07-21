#ifndef RTT_SPGEMM_DMMA_SPGEMM_H_
#define RTT_SPGEMM_DMMA_SPGEMM_H_

#include "common.h"
#include "dmma_low_fill_exact_tile.h"
#include "dmma_tile_early_split.h"
#include "dmma_tiles.h"

#include <cuda_runtime.h>
#include <mma.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

#include <climits>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <type_traits>
#include <utility>
#include <vector>

static constexpr int DMMA_WARPS_PER_BLOCK = 4;
static constexpr int DMMA_THREADS_PER_BLOCK = DMMA_WARPS_PER_BLOCK * WARP_SIZE;
static constexpr int DMMA_SPA_WORDS_PER_WARP = 512;
static constexpr int DMMA_SPA_MAX_TILE_COLUMNS =
    DMMA_SPA_WORDS_PER_WARP * 32;
static constexpr std::size_t DMMA_WIDE_BITSET_SCRATCH_BYTES =
    std::size_t(128) * 1024 * 1024;
/* The ordinary forward-exact prototype shares the same hard scratch budget
 * as the existing wide/oversized path.  The two arenas are live in disjoint
 * phases; no experiment may raise this alias above 128 MiB. */
static constexpr std::size_t DMMA_EXACT_FORWARD_SPA_SCRATCH_BYTES =
    DMMA_WIDE_BITSET_SCRATCH_BYTES;
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
    const uint32_t *row_tile_nnz_sum;
    const uint32_t *col_tile_nnz_sum;
    unsigned long long structural_nnz;
    bool row_tile_nnz_sum_valid;
    bool col_tile_nnz_sum_valid;
    bool low_fill_metadata_overflow;
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
    uint32_t *row_tile_nnz_sum = nullptr;
    uint32_t *col_tile_nnz_sum = nullptr;
};

/* super16_symbolic.cuh consumes the production device-tile view above. */
#define RTT_SPGEMM_DMMA_DEVICE_TILES_H_
#include "super16_symbolic.cuh"

struct DmmaSuper16IndexCache
{
    const int *a_row_ptr = nullptr;
    const int *a_col_idx = nullptr;
    const int *b_row_ptr = nullptr;
    const int *b_col_idx = nullptr;
    int device = -1;
    rtt::super16::OwnedDeviceIndex a_index;
    rtt::super16::OwnedDeviceIndex b_index;

    bool prepare(const DmmaDeviceTiles &a, const DmmaDeviceTiles &b,
                 cudaStream_t stream)
    {
        int current = -1;
        if (cudaGetDevice(&current) != cudaSuccess)
            return false;
        const bool same = current == device && a_row_ptr == a.tile_row_ptr &&
                          a_col_idx == a.tile_col_idx &&
                          b_row_ptr == b.tile_row_ptr &&
                          b_col_idx == b.tile_col_idx && a_index.valid() &&
                          b_index.valid();
        if (same)
            return true;
        a_index.reset();
        b_index.reset();
        if (!rtt::super16::build_device_index(
                a, rtt::super16::OperandRole::A8x4, stream, &a_index) ||
            !rtt::super16::build_device_index(
                b, rtt::super16::OperandRole::B4x8, stream, &b_index))
            return false;
        device = current;
        a_row_ptr = a.tile_row_ptr;
        a_col_idx = a.tile_col_idx;
        b_row_ptr = b.tile_row_ptr;
        b_col_idx = b.tile_col_idx;
        return true;
    }
};

enum DmmaNumericScheduleMode
{
    DMMA_SCHEDULE_DIRECT = 0,
    DMMA_SCHEDULE_SPLIT_CTA = 1,
    DMMA_SCHEDULE_SPLIT_PERSISTENT = 2,
    /* P15: one ordinary CUDA grid contains both fixed-work heavy chunks and
     * unsplit regular C tiles.  CUDA's native CTA scheduler, rather than a
     * second stream, a persistent software queue, or an SM-ID mapping, hands
     * pending work to SMs as resident CTAs complete. */
    DMMA_SCHEDULE_SPLIT_FLAT = 3,
    /* Default-off production-shaped tail queue.  The complete ordinary
     * population keeps the four-warps-per-CTA tile kernel and skips only a
     * sparse exact-symbolic heavy bitset.  After that bulk grid drains, all
     * SMs are available to a bounded persistent queue of K-intersection
     * chunks.  This deliberately does not reuse the row-worker kernel. */
    DMMA_SCHEDULE_TILE_TAIL_QUEUE = 4,
    /* Default-off early-distribution variant.  Heavy exact-symbolic parents
     * are removed from the unchanged four-warp light grid, split before
     * numeric, and exposed through a globally capped equal-priority queue at
     * the same numeric start.  CUDA chooses CTA placement; there is no
     * block-to-SM affinity contract. */
    DMMA_SCHEDULE_TILE_EARLY_SPLIT = 5
};

/* Direct symbolic work is identical for all layouts.  TILE_DYNAMIC retains
 * the ordinary one-C-tile-per-warp grid whose queued CTAs are dynamically
 * placed by CUDA.  ROW_STATIC is a deliberately weak scheduling baseline:
 * one persistent logical worker CTA owns one contiguous equal-row-count
 * block, with no queue, stealing, or cost model.  ROW_DYNAMIC is its pure
 * load-balancing ablation: the launch shape and per-row work are unchanged,
 * but each worker claims its next C-tile row from one global atomic head. */
enum DmmaDirectNumericLayout
{
    DMMA_DIRECT_NUMERIC_TILE_DYNAMIC = 0,
    DMMA_DIRECT_NUMERIC_ROW_STATIC = 1,
    DMMA_DIRECT_NUMERIC_ROW_DYNAMIC = 2
};

static inline const char *dmma_direct_numeric_layout_name(
    DmmaDirectNumericLayout layout)
{
    switch (layout)
    {
    case DMMA_DIRECT_NUMERIC_ROW_STATIC:
        return "row-static-block";
    case DMMA_DIRECT_NUMERIC_ROW_DYNAMIC:
        return "row-dynamic";
    default:
        return "tile-dynamic";
    }
}

enum DmmaPartialReductionMode
{
    DMMA_REDUCTION_ATOMIC = 0,
    DMMA_REDUCTION_WORKSPACE = 1
};

/* The default keeps the proven full light grid byte-for-byte selectable.
 * PERSISTENT_SUFFIX is an explicit critical-window experiment: output tasks
 * before q_begin use a static bulk grid, while regular tasks in the exact
 * output suffix are dequeued by one-warp persistent CTAs. */
enum DmmaLightPolicy
{
    DMMA_LIGHT_STATIC = 0,
    DMMA_LIGHT_PERSISTENT_SUFFIX = 1,
    /* P13: one four-warp persistent kernel owns every non-heavy task.  A
     * bounded symbolic fine-ID queue is drained after the coarse page queue;
     * no second light CUDA stream is involved. */
    DMMA_LIGHT_PERSISTENT_UNIFIED = 2
};

static inline const char *dmma_light_policy_name(DmmaLightPolicy policy)
{
    switch (policy)
    {
    case DMMA_LIGHT_PERSISTENT_SUFFIX:
        return "persistent-suffix";
    case DMMA_LIGHT_PERSISTENT_UNIFIED:
        return "persistent-unified";
    default:
        return "static";
    }
}

enum DmmaUnifiedFallbackReason
{
    DMMA_UNIFIED_FALLBACK_NONE = 0,
    DMMA_UNIFIED_FALLBACK_SYMBOLIC_UNAVAILABLE = 1,
    DMMA_UNIFIED_FALLBACK_FINE_OVERFLOW = 2,
    DMMA_UNIFIED_FALLBACK_SATURATED_LOAD = 3,
    DMMA_UNIFIED_FALLBACK_ZERO_LOAD = 4,
    DMMA_UNIFIED_FALLBACK_TAIL_METADATA = 5
};

static inline const char *dmma_unified_fallback_reason_name(
    DmmaUnifiedFallbackReason reason)
{
    switch (reason)
    {
    case DMMA_UNIFIED_FALLBACK_SYMBOLIC_UNAVAILABLE:
        return "symbolic-unavailable";
    case DMMA_UNIFIED_FALLBACK_FINE_OVERFLOW:
        return "fine-overflow";
    case DMMA_UNIFIED_FALLBACK_SATURATED_LOAD:
        return "saturated-load";
    case DMMA_UNIFIED_FALLBACK_ZERO_LOAD:
        return "zero-load";
    case DMMA_UNIFIED_FALLBACK_TAIL_METADATA:
        return "tail-metadata";
    default:
        return "none";
    }
}

/* Zero suffix workers selects an automatic resource split.  TASKS retains
 * the P11 task-count heuristic, WORK uses the exact symbolic suffix share,
 * and REGULAR_WORK removes parents executed by the independent chunk stream
 * from both sides of the ratio. */
enum DmmaSuffixAutoBasis
{
    DMMA_SUFFIX_AUTO_TASK = 0,
    DMMA_SUFFIX_AUTO_WORK = 1,
    DMMA_SUFFIX_AUTO_REGULAR_WORK = 2
};

static inline const char *dmma_suffix_auto_basis_name(
    DmmaSuffixAutoBasis basis)
{
    switch (basis)
    {
    case DMMA_SUFFIX_AUTO_TASK:
        return "task";
    case DMMA_SUFFIX_AUTO_WORK:
        return "work";
    default:
        return "regular-work";
    }
}

/* Host-side ordering and CUDA stream priorities are independent ablation
 * knobs from the CTA/persistent work distribution.  HEAVY_PRIORITY is the
 * production default: enqueue the long-tail path first and give it the
 * device's greatest available stream priority. */
enum DmmaSplitLaunchPolicy
{
    DMMA_SPLIT_LAUNCH_LIGHT_FIRST = 0,
    DMMA_SPLIT_LAUNCH_HEAVY_FIRST = 1,
    DMMA_SPLIT_LAUNCH_HEAVY_PRIORITY = 2
};

/* The production default keeps admission as two side-channel kernels so its
 * exact-mask launch remains directly comparable with the direct schedule.
 * FUSED_EXACT is an explicit experiment: the exact-mask warp also computes
 * merge counts and appends sparse heavy records, eliminating both the dense
 * upper-filter pass and the maybe-ID replay. */
enum DmmaSymbolicAdmissionMode
{
    DMMA_SYMBOLIC_ADMISSION_SEPARATE = 0,
    DMMA_SYMBOLIC_ADMISSION_FUSED_EXACT = 1
};

static inline const char *dmma_symbolic_admission_name(
    DmmaSymbolicAdmissionMode mode)
{
    return mode == DMMA_SYMBOLIC_ADMISSION_FUSED_EXACT ? "fused-exact" :
                                                         "separate";
}

/* The row-hybrid exact prototype is deliberately independent of numeric
 * scheduling.  Its default is disabled, and every non-enabled reason is a
 * correctness-preserving fallback to the original ordinary exact kernel. */
enum DmmaExactForwardSpaReason
{
    DMMA_EXACT_FORWARD_NOT_REQUESTED = 0,
    DMMA_EXACT_FORWARD_ENABLED = 1,
    DMMA_EXACT_FORWARD_ZERO_SELECTED = 2,
    DMMA_EXACT_FORWARD_CAPACITY = 3,
    DMMA_EXACT_FORWARD_ALLOCATION = 4,
    DMMA_EXACT_FORWARD_ESTIMATE_OVERFLOW = 5,
    DMMA_EXACT_FORWARD_OVERSIZED_EXISTING = 6
};

static inline const char *dmma_exact_forward_spa_reason_name(
    DmmaExactForwardSpaReason reason)
{
    switch (reason)
    {
    case DMMA_EXACT_FORWARD_ENABLED:
        return "enabled";
    case DMMA_EXACT_FORWARD_ZERO_SELECTED:
        return "zero-selected";
    case DMMA_EXACT_FORWARD_CAPACITY:
        return "capacity";
    case DMMA_EXACT_FORWARD_ALLOCATION:
        return "allocation";
    case DMMA_EXACT_FORWARD_ESTIMATE_OVERFLOW:
        return "estimate-overflow";
    case DMMA_EXACT_FORWARD_OVERSIZED_EXISTING:
        return "oversized-existing";
    default:
        return "not-requested";
    }
}

static __host__ __device__ __forceinline__ unsigned long long
dmma_exact_forward_saturating_add(unsigned long long left,
                                  unsigned long long right,
                                  bool *overflow)
{
    if (ULLONG_MAX - left < right)
    {
        if (overflow != nullptr)
            *overflow = true;
        return ULLONG_MAX;
    }
    return left + right;
}

static __host__ __device__ __forceinline__ unsigned long long
dmma_exact_forward_saturating_mul(unsigned long long left,
                                  unsigned long long right,
                                  bool *overflow)
{
    if (left != 0 && right > ULLONG_MAX / left)
    {
        if (overflow != nullptr)
            *overflow = true;
        return ULLONG_MAX;
    }
    return left * right;
}

struct DmmaExactForwardRowEstimate
{
    unsigned long long forward_pairs = 0;
    unsigned long long estimated_candidates = 0;
    unsigned long long reverse_work = 0;
    unsigned long long forward_work = 0;
    bool overflow = false;
    bool selected = false;
};

/* This gate reads only A/B structure available before candidate discovery:
 * A row degree, sum_k degree(B[k,:]), B tile shape/counts and frozen scalar
 * thresholds.  It never reads candidate keep flags, exact masks or C nnz. */
static __host__ __device__ __forceinline__ DmmaExactForwardRowEstimate
dmma_exact_forward_row_estimate(
    int a_degree, unsigned long long forward_pairs, int b_tile_columns,
    int b_tile_count, unsigned long long minimum_forward_pairs,
    double minimum_reverse_over_forward)
{
    DmmaExactForwardRowEstimate result;
    result.forward_pairs = forward_pairs;
    if (a_degree <= 0 || b_tile_columns <= 0 || b_tile_count < 0 ||
        forward_pairs == 0 || minimum_reverse_over_forward < 1.0)
        return result;

    const unsigned long long columns =
        static_cast<unsigned long long>(b_tile_columns);
    result.estimated_candidates =
        forward_pairs < columns ? forward_pairs : columns;
    const unsigned long long average_b_column_degree =
        (static_cast<unsigned long long>(b_tile_count) + columns - 1) /
        columns;
    bool overflow = false;
    const unsigned long long merge_span = dmma_exact_forward_saturating_add(
        static_cast<unsigned long long>(a_degree),
        average_b_column_degree, &overflow);
    result.reverse_work = dmma_exact_forward_saturating_mul(
        result.estimated_candidates, merge_span, &overflow);
    result.forward_work = dmma_exact_forward_saturating_add(
        forward_pairs, columns, &overflow);
    result.forward_work = dmma_exact_forward_saturating_add(
        result.forward_work, result.estimated_candidates, &overflow);
    result.overflow = overflow;
    result.selected =
        !overflow && forward_pairs >= minimum_forward_pairs &&
        static_cast<double>(result.reverse_work) >=
            minimum_reverse_over_forward *
                static_cast<double>(result.forward_work);
    return result;
}

static inline std::size_t dmma_exact_forward_spa_row_bytes(
    int b_tile_columns)
{
    if (b_tile_columns <= 0 ||
        static_cast<std::size_t>(b_tile_columns) >
            SIZE_MAX / sizeof(unsigned long long))
        return 0;
    return static_cast<std::size_t>(b_tile_columns) *
           sizeof(unsigned long long);
}

static inline int dmma_exact_forward_spa_batch_capacity(
    int b_tile_columns, int selected_rows,
    std::size_t scratch_limit = DMMA_EXACT_FORWARD_SPA_SCRATCH_BYTES)
{
    const std::size_t row_bytes =
        dmma_exact_forward_spa_row_bytes(b_tile_columns);
    if (row_bytes == 0 || row_bytes > scratch_limit || selected_rows <= 0)
        return 0;
    const std::size_t capacity = scratch_limit / row_bytes;
    const std::size_t bounded =
        capacity < static_cast<std::size_t>(selected_rows)
            ? capacity
            : static_cast<std::size_t>(selected_rows);
    return bounded > static_cast<std::size_t>(INT_MAX)
               ? INT_MAX
               : static_cast<int>(bounded);
}

/* Host-only proof oracle for tests/debug tooling.  Candidate rows are
 * contiguous by construction; row flags therefore define an exact disjoint
 * partition without inspecting any exact-symbolic result. */
static inline bool dmma_exact_forward_partition_contract(
    int row_count, const int *candidate_row_ptr, const int *forward_flags,
    int candidate_count, unsigned long long *ordinary_candidates,
    unsigned long long *forward_candidates)
{
    if (row_count < 0 || candidate_count < 0 ||
        candidate_row_ptr == nullptr || forward_flags == nullptr ||
        ordinary_candidates == nullptr || forward_candidates == nullptr ||
        candidate_row_ptr[0] != 0 ||
        candidate_row_ptr[row_count] != candidate_count)
        return false;
    unsigned long long ordinary = 0;
    unsigned long long forward = 0;
    for (int row = 0; row < row_count; ++row)
    {
        const int begin = candidate_row_ptr[row];
        const int end = candidate_row_ptr[row + 1];
        if (begin < 0 || end < begin || end > candidate_count ||
            (forward_flags[row] != 0 && forward_flags[row] != 1))
            return false;
        if (forward_flags[row] != 0)
            forward += static_cast<unsigned long long>(end - begin);
        else
            ordinary += static_cast<unsigned long long>(end - begin);
    }
    *ordinary_candidates = ordinary;
    *forward_candidates = forward;
    return ordinary + forward ==
           static_cast<unsigned long long>(candidate_count);
}

/* Why a requested split schedule ultimately did (or did not) enter the
 * sparse-tail numeric path.  Only DMMA_TAIL_GATE_ENABLED launches the light
 * and heavy streams; every other non-NOT_REQUESTED state is a correctness-
 * preserving fallback to the ordinary direct numeric kernel. */
enum DmmaTailGateReason
{
    DMMA_TAIL_GATE_NOT_REQUESTED = 0,
    DMMA_TAIL_GATE_ENABLED = 1,
    DMMA_TAIL_GATE_ZERO_HEAVY = 2,
    DMMA_TAIL_GATE_RECORD_OVERFLOW = 3,
    DMMA_TAIL_GATE_HEAVY_FRACTION = 4,
    DMMA_TAIL_GATE_REUSE_DISABLED = 5,
    DMMA_TAIL_GATE_OVERSIZED_SYMBOLIC = 6,
    DMMA_TAIL_GATE_MAYBE_OVERFLOW = 7
};

static inline const char *dmma_tail_gate_reason_name(DmmaTailGateReason reason)
{
    switch (reason)
    {
    case DMMA_TAIL_GATE_ENABLED:
        return "enabled";
    case DMMA_TAIL_GATE_ZERO_HEAVY:
        return "zero-heavy";
    case DMMA_TAIL_GATE_RECORD_OVERFLOW:
        return "record-overflow";
    case DMMA_TAIL_GATE_HEAVY_FRACTION:
        return "heavy-fraction";
    case DMMA_TAIL_GATE_REUSE_DISABLED:
        return "reuse-disabled";
    case DMMA_TAIL_GATE_OVERSIZED_SYMBOLIC:
        return "oversized-symbolic";
    case DMMA_TAIL_GATE_MAYBE_OVERFLOW:
        return "maybe-overflow";
    default:
        return "not-requested";
    }
}

/* Per-C-tile workload model.  The coefficients are expressed in an arbitrary
 * positive work unit (normally fitted nanoseconds); only ratios and the
 * threshold need to share the same unit. */
struct DmmaTaskCostModel
{
    double intercept = 0.0;
    double scan = 1.0;
    double match = 8.0;
    double output = 0.25;
};

/* CUDA execution infrastructure may be prepared once by a benchmark or
 * library handle and then borrowed by sequential Core calls.  The concrete
 * definition follows the public schedule/stat structures below. */
struct DmmaSplitAsyncState;

struct DmmaNumericScheduleConfig
{
    bool tileflex16_symbolic = false;
    /* Cost-balanced is a direct-numeric scheduling wrapper.  It reorders
     * independent final C tasks by symbolic work and feeds the unchanged
     * hybrid-payload Tensor Core primitive from a persistent warp queue. */
    bool cost_balanced = false;
    int cost_workers_per_sm = 4;
    DmmaNumericScheduleMode mode = DMMA_SCHEDULE_DIRECT;
    DmmaDirectNumericLayout direct_numeric_layout =
        DMMA_DIRECT_NUMERIC_TILE_DYNAMIC;
    /* Row-dynamic reserves this many consecutive C tile rows per global
     * fetch-add.  One is the frozen pure-LB behavior; two/four are explicit
     * queue/locality ablations and are invalid for every other layout. */
    int row_queue_batch = 1;
    /* Default-off pre-numeric layout gate.  When requested, one bounded
     * reduction over the already materialized exact C row pointer estimates
     * the load of the existing contiguous row-static workers.  It selects
     * one of the two existing row-worker kernel modes; it never changes the
     * CTA shape or the work performed within a claimed C tile row. */
    bool row_dynamic_auto = false;
    double row_dynamic_threshold = 1.10;
    DmmaPartialReductionMode reduction = DMMA_REDUCTION_ATOMIC;
    DmmaLightPolicy light_policy = DMMA_LIGHT_STATIC;
    DmmaSplitLaunchPolicy launch_policy =
        DMMA_SPLIT_LAUNCH_HEAVY_PRIORITY;
    DmmaTaskCostModel cost{};
    /* Admission answers whether a task should be split; chunk_target answers
     * how much modeled work each chunk should carry.  Zero preserves the
     * historical coupled behavior by resolving to split_threshold. */
    double split_threshold = 256.0;
    double chunk_target = 0.0;
    int max_chunks = 8;
    /* The flat grid maps one virtual work item to each warp.  Two warps is
     * the production default: unlike a one-warp CTA it can reach 64 resident
     * warps on devices with a 32-block/SM architectural limit, while bounding
     * the within-CTA convoy to one peer warp.  One and four remain explicit
     * ablation choices. */
    int flat_warps_per_cta = 2;
    /* q is defined on final compact output IDs: q_i=i/(N-1).  Zero workers
     * selects an occupancy-derived architecture-local default. */
    double critical_q_min = 0.75;
    int suffix_workers_per_sm = 0;
    DmmaSuffixAutoBasis suffix_auto_basis =
        DMMA_SUFFIX_AUTO_REGULAR_WORK;
    int suffix_queue_batch = 16;
    int suffix_fine_tasks_per_worker = 8;
    /* Unified-light uses an independent coarse page size: 256 avoids millions
     * of global page claims on large task sets while remaining sweepable for
     * the page-convoy tradeoff.  suffix_fine_tasks_per_worker is reused only
     * as a fixed terminal safety window.  A zero fine threshold resolves to
     * half the split-admission threshold. */
    int unified_page_size = 256;
    /* Zero preserves the occupancy-derived P13 policy.  A positive value is
     * an engineering control for the light/heavy resident-CTA tradeoff and
     * is clamped to the kernel's architecture-local occupancy ceiling. */
    int unified_workers_per_sm = 0;
    double unified_fine_threshold = 0.0;
    int unified_fine_capacity = 1 << 20;
    std::size_t workspace_limit_bytes = std::size_t(1024) * 1024 * 1024;
    bool collect_task_stats = false;
    /* Optional ordinary-candidate exact-symbolic row hybrid.  The default
     * leaves dmma_exact_mask_kernel byte-for-byte selectable.  When enabled,
     * a pre-candidate A/B structural gate assigns complete C tile rows either
     * to that ordinary merge or to a bounded forward mask SPA. */
    bool exact_forward_spa = false;
    unsigned long long exact_forward_min_row_pairs = 65536ull;
    double exact_forward_min_ratio = 2.0;
    /* ExactTile-Sparse v1 is a calibration-only, default-off direct path.
     * q is the sole threshold and must be one of {4,8,12,16}.  The first
     * milestone rejects every split/row/tail combination before allocation. */
    bool low_fill_exact_tile = false;
    int low_fill_q = 0;
    /* At most this many exceptional records are appended by exact symbolic.
     * The allocation is additionally capped by candidate_count, so the 1M
     * default bounds the sparse side channel at 16 MiB. */
    int tail_record_capacity = 1 << 20;
    /* Stage-one upper-bound candidates have their own bounded ID channel.
     * Overflow is a safe direct fallback and never truncates scheduling. */
    int maybe_candidate_capacity = 1 << 20;
    /* Sparse tail fusion is profitable only when exceptional tasks are rare.
     * A larger fraction falls back to the uniform direct kernel unless the
     * explicit force_tail_split ablation knob is enabled. */
    double max_heavy_fraction = 0.01;
    bool force_tail_split = false;
    /* Optional process-level execution context.  Streams, timing events and
     * the cached priority range are infrastructure, not matrix-dependent
     * workspace.  A null pointer preserves the self-contained compatibility
     * path, which creates and destroys a local context only if split numeric
     * is actually admitted.  Borrowed contexts must not be used concurrently
     * by multiple Core calls. */
    DmmaSplitAsyncState *split_context = nullptr;
    /* Split schedules normally filter candidates with a safe FP32 upper bound
     * and replay the exact merge only for the bounded maybe-ID population.
     * Turning this off is an ablation/debug path that falls back to direct;
     * it never replays a dense scheduler-side merge. */
    bool reuse_symbolic_task_counts = true;
    DmmaSymbolicAdmissionMode symbolic_admission =
        DMMA_SYMBOLIC_ADMISSION_SEPARATE;
    /* Full per-output workload telemetry.  Persistent suffix auto-W enables
     * it for work-based policies; explicit collection enables it for
     * diagnostics.  Production unified-light instead uses the bounded joint
     * maybe/replay channel above and leaves this false.  When enabled, exact
     * symbolic stores nnz in candidate_nnz[6:0] and saturated quantized task
     * cost in candidate_nnz[30:7], then compact consumes both. */
    bool collect_symbolic_load = false;
    double symbolic_load_quantum = 64.0;
    int symbolic_wave_ctas_per_sm = 4;
    int symbolic_critical_waves = 1;
    /* Core ends when the native device tile output is ready.  Benchmark
     * iterations that do not need validation may leave this false and avoid
     * the post-Core D2H/export entirely; all device output buffers are still
     * released normally before dmma_tilespgemm returns. */
    bool materialize_output = true;
    /* A null path keeps the production kernel and memory path unchanged.
     * Tracing replays the direct grid after the Core stopwatch and records
     * only the configured task sample. */
    const char *task_trace_path = nullptr;
    const char *matrix_name = nullptr;
    unsigned int task_trace_sample_shift = 0;
    unsigned int task_trace_sample_phase = 0;
};

struct DmmaSpGemmStats
{
    double candidate_ms = 0.0;
    double symbolic_ms = 0.0;
    double exact_kernel_ms = 0.0;
    double symbolic_finalize_ms = 0.0;
    double numeric_ms = 0.0;
    double total_ms = 0.0;
    double allocation_ms = 0.0;
    /* scheduler_ms is the host preparation/submission wall time retained for
     * compatibility; scheduler_device_ms spans the default-stream metadata
     * chain through its ready event and is the device-visible Core delay. */
    double scheduler_ms = 0.0;
    bool cost_balanced_requested = false;
    bool cost_balanced_used = false;
    int cost_worker_blocks = 0;
    std::size_t cost_metadata_bytes = 0;
    double scheduler_device_ms = 0.0;
    /* Device time of the cheap upper-bound filter plus the exact-count
     * replay over maybe IDs.  The direct exact-mask kernel is excluded. */
    double admission_device_ms = 0.0;
    double admission_filter_ms = 0.0;
    double admission_count_ms = 0.0;
    bool admission_timing_valid = false;
    bool admission_count_executed = false;
    bool admission_filter_fused_into_exact = false;
    bool admission_count_fused_into_exact = false;
    double light_numeric_ms = 0.0;
    double prefix_numeric_ms = 0.0;
    double suffix_numeric_ms = 0.0;
    double chunk_numeric_ms = 0.0;
    double reduction_ms = 0.0;
    double task_stats_ms = 0.0;
    double task_trace_ms = 0.0;
    double output_copy_ms = 0.0;
    bool output_materialized = false;
    /* Absolute wall-clock endpoint of the native-device-output Core.  The
     * caller uses this with a timestamp taken immediately before the online
     * B update, yielding one continuous B-update+SpGEMM stopwatch while the
     * trace/statistics/output-export work below remains excluded. */
    std::chrono::steady_clock::time_point core_completion_wall{};
    bool core_completion_wall_valid = false;
    unsigned long long candidate_tiles = 0;
    bool exact_forward_spa_requested = false;
    bool exact_forward_spa_used = false;
    DmmaExactForwardSpaReason exact_forward_spa_reason =
        DMMA_EXACT_FORWARD_NOT_REQUESTED;
    int exact_forward_rows = 0;
    int exact_forward_batch_capacity = 0;
    int exact_forward_batches = 0;
    std::size_t exact_forward_scratch_bytes = 0;
    unsigned long long exact_forward_pairs = 0;
    unsigned long long exact_forward_estimated_candidates = 0;
    unsigned long long exact_forward_estimated_reverse_work = 0;
    unsigned long long exact_forward_estimated_forward_work = 0;
    unsigned long long exact_forward_candidates = 0;
    unsigned long long exact_ordinary_candidates = 0;
    bool exact_forward_estimate_overflow = false;
    bool exact_forward_partition_complete = true;
    bool low_fill_exact_tile_requested = false;
    bool low_fill_exact_tile_used = false;
    int low_fill_q = 0;
    std::size_t low_fill_metadata_bytes = 0;
    bool low_fill_global_guard = false;
    DmmaLowFillExactTileReason low_fill_reason =
        DMMA_LOW_FILL_NOT_REQUESTED;
    int output_tiles = 0;
    int output_nnz = 0;
    DmmaDirectNumericLayout direct_numeric_layout =
        DMMA_DIRECT_NUMERIC_TILE_DYNAMIC;
    bool row_static_used = false;
    int row_static_ctas = 0;
    int row_static_unique_sms = 0;
    /* CUDA exposes no portable block-to-SM affinity.  A row-static sample is
     * a strict row-to-SM mapping only when the post-Core CTA SM-ID audit finds
     * one distinct SM for every logical worker CTA. */
    bool row_static_mapping_valid = false;
    bool row_dynamic_used = false;
    int row_dynamic_ctas = 0;
    int row_dynamic_unique_sms = 0;
    /* The same placement audit is retained for a controlled static/dynamic
     * comparison.  Dynamic correctness does not depend on one CTA per SM,
     * but a sample is a strict pure-LB comparison only when both layouts
     * observe the same distinct-SM worker placement. */
    bool row_dynamic_mapping_valid = false;
    int row_dynamic_queue_batch = 1;
    /* Compatibility field: for batch=1 this is byte-for-byte the historical
     * queue-head value.  For every batch it denotes the actual number of
     * atomicAdd reservations, while final_head records the fetched counter. */
    int row_dynamic_claims = 0;
    int row_dynamic_final_head = 0;
    int row_dynamic_expected_claims = 0;
    int row_dynamic_expected_final_head = 0;
    bool row_dynamic_claims_valid = false;
    /* exact-row-ptr-v1 auto-layout telemetry.  All values are generated
     * before numeric and the complete reduction/D2H/dispatch wall time is
     * part of Core.  Diagnostic printing happens only after the endpoint. */
    bool row_gate_requested = false;
    bool row_gate_used = false;
    bool row_gate_valid = false;
    bool row_gate_decision_dynamic = false;
    int row_gate_rows = 0;
    int row_gate_workers = 0;
    int row_gate_zero_workers = 0;
    unsigned long long row_gate_exact_tiles = 0;
    unsigned long long row_gate_load_sum = 0;
    unsigned long long row_gate_load_max = 0;
    double row_gate_load_sum_sq = 0.0;
    double row_gate_static_max_over_mean = 1.0;
    double row_gate_static_cv = 0.0;
    double row_gate_threshold = 1.10;
    double row_gate_reduction_ms = 0.0;
    int heavy_tasks = 0;
    int critical_q_begin = 0;
    int prefix_tasks = 0;
    int suffix_tasks = 0;
    double suffix_task_fraction = 0.0;
    int suffix_workers_per_sm = 0;
    int suffix_worker_blocks = 0;
    DmmaSuffixAutoBasis suffix_auto_basis =
        DMMA_SUFFIX_AUTO_REGULAR_WORK;
    double suffix_auto_fraction = 0.0;
    bool suffix_auto_used_symbolic_work = false;
    bool suffix_auto_fallback_to_tasks = false;
    int suffix_queue_batch = 0;
    int suffix_fine_tasks = 0;
    unsigned long long suffix_queue_atomics = 0;
    bool unified_light_used = false;
    bool unified_sparse_replay_used = false;
    DmmaUnifiedFallbackReason unified_fallback_reason =
        DMMA_UNIFIED_FALLBACK_NONE;
    int unified_worker_blocks = 0;
    int unified_workers_per_sm = 0;
    int unified_coarse_tasks = 0;
    int unified_fine_tasks = 0;
    int unified_fine_queue_tasks = 0;
    int unified_heavy_fine_tasks = 0;
    int unified_fine_capacity = 0;
    bool unified_fine_overflow = false;
    int unified_page_size = 0;
    int unified_coarse_pages = 0;
    unsigned long long unified_coarse_page_claims = 0;
    unsigned long long unified_fine_ticket_claims = 0;
    double unified_coarse_work = 0.0;
    double unified_fine_work = 0.0;
    double unified_heavy_work = 0.0;
    bool unified_coarse_work_available = false;
    std::size_t unified_metadata_bytes = 0;
    /* Number of descriptors that execute at least one merge step.  The
     * partitioner reserves one step for every later descriptor, so this is
     * both the emitted descriptor count and the actual nonempty chunk count. */
    int split_chunks = 0;
    int task_trace_tasks = 0;
    double split_task_fraction = 0.0;
    /* Actual nonempty chunks per heavy task, not the pre-rounding value of
     * ceil(predicted_work / threshold). */
    double average_chunks = 0.0;
    bool flat_grid_used = false;
    int flat_warps_per_cta = 0;
    unsigned int flat_grid_blocks = 0;
    unsigned long long flat_work_items = 0;
    double flat_numeric_ms = 0.0;
    double average_distinct_sms = 0.0;
    /* These two load statistics cover only scheduled heavy chunks and only
     * their splittable K-intersection work.  They are not whole-kernel SM
     * utilization metrics. */
    double sm_work_max_over_mean = 1.0;
    double sm_work_cv = 0.0;
    std::size_t partial_workspace_bytes = 0;
    /* One bit per output task marks work removed from the light grid. */
    std::size_t heavy_flag_bytes = 0;
    /* Bytes in the bounded exact-symbolic sparse tail side channel.  Fused
     * oversized symbolic deliberately uses zero bytes and falls back direct
     * because it has no materialized candidate IDs to map. */
    std::size_t symbolic_task_count_bytes = 0;
    bool symbolic_load_metadata = false;
    std::size_t symbolic_load_metadata_bytes = 0;
    double symbolic_load_quantum = 0.0;
    int symbolic_load_tasks = 0;
    int symbolic_load_saturated_tasks = 0;
    int symbolic_wave_task_capacity = 0;
    int symbolic_predicted_waves = 0;
    int symbolic_critical_tasks = 0;
    int symbolic_critical_tail_tasks = 0;
    double symbolic_total_work = 0.0;
    double symbolic_suffix_work = 0.0;
    double symbolic_split_suffix_work = 0.0;
    double symbolic_suffix_work_fraction = 0.0;
    double symbolic_regular_suffix_work_fraction = 0.0;
    double symbolic_max_task_work = 0.0;
    double symbolic_critical_work = 0.0;
    double symbolic_critical_max_task_work = 0.0;
    double symbolic_critical_tail_work = 0.0;
    double symbolic_critical_work_over_average_wave = 0.0;
    double symbolic_critical_tail_work_fraction = 0.0;
    int maybe_candidate_count = 0;
    int maybe_candidate_capacity = 0;
    bool maybe_candidate_overflow = false;
    bool maybe_candidate_count_is_lower_bound = false;
    double maybe_candidate_fraction = 0.0;
    bool maybe_scope_joint = false;
    int scheduler_reused_count_tasks = 0;
    int tail_record_count = 0;
    int tail_record_capacity = 0;
    bool tail_record_overflow = false;
    /* On overflow, count is the saturated capacity+1 lower bound rather than
     * the exact heavy-task population.  Without overflow it remains exact. */
    bool tail_record_count_is_lower_bound = false;
    double tail_record_fraction = 0.0;
    DmmaTailGateReason tail_gate_reason = DMMA_TAIL_GATE_NOT_REQUESTED;
    bool tail_split_forced = false;
    bool tail_gate_fallback_to_direct = false;
    bool scheduler_reused_symbolic_counts = false;
    bool scheduler_count_merge_fallback = false;
    bool zero_heavy_fallback_to_direct = false;
    bool workspace_fallback_to_atomic = false;
    DmmaNumericScheduleMode schedule_mode = DMMA_SCHEDULE_DIRECT;
    DmmaLightPolicy light_policy = DMMA_LIGHT_STATIC;
    DmmaSymbolicAdmissionMode symbolic_admission =
        DMMA_SYMBOLIC_ADMISSION_SEPARATE;
    DmmaPartialReductionMode reduction_mode = DMMA_REDUCTION_ATOMIC;
    DmmaSplitLaunchPolicy launch_policy =
        DMMA_SPLIT_LAUNCH_HEAVY_PRIORITY;
    /* CUDA priorities are numerically smaller when more urgent.  They are
     * meaningful only when split_streams_used is true. */
    int light_stream_priority = 0;
    int suffix_stream_priority = 0;
    int heavy_stream_priority = 0;
    bool stream_priority_range_supported = false;
    /* Distinguish per-Core infrastructure creation from use of a context
     * created outside the stopwatch. */
    bool split_streams_created = false;
    bool split_streams_used = false;
    /* True only when this Core call entered split numeric and borrowed the
     * caller-provided process-level context.  A precreated but unused context
     * leaves this flag and both stream flags false on direct/fallback calls. */
    bool split_context_reused = false;
    bool early_split_requested = false;
    bool early_split_used = false;
    int early_heavy_worker_block_cap = 0;
    int early_heavy_worker_blocks = 0;
    int early_reduction_worker_blocks = 0;
    int early_heavy_cap_numerator =
        DMMA_TILE_EARLY_SPLIT_CAP_NUMERATOR;
    int early_heavy_cap_denominator =
        DMMA_TILE_EARLY_SPLIT_CAP_DENOMINATOR;
    /* This is always false: CUDA owns placement and the implementation never
     * derives an SM ID to steer a CTA. */
    bool early_split_sm_affinity_claimed = false;
    bool wide_output_unrepresentable = false;
    unsigned long long wide_output_tiles = 0;
    unsigned long long wide_output_nnz = 0;
};

static inline void dmma_print_tail_fusion_diagnostic(
    const char *source, const DmmaNumericScheduleConfig &schedule,
    const DmmaSpGemmStats &stats)
{
    std::printf(
        "DMMA_TAIL_FUSION source=%s tail_count=%d capacity=%d "
        "overflow=%d tail_count_lower_bound=%d "
        "tail_fraction=%.9f max_heavy_fraction=%.9f "
        "force=%d gate_reason=%s symbolic_bytes=%zu reused_tasks=%d "
        "fallback_to_direct=%d admission=two-stage-upper "
        "admission_path=%s filter_fused=%d count_fused=%d "
        "admission_precision=fp32 maybe_scope=%s "
        "maybe_count=%d maybe_capacity=%d "
        "maybe_overflow=%d maybe_count_lower_bound=%d "
        "maybe_fraction=%.9f admission_filter_ms=%.6f "
        "admission_count_ms=%.6f admission_device_ms=%.6f "
        "admission_timing_valid=%d admission_count_executed=%d\n",
        source, stats.tail_record_count, stats.tail_record_capacity,
        stats.tail_record_overflow ? 1 : 0,
        stats.tail_record_count_is_lower_bound ? 1 : 0,
        stats.tail_record_fraction, schedule.max_heavy_fraction,
        stats.tail_split_forced ? 1 : 0,
        dmma_tail_gate_reason_name(stats.tail_gate_reason),
        stats.symbolic_task_count_bytes,
        stats.scheduler_reused_count_tasks,
        stats.tail_gate_fallback_to_direct ? 1 : 0,
        dmma_symbolic_admission_name(stats.symbolic_admission),
        stats.admission_filter_fused_into_exact ? 1 : 0,
        stats.admission_count_fused_into_exact ? 1 : 0,
        stats.maybe_scope_joint ? "joint" : "heavy",
        stats.maybe_candidate_count, stats.maybe_candidate_capacity,
        stats.maybe_candidate_overflow ? 1 : 0,
        stats.maybe_candidate_count_is_lower_bound ? 1 : 0,
        stats.maybe_candidate_fraction, stats.admission_filter_ms,
        stats.admission_count_ms, stats.admission_device_ms,
        stats.admission_timing_valid ? 1 : 0,
        stats.admission_count_executed ? 1 : 0);
}

static inline double dmma_elapsed_ms(const timeval &begin, const timeval &end)
{
    return (end.tv_sec - begin.tv_sec) * 1000.0 +
           (end.tv_usec - begin.tv_usec) / 1000.0;
}

/* Exact critical-window arithmetic is centralized so host tests and the
 * production path cannot silently disagree about q=0/q=1 or odd N. */
static inline int dmma_critical_suffix_begin(int output_tasks, double q_min)
{
    if (output_tasks <= 1)
        return 0;
    const double position =
        q_min * static_cast<double>(output_tasks - 1);
    const long long begin = static_cast<long long>(std::ceil(position));
    return static_cast<int>(
        std::max<long long>(0, std::min<long long>(output_tasks - 1, begin)));
}

static inline int dmma_suffix_fine_task_count(
    int suffix_tasks, int worker_blocks, int fine_tasks_per_worker)
{
    if (suffix_tasks <= 0 || worker_blocks <= 0 ||
        fine_tasks_per_worker <= 0)
        return 0;
    const long long requested =
        static_cast<long long>(worker_blocks) * fine_tasks_per_worker;
    return static_cast<int>(
        std::min<long long>(suffix_tasks, requested));
}

/* Unified-light partitions exact output IDs into three disjoint sets:
 * heavy parents, non-heavy fine IDs, and the remaining coarse IDs.  The
 * terminal predicate protects the endpoint even when quantized cost is tied;
 * the threshold predicate admits unusually costly tasks elsewhere in q's
 * suffix. */
static __host__ __device__ __forceinline__ bool
dmma_unified_is_fine_task(int output, int suffix_begin, int terminal_begin,
                          uint32_t work_units,
                          uint32_t fine_threshold_units)
{
    return output >= suffix_begin &&
           (output >= terminal_begin || work_units >= fine_threshold_units);
}

struct DmmaUnifiedPartitionCounts
{
    int coarse = 0;
    int fine = 0;
    int heavy = 0;
    int heavy_fine = 0;
    bool valid = false;
};

static inline DmmaUnifiedPartitionCounts dmma_unified_partition_counts(
    int total, int potential_fine, int heavy, int heavy_fine)
{
    DmmaUnifiedPartitionCounts result;
    result.valid = total >= 0 && potential_fine >= 0 && heavy >= 0 &&
                   heavy_fine >= 0 && potential_fine <= total &&
                   heavy <= total && heavy_fine <= potential_fine &&
                   heavy_fine <= heavy &&
                   heavy - heavy_fine <= total - potential_fine;
    if (!result.valid)
        return result;
    result.fine = potential_fine - heavy_fine;
    result.heavy = heavy;
    result.heavy_fine = heavy_fine;
    result.coarse = total - potential_fine - (heavy - heavy_fine);
    return result;
}

static inline DmmaUnifiedFallbackReason dmma_unified_symbolic_fallback(
    bool metadata_available, int fine_count, int fine_capacity,
    bool fine_overflow, int saturated_tasks,
    unsigned long long total_units)
{
    if (!metadata_available)
        return DMMA_UNIFIED_FALLBACK_SYMBOLIC_UNAVAILABLE;
    if (fine_overflow || fine_count < 0 || fine_count > fine_capacity)
        return DMMA_UNIFIED_FALLBACK_FINE_OVERFLOW;
    if (saturated_tasks != 0)
        return DMMA_UNIFIED_FALLBACK_SATURATED_LOAD;
    if (total_units == 0)
        return DMMA_UNIFIED_FALLBACK_ZERO_LOAD;
    return DMMA_UNIFIED_FALLBACK_NONE;
}

struct DmmaSuffixAutoSelection
{
    double fraction = 0.0;
    bool used_symbolic_work = false;
    bool fallback_to_tasks = false;
};

/* Select the explainable resource fraction before architecture-specific
 * occupancy clamps.  Saturated per-task metadata is intentionally rejected:
 * clipping a long task would systematically under-allocate its stream. */
static inline DmmaSuffixAutoSelection dmma_select_suffix_auto_fraction(
    DmmaSuffixAutoBasis basis, double task_fraction, double total_work,
    double suffix_work, double split_suffix_work, int saturated_tasks)
{
    DmmaSuffixAutoSelection result;
    result.fraction =
        std::max(0.0, std::min(1.0, task_fraction));
    if (basis == DMMA_SUFFIX_AUTO_TASK)
        return result;

    bool valid = saturated_tasks == 0 && std::isfinite(total_work) &&
                 std::isfinite(suffix_work) &&
                 std::isfinite(split_suffix_work) && total_work > 0.0 &&
                 suffix_work >= 0.0 && suffix_work <= total_work &&
                 split_suffix_work >= 0.0 &&
                 split_suffix_work <= suffix_work;
    double numerator = suffix_work;
    double denominator = total_work;
    if (basis == DMMA_SUFFIX_AUTO_REGULAR_WORK)
    {
        numerator -= split_suffix_work;
        denominator -= split_suffix_work;
        valid = valid && denominator > 0.0 && numerator >= 0.0 &&
                numerator <= denominator;
    }
    if (!valid)
    {
        result.fallback_to_tasks = true;
        return result;
    }
    result.fraction =
        std::max(0.0, std::min(1.0, numerator / denominator));
    result.used_symbolic_work = true;
    return result;
}

static inline int dmma_suffix_auto_requested_workers(double fraction,
                                                     int full_warps)
{
    if (full_warps <= 0)
        return 0;
    const double bounded = std::max(0.0, std::min(1.0, fraction));
    return std::max(1, static_cast<int>(std::ceil(bounded * full_warps)));
}

/* Close the public Core clock with a monotonic timestamp first.  timeval is
 * retained only for the existing internal phase diagnostics. */
static inline void dmma_close_core_endpoint(
    const timeval &total_begin, timeval *total_end, DmmaSpGemmStats *stats)
{
    stats->core_completion_wall = std::chrono::steady_clock::now();
    stats->core_completion_wall_valid = true;
    gettimeofday(total_end, nullptr);
    stats->total_ms = dmma_elapsed_ms(total_begin, *total_end);
}

static inline bool dmma_cuda_ok(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "CUDA error in %s: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

struct DmmaSplitAsyncState
{
    cudaStream_t light_stream = nullptr;
    cudaStream_t suffix_stream = nullptr;
    cudaStream_t heavy_stream = nullptr;
    /* Scheduler preparation is issued on stream zero.  All non-blocking
     * numeric streams explicitly wait for this event before consuming any
     * matrix-dependent metadata. */
    cudaEvent_t metadata_ready = nullptr;
    cudaEvent_t scheduler_start = nullptr;
    cudaEvent_t scheduler_stop = nullptr;
    cudaEvent_t admission_filter_start = nullptr;
    cudaEvent_t admission_filter_stop = nullptr;
    cudaEvent_t admission_count_start = nullptr;
    cudaEvent_t admission_count_stop = nullptr;
    cudaEvent_t numeric_start = nullptr;
    cudaEvent_t numeric_stop = nullptr;
    cudaEvent_t light_start = nullptr;
    cudaEvent_t light_stop = nullptr;
    cudaEvent_t suffix_start = nullptr;
    cudaEvent_t suffix_stop = nullptr;
    cudaEvent_t light_join_stop = nullptr;
    cudaEvent_t chunk_start = nullptr;
    cudaEvent_t chunk_stop = nullptr;
    cudaEvent_t reduction_start = nullptr;
    cudaEvent_t reduction_stop = nullptr;
    int light_priority = 0;
    int suffix_priority = 0;
    int heavy_priority = 0;
    bool priority_range_supported = false;
    DmmaSplitLaunchPolicy launch_policy =
        DMMA_SPLIT_LAUNCH_HEAVY_PRIORITY;
};

static inline bool dmma_read_admission_timing(
    const DmmaSplitAsyncState &state, bool count_event_recorded,
    DmmaSpGemmStats *stats)
{
    if (stats == nullptr)
        return false;
    float elapsed = 0.0f;
    if (!dmma_cuda_ok(cudaEventElapsedTime(
                          &elapsed, state.admission_filter_start,
                          state.admission_filter_stop),
                      "time admission maybe filter"))
        return false;
    stats->admission_filter_ms = static_cast<double>(elapsed);
    if (count_event_recorded)
    {
        if (!dmma_cuda_ok(cudaEventElapsedTime(
                              &elapsed, state.admission_count_start,
                              state.admission_count_stop),
                          "time admission exact-count replay"))
            return false;
        stats->admission_count_ms = static_cast<double>(elapsed);
    }
    stats->admission_device_ms =
        stats->admission_filter_ms + stats->admission_count_ms;
    stats->admission_timing_valid = true;
    return true;
}

static inline bool dmma_split_async_state_ready(
    const DmmaSplitAsyncState &state)
{
    return state.light_stream != nullptr && state.suffix_stream != nullptr &&
           state.heavy_stream != nullptr &&
           state.metadata_ready != nullptr && state.scheduler_start != nullptr &&
           state.scheduler_stop != nullptr &&
           state.admission_filter_start != nullptr &&
           state.admission_filter_stop != nullptr &&
           state.admission_count_start != nullptr &&
           state.admission_count_stop != nullptr &&
           state.numeric_start != nullptr &&
           state.numeric_stop != nullptr && state.light_start != nullptr &&
           state.light_stop != nullptr && state.suffix_start != nullptr &&
           state.suffix_stop != nullptr &&
           state.light_join_stop != nullptr &&
           state.chunk_start != nullptr && state.chunk_stop != nullptr &&
           state.reduction_start != nullptr && state.reduction_stop != nullptr;
}

static inline bool dmma_split_async_state_empty(
    const DmmaSplitAsyncState &state)
{
    return state.light_stream == nullptr && state.suffix_stream == nullptr &&
           state.heavy_stream == nullptr &&
           state.metadata_ready == nullptr && state.scheduler_start == nullptr &&
           state.scheduler_stop == nullptr &&
           state.admission_filter_start == nullptr &&
           state.admission_filter_stop == nullptr &&
           state.admission_count_start == nullptr &&
           state.admission_count_stop == nullptr &&
           state.numeric_start == nullptr &&
           state.numeric_stop == nullptr && state.light_start == nullptr &&
           state.light_stop == nullptr && state.suffix_start == nullptr &&
           state.suffix_stop == nullptr &&
           state.light_join_stop == nullptr &&
           state.chunk_start == nullptr && state.chunk_stop == nullptr &&
           state.reduction_start == nullptr && state.reduction_stop == nullptr;
}

static inline void dmma_destroy_split_async_state(
    DmmaSplitAsyncState *state, bool synchronize_first);

static inline bool dmma_create_split_async_state(
    DmmaSplitLaunchPolicy launch_policy, DmmaSplitAsyncState *state)
{
    if (state == nullptr || !dmma_split_async_state_empty(*state))
        return false;
    int least_priority = 0;
    int greatest_priority = 0;
    if (!dmma_cuda_ok(cudaDeviceGetStreamPriorityRange(
                          &least_priority, &greatest_priority),
                      "query numeric stream priority range"))
        return false;

    /* CUDA reports [greatest, least], with smaller values denoting higher
     * priority.  Zero is the conventional default; clamp it for runtimes
     * exposing a range that does not contain zero. */
    const int common_priority =
        std::max(greatest_priority, std::min(0, least_priority));
    state->light_priority = common_priority;
    state->suffix_priority = common_priority;
    state->heavy_priority = common_priority;
    state->priority_range_supported = greatest_priority < least_priority;
    state->launch_policy = launch_policy;
    if (launch_policy == DMMA_SPLIT_LAUNCH_HEAVY_PRIORITY)
    {
        state->light_priority = least_priority;
        state->suffix_priority = least_priority;
        state->heavy_priority = greatest_priority;
    }

    const bool created =
        dmma_cuda_ok(cudaStreamCreateWithPriority(
                         &state->light_stream, cudaStreamNonBlocking,
                         state->light_priority),
                     "create light numeric stream") &&
        dmma_cuda_ok(cudaStreamCreateWithPriority(
                         &state->suffix_stream, cudaStreamNonBlocking,
                         state->suffix_priority),
                     "create suffix numeric stream") &&
        dmma_cuda_ok(cudaStreamCreateWithPriority(
                         &state->heavy_stream, cudaStreamNonBlocking,
                         state->heavy_priority),
                     "create heavy numeric stream") &&
        dmma_cuda_ok(cudaEventCreateWithFlags(
                         &state->metadata_ready, cudaEventDisableTiming),
                     "create scheduler metadata-ready event") &&
        dmma_cuda_ok(cudaEventCreate(&state->scheduler_start),
                     "create scheduler start event") &&
        dmma_cuda_ok(cudaEventCreate(&state->scheduler_stop),
                     "create scheduler stop event") &&
        dmma_cuda_ok(cudaEventCreate(&state->admission_filter_start),
                     "create admission filter start event") &&
        dmma_cuda_ok(cudaEventCreate(&state->admission_filter_stop),
                     "create admission filter stop event") &&
        dmma_cuda_ok(cudaEventCreate(&state->admission_count_start),
                     "create admission count start event") &&
        dmma_cuda_ok(cudaEventCreate(&state->admission_count_stop),
                     "create admission count stop event") &&
        dmma_cuda_ok(cudaEventCreate(&state->numeric_start),
                     "create numeric start event") &&
        dmma_cuda_ok(cudaEventCreate(&state->numeric_stop),
                     "create numeric stop event") &&
        dmma_cuda_ok(cudaEventCreate(&state->light_start),
                     "create light start event") &&
        dmma_cuda_ok(cudaEventCreate(&state->light_stop),
                     "create light stop event") &&
        dmma_cuda_ok(cudaEventCreate(&state->suffix_start),
                     "create suffix start event") &&
        dmma_cuda_ok(cudaEventCreate(&state->suffix_stop),
                     "create suffix stop event") &&
        dmma_cuda_ok(cudaEventCreate(&state->light_join_stop),
                     "create light join stop event") &&
        dmma_cuda_ok(cudaEventCreate(&state->chunk_start),
                     "create chunk start event") &&
        dmma_cuda_ok(cudaEventCreate(&state->chunk_stop),
                     "create chunk stop event") &&
        dmma_cuda_ok(cudaEventCreate(&state->reduction_start),
                     "create reduction start event") &&
        dmma_cuda_ok(cudaEventCreate(&state->reduction_stop),
                     "create reduction stop event");
    if (!created)
        dmma_destroy_split_async_state(state, false);
    return created;
}

static inline void dmma_destroy_split_async_state(
    DmmaSplitAsyncState *state, bool synchronize_first)
{
    if (state == nullptr)
        return;
    if (synchronize_first &&
        (state->light_stream != nullptr || state->suffix_stream != nullptr ||
         state->heavy_stream != nullptr))
        cudaDeviceSynchronize();
    if (state->metadata_ready != nullptr)
        cudaEventDestroy(state->metadata_ready);
    if (state->scheduler_start != nullptr)
        cudaEventDestroy(state->scheduler_start);
    if (state->scheduler_stop != nullptr)
        cudaEventDestroy(state->scheduler_stop);
    if (state->admission_filter_start != nullptr)
        cudaEventDestroy(state->admission_filter_start);
    if (state->admission_filter_stop != nullptr)
        cudaEventDestroy(state->admission_filter_stop);
    if (state->admission_count_start != nullptr)
        cudaEventDestroy(state->admission_count_start);
    if (state->admission_count_stop != nullptr)
        cudaEventDestroy(state->admission_count_stop);
    if (state->numeric_start != nullptr)
        cudaEventDestroy(state->numeric_start);
    if (state->numeric_stop != nullptr)
        cudaEventDestroy(state->numeric_stop);
    if (state->light_start != nullptr)
        cudaEventDestroy(state->light_start);
    if (state->light_stop != nullptr)
        cudaEventDestroy(state->light_stop);
    if (state->suffix_start != nullptr)
        cudaEventDestroy(state->suffix_start);
    if (state->suffix_stop != nullptr)
        cudaEventDestroy(state->suffix_stop);
    if (state->light_join_stop != nullptr)
        cudaEventDestroy(state->light_join_stop);
    if (state->chunk_start != nullptr)
        cudaEventDestroy(state->chunk_start);
    if (state->chunk_stop != nullptr)
        cudaEventDestroy(state->chunk_stop);
    if (state->reduction_start != nullptr)
        cudaEventDestroy(state->reduction_start);
    if (state->reduction_stop != nullptr)
        cudaEventDestroy(state->reduction_stop);
    if (state->light_stream != nullptr)
        cudaStreamDestroy(state->light_stream);
    if (state->suffix_stream != nullptr)
        cudaStreamDestroy(state->suffix_stream);
    if (state->heavy_stream != nullptr)
        cudaStreamDestroy(state->heavy_stream);
    *state = DmmaSplitAsyncState();
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
    cudaFree(tiles->row_tile_nnz_sum);
    cudaFree(tiles->col_tile_nnz_sum);
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
                                       DmmaOwnedDeviceTiles *device,
                                       bool prepare_low_fill_metadata = false)
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
    unsigned long long structural_nnz = 0;
    if (prepare_low_fill_metadata)
        for (std::size_t tile = 0; tile < tiles; ++tile)
            structural_nnz += static_cast<unsigned long long>(
                __builtin_popcount(host.masks[tile]));
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
    result.view.structural_nnz = structural_nnz;
    if (prepare_low_fill_metadata)
    {
        bool overflow = false;
        if (require_csc)
        {
            std::vector<uint32_t> sums(
                static_cast<std::size_t>(host.tile_col_count), 0u);
            for (int col = 0; col < host.tile_col_count; ++col)
            {
                unsigned long long sum = 0;
                for (int p = host.tile_col_ptr[col];
                     p < host.tile_col_ptr[col + 1]; ++p)
                    sum += static_cast<unsigned long long>(
                        __builtin_popcount(host.masks[host.csc_tile_ids[p]]));
                if (sum > UINT_MAX)
                {
                    overflow = true;
                    break;
                }
                sums[static_cast<std::size_t>(col)] =
                    static_cast<uint32_t>(sum);
            }
            if (!overflow && dmma_upload_array(
                                 &result.col_tile_nnz_sum, sums.data(),
                                 sums.size(), "upload low-fill B column sums"))
                result.view.col_tile_nnz_sum_valid = true;
            else if (!overflow)
            {
                cudaFree(result.col_tile_nnz_sum);
                result.col_tile_nnz_sum = nullptr;
                cudaGetLastError();
            }
        }
        else
        {
            std::vector<uint32_t> sums(
                static_cast<std::size_t>(host.tile_row_count), 0u);
            for (int row = 0; row < host.tile_row_count; ++row)
            {
                unsigned long long sum = 0;
                for (int p = host.tile_row_ptr[row];
                     p < host.tile_row_ptr[row + 1]; ++p)
                    sum += static_cast<unsigned long long>(
                        __builtin_popcount(host.masks[p]));
                if (sum > UINT_MAX)
                {
                    overflow = true;
                    break;
                }
                sums[static_cast<std::size_t>(row)] =
                    static_cast<uint32_t>(sum);
            }
            if (!overflow && dmma_upload_array(
                                 &result.row_tile_nnz_sum, sums.data(),
                                 sums.size(), "upload low-fill A row sums"))
                result.view.row_tile_nnz_sum_valid = true;
            else if (!overflow)
            {
                cudaFree(result.row_tile_nnz_sum);
                result.row_tile_nnz_sum = nullptr;
                cudaGetLastError();
            }
        }
        result.view.low_fill_metadata_overflow = overflow;
    }
    result.view.row_tile_nnz_sum = result.row_tile_nnz_sum;
    result.view.col_tile_nnz_sum = result.col_tile_nnz_sum;
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

/* A uses row-major 8x4 mask bits and B uses column-major 4x8 mask bits.
 * Thus A(row,:) and B(:,col) are contiguous four-bit nibbles at row*4 and
 * col*4.  This host/device helper is the shared layout contract for both
 * candidate exact-symbolic primitives and the full tile-pair product. */
static __host__ __device__ __forceinline__ bool
dmma_tile_pair_output_present(uint32_t mask_a, uint32_t mask_b,
                              int output_position)
{
    const int row = output_position / DMMA_TILE_N;
    const int col = output_position % DMMA_TILE_N;
    const uint32_t a_bits =
        (mask_a >> (row * DMMA_TILE_K)) & uint32_t(0xfu);
    const uint32_t b_bits =
        (mask_b >> (col * DMMA_TILE_K)) & uint32_t(0xfu);
    return (a_bits & b_bits) != 0;
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
                if (dmma_tile_pair_output_present(
                        mask_a, mask_b, output_pos))
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

static __host__ __device__ __forceinline__ int
dmma_merge_scans_from_advances(int advanced_a, int advanced_b, int matches)
{
    const long long scans = static_cast<long long>(advanced_a) +
                            static_cast<long long>(advanced_b) - matches;
    if (scans <= 0)
        return 0;
    return scans > INT_MAX ? INT_MAX : static_cast<int>(scans);
}

/* Split scheduling needs the exact merge trajectory as well as the output
 * mask.  Keep this as a separate primitive/kernel from the production direct
 * path: direct scheduling has no scan/match counters or sparse-append state in
 * its exact-mask kernel's register footprint. */
__device__ __forceinline__ unsigned long long
dmma_candidate_exact_mask_with_counts(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int tile_row, int tile_col,
    int lane, int *scan_steps_result, int *matches_result)
{
    int pa = a.tile_row_ptr[tile_row];
    const int a_end = a.tile_row_ptr[tile_row + 1];
    int pb = b.tile_col_ptr[tile_col];
    const int b_end = b.tile_col_ptr[tile_col + 1];
    unsigned long long local_output = 0;
    int matches = 0;

    while (pa < a_end && pb < b_end)
    {
        const int ka = a.tile_col_idx[pa];
        const int kb = b.tile_row_idx[pb];
        if (ka == kb)
        {
            if (lane == 0)
                ++matches;
            const uint32_t mask_a = a.masks[pa];
            const int tile_b = b.csc_tile_ids[pb];
            const uint32_t mask_b = b.masks[tile_b];
            for (int output_pos = lane; output_pos < DMMA_OUTPUT_ELEMS;
                 output_pos += WARP_SIZE)
            {
                if (dmma_tile_pair_output_present(
                        mask_a, mask_b, output_pos))
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
    if (lane == 0)
    {
        *scan_steps_result = dmma_merge_scans_from_advances(
            pa - a.tile_row_ptr[tile_row],
            pb - b.tile_col_ptr[tile_col], matches);
        *matches_result = matches;
    }
    return dmma_warp_or(local_output);
}

__global__ void dmma_exact_mask_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int candidate_count,
    const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols, uint64_t *candidate_masks,
    int *candidate_keep)
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
        candidate_keep[global_warp] = output_mask != 0;
    }
}

/* ExactTile-Sparse v1 keeps the ordinary one-warp candidate owner.  A local
 * structural gate is chosen once for the task; every matched tile pair then
 * contributes through exactly one sparse-outer-product or fixed 64-position
 * primitive.  The host launches this kernel only after the global guard and
 * all metadata checks have succeeded. */
__device__ __forceinline__ unsigned long long
dmma_candidate_exact_mask_low_fill(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int tile_row, int tile_col,
    int lane, int q)
{
    const uint32_t a_degree = static_cast<uint32_t>(
        a.tile_row_ptr[tile_row + 1] - a.tile_row_ptr[tile_row]);
    const uint32_t b_degree = static_cast<uint32_t>(
        b.tile_col_ptr[tile_col + 1] - b.tile_col_ptr[tile_col]);
    const bool task_sparse = dmma_low_fill_local_guard(
        a.row_tile_nnz_sum[tile_row], a_degree,
        b.col_tile_nnz_sum[tile_col], b_degree, q);
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
            const uint32_t mask_b = b.masks[b.csc_tile_ids[pb]];
            const bool sparse_pair =
                task_sparse && __popc(mask_a) * __popc(mask_b) <=
                                   DMMA_OUTPUT_ELEMS;
            if (sparse_pair)
            {
                if ((mask_a & (uint32_t(1) << lane)) != 0)
                {
                    const int output_row = lane / DMMA_TILE_K;
                    const int k = lane % DMMA_TILE_K;
                    uint32_t matching_b =
                        mask_b & (uint32_t(0x11111111u) << k);
                    while (matching_b != 0)
                    {
                        const int b_position = __ffs(matching_b) - 1;
                        const int output_col = b_position / DMMA_TILE_K;
                        local_output |=
                            1ull << (output_row * DMMA_TILE_N + output_col);
                        matching_b &= matching_b - 1;
                    }
                }
            }
            else
            {
#pragma unroll
                for (int output_pos = lane;
                     output_pos < DMMA_OUTPUT_ELEMS;
                     output_pos += WARP_SIZE)
                    if (dmma_tile_pair_output_present(
                            mask_a, mask_b, output_pos))
                        local_output |= 1ull << output_pos;
            }
            ++pa;
            ++pb;
        }
        else if (ka < kb)
            ++pa;
        else
            ++pb;
    }
    return dmma_warp_or(local_output);
}

__global__ void dmma_exact_mask_low_fill_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int candidate_count,
    const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols, int q,
    uint64_t *__restrict__ candidate_masks,
    int *__restrict__ candidate_keep)
{
    const std::size_t task = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (task >= static_cast<std::size_t>(candidate_count))
        return;
    const unsigned long long mask = dmma_candidate_exact_mask_low_fill(
        a, b, candidate_rows[task], candidate_cols[task], lane, q);
    if (lane == 0)
    {
        candidate_masks[task] = mask;
        candidate_keep[task] = mask != 0;
    }
}

/* Device-side audit record for the optional row-hybrid exact path.  All
 * counters saturate instead of wrapping; an overflow makes the host discard
 * the whole partition and safely relaunch the original ordinary kernel. */
struct DmmaExactForwardSpaDeviceSummary
{
    unsigned long long selected_rows = 0;
    unsigned long long forward_pairs = 0;
    unsigned long long estimated_candidates = 0;
    unsigned long long reverse_work = 0;
    unsigned long long forward_work = 0;
    unsigned long long forward_candidates = 0;
    unsigned long long estimate_overflow = 0;
};

__device__ __forceinline__ bool dmma_atomic_saturating_add_u64(
    unsigned long long *address, unsigned long long value)
{
    unsigned long long observed = atomicCAS(address, 0ull, 0ull);
    while (true)
    {
        const bool overflow = ULLONG_MAX - observed < value;
        const unsigned long long desired =
            overflow ? ULLONG_MAX : observed + value;
        const unsigned long long previous = atomicCAS(address, observed,
                                                       desired);
        if (previous == observed)
            return overflow;
        observed = previous;
    }
}

/* One warp classifies one A tile row.  The gate deliberately does not take
 * candidate rows/columns, exact masks, keep flags, C nnz, or any timing
 * feedback: every input is an A/B structural quantity available before
 * candidate materialisation. */
__global__ void dmma_classify_exact_forward_rows_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b,
    unsigned long long minimum_forward_pairs,
    double minimum_reverse_over_forward,
    int *__restrict__ forward_flags, int *__restrict__ forward_rows,
    DmmaExactForwardSpaDeviceSummary *__restrict__ summary)
{
    const std::size_t row = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (row >= static_cast<std::size_t>(a.tile_row_count))
        return;

    unsigned long long forward_pairs = 0;
    for (int pa = a.tile_row_ptr[row] + lane;
         pa < a.tile_row_ptr[row + 1]; pa += WARP_SIZE)
    {
        const int k_tile = a.tile_col_idx[pa];
        forward_pairs += static_cast<unsigned long long>(
            b.tile_row_ptr[k_tile + 1] - b.tile_row_ptr[k_tile]);
    }
    forward_pairs = dmma_warp_sum_u64(forward_pairs);
    if (lane != 0)
        return;

    const int a_degree =
        a.tile_row_ptr[row + 1] - a.tile_row_ptr[row];
    const DmmaExactForwardRowEstimate estimate =
        dmma_exact_forward_row_estimate(
            a_degree, forward_pairs, b.tile_col_count, b.num_tiles,
            minimum_forward_pairs, minimum_reverse_over_forward);
    forward_flags[row] = estimate.selected ? 1 : 0;
    if (estimate.overflow)
    {
        atomicExch(&summary->estimate_overflow, 1ull);
        return;
    }
    if (!estimate.selected)
        return;

    const unsigned long long slot =
        atomicAdd(&summary->selected_rows, 1ull);
    forward_rows[slot] = static_cast<int>(row);
    bool aggregate_overflow = false;
    aggregate_overflow |= dmma_atomic_saturating_add_u64(
        &summary->forward_pairs, estimate.forward_pairs);
    aggregate_overflow |= dmma_atomic_saturating_add_u64(
        &summary->estimated_candidates, estimate.estimated_candidates);
    aggregate_overflow |= dmma_atomic_saturating_add_u64(
        &summary->reverse_work, estimate.reverse_work);
    aggregate_overflow |= dmma_atomic_saturating_add_u64(
        &summary->forward_work, estimate.forward_work);
    if (aggregate_overflow)
        atomicExch(&summary->estimate_overflow, 1ull);
}

/* Ordinary side of the exact row partition.  Apart from the row flag guard,
 * this is intentionally identical to dmma_exact_mask_kernel. */
__global__ void dmma_exact_mask_partitioned_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int candidate_count,
    const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols,
    const int *__restrict__ forward_flags,
    uint64_t *__restrict__ candidate_masks,
    int *__restrict__ candidate_keep)
{
    const std::size_t candidate =
        dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (candidate >= static_cast<std::size_t>(candidate_count))
        return;
    const int tile_row = candidate_rows[candidate];
    if (forward_flags[tile_row] != 0)
        return;
    const unsigned long long output_mask =
        dmma_candidate_exact_mask(a, b, tile_row,
                                  candidate_cols[candidate], lane);
    if (lane == 0)
    {
        candidate_masks[candidate] = output_mask;
        candidate_keep[candidate] = output_mask != 0;
    }
}

/* Fused-exact records are born in candidate-ID space and mapped after the keep
 * scan.  The default separate path replays after that scan and writes final
 * output IDs directly.  Keeping one compact 16-byte record per exceptional
 * task avoids dense scan/match arrays at both candidate and output scale. */
struct DmmaTailRecord
{
    int id = 0;
    int scans = 0;
    int matches = 0;
    int chunks = 0;
};

struct DmmaTailAppendState
{
    int count = 0;
    int overflow = 0;
};

/* B-values iterations preserve both operand structures, so the exact heavy
 * set is invariant.  Keep a compact host copy between calls; cache hits avoid
 * rebuilding an output-sized count buffer and avoid reserving the configured
 * worst-case record capacity.  Numeric outputs and partial accumulators are
 * never cached. */
struct DmmaTileFlexTailHostCache
{
    int device = -1;
    const int *a_row_ptr = nullptr;
    const int *a_col_idx = nullptr;
    const int *b_row_ptr = nullptr;
    const int *b_col_idx = nullptr;
    int output_tiles = -1;
    double scan = 0.0;
    double match = 0.0;
    double threshold = 0.0;
    double chunk_target = 0.0;
    int max_chunks = 0;
    std::vector<DmmaTailRecord> records;
    bool valid = false;

    bool matches(const DmmaDeviceTiles &a, const DmmaDeviceTiles &b,
                 int current_device, const DmmaNumericScheduleConfig &schedule,
                 double resolved_target) const
    {
        return valid && device == current_device &&
               a_row_ptr == a.tile_row_ptr && a_col_idx == a.tile_col_idx &&
               b_row_ptr == b.tile_row_ptr && b_col_idx == b.tile_col_idx &&
               scan == schedule.cost.scan && match == schedule.cost.match &&
               threshold == schedule.split_threshold &&
               chunk_target == resolved_target &&
               max_chunks == schedule.max_chunks;
    }
};

/* Reserve one bounded sparse-tail record.  Before overflow, each successful
 * caller receives a unique slot in [0, capacity).  The first excess caller
 * raises overflow and all racing excess increments converge to capacity+1;
 * later callers observe overflow and avoid the append counter entirely.
 *
 * In production capacity=min(candidate_count, configured_capacity), so an
 * overflow implies capacity<candidate_count<=INT_MAX and capacity+1 is
 * representable.  The host branch is a sequential model used by regression
 * tests; the device branch is the production implementation. */
static __host__ __device__ __forceinline__ int
dmma_tail_try_reserve_record(DmmaTailAppendState *state, int capacity)
{
    if (state == nullptr || capacity <= 0 || state->overflow != 0)
        return -1;
#if defined(__CUDA_ARCH__)
    const int slot = atomicAdd(&state->count, 1);
    if (slot < capacity)
        return slot;
    atomicExch(&state->overflow, 1);
    /* Every caller that incremented after capacity participates in this
     * convergence, so count is exactly capacity+1 when the kernel ends. */
    atomicMin(&state->count, capacity + 1);
    return -1;
#else
    const int slot = state->count++;
    if (slot < capacity)
        return slot;
    state->overflow = 1;
    state->count = capacity + 1;
    return -1;
#endif
}

/* FP32 is used only for scheduling admission.  Tail records retain exact
 * integer counts, and descriptor boundary construction below continues to
 * use the original FP64 model so chunk coverage and weighting are not
 * approximated a second time. */
static __host__ __device__ __forceinline__ float
dmma_tail_fp32_work_from_counts(
    long long scans, int matches, float scan_cost, float match_cost)
{
    return scan_cost * static_cast<float>(scans) +
           match_cost * static_cast<float>(matches);
}

static __host__ __device__ __forceinline__ int
dmma_tail_chunk_count_from_counts(
    int scans, int matches, float scan_cost, float match_cost,
    float admission_threshold, float chunk_target, int max_chunks)
{
    const float splittable_work = dmma_tail_fp32_work_from_counts(
        static_cast<long long>(scans), matches, scan_cost, match_cost);
    if (!(splittable_work > admission_threshold) ||
        !(chunk_target > 0.0f) || scans <= 1 || max_chunks <= 1)
        return 0;
    const float requested = ceilf(splittable_work / chunk_target);
    int chunks = requested >= static_cast<float>(max_chunks)
                     ? max_chunks
                     : static_cast<int>(requested);
    if (chunks < 2)
        chunks = 2;
    if (chunks > scans)
        chunks = scans;
    return chunks >= 2 ? chunks : 0;
}

/* TileFlex step2 has already compacted final 8x8 C tasks.  Decode the exact
 * merge counts collected during its native finalize and materialize only the
 * sparse heavy set expected by the pre-16x16 tail scheduler.  This replaces
 * the legacy candidate-ID map pass; record IDs are born in final-output
 * space, so the numeric light/heavy partition remains unchanged. */
__global__ void dmma_tileflex_build_tail_records_kernel(
    int task_count, const std::uint64_t *__restrict__ packed_counts,
    float scan_cost, float match_cost, float admission_threshold,
    float chunk_target, int max_chunks,
    DmmaTailRecord *__restrict__ tail_records, int tail_capacity,
    DmmaTailAppendState *__restrict__ tail_state)
{
    const int task = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (task >= task_count)
        return;
    const std::uint64_t packed = packed_counts[task];
    const std::uint32_t scans_wide = static_cast<std::uint32_t>(packed >> 32);
    const std::uint32_t matches_wide = static_cast<std::uint32_t>(packed);
    if (scans_wide > static_cast<std::uint32_t>(INT_MAX) ||
        matches_wide > scans_wide)
    {
        atomicExch(&tail_state->overflow, 1);
        return;
    }
    const int scans = static_cast<int>(scans_wide);
    const int matches = static_cast<int>(matches_wide);
    const int chunks = dmma_tail_chunk_count_from_counts(
        scans, matches, scan_cost, match_cost, admission_threshold,
        chunk_target, max_chunks);
    if (chunks <= 0)
        return;
    const int slot = dmma_tail_try_reserve_record(tail_state, tail_capacity);
    if (slot < 0)
        return;
    DmmaTailRecord record;
    record.id = task;
    record.scans = scans;
    record.matches = matches;
    record.chunks = chunks;
    tail_records[slot] = record;
}

/* Source-compatible coupled-threshold form retained for host tests and
 * callers that do not need the admission/chunk-target ablation. */
static __host__ __device__ __forceinline__ int
dmma_tail_chunk_count_from_counts(
    int scans, int matches, float scan_cost, float match_cost,
    float threshold, int max_chunks)
{
    return dmma_tail_chunk_count_from_counts(
        scans, matches, scan_cost, match_cost, threshold, threshold,
        max_chunks);
}

static constexpr uint32_t DMMA_PACKED_NNZ_BITS = 7u;
static constexpr uint32_t DMMA_PACKED_NNZ_MASK =
    (uint32_t(1) << DMMA_PACKED_NNZ_BITS) - 1u;
/* Keep bit 31 clear so packed values remain nonnegative signed ints. */
static constexpr uint32_t DMMA_PACKED_LOAD_BITS = 24u;
static constexpr uint32_t DMMA_PACKED_LOAD_MAX =
    (uint32_t(1) << DMMA_PACKED_LOAD_BITS) - 1u;

static __host__ __device__ __forceinline__ uint32_t
dmma_quantize_symbolic_load(float work, float quantum)
{
    if (!(work > 0.0f) || !(quantum > 0.0f))
        return 0;
    const float scaled = work / quantum;
    if (!(scaled < static_cast<float>(DMMA_PACKED_LOAD_MAX)))
        return DMMA_PACKED_LOAD_MAX;
    const float rounded = ceilf(scaled);
    return rounded > 0.0f ? static_cast<uint32_t>(rounded) : 0u;
}

static __host__ __device__ __forceinline__ int
dmma_pack_candidate_nnz_load(int nnz, uint32_t load_units)
{
    const uint32_t safe_nnz =
        static_cast<uint32_t>(nnz) & DMMA_PACKED_NNZ_MASK;
    const uint32_t safe_load =
        load_units > DMMA_PACKED_LOAD_MAX ? DMMA_PACKED_LOAD_MAX : load_units;
    return static_cast<int>((safe_load << DMMA_PACKED_NNZ_BITS) | safe_nnz);
}

static __host__ __device__ __forceinline__ int
dmma_unpack_candidate_nnz(int packed)
{
    return static_cast<int>(static_cast<uint32_t>(packed) &
                            DMMA_PACKED_NNZ_MASK);
}

static __host__ __device__ __forceinline__ uint32_t
dmma_unpack_candidate_load(int packed)
{
    return (static_cast<uint32_t>(packed) >> DMMA_PACKED_NNZ_BITS) &
           DMMA_PACKED_LOAD_MAX;
}

/* Load-aware exact symbolic is a dedicated opt-in kernel.  Direct, static,
 * and manually sized suffix schedules keep using dmma_exact_mask_kernel when
 * telemetry is disabled, so their exact-mask register footprint is unchanged.
 * The existing candidate_nnz word temporarily carries both nnz and work; the
 * compact pass restores plain nnz before the offset scan. */
__global__ void dmma_exact_mask_packed_load_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int candidate_count,
    const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols, uint64_t *candidate_masks,
    int *candidate_nnz, int *candidate_keep, float intercept_cost,
    float scan_cost, float match_cost, float output_cost,
    float load_quantum)
{
    const std::size_t candidate =
        dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (candidate >= static_cast<std::size_t>(candidate_count))
        return;

    const int tile_row = candidate_rows[candidate];
    const int tile_col = candidate_cols[candidate];
    int scan_steps = 0;
    int matches = 0;
    const unsigned long long output_mask =
        dmma_candidate_exact_mask_with_counts(
            a, b, tile_row, tile_col, lane, &scan_steps, &matches);
    if (lane != 0)
        return;

    const int nnz = __popcll(output_mask);
    const float work = intercept_cost +
                       scan_cost * static_cast<float>(scan_steps) +
                       match_cost * static_cast<float>(matches) +
                       output_cost * static_cast<float>(nnz);
    candidate_masks[candidate] = output_mask;
    candidate_nnz[candidate] = dmma_pack_candidate_nnz_load(
        nnz, dmma_quantize_symbolic_load(work, load_quantum));
    candidate_keep[candidate] = nnz != 0;
}

/* Safe stage-one test.  For two sorted K-tile lists, the merge loop performs
 * at most lenA+lenB-1 iterations and at most min(lenA,lenB) matches.  With
 * nonnegative fitted coefficients this upper work cannot reject a truly
 * heavy task.  A false positive is resolved by the exact-count stage. */
static __host__ __device__ __forceinline__ bool
dmma_tail_maybe_from_lengths(
    int len_a, int len_b, float scan_cost, float match_cost,
    float threshold)
{
    if (len_a <= 0 || len_b <= 0)
        return false;
    const long long upper_scans =
        static_cast<long long>(len_a) + static_cast<long long>(len_b) - 1;
    const int upper_matches = len_a < len_b ? len_a : len_b;
    const float upper_work = dmma_tail_fp32_work_from_counts(
        upper_scans, upper_matches, scan_cost, match_cost);
    return upper_scans > 1 && upper_work > threshold;
}

/* One cheap thread per materialized candidate.  IDs are appended into a
 * bounded channel independent of the final 16-byte TailRecord channel. */
__global__ void dmma_filter_tail_maybe_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int candidate_count,
    const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols, float scan_cost,
    float match_cost, float threshold, int *__restrict__ maybe_ids,
    int maybe_capacity, DmmaTailAppendState *__restrict__ maybe_state)
{
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t candidate = dmma_global_thread_index();
         candidate < static_cast<std::size_t>(candidate_count);
         candidate += stride)
    {
        const int tile_row = candidate_rows[candidate];
        const int tile_col = candidate_cols[candidate];
        const int len_a = a.tile_row_ptr[tile_row + 1] -
                          a.tile_row_ptr[tile_row];
        const int len_b = b.tile_col_ptr[tile_col + 1] -
                          b.tile_col_ptr[tile_col];
        if (dmma_tail_maybe_from_lengths(
                len_a, len_b, scan_cost, match_cost, threshold))
        {
            const int slot =
                dmma_tail_try_reserve_record(maybe_state, maybe_capacity);
            if (slot >= 0)
                maybe_ids[slot] = static_cast<int>(candidate);
        }
    }
}

/* Production unified-light admission is sparse.  Run this inexpensive pass
 * after the keep scan, when a retained candidate's final output ID is known.
 * The joint list is the union of two independently safe upper bounds:
 *
 *   (1) splittable heavy work, and
 *   (2) full fine-queue work, including beta_0 and beta_output * 64.
 *
 * No ordering between fine_threshold and split_threshold is assumed.  The
 * fixed terminal range is owned implicitly by the numeric scheduler, so an
 * ordinary terminal task is not appended; a terminal task whose heavy upper
 * bound fires is still retained so chunk admission remains complete. */
__global__ void dmma_filter_joint_maybe_output_window_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int candidate_count,
    const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols,
    const uint64_t *__restrict__ candidate_masks,
    const int *__restrict__ candidate_positions, int suffix_begin,
    int terminal_begin, float intercept_cost, float scan_cost,
    float match_cost, float output_cost, float split_threshold,
    float fine_threshold, int *__restrict__ maybe_ids, int maybe_capacity,
    DmmaTailAppendState *__restrict__ maybe_state)
{
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t candidate = dmma_global_thread_index();
         candidate < static_cast<std::size_t>(candidate_count);
         candidate += stride)
    {
        if (candidate_masks[candidate] == 0)
            continue;
        const int output = candidate_positions[candidate];
        const int tile_row = candidate_rows[candidate];
        const int tile_col = candidate_cols[candidate];
        const int len_a = a.tile_row_ptr[tile_row + 1] -
                          a.tile_row_ptr[tile_row];
        const int len_b = b.tile_col_ptr[tile_col + 1] -
                          b.tile_col_ptr[tile_col];
        const bool heavy_upper = dmma_tail_maybe_from_lengths(
            len_a, len_b, scan_cost, match_cost, split_threshold);
        bool fine_upper = false;
        if (output >= suffix_begin && output < terminal_begin &&
            len_a > 0 && len_b > 0)
        {
            const long long upper_scans =
                static_cast<long long>(len_a) +
                static_cast<long long>(len_b) - 1;
            const int upper_matches = len_a < len_b ? len_a : len_b;
            const float upper_work =
                intercept_cost +
                dmma_tail_fp32_work_from_counts(
                    upper_scans, upper_matches, scan_cost, match_cost) +
                output_cost * static_cast<float>(DMMA_OUTPUT_ELEMS);
            fine_upper = upper_work >= fine_threshold;
        }
        if (heavy_upper || fine_upper)
        {
            const int slot =
                dmma_tail_try_reserve_record(maybe_state, maybe_capacity);
            if (slot >= 0)
                maybe_ids[slot] = static_cast<int>(candidate);
        }
    }
}

struct DmmaUnifiedReplaySummary
{
    unsigned long long sparse_fine_units = 0;
    unsigned long long terminal_units = 0;
    unsigned long long heavy_units = 0;
    unsigned long long heavy_fine_units = 0;
    int sparse_fine_count = 0;
    int heavy_fine_count = 0;
    int fine_overflow = 0;
    int saturated_task_count = 0;
};

/* Exact replay over only the bounded joint maybe population.  A candidate may
 * independently append a non-terminal fine output ID and/or a heavy record;
 * overlap is intentional and later resolved by the admitted-heavy bitset.
 * Every write names final compact output space, so no candidate-ID map pass is
 * required. */
__global__ void dmma_replay_joint_maybe_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int maybe_count,
    const int *__restrict__ maybe_ids,
    const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols,
    const uint64_t *__restrict__ candidate_masks,
    const int *__restrict__ candidate_positions, int suffix_begin,
    int terminal_begin, float intercept_cost, float scan_cost,
    float match_cost, float output_cost, float split_threshold,
    float fine_threshold, float chunk_target, int max_chunks,
    float load_quantum, int *__restrict__ fine_task_ids, int fine_capacity,
    uint32_t *__restrict__ fine_flags,
    DmmaUnifiedReplaySummary *__restrict__ replay_summary,
    DmmaTailRecord *__restrict__ tail_records, int tail_capacity,
    DmmaTailAppendState *__restrict__ tail_state)
{
    const std::size_t maybe = dmma_global_thread_index();
    if (maybe >= static_cast<std::size_t>(maybe_count))
        return;

    const int candidate = maybe_ids[maybe];
    const uint64_t output_mask = candidate_masks[candidate];
    if (output_mask == 0)
        return;
    const int output = candidate_positions[candidate];
    const int tile_row = candidate_rows[candidate];
    const int tile_col = candidate_cols[candidate];
    int pa = a.tile_row_ptr[tile_row];
    const int a_begin = pa;
    const int a_end = a.tile_row_ptr[tile_row + 1];
    int pb = b.tile_col_ptr[tile_col];
    const int b_begin = pb;
    const int b_end = b.tile_col_ptr[tile_col + 1];
    int matches = 0;
    while (pa < a_end && pb < b_end)
    {
        const int ka = a.tile_col_idx[pa];
        const int kb = b.tile_row_idx[pb];
        if (ka == kb)
        {
            if (matches < INT_MAX)
                ++matches;
            ++pa;
            ++pb;
        }
        else if (ka < kb)
            ++pa;
        else
            ++pb;
    }
    const int scans = dmma_merge_scans_from_advances(
        pa - a_begin, pb - b_begin, matches);
    const int nnz = __popcll(output_mask);
    const float splittable_work = dmma_tail_fp32_work_from_counts(
        static_cast<long long>(scans), matches, scan_cost, match_cost);
    const float full_work = intercept_cost + splittable_work +
                            output_cost * static_cast<float>(nnz);
    const uint32_t units =
        dmma_quantize_symbolic_load(full_work, load_quantum);
    const bool terminal = output >= terminal_begin;
    const bool sparse_fine = output >= suffix_begin && !terminal &&
                             full_work >= fine_threshold;
    const int chunks = dmma_tail_chunk_count_from_counts(
        scans, matches, scan_cost, match_cost, split_threshold, chunk_target,
        max_chunks);
    const bool heavy = chunks > 0;

    if (sparse_fine)
    {
        const int slot = atomicAdd(&replay_summary->sparse_fine_count, 1);
        atomicAdd(&replay_summary->sparse_fine_units,
                  static_cast<unsigned long long>(units));
        if (slot >= 0 && slot < fine_capacity)
        {
            fine_task_ids[slot] = output;
            atomicOr(fine_flags +
                         (static_cast<unsigned int>(output) >> 5),
                     1u << (static_cast<unsigned int>(output) & 31u));
        }
        else
        {
            atomicExch(&replay_summary->fine_overflow, 1);
        }
    }
    const bool potentially_fine = sparse_fine || terminal;
    if (heavy)
    {
        atomicAdd(&replay_summary->heavy_units,
                  static_cast<unsigned long long>(units));
        if (potentially_fine)
        {
            atomicAdd(&replay_summary->heavy_fine_count, 1);
            atomicAdd(&replay_summary->heavy_fine_units,
                      static_cast<unsigned long long>(units));
        }
        const int slot =
            dmma_tail_try_reserve_record(tail_state, tail_capacity);
        if (slot >= 0)
        {
            DmmaTailRecord record;
            record.id = output;
            record.scans = scans;
            record.matches = matches;
            record.chunks = chunks;
            tail_records[slot] = record;
        }
    }
    /* Terminal tasks are summarized by the bounded terminal kernel below;
     * avoiding the increment here prevents double-counting terminal-heavy
     * saturation. */
    if (!terminal && units == DMMA_PACKED_LOAD_MAX)
        atomicAdd(&replay_summary->saturated_task_count, 1);
}

/* The terminal queue is intentionally implicit rather than materialized in
 * the joint maybe/fine channels.  It is small (a fixed task window), so one
 * warp per terminal output can recover exact model telemetry without a dense
 * candidate-sized array. */
__global__ void dmma_summarize_unified_terminal_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int output_tile_count,
    int terminal_begin, const int *__restrict__ output_rows,
    const int *__restrict__ output_cols,
    const uint64_t *__restrict__ output_masks, float intercept_cost,
    float scan_cost, float match_cost, float output_cost, float load_quantum,
    DmmaUnifiedReplaySummary *__restrict__ replay_summary)
{
    const int local = static_cast<int>(dmma_global_thread_index() / WARP_SIZE);
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int task = terminal_begin + local;
    if (task >= output_tile_count)
        return;
    const int tile_row = output_rows[task];
    const int tile_col = output_cols[task];
    int pa = a.tile_row_ptr[tile_row];
    const int a_begin = pa;
    const int a_end = a.tile_row_ptr[tile_row + 1];
    int pb = b.tile_col_ptr[tile_col];
    const int b_begin = pb;
    const int b_end = b.tile_col_ptr[tile_col + 1];
    int matches = 0;
    if (lane == 0)
    {
        while (pa < a_end && pb < b_end)
        {
            const int ka = a.tile_col_idx[pa];
            const int kb = b.tile_row_idx[pb];
            if (ka == kb)
            {
                if (matches < INT_MAX)
                    ++matches;
                ++pa;
                ++pb;
            }
            else if (ka < kb)
                ++pa;
            else
                ++pb;
        }
        const int scans = dmma_merge_scans_from_advances(
            pa - a_begin, pb - b_begin, matches);
        const float full_work =
            intercept_cost +
            dmma_tail_fp32_work_from_counts(
                static_cast<long long>(scans), matches, scan_cost,
                match_cost) +
            output_cost * static_cast<float>(__popcll(output_masks[task]));
        const uint32_t units =
            dmma_quantize_symbolic_load(full_work, load_quantum);
        atomicAdd(&replay_summary->terminal_units,
                  static_cast<unsigned long long>(units));
        if (units == DMMA_PACKED_LOAD_MAX)
            atomicAdd(&replay_summary->saturated_task_count, 1);
    }
}

/* Replay only maybe IDs that survive exact symbolic and belong to the final
 * output scheduling window.  candidate_positions is the in-place exclusive
 * scan of candidate_keep, so structural presence must be read from the mask,
 * not from candidate_positions.  Records are born in final-output ID space;
 * the separate path therefore needs neither a map pass nor a window buffer. */
__global__ void dmma_count_tail_maybe_output_window_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int maybe_count,
    const int *__restrict__ maybe_ids,
    const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols,
    const uint64_t *__restrict__ candidate_masks,
    const int *__restrict__ candidate_positions, int output_begin,
    float scan_cost, float match_cost, float admission_threshold,
    float chunk_target, int max_chunks,
    DmmaTailRecord *__restrict__ tail_records, int tail_capacity,
    DmmaTailAppendState *__restrict__ tail_state)
{
    const std::size_t maybe = dmma_global_thread_index();
    if (maybe >= static_cast<std::size_t>(maybe_count))
        return;

    const int candidate = maybe_ids[maybe];
    if (candidate_masks[candidate] == 0)
        return;
    const int output = candidate_positions[candidate];
    if (output < output_begin)
        return;
    const int tile_row = candidate_rows[candidate];
    const int tile_col = candidate_cols[candidate];
    int pa = a.tile_row_ptr[tile_row];
    const int a_end = a.tile_row_ptr[tile_row + 1];
    int pb = b.tile_col_ptr[tile_col];
    const int b_end = b.tile_col_ptr[tile_col + 1];
    int scan_steps = 0;
    int matches = 0;
    while (pa < a_end && pb < b_end)
    {
        if (scan_steps < INT_MAX)
            ++scan_steps;
        const int ka = a.tile_col_idx[pa];
        const int kb = b.tile_row_idx[pb];
        if (ka == kb)
        {
            ++matches;
            ++pa;
            ++pb;
        }
        else if (ka < kb)
            ++pa;
        else
            ++pb;
    }
    const int chunks = dmma_tail_chunk_count_from_counts(
        scan_steps, matches, scan_cost, match_cost, admission_threshold,
        chunk_target, max_chunks);
    if (chunks > 0)
    {
        const int slot =
            dmma_tail_try_reserve_record(tail_state, tail_capacity);
        if (slot >= 0)
        {
            DmmaTailRecord record;
            record.id = output;
            record.scans = scan_steps;
            record.matches = matches;
            record.chunks = chunks;
            tail_records[slot] = record;
        }
    }
}

/* Experimental single-pass exact symbolic.  The upper-bound predicate is
 * evaluated inside the warp that already owns the candidate; the same exact
 * merge used to construct the structural mask supplies scans/matches, so no
 * maybe-ID storage or replay pass is required.  maybe_state has logical
 * capacity=candidate_count and records only the exact maybe population. */
__global__ void dmma_exact_mask_fused_tail_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int candidate_count,
    const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols, uint64_t *candidate_masks,
    int *candidate_nnz, int *candidate_keep, float intercept_cost,
    float scan_cost, float match_cost, float output_cost,
    float admission_threshold, float chunk_target, int max_chunks,
    float load_quantum, int pack_load,
    DmmaTailAppendState *__restrict__ maybe_state, int count_maybe,
    DmmaTailRecord *__restrict__ tail_records, int tail_capacity,
    DmmaTailAppendState *__restrict__ tail_state)
{
    const std::size_t candidate =
        dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (candidate >= static_cast<std::size_t>(candidate_count))
        return;

    const int tile_row = candidate_rows[candidate];
    const int tile_col = candidate_cols[candidate];
    const int len_a = a.tile_row_ptr[tile_row + 1] -
                      a.tile_row_ptr[tile_row];
    const int len_b = b.tile_col_ptr[tile_col + 1] -
                      b.tile_col_ptr[tile_col];
    const bool maybe = dmma_tail_maybe_from_lengths(
        len_a, len_b, scan_cost, match_cost, admission_threshold);
    int scan_steps = 0;
    int matches = 0;
    const unsigned long long output_mask =
        dmma_candidate_exact_mask_with_counts(
            a, b, tile_row, tile_col, lane, &scan_steps, &matches);
    if (lane != 0)
        return;

    const int nnz = __popcll(output_mask);
    candidate_masks[candidate] = output_mask;
    candidate_keep[candidate] = nnz != 0;
    if (candidate_nnz != nullptr && pack_load != 0)
    {
        const float work = intercept_cost +
                           scan_cost * static_cast<float>(scan_steps) +
                           match_cost * static_cast<float>(matches) +
                           output_cost * static_cast<float>(nnz);
        candidate_nnz[candidate] = dmma_pack_candidate_nnz_load(
            nnz, dmma_quantize_symbolic_load(work, load_quantum));
    }
    else if (candidate_nnz != nullptr)
    {
        candidate_nnz[candidate] = nnz;
    }

    if (!maybe)
        return;
    /* tile-tail-queue already performs the exact counted merge for every
     * candidate.  It has no use for the loose maybe population, whose global
     * atomic counter can be pathological on dense products such as
     * human_gene2.  Legacy fused experiments retain that telemetry. */
    if (count_maybe != 0)
        (void)dmma_tail_try_reserve_record(maybe_state, candidate_count);
    if (nnz == 0)
        return;
    const int chunks = dmma_tail_chunk_count_from_counts(
        scan_steps, matches, scan_cost, match_cost, admission_threshold,
        chunk_target, max_chunks);
    if (chunks <= 0)
        return;
    const int slot =
        dmma_tail_try_reserve_record(tail_state, tail_capacity);
    if (slot >= 0)
    {
        DmmaTailRecord record;
        record.id = static_cast<int>(candidate);
        record.scans = scan_steps;
        record.matches = matches;
        record.chunks = chunks;
        tail_records[slot] = record;
    }
}

__global__ void dmma_map_tail_records_to_outputs_kernel(
    int tail_count, const int *__restrict__ candidate_positions,
    DmmaTailRecord *__restrict__ tail_records)
{
    const int tail = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tail >= tail_count)
        return;
    DmmaTailRecord record = tail_records[tail];
    record.id = candidate_positions[record.id];
    tail_records[tail] = record;
}

/* q is meaningful only after the exact keep scan has assigned compact output
 * IDs.  Read and write arrays must not alias: append-order compaction can
 * otherwise overwrite an input record that another thread has not read. */
__global__ void dmma_filter_tail_output_suffix_kernel(
    int tail_count, int suffix_begin,
    const DmmaTailRecord *__restrict__ input_records,
    DmmaTailRecord *__restrict__ output_records,
    DmmaTailAppendState *__restrict__ output_state)
{
    const int tail = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tail >= tail_count)
        return;
    const DmmaTailRecord record = input_records[tail];
    if (record.id < suffix_begin)
        return;
    const int slot = dmma_tail_try_reserve_record(output_state, tail_count);
    if (slot >= 0)
        output_records[slot] = record;
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

static __host__ __device__ __forceinline__ int
dmma_exact_popcount_u32(uint32_t value)
{
#ifdef __CUDA_ARCH__
    return __popc(value);
#else
    return __builtin_popcount(value);
#endif
}

static __host__ __device__ __forceinline__ int
dmma_exact_ffs_u32(uint32_t value)
{
#ifdef __CUDA_ARCH__
    return __ffs(value);
#else
    return value == 0 ? 0 : __builtin_ctz(value) + 1;
#endif
}

/* Form the exact Boolean product of one 8x4 A mask and one 4x8 B
 * mask. B's physical mask is column-major (four k bits per output column),
 * matching dmma_candidate_exact_mask above.  Host availability lets the
 * contract test compare this exact forward primitive against a scalar
 * element-by-element oracle without launching a GPU. */
static __host__ __device__ __forceinline__ unsigned long long
dmma_tile_pair_exact_mask(
    uint32_t mask_a, uint32_t mask_b)
{
    unsigned long long output = 0;
    if (dmma_exact_popcount_u32(mask_a) *
            dmma_exact_popcount_u32(mask_b) <=
        DMMA_OUTPUT_ELEMS)
    {
        uint32_t remaining_a = mask_a;
        while (remaining_a != 0)
        {
            const int position_a = dmma_exact_ffs_u32(remaining_a) - 1;
            const int row = position_a / DMMA_TILE_K;
            const int k = position_a % DMMA_TILE_K;
            uint32_t matching_b =
                mask_b & (uint32_t(0x11111111u) << k);
            while (matching_b != 0)
            {
                const int position_b = dmma_exact_ffs_u32(matching_b) - 1;
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
        unsigned int output_columns = 0;
#pragma unroll
        for (int col = 0; col < DMMA_TILE_N; ++col)
        {
            const int output_position = row * DMMA_TILE_N + col;
            output_columns |= dmma_tile_pair_output_present(
                                  mask_a, mask_b, output_position)
                                  ? 1u << col
                                  : 0u;
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

/* Forward side of the ordinary-candidate row partition.  One CTA owns one
 * selected C tile row, builds a full exact mask SPA for that row, then writes
 * only the already-materialised candidate interval of the same row.  Because
 * candidate_row_ptr is contiguous and the ordinary kernel rejects the same
 * row flag, each candidate mask/keep pair has exactly one writer. */
__global__ void dmma_exact_forward_spa_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b,
    const int *__restrict__ forward_rows, int batch_row_begin,
    int batch_row_count, int output_tile_columns,
    unsigned long long *__restrict__ mask_spas,
    const int *__restrict__ candidate_row_ptr,
    const int *__restrict__ candidate_cols,
    uint64_t *__restrict__ candidate_masks,
    int *__restrict__ candidate_keep,
    DmmaExactForwardSpaDeviceSummary *__restrict__ summary)
{
    const int batch_row = static_cast<int>(blockIdx.x);
    if (batch_row >= batch_row_count)
        return;
    const int row = forward_rows[batch_row_begin + batch_row];
    unsigned long long *masks =
        mask_spas + static_cast<std::size_t>(batch_row) *
                        output_tile_columns;
    dmma_exact_mask_spa_clear_and_mark(a, b, row, output_tile_columns,
                                       masks);

    const int candidate_begin = candidate_row_ptr[row];
    const int candidate_end = candidate_row_ptr[row + 1];
    for (int candidate = candidate_begin + threadIdx.x;
         candidate < candidate_end; candidate += blockDim.x)
    {
        const int col = candidate_cols[candidate];
        const unsigned long long mask = masks[col];
        candidate_masks[candidate] = mask;
        candidate_keep[candidate] = mask != 0;
    }
    if (threadIdx.x == 0)
        dmma_atomic_saturating_add_u64(
            &summary->forward_candidates,
            static_cast<unsigned long long>(candidate_end -
                                             candidate_begin));
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
    output_nnz[output] = __popcll(candidate_masks[candidate]);
}

struct DmmaSymbolicLoadSummary
{
    unsigned long long total_units = 0;
    unsigned long long suffix_units = 0;
    unsigned long long fine_units = 0;
    unsigned long long split_units = 0;
    unsigned long long split_suffix_units = 0;
    unsigned long long split_fine_units = 0;
    unsigned long long critical_units = 0;
    unsigned long long critical_tail_units = 0;
    unsigned int max_task_units = 0;
    unsigned int critical_max_task_units = 0;
    int task_count = 0;
    int suffix_task_count = 0;
    int fine_task_count = 0;
    int split_fine_task_count = 0;
    int fine_overflow = 0;
    int critical_task_count = 0;
    int critical_tail_task_count = 0;
    int saturated_task_count = 0;
};

/* Consume packed candidate_nnz in the compact pass so per-C-tile workload
 * metadata costs no dense allocation and no additional candidate traversal.
 * Only this optional kernel understands packed values; the production
 * compact kernel above remains byte-for-byte independent of the experiment. */
__global__ void dmma_compact_candidates_with_load_kernel(
    int candidate_count, const int *__restrict__ candidate_rows,
    const int *__restrict__ candidate_cols,
    const uint64_t *__restrict__ candidate_masks,
    const int *__restrict__ packed_candidate_nnz,
    const int *__restrict__ candidate_positions, int suffix_task_begin,
    int critical_task_begin, int fine_suffix_begin, int fine_terminal_begin,
    uint32_t fine_threshold_units, int *__restrict__ fine_task_ids,
    int fine_capacity, uint32_t *__restrict__ fine_flags,
    int *output_rows, int *output_cols,
    uint64_t *output_masks,
    int *output_nnz, DmmaSymbolicLoadSummary *__restrict__ summary)
{
    const std::size_t candidate = dmma_global_thread_index();
    const bool kept = candidate < static_cast<std::size_t>(candidate_count) &&
                      candidate_masks[candidate] != 0;
    int output = 0;
    unsigned int work = 0;
    if (kept)
    {
        output = candidate_positions[candidate];
        const int packed = packed_candidate_nnz[candidate];
        const int nnz = dmma_unpack_candidate_nnz(packed);
        work = dmma_unpack_candidate_load(packed);
        output_rows[output] = candidate_rows[candidate];
        output_cols[output] = candidate_cols[candidate];
        output_masks[output] = candidate_masks[candidate];
        output_nnz[output] = nnz;

        if (fine_task_ids != nullptr && fine_flags != nullptr &&
            dmma_unified_is_fine_task(
                output, fine_suffix_begin, fine_terminal_begin, work,
                fine_threshold_units))
        {
            const int slot = atomicAdd(&summary->fine_task_count, 1);
            atomicAdd(&summary->fine_units,
                      static_cast<unsigned long long>(work));
            if (slot >= 0 && slot < fine_capacity)
            {
                fine_task_ids[slot] = output;
                atomicOr(fine_flags +
                             (static_cast<unsigned int>(output) >> 5),
                         1u << (static_cast<unsigned int>(output) & 31u));
            }
            else
            {
                atomicExch(&summary->fine_overflow, 1);
            }
        }
    }

    const bool suffix = kept && output >= suffix_task_begin;
    const bool critical = kept && output >= critical_task_begin;
    unsigned long long total_units = kept ? work : 0u;
    unsigned long long suffix_units = suffix ? work : 0u;
    unsigned long long critical_units = critical ? work : 0u;
    unsigned int max_units = kept ? work : 0u;
    unsigned int critical_max_units = critical ? work : 0u;
    int task_count = kept ? 1 : 0;
    int suffix_count = suffix ? 1 : 0;
    int critical_count = critical ? 1 : 0;
    int saturated_count =
        kept && work == DMMA_PACKED_LOAD_MAX ? 1 : 0;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        total_units += __shfl_down_sync(0xffffffffu, total_units, offset);
        suffix_units +=
            __shfl_down_sync(0xffffffffu, suffix_units, offset);
        critical_units +=
            __shfl_down_sync(0xffffffffu, critical_units, offset);
        max_units = max(
            max_units,
            __shfl_down_sync(0xffffffffu, max_units, offset));
        critical_max_units = max(
            critical_max_units,
            __shfl_down_sync(0xffffffffu, critical_max_units, offset));
        task_count += __shfl_down_sync(0xffffffffu, task_count, offset);
        suffix_count +=
            __shfl_down_sync(0xffffffffu, suffix_count, offset);
        critical_count +=
            __shfl_down_sync(0xffffffffu, critical_count, offset);
        saturated_count +=
            __shfl_down_sync(0xffffffffu, saturated_count, offset);
    }

    static constexpr int kMaxCompactWarps =
        DMMA_WIDE_BLOCK_THREADS / WARP_SIZE;
    __shared__ unsigned long long warp_total[kMaxCompactWarps];
    __shared__ unsigned long long warp_suffix[kMaxCompactWarps];
    __shared__ unsigned long long warp_critical[kMaxCompactWarps];
    __shared__ unsigned int warp_max[kMaxCompactWarps];
    __shared__ unsigned int warp_critical_max[kMaxCompactWarps];
    __shared__ int warp_tasks[kMaxCompactWarps];
    __shared__ int warp_suffix_tasks[kMaxCompactWarps];
    __shared__ int warp_critical_tasks[kMaxCompactWarps];
    __shared__ int warp_saturated[kMaxCompactWarps];
    if (lane == 0)
    {
        warp_total[warp] = total_units;
        warp_suffix[warp] = suffix_units;
        warp_critical[warp] = critical_units;
        warp_max[warp] = max_units;
        warp_critical_max[warp] = critical_max_units;
        warp_tasks[warp] = task_count;
        warp_suffix_tasks[warp] = suffix_count;
        warp_critical_tasks[warp] = critical_count;
        warp_saturated[warp] = saturated_count;
    }
    __syncthreads();

    if (warp == 0)
    {
        const int warp_count = blockDim.x / WARP_SIZE;
        total_units = lane < warp_count ? warp_total[lane] : 0u;
        suffix_units = lane < warp_count ? warp_suffix[lane] : 0u;
        critical_units = lane < warp_count ? warp_critical[lane] : 0u;
        max_units = lane < warp_count ? warp_max[lane] : 0u;
        critical_max_units =
            lane < warp_count ? warp_critical_max[lane] : 0u;
        task_count = lane < warp_count ? warp_tasks[lane] : 0;
        suffix_count =
            lane < warp_count ? warp_suffix_tasks[lane] : 0;
        critical_count =
            lane < warp_count ? warp_critical_tasks[lane] : 0;
        saturated_count =
            lane < warp_count ? warp_saturated[lane] : 0;
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        {
            total_units +=
                __shfl_down_sync(0xffffffffu, total_units, offset);
            suffix_units +=
                __shfl_down_sync(0xffffffffu, suffix_units, offset);
            critical_units +=
                __shfl_down_sync(0xffffffffu, critical_units, offset);
            max_units = max(
                max_units,
                __shfl_down_sync(0xffffffffu, max_units, offset));
            critical_max_units = max(
                critical_max_units,
                __shfl_down_sync(0xffffffffu, critical_max_units, offset));
            task_count +=
                __shfl_down_sync(0xffffffffu, task_count, offset);
            suffix_count +=
                __shfl_down_sync(0xffffffffu, suffix_count, offset);
            critical_count +=
                __shfl_down_sync(0xffffffffu, critical_count, offset);
            saturated_count +=
                __shfl_down_sync(0xffffffffu, saturated_count, offset);
        }
        if (lane == 0)
        {
            if (total_units != 0)
                atomicAdd(&summary->total_units, total_units);
            if (suffix_units != 0)
                atomicAdd(&summary->suffix_units, suffix_units);
            if (critical_units != 0)
                atomicAdd(&summary->critical_units, critical_units);
            atomicMax(&summary->max_task_units, max_units);
            atomicMax(&summary->critical_max_task_units,
                      critical_max_units);
            atomicAdd(&summary->task_count, task_count);
            atomicAdd(&summary->suffix_task_count, suffix_count);
            atomicAdd(&summary->critical_task_count, critical_count);
            atomicAdd(&summary->saturated_task_count, saturated_count);
        }
    }
}

/* Tail records are sparse and have already been mapped from candidate IDs to
 * output-task IDs.  This small pass measures how much exact splittable work
 * actually lies in the predicted endpoint window. */
__global__ void dmma_summarize_critical_tail_kernel(
    int tail_count, const DmmaTailRecord *__restrict__ tail_records,
    int output_task_count, int suffix_task_begin, int critical_task_begin,
    const int *__restrict__ output_nnz, float intercept_cost,
    float scan_cost, float match_cost, float output_cost,
    float load_quantum, const uint32_t *__restrict__ fine_flags,
    DmmaSymbolicLoadSummary *__restrict__ summary)
{
    const int tail = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tail >= tail_count)
        return;
    const DmmaTailRecord record = tail_records[tail];
    if (record.id < 0 || record.id >= output_task_count)
        return;
    const float splittable_work = dmma_tail_fp32_work_from_counts(
        static_cast<long long>(record.scans), record.matches,
        scan_cost, match_cost);
    const float full_work =
        intercept_cost + splittable_work +
        output_cost * static_cast<float>(output_nnz[record.id]);
    const unsigned int full_units =
        dmma_quantize_symbolic_load(full_work, load_quantum);
    atomicAdd(&summary->split_units,
              static_cast<unsigned long long>(full_units));
    if (fine_flags != nullptr &&
        (fine_flags[static_cast<unsigned int>(record.id) >> 5] &
         (1u << (static_cast<unsigned int>(record.id) & 31u))) != 0)
    {
        atomicAdd(&summary->split_fine_units,
                  static_cast<unsigned long long>(full_units));
        atomicAdd(&summary->split_fine_task_count, 1);
    }
    if (record.id >= suffix_task_begin)
    {
        atomicAdd(&summary->split_suffix_units,
                  static_cast<unsigned long long>(full_units));
    }
    if (record.id >= critical_task_begin)
    {
        const unsigned int units =
            dmma_quantize_symbolic_load(splittable_work, load_quantum);
        atomicAdd(&summary->critical_tail_units,
                  static_cast<unsigned long long>(units));
        atomicAdd(&summary->critical_tail_task_count, 1);
    }
}

static __host__ __device__ __forceinline__ int
dmma_output_row_boundary_from_candidate_prefix(
    int candidate_boundary, const int *candidate_positions)
{
    return candidate_positions[candidate_boundary];
}

/* Candidate IDs are row-major contiguous.  The keep exclusive prefix at the
 * first candidate of row r is therefore exactly C.row_ptr[r].  Transform the
 * candidate row pointer in place, including its sentinel, and avoid an
 * output-sized atomic histogram plus a second scan/allocation. */
__global__ void dmma_gather_output_row_ptr_kernel(
    int tile_row_count, int *__restrict__ candidate_row_ptr,
    const int *__restrict__ candidate_positions)
{
    const std::size_t row = dmma_global_thread_index();
    if (row <= static_cast<std::size_t>(tile_row_count))
    {
        const int candidate_boundary = candidate_row_ptr[row];
        candidate_row_ptr[row] =
            dmma_output_row_boundary_from_candidate_prefix(
                candidate_boundary, candidate_positions);
    }
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

struct DmmaChunkDescriptor
{
    int task = 0;
    int pa_begin = 0;
    int pb_begin = 0;
    int merge_steps = 0;
};

__device__ __forceinline__ uint32_t dmma_schedule_smid()
{
    uint32_t value = 0;
    asm volatile("mov.u32 %0, %smid;" : "=r"(value));
    return value;
}

struct DmmaTaskTraceRecord
{
    uint64_t start_ns = 0;
    uint64_t end_ns = 0;
    uint32_t sm_id = 0;
    int task_id = 0;
    int scan_steps = 0;
    int matches = 0;
    int output_nnz = 0;
};

__device__ __forceinline__ uint64_t dmma_task_trace_globaltimer_ns()
{
    /* PTX %globaltimer is device-wide and expressed in nanoseconds. */
    uint64_t value = 0;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(value) :: "memory");
    return value;
}

/*
 * Untimed profiling replay for direct C-tile tasks.  The production numeric
 * kernel is intentionally left untouched: this kernel is launched only when
 * task_trace_path is non-null, after the Core stopwatch has stopped.  The
 * complete grid repeats the same merge, DMMA, and compressed-output stores as
 * dmma_numeric_kernel so sampling does not change scheduler pressure.  Only
 * selected warps emit records, retaining their original task IDs.
 */
__global__ void dmma_direct_task_trace_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int output_tile_count,
    const int *__restrict__ output_rows,
    const int *__restrict__ output_cols,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    unsigned int sample_stride, unsigned int sample_phase,
    int sampled_task_count, unsigned char *__restrict__ trace_row_ptr,
    unsigned char *__restrict__ trace_col_idx,
    MAT_VAL_TYPE *__restrict__ trace_values,
    DmmaTaskTraceRecord *__restrict__ records)
{
    const std::size_t task_wide =
        dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    if (task_wide >= static_cast<std::size_t>(output_tile_count))
        return;
    const int task = static_cast<int>(task_wide);
    const bool sampled =
        task_wide >= static_cast<std::size_t>(sample_phase) &&
        (task_wide - static_cast<std::size_t>(sample_phase)) %
                static_cast<std::size_t>(sample_stride) ==
            0;
    const std::size_t sample_index =
        sampled ? (task_wide - static_cast<std::size_t>(sample_phase)) /
                      static_cast<std::size_t>(sample_stride)
                : 0;

#if __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;
    __shared__ MAT_VAL_TYPE shared_a[DMMA_WARPS_PER_BLOCK *
                                     DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE shared_b[DMMA_WARPS_PER_BLOCK *
                                     DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE shared_c[DMMA_WARPS_PER_BLOCK *
                                     DMMA_OUTPUT_ELEMS];
    MAT_VAL_TYPE *tile_a_values =
        shared_a + local_warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_b_values =
        shared_b + local_warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_c_values =
        shared_c + local_warp * DMMA_OUTPUT_ELEMS;

    uint64_t start_ns = 0;
    uint32_t sm_id = 0;
    if (sampled && lane == 0)
    {
        sm_id = dmma_schedule_smid();
        start_ns = dmma_task_trace_globaltimer_ns();
    }
    __syncwarp();

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

    const int tile_row = output_rows[task];
    const int tile_col = output_cols[task];
    int pa = a.tile_row_ptr[tile_row];
    const int a_end = a.tile_row_ptr[tile_row + 1];
    int pb = b.tile_col_ptr[tile_col];
    const int b_end = b.tile_col_ptr[tile_col + 1];
    int scan_steps = 0;
    int matches = 0;
    while (pa < a_end && pb < b_end)
    {
        ++scan_steps;
        const int ka = a.tile_col_idx[pa];
        const int kb = b.tile_row_idx[pb];
        if (ka == kb)
        {
            ++matches;
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
            ++pa;
        else
            ++pb;
    }

    wmma::store_matrix_sync(tile_c_values, fragment_c, DMMA_TILE_N,
                            wmma::mem_row_major);
    __syncwarp();
    const uint64_t mask = output_masks[task];
    if (lane < DMMA_TILE_M)
    {
        const uint64_t preceding =
            lane == 0 ? 0ull
                      : mask & ((1ull << (lane * DMMA_TILE_N)) - 1ull);
        trace_row_ptr[task_wide * DMMA_TILE_M + lane] =
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
        const std::size_t trace_index =
            static_cast<std::size_t>(output_offsets[task]) + rank;
        trace_col_idx[trace_index] =
            static_cast<unsigned char>(position % DMMA_TILE_N);
        trace_values[trace_index] = tile_c_values[position];
    }
    __syncwarp();
    if (sampled && lane == 0 &&
        sample_index < static_cast<std::size_t>(sampled_task_count))
    {
        DmmaTaskTraceRecord record;
        record.start_ns = start_ns;
        record.end_ns = dmma_task_trace_globaltimer_ns();
        record.sm_id = sm_id;
        record.task_id = task;
        record.scan_steps = scan_steps;
        record.matches = matches;
        record.output_nnz =
            output_offsets[task + 1] - output_offsets[task];
        records[sample_index] = record;
    }
#else
    if (sampled && lane == 0 &&
        sample_index < static_cast<std::size_t>(sampled_task_count))
    {
        DmmaTaskTraceRecord record;
        record.start_ns = dmma_task_trace_globaltimer_ns();
        record.end_ns = record.start_ns + 1;
        record.sm_id = dmma_schedule_smid();
        record.task_id = task;
        record.output_nnz =
            output_offsets[task + 1] - output_offsets[task];
        records[sample_index] = record;
    }
#endif
}

static inline bool dmma_write_all_fd(int descriptor, const char *data,
                                     std::size_t bytes)
{
    std::size_t written = 0;
    while (written < bytes)
    {
        const ssize_t result =
            ::write(descriptor, data + written, bytes - written);
        if (result < 0)
        {
            if (errno == EINTR)
                continue;
            return false;
        }
        if (result == 0)
            return false;
        written += static_cast<std::size_t>(result);
    }
    return true;
}

static inline std::string dmma_csv_field(const char *text)
{
    const char *value = text == nullptr ? "unnamed" : text;
    bool quote = *value == '\0';
    for (const char *cursor = value; *cursor != '\0'; ++cursor)
        quote = quote || *cursor == ',' || *cursor == '"' ||
                *cursor == '\n' || *cursor == '\r';
    if (!quote)
        return std::string(value);
    std::string result(1, '"');
    for (const char *cursor = value; *cursor != '\0'; ++cursor)
    {
        if (*cursor == '"')
            result.push_back('"');
        result.push_back(*cursor);
    }
    result.push_back('"');
    return result;
}

static inline bool dmma_append_task_trace_csv(
    const char *path, const char *matrix_name,
    const std::vector<DmmaTaskTraceRecord> &records)
{
    if (path == nullptr || *path == '\0' || records.empty())
        return records.empty() || (path != nullptr && *path != '\0');
    static constexpr const char header[] =
        "matrix,task_id,sm_id,start_ns,end_ns,scan_steps,matches,"
        "output_nnz,schedule\n";
    const int descriptor =
        ::open(path, O_CREAT | O_RDWR | O_APPEND, S_IRUSR | S_IWUSR |
                                                   S_IRGRP | S_IROTH);
    if (descriptor < 0)
    {
        std::fprintf(stderr, "Cannot open DMMA task trace CSV %s: %s\n",
                     path, std::strerror(errno));
        return false;
    }
    bool locked = false;
    bool ok = true;
    if (::flock(descriptor, LOCK_EX) != 0)
        ok = false;
    else
        locked = true;
    struct stat metadata{};
    if (ok && ::fstat(descriptor, &metadata) != 0)
        ok = false;
    if (ok && metadata.st_size == 0)
        ok = dmma_write_all_fd(descriptor, header, sizeof(header) - 1);
    else if (ok)
    {
        char existing_header[sizeof(header) - 1] = {};
        const ssize_t bytes =
            ::pread(descriptor, existing_header, sizeof(existing_header), 0);
        if (bytes != static_cast<ssize_t>(sizeof(existing_header)) ||
            std::memcmp(existing_header, header, sizeof(existing_header)) != 0)
        {
            std::fprintf(stderr,
                         "DMMA task trace CSV has an incompatible header: %s\n",
                         path);
            ok = false;
        }
    }

    try
    {
        const std::string matrix = dmma_csv_field(matrix_name);
        uint64_t origin = records.front().start_ns;
        for (const DmmaTaskTraceRecord &record : records)
            origin = std::min(origin, record.start_ns);
        std::string batch;
        batch.reserve(1024 * 1024);
        for (const DmmaTaskTraceRecord &record : records)
        {
            if (!ok)
                break;
            if (record.end_ns <= record.start_ns ||
                record.scan_steps < 0 || record.matches < 0 ||
                record.output_nnz < 0)
            {
                std::fprintf(stderr,
                             "Invalid DMMA task trace record for task %d.\n",
                             record.task_id);
                ok = false;
                break;
            }
            char numeric[256];
            const int length = std::snprintf(
                numeric, sizeof(numeric),
                ",%d,%u,%llu,%llu,%d,%d,%d,direct\n", record.task_id,
                record.sm_id,
                static_cast<unsigned long long>(record.start_ns - origin),
                static_cast<unsigned long long>(record.end_ns - origin),
                record.scan_steps, record.matches, record.output_nnz);
            if (length <= 0 || length >= static_cast<int>(sizeof(numeric)))
            {
                ok = false;
                break;
            }
            if (batch.size() + matrix.size() +
                    static_cast<std::size_t>(length) >
                1024 * 1024)
            {
                ok = dmma_write_all_fd(descriptor, batch.data(), batch.size());
                batch.clear();
            }
            batch.append(matrix);
            batch.append(numeric, static_cast<std::size_t>(length));
        }
        if (ok && !batch.empty())
            ok = dmma_write_all_fd(descriptor, batch.data(), batch.size());
    }
    catch (const std::exception &error)
    {
        std::fprintf(stderr, "Cannot format DMMA task trace CSV: %s\n",
                     error.what());
        ok = false;
    }
    catch (...)
    {
        std::fprintf(stderr, "Cannot format DMMA task trace CSV.\n");
        ok = false;
    }

    if (locked)
        (void)::flock(descriptor, LOCK_UN);
    if (::close(descriptor) != 0)
        ok = false;
    return ok;
}

static inline bool dmma_profile_direct_task_trace(
    const DmmaDeviceTiles &a, const DmmaDeviceTiles &b,
    int output_tile_count,
    const int *d_output_rows, const int *d_output_cols,
    const uint64_t *d_output_masks, const int *d_output_offsets,
    unsigned char *d_output_tile_row_ptr,
    unsigned char *d_output_value_cols, MAT_VAL_TYPE *d_output_values,
    const DmmaNumericScheduleConfig &schedule, DmmaSpGemmStats *stats)
{
    if (schedule.task_trace_path == nullptr)
        return true;
    if (*schedule.task_trace_path == '\0' || schedule.matrix_name == nullptr ||
        *schedule.matrix_name == '\0')
    {
        std::fprintf(stderr,
                     "DMMA task trace requires non-empty path and matrix name.\n");
        return false;
    }
    if (schedule.mode != DMMA_SCHEDULE_DIRECT)
    {
        std::fprintf(stderr,
                     "DMMA task trace currently supports direct schedule only.\n");
        return false;
    }
    if (schedule.task_trace_sample_shift > 30)
    {
        std::fprintf(stderr, "DMMA task trace sample shift exceeds 30.\n");
        return false;
    }
    const unsigned int stride =
        1u << schedule.task_trace_sample_shift;
    const unsigned int phase = schedule.task_trace_sample_phase;
    if (phase >= stride)
    {
        std::fprintf(stderr,
                     "DMMA task trace sample phase must be below stride.\n");
        return false;
    }
    const int sampled_task_count =
        output_tile_count <= static_cast<int>(phase)
            ? 0
            : static_cast<int>(
                  (static_cast<unsigned int>(output_tile_count - 1) - phase) /
                      stride +
                  1u);
    if (stats != nullptr)
        stats->task_trace_tasks = sampled_task_count;
    if (sampled_task_count == 0)
        return true;

    DmmaTaskTraceRecord *d_records = nullptr;
    auto cleanup = [&]() {
        cudaFree(d_records);
    };
    const std::size_t sampled = static_cast<std::size_t>(sampled_task_count);
    if (!dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_records),
                                 sampled * sizeof(DmmaTaskTraceRecord)),
                      "allocate task trace records"))
    {
        cleanup();
        return false;
    }

    unsigned int blocks = 0;
    if (!dmma_launch_blocks(static_cast<std::size_t>(output_tile_count),
                            DMMA_WARPS_PER_BLOCK, &blocks,
                            "direct task trace"))
    {
        cleanup();
        return false;
    }
    dmma_direct_task_trace_kernel<<<blocks, DMMA_THREADS_PER_BLOCK>>>(
        a, b, output_tile_count, d_output_rows, d_output_cols,
        d_output_masks, d_output_offsets, stride, phase,
        sampled_task_count, d_output_tile_row_ptr, d_output_value_cols,
        d_output_values, d_records);
    if (!dmma_cuda_ok(cudaGetLastError(), "launch direct task trace") ||
        !dmma_cuda_ok(cudaDeviceSynchronize(), "complete direct task trace"))
    {
        cleanup();
        return false;
    }

    std::vector<DmmaTaskTraceRecord> records;
    try
    {
        records.resize(sampled);
    }
    catch (...)
    {
        std::fprintf(stderr, "Cannot allocate host DMMA task trace records.\n");
        cleanup();
        return false;
    }
    const bool copied = dmma_cuda_ok(
        cudaMemcpy(records.data(), d_records,
                   sampled * sizeof(DmmaTaskTraceRecord),
                   cudaMemcpyDeviceToHost),
        "copy direct task trace records");
    cleanup();
    if (!copied)
        return false;

    for (int sample = 0; sample < sampled_task_count; ++sample)
    {
        const int expected_task = static_cast<int>(
            phase + static_cast<unsigned int>(sample) * stride);
        if (records[static_cast<std::size_t>(sample)].task_id != expected_task)
        {
            std::fprintf(stderr,
                         "DMMA task trace returned a mismatched task ID.\n");
            return false;
        }
    }
    return dmma_append_task_trace_csv(schedule.task_trace_path,
                                      schedule.matrix_name, records);
}

/* Replay the exact merge trajectory and divide it by accumulated model work.
 * A descriptor stores a starting pointer pair plus a merge-step count.  This
 * representation remains correct when only one of the two pointers advances
 * at a chunk boundary. */
__global__ void dmma_prepare_sparse_tail_metadata_kernel(
    int heavy_task_count, const DmmaTailRecord *__restrict__ tail_records,
    int flag_task_begin, int flag_task_count, int max_chunks,
    uint32_t *__restrict__ heavy_flags,
    int *__restrict__ heavy_chunk_counts,
    int *__restrict__ invalid_record)
{
    const int heavy = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (heavy >= heavy_task_count)
        return;
    const DmmaTailRecord record = tail_records[heavy];
    const int local = record.id - flag_task_begin;
    if (local < 0 || local >= flag_task_count || record.scans < 2 ||
        record.matches < 0 || record.matches > record.scans ||
        record.chunks < 2 || record.chunks > max_chunks ||
        record.chunks > record.scans)
    {
        heavy_chunk_counts[heavy] = 0;
        atomicExch(invalid_record, 1);
        return;
    }
    const uint32_t bit = 1u << (static_cast<unsigned int>(local) & 31u);
    const uint32_t old = atomicOr(
        heavy_flags + (static_cast<unsigned int>(local) >> 5), bit);
    if ((old & bit) != 0)
    {
        heavy_chunk_counts[heavy] = 0;
        atomicExch(invalid_record, 1);
        return;
    }
    heavy_chunk_counts[heavy] = record.chunks;
}

/* The sparse-tail admission rule clamps chunks to merge_steps.  While
 * following weighted targets, one high-weight match can nevertheless jump
 * across several targets at once.  Limit the current chunk's exclusive end
 * so that every later chunk retains at least one merge step.  This common
 * host/device predicate is also used by the boundary regression test. */
static __host__ __device__ __forceinline__ bool
dmma_chunk_can_consume_merge_step(int consumed_steps, int total_steps,
                                  int chunk, int chunks)
{
    const int later_chunks = chunks - chunk - 1;
    return consumed_steps < total_steps - later_chunks;
}

__global__ void dmma_emit_chunk_descriptors_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int heavy_task_count,
    const DmmaTailRecord *__restrict__ tail_records,
    const int *__restrict__ output_rows,
    const int *__restrict__ output_cols,
    const int *__restrict__ chunk_offsets,
    DmmaTaskCostModel model, DmmaChunkDescriptor *__restrict__ descriptors)
{
    const int heavy = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (heavy >= heavy_task_count)
        return;
    const DmmaTailRecord record = tail_records[heavy];
    const int task = record.id;
    const int first = chunk_offsets[heavy];
    const int chunks = chunk_offsets[heavy + 1] - first;
    const int tile_row = output_rows[task];
    const int tile_col = output_cols[task];
    int pa = a.tile_row_ptr[tile_row];
    int pb = b.tile_col_ptr[tile_col];
    const int a_end = a.tile_row_ptr[tile_row + 1];
    const int b_end = b.tile_col_ptr[tile_col + 1];
    const double merge_work =
        model.scan * static_cast<double>(record.scans) +
        model.match * static_cast<double>(record.matches);
    double consumed = 0.0;
    int consumed_steps = 0;

    for (int chunk = 0; chunk < chunks; ++chunk)
    {
        DmmaChunkDescriptor descriptor;
        descriptor.task = task;
        descriptor.pa_begin = pa;
        descriptor.pb_begin = pb;
        const double target = chunk + 1 == chunks
                                  ? merge_work + 1.0
                                  : merge_work * static_cast<double>(chunk + 1) /
                                        static_cast<double>(chunks);
        int steps = 0;
        while (pa < a_end && pb < b_end &&
               dmma_chunk_can_consume_merge_step(
                   consumed_steps, record.scans, chunk, chunks) &&
               (chunk + 1 == chunks || consumed < target))
        {
            ++steps;
            ++consumed_steps;
            consumed += model.scan;
            const int ka = a.tile_col_idx[pa];
            const int kb = b.tile_row_idx[pb];
            if (ka == kb)
            {
                consumed += model.match;
                ++pa;
                ++pb;
            }
            else if (ka < kb)
                ++pa;
            else
                ++pb;
        }
        /* A previous high-weight match may already have crossed this target.
         * Consume the step reserved for this chunk instead of publishing an
         * empty descriptor. */
        if (steps == 0 && pa < a_end && pb < b_end &&
            dmma_chunk_can_consume_merge_step(
                consumed_steps, record.scans, chunk, chunks))
        {
            ++steps;
            ++consumed_steps;
            consumed += model.scan;
            const int ka = a.tile_col_idx[pa];
            const int kb = b.tile_row_idx[pb];
            if (ka == kb)
            {
                consumed += model.match;
                ++pa;
                ++pb;
            }
            else if (ka < kb)
                ++pa;
            else
                ++pb;
        }
        descriptor.merge_steps = steps;
        descriptors[first + chunk] = descriptor;
    }
}

__global__ void dmma_emit_output_layout_kernel(
    int task_count, const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    unsigned char *__restrict__ output_row_ptr,
    unsigned char *__restrict__ output_col_idx)
{
    const std::size_t task = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (task >= static_cast<std::size_t>(task_count))
        return;
    const uint64_t mask = output_masks[task];
    const int output_begin = output_offsets[task];
    if (lane < DMMA_TILE_M)
    {
        const uint64_t preceding =
            lane == 0 ? 0ull : mask & ((1ull << (lane * DMMA_TILE_N)) - 1ull);
        output_row_ptr[task * DMMA_TILE_M + lane] =
            static_cast<unsigned char>(__popcll(preceding));
    }
    for (int position = lane; position < DMMA_OUTPUT_ELEMS;
         position += WARP_SIZE)
    {
        if ((mask & (1ull << position)) == 0)
            continue;
        const uint64_t lower =
            position == 0 ? 0ull : mask & ((1ull << position) - 1ull);
        output_col_idx[output_begin + __popcll(lower)] =
            static_cast<unsigned char>(position % DMMA_TILE_N);
    }
}

/* Heavy tasks do not pass through the light kernel.  Prepare only their
 * compressed layout and, for the atomic backend, clear only the scalar slots
 * that chunk CTAs will accumulate into.  Light tasks overwrite their own
 * disjoint layout/value ranges in dmma_numeric_light_kernel. */
__global__ void dmma_prepare_heavy_output_kernel(
    int heavy_task_count,
    const DmmaTailRecord *__restrict__ tail_records,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    unsigned char *__restrict__ output_row_ptr,
    unsigned char *__restrict__ output_col_idx,
    MAT_VAL_TYPE *__restrict__ output_values, int clear_values)
{
    const std::size_t heavy = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (heavy >= static_cast<std::size_t>(heavy_task_count))
        return;
    const int task = tail_records[heavy].id;
    const uint64_t mask = output_masks[task];
    const int output_begin = output_offsets[task];
    if (lane < DMMA_TILE_M)
    {
        const uint64_t preceding =
            lane == 0 ? 0ull
                      : mask & ((1ull << (lane * DMMA_TILE_N)) - 1ull);
        output_row_ptr[task * DMMA_TILE_M + lane] =
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
        if (clear_values)
            output_values[output_begin + rank] = MAT_VAL_TYPE(0);
    }
}

__device__ __forceinline__ void dmma_accumulate_chunk(
    DmmaDeviceTiles a, DmmaDeviceTiles b,
    const DmmaChunkDescriptor &descriptor, int lane,
    MAT_VAL_TYPE *tile_a_values, MAT_VAL_TYPE *tile_b_values,
    MAT_VAL_TYPE *tile_c_values)
{
#if __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;
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
    int pa = descriptor.pa_begin;
    int pb = descriptor.pb_begin;
    for (int step = 0; step < descriptor.merge_steps; ++step)
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
            ++pa;
        else
            ++pb;
    }
    wmma::store_matrix_sync(tile_c_values, fragment_c, DMMA_TILE_N,
                            wmma::mem_row_major);
    __syncwarp();
#endif
}

__device__ __forceinline__ void dmma_store_chunk_result(
    int chunk, const DmmaChunkDescriptor &descriptor, int lane,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets, MAT_VAL_TYPE *output_values,
    MAT_VAL_TYPE *partial_workspace, MAT_VAL_TYPE *tile_c_values,
    int workspace_mode)
{
    if (workspace_mode)
    {
        for (int position = lane; position < DMMA_OUTPUT_ELEMS;
             position += WARP_SIZE)
            partial_workspace[static_cast<std::size_t>(chunk) *
                                  DMMA_OUTPUT_ELEMS +
                              position] = tile_c_values[position];
        return;
    }
    const int task = descriptor.task;
    const uint64_t mask = output_masks[task];
    const int output_begin = output_offsets[task];
    for (int position = lane; position < DMMA_OUTPUT_ELEMS;
         position += WARP_SIZE)
    {
        if ((mask & (1ull << position)) == 0)
            continue;
        const uint64_t lower =
            position == 0 ? 0ull : mask & ((1ull << position) - 1ull);
        atomicAdd(output_values + output_begin + __popcll(lower),
                  tile_c_values[position]);
    }
}

/* One independent one-warp CTA per chunk: chunks from one parent C tile are
 * independent CUDA scheduling units and may execute on distinct SMs. */
__global__ void dmma_chunk_cta_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int chunk_count,
    const DmmaChunkDescriptor *__restrict__ descriptors,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets, MAT_VAL_TYPE *output_values,
    MAT_VAL_TYPE *partial_workspace, uint32_t *__restrict__ chunk_sm_ids,
    int workspace_mode)
{
    const int chunk = static_cast<int>(blockIdx.x);
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (chunk >= chunk_count)
        return;
    __shared__ MAT_VAL_TYPE tile_a_values[DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE tile_b_values[DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE tile_c_values[DMMA_OUTPUT_ELEMS];
    const DmmaChunkDescriptor descriptor = descriptors[chunk];
    dmma_accumulate_chunk(a, b, descriptor, lane, tile_a_values,
                          tile_b_values, tile_c_values);
    dmma_store_chunk_result(chunk, descriptor, lane, output_masks,
                            output_offsets, output_values, partial_workspace,
                            tile_c_values, workspace_mode);
    if (lane == 0 && chunk_sm_ids != nullptr)
        chunk_sm_ids[chunk] = dmma_schedule_smid();
}

/* Persistent worker CTAs cover the device and dequeue heavy chunks.  Four
 * independent warps per CTA keep the original DMMA resource shape. */
__global__ void dmma_chunk_persistent_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int chunk_count,
    const DmmaChunkDescriptor *__restrict__ descriptors,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets, MAT_VAL_TYPE *output_values,
    MAT_VAL_TYPE *partial_workspace, uint32_t *__restrict__ chunk_sm_ids,
    int *queue_head, int workspace_mode)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    __shared__ __align__(32) MAT_VAL_TYPE
        shared_a[DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ __align__(32) MAT_VAL_TYPE
        shared_b[DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ __align__(32) MAT_VAL_TYPE
        shared_c[DMMA_WARPS_PER_BLOCK * DMMA_OUTPUT_ELEMS];
    MAT_VAL_TYPE *tile_a_values = shared_a + local_warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_b_values = shared_b + local_warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_c_values = shared_c + local_warp * DMMA_OUTPUT_ELEMS;
    while (true)
    {
        int chunk = 0;
        if (lane == 0)
            chunk = atomicAdd(queue_head, 1);
        chunk = __shfl_sync(0xffffffffu, chunk, 0);
        if (chunk >= chunk_count)
            break;
        const DmmaChunkDescriptor descriptor = descriptors[chunk];
        dmma_accumulate_chunk(a, b, descriptor, lane, tile_a_values,
                              tile_b_values, tile_c_values);
        dmma_store_chunk_result(chunk, descriptor, lane, output_masks,
                                output_offsets, output_values,
                                partial_workspace, tile_c_values,
                                workspace_mode);
        if (lane == 0 && chunk_sm_ids != nullptr)
            chunk_sm_ids[chunk] = dmma_schedule_smid();
    }
}

__device__ __forceinline__ void dmma_numeric_regular_task(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int output_tile_count,
    const int *__restrict__ output_rows, const int *__restrict__ output_cols,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    unsigned char *__restrict__ output_row_ptr,
    unsigned char *__restrict__ output_col_idx,
    MAT_VAL_TYPE *__restrict__ output_values, int task, int lane,
    MAT_VAL_TYPE *tile_a_values, MAT_VAL_TYPE *tile_b_values,
    MAT_VAL_TYPE *tile_c_values)
{
#if __CUDA_ARCH__ >= 800
    namespace wmma = nvcuda::wmma;
    if (task < 0 || task >= output_tile_count)
        return;

    wmma::fragment<wmma::matrix_a, DMMA_TILE_M, DMMA_TILE_N, DMMA_TILE_K,
                   MAT_VAL_TYPE, wmma::row_major>
        fragment_a;
    wmma::fragment<wmma::matrix_b, DMMA_TILE_M, DMMA_TILE_N, DMMA_TILE_K,
                   MAT_VAL_TYPE, wmma::col_major>
        fragment_b;
    wmma::fragment<wmma::accumulator, DMMA_TILE_M, DMMA_TILE_N,
                   DMMA_TILE_K, MAT_VAL_TYPE>
        fragment_c;
    wmma::fill_fragment(fragment_c, MAT_VAL_TYPE(0));

    const int tile_row = output_rows[task];
    const int tile_col = output_cols[task];
    int pa = a.tile_row_ptr[tile_row];
    int pb = b.tile_col_ptr[tile_col];
    const int a_end = a.tile_row_ptr[tile_row + 1];
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
            ++pa;
        else
            ++pb;
    }
    wmma::store_matrix_sync(tile_c_values, fragment_c, DMMA_TILE_N,
                            wmma::mem_row_major);
    __syncwarp();

    const uint64_t mask = output_masks[task];
    const int output_begin = output_offsets[task];
    if (lane < DMMA_TILE_M)
    {
        const uint64_t preceding =
            lane == 0 ? 0ull
                      : mask & ((1ull << (lane * DMMA_TILE_N)) - 1ull);
        output_row_ptr[static_cast<std::size_t>(task) * DMMA_TILE_M + lane] =
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

/* Cost-balanced wrapper around the unchanged regular numeric primitive.
 * Symbolic supplies a task permutation ordered by predicted hybrid-payload
 * work.  Each warp independently dequeues one task, removing the four-task
 * CTA convoy of the uniform grid while preserving all A/B decoding and MMA
 * instructions inside dmma_numeric_regular_task. */
__global__ void dmma_numeric_cost_queue_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int output_tile_count,
    const int *__restrict__ task_order, int *__restrict__ queue_head,
    const int *__restrict__ output_rows, const int *__restrict__ output_cols,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    unsigned char *__restrict__ output_row_ptr,
    unsigned char *__restrict__ output_col_idx,
    MAT_VAL_TYPE *__restrict__ output_values)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    __shared__ MAT_VAL_TYPE
        shared_a[DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE
        shared_b[DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE
        shared_c[DMMA_WARPS_PER_BLOCK * DMMA_OUTPUT_ELEMS];
    MAT_VAL_TYPE *tile_a_values =
        shared_a + local_warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_b_values =
        shared_b + local_warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_c_values =
        shared_c + local_warp * DMMA_OUTPUT_ELEMS;
    while (true)
    {
        int position = 0;
        if (lane == 0)
            position = atomicAdd(queue_head, 1);
        position = __shfl_sync(0xffffffffu, position, 0);
        if (position >= output_tile_count)
            break;
        const int task = task_order[position];
        dmma_numeric_regular_task(
            a, b, output_tile_count, output_rows, output_cols, output_masks,
            output_offsets, output_row_ptr, output_col_idx, output_values,
            task, lane, tile_a_values, tile_b_values, tile_c_values);
    }
}

__device__ __forceinline__ void dmma_numeric_low_fill_sparse_task(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int task, int tile_row,
    int tile_col, int lane, uint64_t mask, int output_begin,
    unsigned char *__restrict__ output_row_ptr,
    unsigned char *__restrict__ output_col_idx,
    MAT_VAL_TYPE *__restrict__ output_values)
{
#if __CUDA_ARCH__ >= 800
    const int positions[2] = {lane, lane + WARP_SIZE};
    MAT_VAL_TYPE accumulators[2] = {MAT_VAL_TYPE(0), MAT_VAL_TYPE(0)};
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
            const uint32_t mask_a = a.masks[pa];
            const uint32_t mask_b = b.masks[tile_b];
#pragma unroll
            for (int owner = 0; owner < 2; ++owner)
            {
                const int position = positions[owner];
                if ((mask & (1ull << position)) == 0)
                    continue;
                const int output_row = position / DMMA_TILE_N;
                const int output_col = position % DMMA_TILE_N;
                uint32_t intersection =
                    ((mask_a >> (output_row * DMMA_TILE_K)) & 0xfu) &
                    ((mask_b >> (output_col * DMMA_TILE_K)) & 0xfu);
                while (intersection != 0)
                {
                    const int k = __ffs(intersection) - 1;
                    accumulators[owner] = fma(
                        dmma_decode_value(
                            a, pa, output_row * DMMA_TILE_K + k),
                        dmma_decode_value(
                            b, tile_b, output_col * DMMA_TILE_K + k),
                        accumulators[owner]);
                    intersection &= intersection - 1;
                }
            }
            ++pa;
            ++pb;
        }
        else if (ka < kb)
            ++pa;
        else
            ++pb;
    }

    if (lane < DMMA_TILE_M)
    {
        const uint64_t preceding =
            lane == 0 ? 0ull
                      : mask & ((1ull << (lane * DMMA_TILE_N)) - 1ull);
        output_row_ptr[static_cast<std::size_t>(task) * DMMA_TILE_M + lane] =
            static_cast<unsigned char>(__popcll(preceding));
    }
#pragma unroll
    for (int owner = 0; owner < 2; ++owner)
    {
        const int position = positions[owner];
        if ((mask & (1ull << position)) == 0)
            continue;
        const uint64_t lower =
            position == 0 ? 0ull : mask & ((1ull << position) - 1ull);
        const int rank = __popcll(lower);
        output_col_idx[output_begin + rank] =
            static_cast<unsigned char>(position % DMMA_TILE_N);
        output_values[output_begin + rank] = accumulators[owner];
    }
#endif
}

/* One warp makes a uniform, pre-write decision.  The high-fill branch calls
 * the existing production regular task exactly once; the sparse branch owns
 * the same native output locations and requires no atomic or workspace. */
__global__ void dmma_numeric_low_fill_exact_tile_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int output_tile_count,
    const int *__restrict__ output_rows,
    const int *__restrict__ output_cols,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets, int q,
    unsigned char *__restrict__ output_row_ptr,
    unsigned char *__restrict__ output_col_idx,
    MAT_VAL_TYPE *__restrict__ output_values
#ifdef DMMA_ENABLE_TIMELINE_TRACE
    , DmmaTimelineView timeline
#endif
    )
{
#if __CUDA_ARCH__ >= 800
    const std::size_t task = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    if (task >= static_cast<std::size_t>(output_tile_count))
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

    __shared__ MAT_VAL_TYPE shared_a[
        DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE shared_b[
        DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE shared_c[
        DMMA_WARPS_PER_BLOCK * DMMA_OUTPUT_ELEMS];
    const int tile_row = output_rows[task];
    const int tile_col = output_cols[task];
    const uint64_t mask = output_masks[task];
    const uint32_t a_degree = static_cast<uint32_t>(
        a.tile_row_ptr[tile_row + 1] - a.tile_row_ptr[tile_row]);
    const uint32_t b_degree = static_cast<uint32_t>(
        b.tile_col_ptr[tile_col + 1] - b.tile_col_ptr[tile_col]);
    const int output_fill = __popcll(mask);
    const bool sparse = output_fill > 0 && output_fill <= q &&
        dmma_low_fill_local_guard(
            a.row_tile_nnz_sum[tile_row], a_degree,
            b.col_tile_nnz_sum[tile_col], b_degree, q);

    if (sparse)
        dmma_numeric_low_fill_sparse_task(
            a, b, static_cast<int>(task), tile_row, tile_col, lane, mask,
            output_offsets[task], output_row_ptr, output_col_idx,
            output_values);
    else
        dmma_numeric_regular_task(
            a, b, output_tile_count, output_rows, output_cols, output_masks,
            output_offsets, output_row_ptr, output_col_idx, output_values,
            static_cast<int>(task), lane,
            shared_a + local_warp * DMMA_INPUT_ELEMS,
            shared_b + local_warp * DMMA_INPUT_ELEMS,
            shared_c + local_warp * DMMA_OUTPUT_ELEMS);

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

static __host__ __device__ __forceinline__ int
dmma_row_static_worker_count(int c_tile_rows, int sm_count)
{
    if (c_tile_rows <= 0 || sm_count <= 0)
        return 0;
    return c_tile_rows < sm_count ? c_tile_rows : sm_count;
}

static __host__ __device__ __forceinline__ int
dmma_row_static_worker_row_begin(int worker, int c_tile_rows,
                                 int worker_count)
{
    if (worker < 0 || worker > worker_count || c_tile_rows < 0 ||
        worker_count <= 0)
        return -1;
    return static_cast<int>(
        static_cast<long long>(worker) * c_tile_rows / worker_count);
}

static __host__ __device__ __forceinline__ int
dmma_row_static_owner_cta(int c_tile_row, int c_tile_rows,
                          int worker_count)
{
    if (c_tile_row < 0 || c_tile_row >= c_tile_rows ||
        worker_count <= 0)
        return -1;
    /* Invert [floor(pR/P), floor((p+1)R/P)).  P<=R, so every interval is
     * non-empty and the ceil expression selects exactly one owner. */
    return static_cast<int>(
        (static_cast<long long>(c_tile_row + 1) * worker_count - 1) /
        c_tile_rows);
}

static __host__ __device__ __forceinline__ int
dmma_row_static_owner_warp(int task, int row_begin)
{
    if (task < row_begin)
        return -1;
    return (task - row_begin) % DMMA_WARPS_PER_BLOCK;
}

static __host__ __device__ __forceinline__ int
dmma_row_queue_batch_valid(int row_queue_batch)
{
    return row_queue_batch == 1 || row_queue_batch == 2 ||
           row_queue_batch == 4;
}

/* exact-row-ptr-v1 is intentionally A100-scoped for the frozen experiment:
 * the existing row-worker launch uses P=min(S,R), and the target has 108 SMs.
 * Keeping the bound explicit prevents an unreviewed architecture-dependent
 * change in either the reduction or the worker launch shape. */
constexpr int DMMA_ROW_GATE_MAX_WORKERS = 108;
constexpr int DMMA_ROW_GATE_MAX_THREADS = 128;

struct DmmaRowGateDeviceSummary
{
    unsigned long long load_sum = 0;
    unsigned long long load_max = 0;
    unsigned long long exact_tiles = 0;
    double load_sum_sq = 0.0;
    int rows = 0;
    int workers = 0;
    int zero_workers = 0;
    int invalid_intervals = 0;
};

struct DmmaRowGateFeatures
{
    bool valid = false;
    double static_max_over_mean = 1.0;
    double static_cv = 0.0;
};

static __host__ __device__ constexpr int
dmma_row_gate_reduction_threads(int workers)
{
    if (workers <= 0 || workers > DMMA_ROW_GATE_MAX_WORKERS)
        return 0;
    int threads = 1;
    while (threads < workers)
        threads <<= 1;
    return threads <= DMMA_ROW_GATE_MAX_THREADS ? threads : 0;
}

/* CPU reference and fixture helper.  Production consumes exactly the same
 * two row-pointer endpoints per logical worker in the CUDA kernel below. */
static inline DmmaRowGateDeviceSummary dmma_row_gate_host_summary(
    const int *output_row_ptr, int rows, int workers)
{
    DmmaRowGateDeviceSummary summary{};
    summary.rows = rows;
    summary.workers = workers;
    if (output_row_ptr == nullptr || rows <= 0 || workers <= 0 ||
        workers > DMMA_ROW_GATE_MAX_WORKERS || workers > rows)
    {
        summary.invalid_intervals = 1;
        return summary;
    }
    const int terminal = output_row_ptr[rows];
    if (terminal < 0)
        ++summary.invalid_intervals;
    else
        summary.exact_tiles = static_cast<unsigned long long>(terminal);
    for (int worker = 0; worker < workers; ++worker)
    {
        const int row_begin = dmma_row_static_worker_row_begin(
            worker, rows, workers);
        const int row_end = dmma_row_static_worker_row_begin(
            worker + 1, rows, workers);
        if (row_begin < 0 || row_end <= row_begin || row_end > rows ||
            output_row_ptr[row_begin] < 0 ||
            output_row_ptr[row_end] < output_row_ptr[row_begin])
        {
            ++summary.invalid_intervals;
            continue;
        }
        const unsigned long long load =
            static_cast<unsigned long long>(output_row_ptr[row_end] -
                                            output_row_ptr[row_begin]);
        summary.load_sum += load;
        summary.load_max = summary.load_max < load ? load : summary.load_max;
        const double load_double = static_cast<double>(load);
        summary.load_sum_sq += load_double * load_double;
        summary.zero_workers += load == 0 ? 1 : 0;
    }
    return summary;
}

static inline DmmaRowGateFeatures dmma_row_gate_features(
    const DmmaRowGateDeviceSummary &summary, int expected_rows,
    int expected_workers, int expected_exact_tiles)
{
    DmmaRowGateFeatures features{};
    const bool structural_valid =
        expected_rows > 0 && expected_workers > 0 &&
        expected_workers <= DMMA_ROW_GATE_MAX_WORKERS &&
        expected_workers <= expected_rows && expected_exact_tiles >= 0 &&
        summary.rows == expected_rows &&
        summary.workers == expected_workers &&
        summary.invalid_intervals == 0 && summary.zero_workers >= 0 &&
        summary.zero_workers <= expected_workers &&
        summary.exact_tiles ==
            static_cast<unsigned long long>(expected_exact_tiles) &&
        summary.load_sum == summary.exact_tiles &&
        summary.load_max <= summary.load_sum &&
        std::isfinite(summary.load_sum_sq) && summary.load_sum_sq >= 0.0;
    if (!structural_valid)
        return features;
    if (summary.load_sum == 0)
    {
        features.valid = true;
        return features;
    }
    const double mean = static_cast<double>(summary.load_sum) /
                        static_cast<double>(expected_workers);
    double variance = summary.load_sum_sq /
                          static_cast<double>(expected_workers) -
                      mean * mean;
    if (variance < 0.0 && variance > -1.0e-9 * mean * mean)
        variance = 0.0;
    if (!(mean > 0.0) || variance < 0.0 || !std::isfinite(variance))
        return features;
    features.static_max_over_mean =
        static_cast<double>(summary.load_max) / mean;
    features.static_cv = std::sqrt(variance) / mean;
    features.valid = std::isfinite(features.static_max_over_mean) &&
                     features.static_max_over_mean >= 1.0 &&
                     std::isfinite(features.static_cv) &&
                     features.static_cv >= 0.0;
    return features;
}

static inline bool dmma_row_gate_select_dynamic(
    const DmmaRowGateDeviceSummary &summary,
    const DmmaRowGateFeatures &features, double threshold)
{
    return features.valid && summary.load_sum > 0 &&
           std::isfinite(threshold) && threshold >= 1.0 &&
           features.static_max_over_mean >= threshold;
}

__global__ void dmma_row_gate_exact_row_ptr_kernel(
    const int *__restrict__ output_row_ptr, int rows, int workers,
    DmmaRowGateDeviceSummary *__restrict__ output)
{
    __shared__ unsigned long long load_sum[DMMA_ROW_GATE_MAX_THREADS];
    __shared__ unsigned long long load_max[DMMA_ROW_GATE_MAX_THREADS];
    __shared__ double load_sum_sq[DMMA_ROW_GATE_MAX_THREADS];
    __shared__ int zero_workers[DMMA_ROW_GATE_MAX_THREADS];
    __shared__ int invalid_intervals[DMMA_ROW_GATE_MAX_THREADS];
    const int thread = static_cast<int>(threadIdx.x);
    unsigned long long load = 0;
    int invalid = 0;
    if (thread < workers)
    {
        const int row_begin = dmma_row_static_worker_row_begin(
            thread, rows, workers);
        const int row_end = dmma_row_static_worker_row_begin(
            thread + 1, rows, workers);
        const int begin_value = row_begin >= 0 && row_begin <= rows
                                    ? output_row_ptr[row_begin]
                                    : -1;
        const int end_value = row_end >= 0 && row_end <= rows
                                  ? output_row_ptr[row_end]
                                  : -1;
        if (row_begin < 0 || row_end <= row_begin || row_end > rows ||
            begin_value < 0 || end_value < begin_value)
            invalid = 1;
        else
            load = static_cast<unsigned long long>(end_value - begin_value);
    }
    load_sum[thread] = load;
    load_max[thread] = load;
    const double load_double = static_cast<double>(load);
    load_sum_sq[thread] = load_double * load_double;
    zero_workers[thread] = thread < workers && load == 0 ? 1 : 0;
    invalid_intervals[thread] = invalid;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) >> 1; stride > 0;
         stride >>= 1)
    {
        if (thread < stride)
        {
            load_sum[thread] += load_sum[thread + stride];
            load_max[thread] =
                load_max[thread] < load_max[thread + stride]
                    ? load_max[thread + stride]
                    : load_max[thread];
            load_sum_sq[thread] += load_sum_sq[thread + stride];
            zero_workers[thread] += zero_workers[thread + stride];
            invalid_intervals[thread] += invalid_intervals[thread + stride];
        }
        __syncthreads();
    }
    if (thread == 0)
    {
        output->load_sum = load_sum[0];
        output->load_max = load_max[0];
        output->load_sum_sq = load_sum_sq[0];
        output->exact_tiles = output_row_ptr[rows] >= 0
                                  ? static_cast<unsigned long long>(
                                        output_row_ptr[rows])
                                  : 0ull;
        output->rows = rows;
        output->workers = workers;
        output->zero_workers = zero_workers[0];
        output->invalid_intervals =
            invalid_intervals[0] + (output_row_ptr[rows] < 0 ? 1 : 0);
    }
}

static __host__ __device__ __forceinline__ int
dmma_row_dynamic_expected_atomic_claims(int c_tile_rows, int worker_count,
                                        int row_queue_batch)
{
    if (c_tile_rows < 0 || worker_count < 0 ||
        !dmma_row_queue_batch_valid(row_queue_batch))
        return -1;
    const long long valid_batches =
        (static_cast<long long>(c_tile_rows) + row_queue_batch - 1) /
        row_queue_batch;
    const long long claims = valid_batches + worker_count;
    return claims <= INT_MAX ? static_cast<int>(claims) : -1;
}

static __host__ __device__ __forceinline__ int
dmma_row_dynamic_expected_final_head(int c_tile_rows, int worker_count,
                                     int row_queue_batch)
{
    const int claims = dmma_row_dynamic_expected_atomic_claims(
        c_tile_rows, worker_count, row_queue_batch);
    if (claims < 0)
        return -1;
    const long long head =
        static_cast<long long>(claims) * row_queue_batch;
    return head <= INT_MAX ? static_cast<int>(head) : -1;
}

/* Frozen batch=1 compatibility helper. */
static __host__ __device__ __forceinline__ int
dmma_row_dynamic_expected_claims(int c_tile_rows, int worker_count)
{
    return dmma_row_dynamic_expected_atomic_claims(c_tile_rows, worker_count,
                                                   1);
}

template <bool VisitOnly>
__device__ __forceinline__ void dmma_numeric_row_worker_process_row(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int c_tile_row_count,
    int output_tile_count, const int *__restrict__ output_tile_row_ptr,
    const int *__restrict__ output_rows, const int *__restrict__ output_cols,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    unsigned char *__restrict__ output_row_ptr,
    unsigned char *__restrict__ output_col_idx,
    MAT_VAL_TYPE *__restrict__ output_values,
    int *__restrict__ row_visit_counts,
    int *__restrict__ task_visit_counts, int row, int lane, int local_warp,
    MAT_VAL_TYPE *__restrict__ tile_a_values,
    MAT_VAL_TYPE *__restrict__ tile_b_values,
    MAT_VAL_TYPE *__restrict__ tile_c_values)
{
    (void)c_tile_row_count;
    if constexpr (VisitOnly)
    {
        if (lane == 0 && local_warp == 0 && row_visit_counts != nullptr)
            atomicAdd(row_visit_counts + row, 1);
    }
    const int begin = output_tile_row_ptr[row];
    const int end = output_tile_row_ptr[row + 1];
    for (int task = begin + local_warp; task < end;
         task += DMMA_WARPS_PER_BLOCK)
    {
        if constexpr (VisitOnly)
        {
            if (lane == 0 && task_visit_counts != nullptr)
                atomicAdd(task_visit_counts + task, 1);
        }
        else
        {
            dmma_numeric_regular_task(
                a, b, output_tile_count, output_rows, output_cols,
                output_masks, output_offsets, output_row_ptr,
                output_col_idx, output_values, task, lane,
                tile_a_values, tile_b_values, tile_c_values);
        }
    }
}

/* One binary implements the controlled row-static-block/row-dynamic pair, so
 * both modes have exactly the same compiled register and shared-memory
 * resources.  Both launch P=min(S,R) 128-thread/four-warp worker CTAs and use
 * the same per-row primitive above.  Static CTA p owns the contiguous block
 * [floor(pR/P),floor((p+1)R/P)); dynamic differs only by claiming each next
 * row batch with atomicAdd(next_row,row_queue_batch).  Batch one is the pure
 * row-level LB anchor; batch two/four only coarsen reservation granularity for
 * the queue/locality ablation.  Neither path uses a cost model, row sort,
 * heavy-tile split, work reuse, oversubscribed grid, or SM affinity.
 *
 * The dynamic claimed-row broadcast aliases shared_a only between rows.  The
 * second CTA barrier makes every thread capture the row before warp 0 reuses
 * that location for decoded A values.  CUDA placement remains non-contractual
 * and is audited after the Core endpoint for both modes. */
template <bool VisitOnly = false>
__global__ void dmma_numeric_row_worker_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int c_tile_row_count,
    int output_tile_count, const int *__restrict__ output_tile_row_ptr,
    const int *__restrict__ output_rows, const int *__restrict__ output_cols,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    unsigned char *__restrict__ output_row_ptr,
    unsigned char *__restrict__ output_col_idx,
    MAT_VAL_TYPE *__restrict__ output_values,
    int dynamic_rows, int row_queue_batch, int *__restrict__ next_row,
    uint32_t *__restrict__ worker_sm_ids,
    int *__restrict__ row_visit_counts,
    int *__restrict__ task_visit_counts)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    if (threadIdx.x == 0 && worker_sm_ids != nullptr)
        worker_sm_ids[blockIdx.x] = dmma_schedule_smid();

    __shared__ MAT_VAL_TYPE
        shared_a[DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE
        shared_b[DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE
        shared_c[DMMA_WARPS_PER_BLOCK * DMMA_OUTPUT_ELEMS];
    MAT_VAL_TYPE *tile_a_values =
        shared_a + local_warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_b_values =
        shared_b + local_warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_c_values =
        shared_c + local_warp * DMMA_OUTPUT_ELEMS;
    int *claimed_row_slot = reinterpret_cast<int *>(shared_a);

    if (dynamic_rows != 0)
    {
        while (true)
        {
            if (threadIdx.x == 0)
                *claimed_row_slot = atomicAdd(next_row, row_queue_batch);
            __syncthreads();
            const int batch_begin = *claimed_row_slot;
            __syncthreads();
            if (batch_begin >= c_tile_row_count)
                break;
            const int batch_end =
                batch_begin < c_tile_row_count - row_queue_batch
                    ? batch_begin + row_queue_batch
                    : c_tile_row_count;
            for (int row = batch_begin; row < batch_end; ++row)
                dmma_numeric_row_worker_process_row<VisitOnly>(
                    a, b, c_tile_row_count, output_tile_count,
                    output_tile_row_ptr, output_rows, output_cols,
                    output_masks, output_offsets, output_row_ptr,
                    output_col_idx, output_values, row_visit_counts,
                    task_visit_counts, row, lane, local_warp, tile_a_values,
                    tile_b_values, tile_c_values);
        }
    }
    else
    {
        const int first_row = dmma_row_static_worker_row_begin(
            static_cast<int>(blockIdx.x), c_tile_row_count,
            static_cast<int>(gridDim.x));
        const int last_row = dmma_row_static_worker_row_begin(
            static_cast<int>(blockIdx.x) + 1, c_tile_row_count,
            static_cast<int>(gridDim.x));
        for (int row = first_row; row < last_row; ++row)
            dmma_numeric_row_worker_process_row<VisitOnly>(
                a, b, c_tile_row_count, output_tile_count,
                output_tile_row_ptr, output_rows, output_cols, output_masks,
                output_offsets, output_row_ptr, output_col_idx, output_values,
                row_visit_counts, task_visit_counts, row, lane, local_warp,
                tile_a_values, tile_b_values, tile_c_values);
    }
}

/* P17 transpose-packed layout.  Let R=ceil(chunk_count/W).  The first R CTAs
 * contain only chunk work: warp w in block b reads descriptor w*R+b, i.e. one
 * position from each contiguous descriptor partition.  With parent-major
 * descriptors this transposition reduces the chance that adjacent siblings
 * share a CTA, but it does not guarantee that every CTA contains different
 * parents.  Keeping regular work out of these CTAs removes P16's within-CTA
 * chunk/regular convoy.  The remaining ceil(output_tile_count/W) CTAs contain
 * only regular tasks, numbered contiguously by warp.  W=1 degenerates to a
 * chunk-only CTA prefix followed by ordinary one-warp regular CTAs.
 *
 * CUDA does not contractually map different CTAs to different SMs; this
 * layout deliberately makes no SM-affinity claim. */
static __host__ __device__ __forceinline__ unsigned long long
dmma_flat_transpose_chunk_rows_wide(int chunk_count, int warps_per_cta)
{
    if (chunk_count < 0 ||
        (warps_per_cta != 1 && warps_per_cta != 2 && warps_per_cta != 4))
        return 0;
    const unsigned long long chunks =
        static_cast<unsigned int>(chunk_count);
    return chunks / static_cast<unsigned int>(warps_per_cta) +
           (chunks % static_cast<unsigned int>(warps_per_cta) != 0
                ? 1ull
                : 0ull);
}

static __host__ __device__ __forceinline__ unsigned long long
dmma_flat_transpose_grid_blocks_wide(int chunk_count, int output_tile_count,
                                     int warps_per_cta)
{
    if (chunk_count < 0 || output_tile_count < 0 ||
        (warps_per_cta != 1 && warps_per_cta != 2 && warps_per_cta != 4))
        return 0;
    const unsigned long long outputs =
        static_cast<unsigned int>(output_tile_count);
    const unsigned long long regular_rows =
        outputs / static_cast<unsigned int>(warps_per_cta) +
        (outputs % static_cast<unsigned int>(warps_per_cta) != 0
                ? 1ull
                : 0ull);
    return dmma_flat_transpose_chunk_rows_wide(
               chunk_count, warps_per_cta) +
           regular_rows;
}

template <int WarpsPerCta>
static __host__ __device__ __forceinline__ unsigned long long
dmma_flat_transpose_chunk_id(unsigned int block, int local_warp,
                             int chunk_count)
{
    static_assert(WarpsPerCta == 1 || WarpsPerCta == 2 ||
                      WarpsPerCta == 4,
                  "transpose-flat supports 1, 2, or 4 warps per CTA");
    if (chunk_count < 0 || local_warp < 0 || local_warp >= WarpsPerCta)
        return ULLONG_MAX;
    const unsigned long long rows =
        dmma_flat_transpose_chunk_rows_wide(chunk_count, WarpsPerCta);
    if (static_cast<unsigned long long>(block) >= rows)
        return ULLONG_MAX;
    return static_cast<unsigned long long>(local_warp) * rows + block;
}

template <int WarpsPerCta>
static __host__ __device__ __forceinline__ unsigned long long
dmma_flat_transpose_regular_task(unsigned int block, int local_warp,
                                 int chunk_count)
{
    static_assert(WarpsPerCta == 1 || WarpsPerCta == 2 ||
                      WarpsPerCta == 4,
                  "transpose-flat supports 1, 2, or 4 warps per CTA");
    if (chunk_count < 0 || local_warp < 0 || local_warp >= WarpsPerCta)
        return ULLONG_MAX;
    const unsigned long long rows =
        dmma_flat_transpose_chunk_rows_wide(chunk_count, WarpsPerCta);
    if (static_cast<unsigned long long>(block) < rows)
        return ULLONG_MAX;
    return (static_cast<unsigned long long>(block) - rows) *
               static_cast<unsigned int>(WarpsPerCta) +
           static_cast<unsigned int>(local_warp);
}

template <int WarpsPerCta, bool VisitOnly = false>
__global__ __launch_bounds__(
    WarpsPerCta * WARP_SIZE,
    WarpsPerCta == 4 ? 16 : 32) void dmma_numeric_flat_mixed_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int chunk_count,
    const DmmaChunkDescriptor *__restrict__ descriptors,
    int output_tile_count,
    const int *__restrict__ output_rows, const int *__restrict__ output_cols,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    const uint32_t *__restrict__ heavy_flags,
    unsigned char *__restrict__ output_row_ptr,
    unsigned char *__restrict__ output_col_idx,
    MAT_VAL_TYPE *__restrict__ output_values,
    MAT_VAL_TYPE *__restrict__ partial_workspace,
    uint32_t *__restrict__ chunk_sm_ids, int workspace_mode,
    int *__restrict__ regular_visit_counts,
    int *__restrict__ chunk_visit_counts)
{
    static_assert(WarpsPerCta == 1 || WarpsPerCta == 2 ||
                      WarpsPerCta == 4,
                  "flat mixed grid supports 1, 2, or 4 warps per CTA");
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    if (local_warp >= WarpsPerCta)
        return;

    __shared__ __align__(32) MAT_VAL_TYPE
        shared_a[WarpsPerCta * DMMA_INPUT_ELEMS];
    __shared__ __align__(32) MAT_VAL_TYPE
        shared_b[WarpsPerCta * DMMA_INPUT_ELEMS];
    __shared__ __align__(32) MAT_VAL_TYPE
        shared_c[WarpsPerCta * DMMA_OUTPUT_ELEMS];
    MAT_VAL_TYPE *tile_a_values =
        shared_a + local_warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_b_values =
        shared_b + local_warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_c_values =
        shared_c + local_warp * DMMA_OUTPUT_ELEMS;

    const unsigned long long chunk_rows =
        dmma_flat_transpose_chunk_rows_wide(chunk_count, WarpsPerCta);
    if (static_cast<unsigned long long>(blockIdx.x) < chunk_rows)
    {
        const unsigned long long chunk_wide =
            dmma_flat_transpose_chunk_id<WarpsPerCta>(
                static_cast<unsigned int>(blockIdx.x), local_warp,
                chunk_count);
        if (chunk_wide >= static_cast<unsigned int>(chunk_count))
            return;
        const int chunk = static_cast<int>(chunk_wide);
        if constexpr (VisitOnly)
        {
            if (lane == 0)
                atomicAdd(chunk_visit_counts + chunk, 1);
        }
        else
        {
            const DmmaChunkDescriptor descriptor = descriptors[chunk];
            dmma_accumulate_chunk(a, b, descriptor, lane, tile_a_values,
                                  tile_b_values, tile_c_values);
            dmma_store_chunk_result(
                chunk, descriptor, lane, output_masks, output_offsets,
                output_values, partial_workspace, tile_c_values,
                workspace_mode);
            if (lane == 0 && chunk_sm_ids != nullptr)
                chunk_sm_ids[chunk] = dmma_schedule_smid();
        }
        return;
    }

    const unsigned long long task_wide =
        dmma_flat_transpose_regular_task<WarpsPerCta>(
            static_cast<unsigned int>(blockIdx.x), local_warp, chunk_count);
    if (task_wide >= static_cast<unsigned int>(output_tile_count))
        return;
    const int task = static_cast<int>(task_wide);
    int heavy = 0;
    if (lane == 0)
        heavy = (heavy_flags[static_cast<unsigned int>(task) >> 5] &
                 (1u << (static_cast<unsigned int>(task) & 31u))) != 0;
    heavy = __shfl_sync(0xffffffffu, heavy, 0);
    if (heavy)
        return;
    if constexpr (VisitOnly)
    {
        if (lane == 0)
            atomicAdd(regular_visit_counts + task, 1);
    }
    else
    {
        dmma_numeric_regular_task(
            a, b, output_tile_count, output_rows, output_cols, output_masks,
            output_offsets, output_row_ptr, output_col_idx, output_values,
            task, lane, tile_a_values, tile_b_values, tile_c_values);
    }
}

__global__ void dmma_numeric_light_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int output_tile_count,
    const int *__restrict__ output_rows, const int *__restrict__ output_cols,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    const uint32_t *__restrict__ heavy_flags,
    unsigned char *__restrict__ output_row_ptr,
    unsigned char *__restrict__ output_col_idx,
    MAT_VAL_TYPE *__restrict__ output_values)
{
    const std::size_t global_warp = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    if (global_warp >= static_cast<std::size_t>(output_tile_count))
        return;
    const unsigned int task = static_cast<unsigned int>(global_warp);
    if ((heavy_flags[task >> 5] & (1u << (task & 31u))) != 0)
        return;
    __shared__ MAT_VAL_TYPE shared_a[DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE shared_b[DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE shared_c[DMMA_WARPS_PER_BLOCK * DMMA_OUTPUT_ELEMS];
    dmma_numeric_regular_task(
        a, b, output_tile_count, output_rows, output_cols, output_masks,
        output_offsets, output_row_ptr, output_col_idx, output_values,
        static_cast<int>(task), lane,
        shared_a + local_warp * DMMA_INPUT_ELEMS,
        shared_b + local_warp * DMMA_INPUT_ELEMS,
        shared_c + local_warp * DMMA_OUTPUT_ELEMS);
}

/* The prefix contains no admitted heavy parent after exact output-window
 * filtering, so it preserves a branch-free four-warps-per-CTA bulk grid. */
__global__ void dmma_numeric_prefix_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int output_tile_count,
    int prefix_task_count,
    const int *__restrict__ output_rows, const int *__restrict__ output_cols,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    unsigned char *__restrict__ output_row_ptr,
    unsigned char *__restrict__ output_col_idx,
    MAT_VAL_TYPE *__restrict__ output_values)
{
    const int task = static_cast<int>(dmma_global_thread_index() / WARP_SIZE);
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int local_warp = threadIdx.x / WARP_SIZE;
    if (task >= prefix_task_count)
        return;
    __shared__ MAT_VAL_TYPE shared_a[DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE shared_b[DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE shared_c[DMMA_WARPS_PER_BLOCK * DMMA_OUTPUT_ELEMS];
    dmma_numeric_regular_task(
        a, b, output_tile_count, output_rows, output_cols, output_masks,
        output_offsets, output_row_ptr, output_col_idx, output_values, task,
        lane, shared_a + local_warp * DMMA_INPUT_ELEMS,
        shared_b + local_warp * DMMA_INPUT_ELEMS,
        shared_c + local_warp * DMMA_OUTPUT_ELEMS);
}

/* One warp owns each persistent CTA.  Queue acquisition and termination are
 * warp-local; there is deliberately no CTA-wide barrier that lets one slow
 * task convoy unrelated workers.  The final fine range uses unit tickets to
 * bound residual batch skew. */
__global__ void dmma_numeric_suffix_persistent_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int output_tile_count,
    int suffix_begin, int suffix_task_count, int bulk_task_count,
    int queue_batch,
    const int *__restrict__ output_rows, const int *__restrict__ output_cols,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    const uint32_t *__restrict__ heavy_suffix_flags,
    unsigned long long *__restrict__ bulk_head,
    unsigned long long *__restrict__ fine_head,
    unsigned char *__restrict__ output_row_ptr,
    unsigned char *__restrict__ output_col_idx,
    MAT_VAL_TYPE *__restrict__ output_values)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    __shared__ MAT_VAL_TYPE tile_a_values[DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE tile_b_values[DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE tile_c_values[DMMA_OUTPUT_ELEMS];

    while (true)
    {
        unsigned long long base = 0;
        if (lane == 0)
            base = atomicAdd(bulk_head,
                             static_cast<unsigned long long>(queue_batch));
        base = __shfl_sync(0xffffffffu, base, 0);
        if (base >= static_cast<unsigned long long>(bulk_task_count))
            break;
        const unsigned long long unclipped_end =
            base + static_cast<unsigned long long>(queue_batch);
        const int end = static_cast<int>(
            unclipped_end < static_cast<unsigned long long>(bulk_task_count)
                ? unclipped_end
                : static_cast<unsigned long long>(bulk_task_count));
        for (int local = static_cast<int>(base); local < end; ++local)
        {
            int heavy = 0;
            if (lane == 0)
                heavy = (heavy_suffix_flags[local >> 5] &
                         (1u << (local & 31))) != 0;
            heavy = __shfl_sync(0xffffffffu, heavy, 0);
            if (!heavy)
                dmma_numeric_regular_task(
                    a, b, output_tile_count, output_rows, output_cols,
                    output_masks, output_offsets, output_row_ptr,
                    output_col_idx, output_values, suffix_begin + local,
                    lane, tile_a_values, tile_b_values, tile_c_values);
        }
    }

    while (true)
    {
        unsigned long long offset = 0;
        if (lane == 0)
            offset = atomicAdd(fine_head, 1ull);
        offset = __shfl_sync(0xffffffffu, offset, 0);
        const unsigned long long local_wide =
            static_cast<unsigned long long>(bulk_task_count) + offset;
        if (local_wide >=
            static_cast<unsigned long long>(suffix_task_count))
            break;
        const int local = static_cast<int>(local_wide);
        int heavy = 0;
        if (lane == 0)
            heavy = (heavy_suffix_flags[local >> 5] &
                     (1u << (local & 31))) != 0;
        heavy = __shfl_sync(0xffffffffu, heavy, 0);
        if (!heavy)
            dmma_numeric_regular_task(
                a, b, output_tile_count, output_rows, output_cols,
                output_masks, output_offsets, output_row_ptr,
                output_col_idx, output_values, suffix_begin + local, lane,
                tile_a_values, tile_b_values, tile_c_values);
    }
}

/* P13 unified light scheduler.  Exactly one 128-thread/four-warp persistent
 * kernel owns every non-heavy output tile.  CTAs acquire coarse ID pages with
 * one global atomic and distribute the page through shared memory; symbolic
 * fine IDs use unit warp tickets after the coarse queue drains.  A task in the
 * fine bitset is skipped by coarse, and a task in the heavy bitset is skipped
 * by both light queues, so {coarse,fine,heavy} is an exactly-once partition. */
template <bool VisitOnly = false>
__global__ void dmma_numeric_unified_persistent_light_kernel(
    DmmaDeviceTiles a, DmmaDeviceTiles b, int output_tile_count,
    int sparse_fine_task_count, int terminal_begin, int coarse_page_size,
    const int *__restrict__ fine_task_ids,
    const uint32_t *__restrict__ fine_flags,
    const int *__restrict__ output_rows, const int *__restrict__ output_cols,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    const uint32_t *__restrict__ heavy_flags,
    unsigned long long *__restrict__ coarse_head,
    unsigned long long *__restrict__ fine_head,
    unsigned char *__restrict__ output_row_ptr,
    unsigned char *__restrict__ output_col_idx,
    MAT_VAL_TYPE *__restrict__ output_values,
    int *__restrict__ visit_counts)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x / WARP_SIZE;
    __shared__ MAT_VAL_TYPE shared_a[DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE shared_b[DMMA_WARPS_PER_BLOCK * DMMA_INPUT_ELEMS];
    __shared__ MAT_VAL_TYPE shared_c[DMMA_WARPS_PER_BLOCK * DMMA_OUTPUT_ELEMS];
    /* Keep the three WMMA buffers first: every base remains 32-byte aligned.
     * Placing the 8-byte queue header first would misalign shared_a. */
    __shared__ unsigned long long page_base;
    MAT_VAL_TYPE *tile_a_values =
        shared_a + warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_b_values =
        shared_b + warp * DMMA_INPUT_ELEMS;
    MAT_VAL_TYPE *tile_c_values =
        shared_c + warp * DMMA_OUTPUT_ELEMS;

    while (true)
    {
        if (threadIdx.x == 0)
            page_base = atomicAdd(
                coarse_head,
                static_cast<unsigned long long>(coarse_page_size));
        __syncthreads();
        const unsigned long long base = page_base;
        if (base >= static_cast<unsigned long long>(output_tile_count))
            break;
        const unsigned long long unclipped_end =
            base + static_cast<unsigned long long>(coarse_page_size);
        const int end = static_cast<int>(
            unclipped_end < static_cast<unsigned long long>(output_tile_count)
                ? unclipped_end
                : static_cast<unsigned long long>(output_tile_count));
        for (int task = static_cast<int>(base) + warp; task < end;
             task += DMMA_WARPS_PER_BLOCK)
        {
            int excluded = 0;
            if (lane == 0)
            {
                const uint32_t bit = 1u << (task & 31);
                excluded =
                    task >= terminal_begin ||
                    (((fine_flags[static_cast<unsigned int>(task) >> 5] |
                       heavy_flags[static_cast<unsigned int>(task) >> 5]) &
                      bit) != 0);
            }
            excluded = __shfl_sync(0xffffffffu, excluded, 0);
            if (!excluded)
            {
                if constexpr (VisitOnly)
                {
                    if (lane == 0)
                        atomicAdd(visit_counts + task, 1);
                }
                else
                {
                    dmma_numeric_regular_task(
                        a, b, output_tile_count, output_rows, output_cols,
                        output_masks, output_offsets, output_row_ptr,
                        output_col_idx, output_values, task, lane,
                        tile_a_values, tile_b_values, tile_c_values);
                }
            }
        }
        __syncthreads();
    }

    while (true)
    {
        const int terminal_task_count = output_tile_count - terminal_begin;
        const int fine_task_count =
            sparse_fine_task_count + terminal_task_count;
        unsigned long long ticket = 0;
        if (lane == 0)
            ticket = atomicAdd(fine_head, 1ull);
        ticket = __shfl_sync(0xffffffffu, ticket, 0);
        if (ticket >= static_cast<unsigned long long>(fine_task_count))
            break;
        const int ticket_int = static_cast<int>(ticket);
        const int task =
            ticket_int < sparse_fine_task_count
                ? fine_task_ids[ticket_int]
                : terminal_begin + (ticket_int - sparse_fine_task_count);
        int heavy = 0;
        if (lane == 0)
            heavy = (heavy_flags[static_cast<unsigned int>(task) >> 5] &
                     (1u << (task & 31))) != 0;
        heavy = __shfl_sync(0xffffffffu, heavy, 0);
        if (!heavy)
        {
            if constexpr (VisitOnly)
            {
                if (lane == 0)
                    atomicAdd(visit_counts + task, 1);
            }
            else
            {
                dmma_numeric_regular_task(
                    a, b, output_tile_count, output_rows, output_cols,
                    output_masks, output_offsets, output_row_ptr,
                    output_col_idx, output_values, task, lane, tile_a_values,
                    tile_b_values, tile_c_values);
            }
        }
    }
}

__device__ __forceinline__ void dmma_reduce_chunk_workspace_task(
    int heavy,
    const DmmaTailRecord *__restrict__ tail_records,
    const int *__restrict__ chunk_offsets,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    const MAT_VAL_TYPE *__restrict__ partial_workspace,
    MAT_VAL_TYPE *__restrict__ output_values, int lane)
{
    const int task = tail_records[heavy].id;
    const int first = chunk_offsets[heavy];
    const int last = chunk_offsets[heavy + 1];
    const uint64_t mask = output_masks[task];
    const int output_begin = output_offsets[task];
    for (int position = lane; position < DMMA_OUTPUT_ELEMS;
         position += WARP_SIZE)
    {
        if ((mask & (1ull << position)) == 0)
            continue;
        MAT_VAL_TYPE sum = MAT_VAL_TYPE(0);
        for (int chunk = first; chunk < last; ++chunk)
            sum += partial_workspace[static_cast<std::size_t>(chunk) *
                                         DMMA_OUTPUT_ELEMS +
                                     position];
        const uint64_t lower =
            position == 0 ? 0ull : mask & ((1ull << position) - 1ull);
        output_values[output_begin + __popcll(lower)] = sum;
    }
}

__global__ void dmma_reduce_chunk_workspace_kernel(
    int heavy_task_count,
    const DmmaTailRecord *__restrict__ tail_records,
    const int *__restrict__ chunk_offsets,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    const MAT_VAL_TYPE *__restrict__ partial_workspace,
    MAT_VAL_TYPE *__restrict__ output_values)
{
    const std::size_t heavy = dmma_global_thread_index() / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (heavy >= static_cast<std::size_t>(heavy_task_count))
        return;
    dmma_reduce_chunk_workspace_task(
        static_cast<int>(heavy), tail_records, chunk_offsets, output_masks,
        output_offsets, partial_workspace, output_values, lane);
}

/* The early mode keeps reduction under the same global CTA-pressure cap as
 * its chunk queue. Four independent warps claim heavy parents; resetting the
 * shared queue head is ordered after chunks on the same heavy stream. */
__global__ void dmma_reduce_chunk_workspace_persistent_kernel(
    int heavy_task_count,
    const DmmaTailRecord *__restrict__ tail_records,
    const int *__restrict__ chunk_offsets,
    const uint64_t *__restrict__ output_masks,
    const int *__restrict__ output_offsets,
    const MAT_VAL_TYPE *__restrict__ partial_workspace,
    MAT_VAL_TYPE *__restrict__ output_values, int *queue_head)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    while (true)
    {
        int heavy = 0;
        if (lane == 0)
            heavy = atomicAdd(queue_head, 1);
        heavy = __shfl_sync(0xffffffffu, heavy, 0);
        if (heavy >= heavy_task_count)
            break;
        dmma_reduce_chunk_workspace_task(
            heavy, tail_records, chunk_offsets, output_masks,
            output_offsets, partial_workspace, output_values, lane);
    }
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
                          "clear empty fused output nnz") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(),
                          "complete empty fused output metadata"))
            goto failure;
        /* Publish only device-ready empty output metadata.  The caller may
         * close Core immediately after this helper returns. */
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
 * matrices.  Numeric work uses either the uniform direct kernel or the gated
 * sparse-tail light/chunk path below.
 */
static inline bool dmma_tilespgemm(const DmmaDeviceTiles &a,
                                   const DmmaDeviceTiles &b,
                                   SMatrix *output, DmmaSpGemmStats *stats,
                                   const DmmaNumericScheduleConfig *schedule_config = nullptr)
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
    DmmaNumericScheduleConfig schedule;
    if (schedule_config != nullptr)
        schedule = *schedule_config;
    if (schedule.max_chunks < 2)
        schedule.max_chunks = 2;
    if (schedule.max_chunks > 32)
        schedule.max_chunks = 32;
    const double resolved_chunk_target =
        schedule.chunk_target > 0.0 ? schedule.chunk_target :
                                      schedule.split_threshold;
    const double resolved_unified_fine_threshold =
        schedule.unified_fine_threshold > 0.0
            ? schedule.unified_fine_threshold
            : schedule.split_threshold * 0.5;
    if ((schedule.mode != DMMA_SCHEDULE_DIRECT &&
         schedule.mode != DMMA_SCHEDULE_SPLIT_CTA &&
         schedule.mode != DMMA_SCHEDULE_SPLIT_PERSISTENT &&
         schedule.mode != DMMA_SCHEDULE_SPLIT_FLAT &&
         schedule.mode != DMMA_SCHEDULE_TILE_TAIL_QUEUE &&
         schedule.mode != DMMA_SCHEDULE_TILE_EARLY_SPLIT) ||
        (schedule.direct_numeric_layout !=
             DMMA_DIRECT_NUMERIC_TILE_DYNAMIC &&
         schedule.direct_numeric_layout !=
             DMMA_DIRECT_NUMERIC_ROW_STATIC &&
         schedule.direct_numeric_layout !=
             DMMA_DIRECT_NUMERIC_ROW_DYNAMIC) ||
        (schedule.mode != DMMA_SCHEDULE_DIRECT &&
         schedule.direct_numeric_layout !=
             DMMA_DIRECT_NUMERIC_TILE_DYNAMIC) ||
        !dmma_row_queue_batch_valid(schedule.row_queue_batch) ||
        (!schedule.row_dynamic_auto &&
         schedule.direct_numeric_layout !=
             DMMA_DIRECT_NUMERIC_ROW_DYNAMIC &&
         schedule.row_queue_batch != 1) ||
        (schedule.row_dynamic_auto &&
         schedule.mode != DMMA_SCHEDULE_DIRECT) ||
        (schedule.cost_balanced &&
         (!schedule.tileflex16_symbolic ||
          schedule.mode != DMMA_SCHEDULE_DIRECT ||
          schedule.row_dynamic_auto ||
          schedule.direct_numeric_layout !=
              DMMA_DIRECT_NUMERIC_TILE_DYNAMIC)) ||
        schedule.cost_workers_per_sm < 1 ||
        schedule.cost_workers_per_sm > 16 ||
        !std::isfinite(schedule.row_dynamic_threshold) ||
        schedule.row_dynamic_threshold < 1.0 ||
        !(schedule.split_threshold > 0.0) ||
        !std::isfinite(schedule.split_threshold) ||
        schedule.split_threshold > FLT_MAX ||
        !(schedule.chunk_target >= 0.0) ||
        !std::isfinite(schedule.chunk_target) ||
        schedule.chunk_target > FLT_MAX ||
        !(resolved_chunk_target > 0.0) ||
        resolved_chunk_target > FLT_MAX ||
        (schedule.light_policy != DMMA_LIGHT_STATIC &&
         schedule.light_policy != DMMA_LIGHT_PERSISTENT_SUFFIX &&
         schedule.light_policy != DMMA_LIGHT_PERSISTENT_UNIFIED) ||
        (schedule.mode == DMMA_SCHEDULE_SPLIT_FLAT &&
         schedule.light_policy != DMMA_LIGHT_STATIC) ||
        (schedule.mode == DMMA_SCHEDULE_TILE_TAIL_QUEUE &&
         (schedule.light_policy != DMMA_LIGHT_STATIC ||
          schedule.symbolic_admission !=
              DMMA_SYMBOLIC_ADMISSION_FUSED_EXACT)) ||
        (schedule.mode == DMMA_SCHEDULE_TILE_EARLY_SPLIT &&
         (schedule.light_policy != DMMA_LIGHT_STATIC ||
          schedule.symbolic_admission !=
              DMMA_SYMBOLIC_ADMISSION_FUSED_EXACT ||
          schedule.launch_policy != DMMA_SPLIT_LAUNCH_HEAVY_FIRST)) ||
        !(schedule.critical_q_min >= 0.0) ||
        !(schedule.critical_q_min <= 1.0) ||
        !std::isfinite(schedule.critical_q_min) ||
        schedule.suffix_workers_per_sm < 0 ||
        schedule.suffix_workers_per_sm > 32 ||
        (schedule.suffix_auto_basis != DMMA_SUFFIX_AUTO_TASK &&
         schedule.suffix_auto_basis != DMMA_SUFFIX_AUTO_WORK &&
         schedule.suffix_auto_basis != DMMA_SUFFIX_AUTO_REGULAR_WORK) ||
        schedule.suffix_queue_batch < 1 ||
        schedule.suffix_queue_batch > 1024 ||
        schedule.suffix_fine_tasks_per_worker < 0 ||
        schedule.suffix_fine_tasks_per_worker > 1024 ||
        schedule.unified_page_size < 1 ||
        schedule.unified_page_size > 4096 ||
        (schedule.flat_warps_per_cta != 1 &&
         schedule.flat_warps_per_cta != 2 &&
         schedule.flat_warps_per_cta != 4) ||
        schedule.unified_workers_per_sm < 0 ||
        schedule.unified_workers_per_sm > 32 ||
        !(schedule.unified_fine_threshold >= 0.0) ||
        !std::isfinite(schedule.unified_fine_threshold) ||
        schedule.unified_fine_threshold > FLT_MAX ||
        !(resolved_unified_fine_threshold > 0.0) ||
        resolved_unified_fine_threshold > FLT_MAX ||
        schedule.unified_fine_capacity < 1 ||
        schedule.tail_record_capacity < 1 ||
        schedule.maybe_candidate_capacity < 1 ||
        !(schedule.max_heavy_fraction >= 0.0) ||
        !(schedule.max_heavy_fraction <= 1.0) ||
        !std::isfinite(schedule.max_heavy_fraction) ||
        (schedule.launch_policy != DMMA_SPLIT_LAUNCH_LIGHT_FIRST &&
         schedule.launch_policy != DMMA_SPLIT_LAUNCH_HEAVY_FIRST &&
         schedule.launch_policy != DMMA_SPLIT_LAUNCH_HEAVY_PRIORITY) ||
        !(schedule.cost.scan >= 0.0) ||
        !std::isfinite(schedule.cost.scan) ||
        schedule.cost.scan > FLT_MAX ||
        !(schedule.cost.match >= 0.0) ||
        !std::isfinite(schedule.cost.match) ||
        schedule.cost.match > FLT_MAX ||
        !(schedule.cost.output >= 0.0) ||
        !std::isfinite(schedule.cost.output) ||
        schedule.cost.output > FLT_MAX ||
        !(schedule.cost.intercept >= 0.0) ||
        !std::isfinite(schedule.cost.intercept) ||
        schedule.cost.intercept > FLT_MAX ||
        (schedule.symbolic_admission !=
             DMMA_SYMBOLIC_ADMISSION_SEPARATE &&
         schedule.symbolic_admission !=
             DMMA_SYMBOLIC_ADMISSION_FUSED_EXACT) ||
        schedule.exact_forward_min_row_pairs == 0 ||
        !(schedule.exact_forward_min_ratio >= 1.0) ||
        !std::isfinite(schedule.exact_forward_min_ratio) ||
        (schedule.exact_forward_spa &&
         (schedule.mode != DMMA_SCHEDULE_DIRECT ||
          schedule.row_dynamic_auto ||
          schedule.direct_numeric_layout !=
              DMMA_DIRECT_NUMERIC_TILE_DYNAMIC ||
          schedule.symbolic_admission !=
              DMMA_SYMBOLIC_ADMISSION_SEPARATE ||
          schedule.collect_symbolic_load)) ||
        (!schedule.low_fill_exact_tile && schedule.low_fill_q != 0) ||
        (schedule.low_fill_exact_tile &&
         (!dmma_low_fill_q_valid(schedule.low_fill_q) ||
          schedule.mode != DMMA_SCHEDULE_DIRECT ||
          schedule.row_dynamic_auto ||
          schedule.direct_numeric_layout !=
              DMMA_DIRECT_NUMERIC_TILE_DYNAMIC ||
          schedule.exact_forward_spa ||
          schedule.symbolic_admission !=
              DMMA_SYMBOLIC_ADMISSION_SEPARATE ||
          schedule.collect_symbolic_load)) ||
        !(schedule.symbolic_load_quantum > 0.0) ||
        !std::isfinite(schedule.symbolic_load_quantum) ||
        schedule.symbolic_load_quantum > FLT_MAX ||
        schedule.symbolic_wave_ctas_per_sm < 1 ||
        schedule.symbolic_critical_waves < 1)
        return false;
    stats->schedule_mode = schedule.mode;
    stats->cost_balanced_requested = schedule.cost_balanced;
    stats->direct_numeric_layout = schedule.direct_numeric_layout;
    stats->row_dynamic_queue_batch = schedule.row_queue_batch;
    stats->row_gate_requested = schedule.row_dynamic_auto;
    stats->row_gate_threshold = schedule.row_dynamic_threshold;
    stats->light_policy = schedule.light_policy;
    stats->suffix_auto_basis = schedule.suffix_auto_basis;
    stats->reduction_mode = schedule.reduction;
    stats->launch_policy = schedule.launch_policy;
    stats->symbolic_admission = schedule.symbolic_admission;
    stats->exact_forward_spa_requested = schedule.exact_forward_spa;
    stats->exact_forward_spa_reason =
        schedule.exact_forward_spa ? DMMA_EXACT_FORWARD_ZERO_SELECTED
                                   : DMMA_EXACT_FORWARD_NOT_REQUESTED;
    stats->low_fill_exact_tile_requested = schedule.low_fill_exact_tile;
    stats->low_fill_q = schedule.low_fill_q;
    stats->tail_split_forced = schedule.force_tail_split;
    stats->early_split_requested =
        schedule.mode == DMMA_SCHEDULE_TILE_EARLY_SPLIT;
    if (schedule.mode != DMMA_SCHEDULE_DIRECT &&
        !schedule.reuse_symbolic_task_counts)
        stats->tail_gate_reason = DMMA_TAIL_GATE_REUSE_DISABLED;
    bool low_fill_ready = false;
    if (schedule.low_fill_exact_tile)
    {
        std::size_t metadata_bytes = 0;
        const bool metadata_size_valid = dmma_low_fill_metadata_bytes(
            a.tile_row_count, b.tile_col_count, &metadata_bytes);
        stats->low_fill_metadata_bytes =
            metadata_size_valid ? metadata_bytes : 0;
        if (a.low_fill_metadata_overflow ||
            b.low_fill_metadata_overflow)
            stats->low_fill_reason = DMMA_LOW_FILL_METADATA_OVERFLOW;
        else if (!metadata_size_valid ||
                 metadata_bytes > DMMA_LOW_FILL_METADATA_BUDGET_BYTES)
            stats->low_fill_reason = DMMA_LOW_FILL_METADATA_BUDGET;
        else if (!a.row_tile_nnz_sum_valid ||
                 !b.col_tile_nnz_sum_valid ||
                 (a.tile_row_count > 0 &&
                  a.row_tile_nnz_sum == nullptr) ||
                 (b.tile_col_count > 0 &&
                  b.col_tile_nnz_sum == nullptr) ||
                 (a.num_tiles > 0 && a.structural_nnz == 0) ||
                 (b.num_tiles > 0 && b.structural_nnz == 0) ||
                 a.structural_nnz >
                     32ull * static_cast<unsigned long long>(a.num_tiles) ||
                 b.structural_nnz >
                     32ull * static_cast<unsigned long long>(b.num_tiles))
            stats->low_fill_reason = DMMA_LOW_FILL_METADATA_MISSING;
        else
        {
            stats->low_fill_global_guard = dmma_low_fill_global_guard(
                a.structural_nnz,
                static_cast<unsigned long long>(a.num_tiles),
                b.structural_nnz,
                static_cast<unsigned long long>(b.num_tiles),
                schedule.low_fill_q);
            if (stats->low_fill_global_guard)
            {
                low_fill_ready = true;
                stats->low_fill_reason = DMMA_LOW_FILL_ENABLED;
            }
            else
                stats->low_fill_reason =
                    DMMA_LOW_FILL_GLOBAL_GATE_REJECTED;
        }
    }
    timeval total_begin{}, total_end{}, begin{}, end{};

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
    int *d_exact_forward_flags = nullptr;
    int *d_exact_forward_rows = nullptr;
    unsigned long long *d_exact_forward_mask_spas = nullptr;
    DmmaExactForwardSpaDeviceSummary *d_exact_forward_summary = nullptr;
    DmmaExactForwardSpaDeviceSummary exact_forward_summary{};
    int exact_forward_row_count = 0;
    int exact_forward_batch_capacity = 0;
    int *d_maybe_candidate_ids = nullptr;
    DmmaTailAppendState *d_maybe_append_state = nullptr;
    DmmaTailRecord *d_tail_records = nullptr;
    DmmaTailRecord *d_window_tail_records = nullptr;
    DmmaTailAppendState *d_tail_append_state = nullptr;
    DmmaSymbolicLoadSummary *d_symbolic_load_summary = nullptr;
    DmmaUnifiedReplaySummary *d_unified_replay_summary = nullptr;
    int *d_output_rows = nullptr;
    int *d_output_cols = nullptr;
    uint64_t *d_output_masks = nullptr;
    int *d_output_nnz = nullptr;
    int *d_output_row_ptr = nullptr;
    std::uint64_t *d_output_numeric_work = nullptr;
    std::uint64_t *d_cost_sort_keys = nullptr;
    int *d_cost_task_order = nullptr;
    int *d_cost_queue_head = nullptr;
    int cost_worker_blocks = 0;
    int output_tile_count = 0;
    int output_nnz = 0;
    rtt::super16::SymbolicOutput super16_output;
    unsigned char *d_output_tile_row_ptr = nullptr;
    unsigned char *d_output_value_cols = nullptr;
    MAT_VAL_TYPE *d_output_values = nullptr;
    uint32_t *d_heavy_flags = nullptr;
    int *d_unified_fine_ids = nullptr;
    uint32_t *d_unified_fine_flags = nullptr;
    unsigned long long *d_unified_coarse_head = nullptr;
    unsigned long long *d_unified_fine_head = nullptr;
    int *d_invalid_tail_records = nullptr;
    int *d_chunk_offsets = nullptr;
    DmmaChunkDescriptor *d_chunk_descriptors = nullptr;
    MAT_VAL_TYPE *d_partial_workspace = nullptr;
    uint32_t *d_chunk_sm_ids = nullptr;
    uint32_t *d_row_worker_sm_ids = nullptr;
    int *d_row_dynamic_next_row = nullptr;
    DmmaRowGateDeviceSummary *d_row_gate_summary = nullptr;
    int row_worker_count = 0;
    int row_worker_device_sm_count = 0;
    int *d_persistent_queue_head = nullptr;
    unsigned long long *d_suffix_bulk_head = nullptr;
    unsigned long long *d_suffix_fine_head = nullptr;
    int critical_q_begin = 0;
    int prefix_task_count = 0;
    int suffix_task_count = 0;
    int suffix_worker_blocks = 0;
    int suffix_workers_per_sm = 0;
    int suffix_fine_tasks = 0;
    int suffix_bulk_tasks = 0;
    int unified_fine_capacity = 0;
    int unified_fine_task_count = 0;
    int unified_sparse_fine_task_count = 0;
    int unified_worker_blocks = 0;
    int unified_terminal_begin = 0;
    uint32_t unified_fine_threshold_units = 0;
    bool unified_symbolic_ready = false;
    bool unified_light_active = false;
    int heavy_task_count = 0;
    int split_chunk_count = 0;
    int tail_record_capacity = 0;
    int tail_record_count = 0;
    bool tail_record_overflow = false;
    int maybe_candidate_capacity = 0;
    int maybe_candidate_count = 0;
    bool maybe_candidate_overflow = false;
    int symbolic_critical_task_begin = 0;
    bool admission_count_event_recorded = false;
    bool use_direct_numeric = schedule.mode == DMMA_SCHEDULE_DIRECT;
    const bool row_dynamic_auto_requested =
        use_direct_numeric && schedule.row_dynamic_auto;
    bool row_static_requested =
        use_direct_numeric && !row_dynamic_auto_requested &&
        schedule.direct_numeric_layout == DMMA_DIRECT_NUMERIC_ROW_STATIC;
    bool row_dynamic_requested =
        use_direct_numeric && !row_dynamic_auto_requested &&
        schedule.direct_numeric_layout == DMMA_DIRECT_NUMERIC_ROW_DYNAMIC;
    bool row_worker_requested =
        row_dynamic_auto_requested || row_static_requested ||
        row_dynamic_requested;
    const bool flat_grid_requested =
        schedule.mode == DMMA_SCHEDULE_SPLIT_FLAT;
    const bool tile_tail_queue_requested =
        schedule.mode == DMMA_SCHEDULE_TILE_TAIL_QUEUE;
    const bool tile_early_split_requested =
        schedule.mode == DMMA_SCHEDULE_TILE_EARLY_SPLIT;
    bool sparse_tail_records_ready = false;
    const bool persistent_suffix_requested =
        schedule.mode != DMMA_SCHEDULE_DIRECT &&
        schedule.light_policy == DMMA_LIGHT_PERSISTENT_SUFFIX;
    const bool unified_light_requested =
        schedule.mode != DMMA_SCHEDULE_DIRECT &&
        schedule.light_policy == DMMA_LIGHT_PERSISTENT_UNIFIED;
    /* Both suffix-persistent execution and the flat late-wave scheduler split
     * only the final exact-output window.  q is not known until after the keep
     * scan, so separate admission applies this bound in exact replay while
     * fused admission applies it after candidate-to-output mapping. */
    const bool late_output_window_requested =
        persistent_suffix_requested || flat_grid_requested;
    const bool collect_sparse_tail_records =
        schedule.mode != DMMA_SCHEDULE_DIRECT &&
        schedule.reuse_symbolic_task_counts;
    /* Unified production uses the bounded post-scan joint replay.  The fused
     * full-candidate counter remains available to legacy and explicit dense
     * telemetry paths, but is deliberately not selected here. */
    const bool unified_joint_replay =
        unified_light_requested && !schedule.collect_symbolic_load;
    stats->maybe_scope_joint = unified_joint_replay;
    const bool fused_symbolic_admission =
        collect_sparse_tail_records &&
        !unified_joint_replay &&
        schedule.symbolic_admission ==
            DMMA_SYMBOLIC_ADMISSION_FUSED_EXACT;
    const bool auto_suffix_symbolic_load =
        persistent_suffix_requested && schedule.suffix_workers_per_sm == 0 &&
        schedule.suffix_auto_basis != DMMA_SUFFIX_AUTO_TASK;
    const bool collect_symbolic_load =
        auto_suffix_symbolic_load || schedule.collect_symbolic_load;
    DmmaPartialReductionMode effective_reduction = schedule.reduction;
    DmmaSplitAsyncState local_split_async{};
    const bool split_context_borrowed = schedule.split_context != nullptr;
    DmmaSplitAsyncState &split_async =
        split_context_borrowed ? *schedule.split_context : local_split_async;
    const bool admission_events_ready =
        split_context_borrowed && dmma_split_async_state_ready(split_async) &&
        split_async.launch_policy == schedule.launch_policy;
    if (collect_sparse_tail_records && split_context_borrowed &&
        !admission_events_ready)
    {
        std::fprintf(stderr,
                     "Borrowed split context is uninitialized or uses "
                     "a mismatched launch policy.\n");
        return false;
    }
    unsigned int numeric_blocks = 0;
    unsigned int heavy_task_blocks = 0;
#ifdef DMMA_ENABLE_TIMELINE_TRACE
    DmmaTimelineView timeline{};
    std::size_t timeline_slots = 0;
    const char *timeline_path = nullptr;
#endif
    /* Row-worker experiment infrastructure is prepared before the internal
     * Core boundary: SM-placement audit storage and one identically prepared
     * control head for both layouts (unused by static).  The timed kernels perform one
     * uint32 SM-ID store per CTA; dynamic additionally performs only the row
     * claims intrinsic to the scheduling policy. */
    if (row_worker_requested && a.tile_row_count > 0)
    {
        int device = 0;
        if (!dmma_cuda_ok(cudaGetDevice(&device),
                          "read row-worker CUDA device") ||
            !dmma_cuda_ok(cudaDeviceGetAttribute(
                              &row_worker_device_sm_count,
                              cudaDevAttrMultiProcessorCount, device),
                          "read row-worker SM count"))
            return false;
        row_worker_count = dmma_row_static_worker_count(
            a.tile_row_count, row_worker_device_sm_count);
        if (row_worker_count <= 0 ||
            (row_dynamic_auto_requested &&
             row_worker_count > DMMA_ROW_GATE_MAX_WORKERS) ||
            dmma_row_dynamic_expected_final_head(
                a.tile_row_count, row_worker_count,
                schedule.row_queue_batch) < 0 ||
            !dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(
                                  &d_row_worker_sm_ids),
                              static_cast<std::size_t>(
                                  row_worker_count) *
                                  sizeof(uint32_t)),
                          "allocate row-worker SM IDs") ||
            !dmma_cuda_ok(cudaMemset(
                              d_row_worker_sm_ids, 0xff,
                              static_cast<std::size_t>(
                                  row_worker_count) *
                                  sizeof(uint32_t)),
                          "clear row-worker SM IDs") ||
            !dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(
                                  &d_row_dynamic_next_row),
                              sizeof(*d_row_dynamic_next_row)),
                          "allocate row-worker control head") ||
            !dmma_cuda_ok(cudaMemset(
                              d_row_dynamic_next_row, 0,
                              sizeof(*d_row_dynamic_next_row)),
                          "initialize row-worker control head") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(),
                          "complete untimed row-worker setup"))
        {
            cudaFree(d_row_worker_sm_ids);
            cudaFree(d_row_dynamic_next_row);
            return false;
        }
    }
    gettimeofday(&total_begin, nullptr);
    if (schedule.tileflex16_symbolic)
    {
        /* TileFlex produces the same final 8x8 task ABI used by the original
         * numeric path.  Direct and the two audited sparse-heavy schedulers
         * are supported; broader split/unified experiments remain closed. */
        if ((schedule.mode != DMMA_SCHEDULE_DIRECT &&
             schedule.mode != DMMA_SCHEDULE_TILE_TAIL_QUEUE &&
             schedule.mode != DMMA_SCHEDULE_TILE_EARLY_SPLIT) ||
            schedule.light_policy != DMMA_LIGHT_STATIC ||
            (schedule.mode == DMMA_SCHEDULE_DIRECT &&
             schedule.symbolic_admission !=
                 DMMA_SYMBOLIC_ADMISSION_SEPARATE) ||
            schedule.collect_symbolic_load || schedule.exact_forward_spa ||
            schedule.low_fill_exact_tile)
        {
            std::fprintf(stderr,
                         "tileflex16 symbolic supports direct or audited "
                         "tile-tail/early-split with static light policy.\n");
            goto failure;
        }
        static DmmaSuper16IndexCache cache;
        static DmmaTileFlexTailHostCache tail_cache;
        const bool tileflex_tail_cache_enabled = [] {
            const char *value = std::getenv("RTT_TILEFLEX_TAIL_CACHE");
            return value != nullptr && std::strcmp(value, "1") == 0;
        }();
        int tileflex_device = -1;
        const bool tail_cache_device_valid =
            dmma_cuda_ok(cudaGetDevice(&tileflex_device),
                         "read TileFlex tail-cache device");
        if (!tail_cache_device_valid)
            goto failure;
        bool tileflex_tail_cache_hit =
            tileflex_tail_cache_enabled && collect_sparse_tail_records &&
            tail_cache.matches(a, b, tileflex_device, schedule,
                               resolved_chunk_target);
        if (!cache.prepare(a, b, nullptr) ||
            !rtt::super16::run_parent_symbolic(
                a, b, cache.a_index.view(), cache.b_index.view(), nullptr,
                &super16_output,
                schedule.cost_balanced ||
                    (collect_sparse_tail_records &&
                     !tileflex_tail_cache_hit)))
            goto failure;
        d_output_row_ptr = super16_output.native.row_ptr.release();
        d_output_rows = super16_output.native.rows.release();
        d_output_cols = super16_output.native.cols.release();
        d_output_masks = super16_output.native.masks.release();
        d_output_nnz = super16_output.native.nnz_offsets.release();
        d_output_numeric_work =
            super16_output.native.numeric_work.release();
        output_tile_count = super16_output.native.tile_count;
        output_nnz = super16_output.native.nnz;
        stats->candidate_ms = super16_output.metrics.candidate_queue_ms;
        stats->symbolic_ms = std::max(
            0.0, super16_output.metrics.symbolic_ms - stats->candidate_ms);
        stats->exact_kernel_ms = super16_output.metrics.exact_queue_ms;
        stats->symbolic_finalize_ms = super16_output.metrics.finalize_ms;
        stats->candidate_tiles =
            super16_output.metrics.parent_c_candidates;
        stats->output_tiles = output_tile_count;
        stats->output_nnz = output_nnz;
        if (collect_sparse_tail_records && output_tile_count > 0)
        {
            timeval tail_begin{}, tail_end{};
            gettimeofday(&tail_begin, nullptr);
            if (tileflex_tail_cache_hit &&
                tail_cache.output_tiles != output_tile_count)
                tileflex_tail_cache_hit = false;
            /* A non-forced run is rejected as soon as the exact heavy
             * fraction exceeds max_heavy_fraction.  Reserving one record
             * beyond that admissible count is sufficient to prove the gate
             * must fall back, and avoids the old O(|C_tiles|)/1M-record
             * speculative allocation on every single-shot call. */
            const double admissible_records_wide = std::floor(
                static_cast<double>(output_tile_count) *
                schedule.max_heavy_fraction);
            const int admissible_records = static_cast<int>(std::min(
                admissible_records_wide,
                static_cast<double>(std::numeric_limits<int>::max() - 1)));
            const int gate_detection_capacity = schedule.force_tail_split
                ? output_tile_count
                : std::min(output_tile_count, admissible_records + 1);
            tail_record_capacity = std::min(
                gate_detection_capacity, schedule.tail_record_capacity);
            const std::size_t capacity_record_bytes =
                static_cast<std::size_t>(tail_record_capacity) *
                sizeof(DmmaTailRecord);
            const std::size_t cached_record_bytes =
                tileflex_tail_cache_hit
                    ? tail_cache.records.size() * sizeof(DmmaTailRecord)
                    : 0;
            if (tail_record_capacity <= 0 ||
                (!tileflex_tail_cache_hit &&
                 d_output_numeric_work == nullptr) ||
                !dmma_cuda_ok(cudaMalloc(
                                  reinterpret_cast<void **>(&d_tail_records),
                                  tileflex_tail_cache_hit
                                      ? std::max<std::size_t>(
                                            cached_record_bytes, 1)
                                      : capacity_record_bytes),
                              "allocate TileFlex tail records"))
                goto symbolic_failure;
            if (tileflex_tail_cache_hit)
            {
                tail_record_count =
                    static_cast<int>(tail_cache.records.size());
                if (cached_record_bytes > 0 &&
                    !dmma_cuda_ok(cudaMemcpy(
                                      d_tail_records,
                                      tail_cache.records.data(),
                                      cached_record_bytes,
                                      cudaMemcpyHostToDevice),
                                  "upload cached TileFlex tail records"))
                    goto symbolic_failure;
                tail_record_overflow = false;
                if (admission_events_ready &&
                    (!dmma_cuda_ok(cudaEventRecord(
                                       split_async.admission_filter_start, 0),
                                   "record cached TileFlex filter start") ||
                     !dmma_cuda_ok(cudaEventRecord(
                                       split_async.admission_filter_stop, 0),
                                   "record cached TileFlex filter stop") ||
                     !dmma_cuda_ok(cudaEventRecord(
                                       split_async.admission_count_start, 0),
                                   "record cached TileFlex count start") ||
                     !dmma_cuda_ok(cudaEventRecord(
                                       split_async.admission_count_stop, 0),
                                   "record cached TileFlex count stop")))
                    goto symbolic_failure;
                admission_count_event_recorded = admission_events_ready;
            }
            else if (!dmma_cuda_ok(cudaMalloc(
                                  reinterpret_cast<void **>(
                                      &d_tail_append_state),
                                  sizeof(DmmaTailAppendState)),
                              "allocate TileFlex tail state") ||
                !dmma_cuda_ok(cudaMemset(d_tail_append_state, 0,
                                         sizeof(DmmaTailAppendState)),
                              "clear TileFlex tail state"))
                goto symbolic_failure;
            if (!tileflex_tail_cache_hit)
            {
            unsigned int tail_blocks = 0;
            if (!dmma_launch_blocks(
                    static_cast<std::size_t>(output_tile_count), 256,
                    &tail_blocks, "TileFlex tail admission"))
                goto symbolic_failure;
            /* TileFlex needs no loose maybe-ID pass, but the borrowed split
             * context's timing ABI still requires a valid filter interval. */
            if (admission_events_ready &&
                (!dmma_cuda_ok(cudaEventRecord(
                                   split_async.admission_filter_start, 0),
                               "record TileFlex empty filter start") ||
                 !dmma_cuda_ok(cudaEventRecord(
                                   split_async.admission_filter_stop, 0),
                               "record TileFlex empty filter stop") ||
                 !dmma_cuda_ok(cudaEventRecord(
                                   split_async.admission_count_start, 0),
                               "record TileFlex tail admission start")))
                goto symbolic_failure;
            dmma_tileflex_build_tail_records_kernel<<<tail_blocks, 256>>>(
                output_tile_count, d_output_numeric_work,
                static_cast<float>(schedule.cost.scan),
                static_cast<float>(schedule.cost.match),
                static_cast<float>(schedule.split_threshold),
                static_cast<float>(resolved_chunk_target),
                schedule.max_chunks, d_tail_records, tail_record_capacity,
                d_tail_append_state);
            if (admission_events_ready &&
                !dmma_cuda_ok(cudaEventRecord(
                                  split_async.admission_count_stop, 0),
                              "record TileFlex tail admission stop"))
                goto symbolic_failure;
            admission_count_event_recorded = admission_events_ready;
            DmmaTailAppendState tail_state{};
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "launch TileFlex tail admission") ||
                !dmma_cuda_ok(cudaMemcpy(
                                  &tail_state, d_tail_append_state,
                                  sizeof(tail_state), cudaMemcpyDeviceToHost),
                              "read TileFlex tail state"))
                goto symbolic_failure;
            tail_record_count = tail_state.count;
            tail_record_overflow = tail_state.overflow != 0 ||
                                   tail_record_count > tail_record_capacity;
            if (tileflex_tail_cache_enabled && !tail_record_overflow)
            {
                tail_cache.records.resize(
                    static_cast<std::size_t>(tail_record_count));
                const std::size_t used_record_bytes =
                    tail_cache.records.size() * sizeof(DmmaTailRecord);
                if (used_record_bytes > 0 &&
                    !dmma_cuda_ok(cudaMemcpy(
                                      tail_cache.records.data(),
                                      d_tail_records, used_record_bytes,
                                      cudaMemcpyDeviceToHost),
                                  "cache TileFlex tail records"))
                    goto symbolic_failure;
                tail_cache.device = tileflex_device;
                tail_cache.a_row_ptr = a.tile_row_ptr;
                tail_cache.a_col_idx = a.tile_col_idx;
                tail_cache.b_row_ptr = b.tile_row_ptr;
                tail_cache.b_col_idx = b.tile_col_idx;
                tail_cache.output_tiles = output_tile_count;
                tail_cache.scan = schedule.cost.scan;
                tail_cache.match = schedule.cost.match;
                tail_cache.threshold = schedule.split_threshold;
                tail_cache.chunk_target = resolved_chunk_target;
                tail_cache.max_chunks = schedule.max_chunks;
                tail_cache.valid = true;
            }
            }
            stats->tail_record_capacity = tail_record_capacity;
            stats->tail_record_count = tail_record_count;
            stats->tail_record_overflow = tail_record_overflow;
            stats->tail_record_count_is_lower_bound = tail_record_overflow;
            stats->tail_record_fraction =
                static_cast<double>(tail_record_count) / output_tile_count;
            stats->symbolic_task_count_bytes =
                (tileflex_tail_cache_hit ? cached_record_bytes
                                         : capacity_record_bytes) +
                (tileflex_tail_cache_hit ? 0
                                         : sizeof(DmmaTailAppendState)) +
                (tileflex_tail_cache_hit
                     ? 0
                     : static_cast<std::size_t>(output_tile_count) *
                           sizeof(std::uint64_t));
            if (tail_record_overflow)
                stats->tail_gate_reason = DMMA_TAIL_GATE_RECORD_OVERFLOW;
            else if (tail_record_count == 0)
                stats->tail_gate_reason = DMMA_TAIL_GATE_ZERO_HEAVY;
            else if (!schedule.force_tail_split &&
                     stats->tail_record_fraction >
                         schedule.max_heavy_fraction)
                stats->tail_gate_reason = DMMA_TAIL_GATE_HEAVY_FRACTION;
            else
            {
                stats->tail_gate_reason = DMMA_TAIL_GATE_ENABLED;
                sparse_tail_records_ready = true;
            }
            gettimeofday(&tail_end, nullptr);
            stats->symbolic_finalize_ms +=
                dmma_elapsed_ms(tail_begin, tail_end);
        }
        goto numeric_phase;
    }
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
        if (low_fill_ready)
        {
            low_fill_ready = false;
            stats->low_fill_reason = DMMA_LOW_FILL_OVERSIZED_SYMBOLIC;
        }
        if (schedule.exact_forward_spa)
        {
            stats->exact_forward_spa_reason =
                DMMA_EXACT_FORWARD_OVERSIZED_EXISTING;
            stats->exact_forward_partition_complete = false;
        }
        if (schedule.mode != DMMA_SCHEDULE_DIRECT)
            stats->tail_gate_reason =
                DMMA_TAIL_GATE_OVERSIZED_SYMBOLIC;
        gettimeofday(&end, nullptr);
        stats->candidate_ms = dmma_elapsed_ms(begin, end);
        stats->candidate_tiles = candidate_count_total;
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
            dmma_close_core_endpoint(total_begin, &total_end, stats);
            if (schedule.mode != DMMA_SCHEDULE_DIRECT)
                dmma_print_tail_fusion_diagnostic(
                    "oversized-empty", schedule, *stats);
            if (schedule.materialize_output)
            {
                gettimeofday(&begin, nullptr);
                if (!dmma_copy_output_to_host(
                        a.rows, b.cols, a.tile_row_count, b.tile_col_count,
                        0, 0, d_output_row_ptr, nullptr, d_output_nnz,
                        nullptr, nullptr, nullptr, output))
                    goto symbolic_failure;
                gettimeofday(&end, nullptr);
                stats->output_copy_ms = dmma_elapsed_ms(begin, end);
                stats->output_materialized = true;
            }
            cudaFree(d_output_nnz);
            cudaFree(d_output_row_ptr);
            cudaFree(d_row_worker_sm_ids);
            cudaFree(d_row_dynamic_next_row);
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
        if (schedule.mode != DMMA_SCHEDULE_DIRECT)
            stats->tail_gate_reason = DMMA_TAIL_GATE_ZERO_HEAVY;
        dmma_close_core_endpoint(total_begin, &total_end, stats);
        if (schedule.mode != DMMA_SCHEDULE_DIRECT)
            dmma_print_tail_fusion_diagnostic(
                "zero-candidate", schedule, *stats);
        if (schedule.materialize_output)
        {
            gettimeofday(&begin, nullptr);
            if (!dmma_copy_output_to_host(
                    a.rows, b.cols, a.tile_row_count,
                    b.tile_col_count, 0, 0, d_candidate_row_ptr, nullptr,
                    d_candidate_row_ptr, nullptr, nullptr, nullptr, output))
                goto failure;
            gettimeofday(&end, nullptr);
            stats->output_copy_ms = dmma_elapsed_ms(begin, end);
            stats->output_materialized = true;
        }
        cudaFree(d_candidate_row_ptr);
        cudaFree(d_row_worker_sm_ids);
        cudaFree(d_row_dynamic_next_row);
        return true;
    }

    gettimeofday(&begin, nullptr);
    if (!dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_candidate_masks),
                                 static_cast<std::size_t>(candidate_count) *
                                     sizeof(uint64_t)),
                      "allocate candidate masks") ||
        !dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_candidate_keep),
                                 (static_cast<std::size_t>(candidate_count) + 1) *
                                     sizeof(int)),
                      "allocate candidate keep scan") ||
        !dmma_cuda_ok(cudaMemset(d_candidate_keep, 0,
                                 (static_cast<std::size_t>(candidate_count) + 1) *
                                     sizeof(int)),
                      "clear candidate keep scan"))
        goto symbolic_failure;
    stats->exact_ordinary_candidates =
        static_cast<unsigned long long>(candidate_count);

    /* Classify complete rows before exact masking.  Although candidate arrays
     * have already been materialised at this integration point, neither the
     * gate kernel nor its estimator receives or reads them; its decision is
     * therefore reproducible from pre-candidate A/B structure alone. */
    if (schedule.exact_forward_spa)
    {
        cudaError_t optional_allocation = cudaMalloc(
            reinterpret_cast<void **>(&d_exact_forward_flags),
            static_cast<std::size_t>(a.tile_row_count) * sizeof(int));
        if (optional_allocation == cudaSuccess)
            optional_allocation = cudaMalloc(
                reinterpret_cast<void **>(&d_exact_forward_rows),
                static_cast<std::size_t>(a.tile_row_count) * sizeof(int));
        if (optional_allocation == cudaSuccess)
            optional_allocation = cudaMalloc(
                reinterpret_cast<void **>(&d_exact_forward_summary),
                sizeof(DmmaExactForwardSpaDeviceSummary));
        if (optional_allocation != cudaSuccess)
        {
            std::fprintf(
                stderr,
                "Optional exact forward-SPA metadata allocation failed; "
                "falling back to ordinary exact: %s.\n",
                cudaGetErrorString(optional_allocation));
            (void)cudaGetLastError();
            cudaFree(d_exact_forward_flags);
            cudaFree(d_exact_forward_rows);
            cudaFree(d_exact_forward_summary);
            d_exact_forward_flags = nullptr;
            d_exact_forward_rows = nullptr;
            d_exact_forward_summary = nullptr;
            stats->exact_forward_spa_reason =
                DMMA_EXACT_FORWARD_ALLOCATION;
        }
        else
        {
            if (!dmma_cuda_ok(
                    cudaMemset(d_exact_forward_flags, 0,
                               static_cast<std::size_t>(a.tile_row_count) *
                                   sizeof(int)),
                    "clear exact forward row flags") ||
                !dmma_cuda_ok(
                    cudaMemset(d_exact_forward_summary, 0,
                               sizeof(DmmaExactForwardSpaDeviceSummary)),
                    "clear exact forward summary"))
                goto symbolic_failure;
            unsigned int classify_blocks = 0;
            if (!dmma_launch_blocks(
                    static_cast<std::size_t>(a.tile_row_count),
                    DMMA_WARPS_PER_BLOCK, &classify_blocks,
                    "classify exact forward rows"))
                goto symbolic_failure;
            dmma_classify_exact_forward_rows_kernel
                <<<classify_blocks, DMMA_THREADS_PER_BLOCK>>>(
                    a, b, schedule.exact_forward_min_row_pairs,
                    schedule.exact_forward_min_ratio,
                    d_exact_forward_flags, d_exact_forward_rows,
                    d_exact_forward_summary);
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "launch exact forward row classifier") ||
                !dmma_cuda_ok(cudaDeviceSynchronize(),
                              "classify exact forward rows") ||
                !dmma_cuda_ok(cudaMemcpy(
                                  &exact_forward_summary,
                                  d_exact_forward_summary,
                                  sizeof(exact_forward_summary),
                                  cudaMemcpyDeviceToHost),
                              "read exact forward row summary"))
                goto symbolic_failure;

            if (exact_forward_summary.selected_rows >
                static_cast<unsigned long long>(a.tile_row_count))
            {
                std::fprintf(stderr,
                             "Invalid exact forward selected-row count.\n");
                goto symbolic_failure;
            }
            exact_forward_row_count = static_cast<int>(
                exact_forward_summary.selected_rows);
            stats->exact_forward_rows = exact_forward_row_count;
            stats->exact_forward_pairs =
                exact_forward_summary.forward_pairs;
            stats->exact_forward_estimated_candidates =
                exact_forward_summary.estimated_candidates;
            stats->exact_forward_estimated_reverse_work =
                exact_forward_summary.reverse_work;
            stats->exact_forward_estimated_forward_work =
                exact_forward_summary.forward_work;
            stats->exact_forward_estimate_overflow =
                exact_forward_summary.estimate_overflow != 0;

            if (stats->exact_forward_estimate_overflow)
            {
                stats->exact_forward_spa_reason =
                    DMMA_EXACT_FORWARD_ESTIMATE_OVERFLOW;
            }
            else if (exact_forward_row_count == 0)
            {
                stats->exact_forward_spa_reason =
                    DMMA_EXACT_FORWARD_ZERO_SELECTED;
            }
            else
            {
                exact_forward_batch_capacity =
                    dmma_exact_forward_spa_batch_capacity(
                        b.tile_col_count, exact_forward_row_count);
                stats->exact_forward_batch_capacity =
                    exact_forward_batch_capacity;
                if (exact_forward_batch_capacity == 0)
                {
                    stats->exact_forward_spa_reason =
                        DMMA_EXACT_FORWARD_CAPACITY;
                }
                else
                {
                    const std::size_t row_bytes =
                        dmma_exact_forward_spa_row_bytes(
                            b.tile_col_count);
                    const std::size_t scratch_bytes =
                        row_bytes * static_cast<std::size_t>(
                                        exact_forward_batch_capacity);
                    cudaError_t scratch_status = cudaMalloc(
                        reinterpret_cast<void **>(
                            &d_exact_forward_mask_spas),
                        scratch_bytes);
                    if (scratch_status != cudaSuccess)
                    {
                        std::fprintf(
                            stderr,
                            "Optional exact forward-SPA scratch allocation "
                            "failed; falling back to ordinary exact: %s.\n",
                            cudaGetErrorString(scratch_status));
                        (void)cudaGetLastError();
                        stats->exact_forward_spa_reason =
                            DMMA_EXACT_FORWARD_ALLOCATION;
                    }
                    else
                    {
                        stats->exact_forward_spa_used = true;
                        stats->exact_forward_spa_reason =
                            DMMA_EXACT_FORWARD_ENABLED;
                        stats->exact_forward_scratch_bytes = scratch_bytes;
                        stats->exact_forward_batches =
                            (exact_forward_row_count +
                             exact_forward_batch_capacity - 1) /
                            exact_forward_batch_capacity;
                    }
                }
            }

            if (!stats->exact_forward_spa_used)
            {
                cudaFree(d_exact_forward_flags);
                cudaFree(d_exact_forward_rows);
                cudaFree(d_exact_forward_summary);
                cudaFree(d_exact_forward_mask_spas);
                d_exact_forward_flags = nullptr;
                d_exact_forward_rows = nullptr;
                d_exact_forward_summary = nullptr;
                d_exact_forward_mask_spas = nullptr;
            }
        }
    }
    /* candidate_nnz is only a transport word for the optional packed load
     * telemetry.  The ordinary structural path derives nnz from its retained
     * 64-bit mask during compaction and therefore avoids this candidate-scale
     * allocation and write stream entirely. */
    if (collect_symbolic_load &&
        !dmma_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&d_candidate_nnz),
                                 static_cast<std::size_t>(candidate_count) *
                                     sizeof(int)),
                      "allocate packed candidate nnz/load"))
        goto symbolic_failure;
    if (collect_sparse_tail_records)
    {
        tail_record_capacity =
            std::min(candidate_count, schedule.tail_record_capacity);
        /* Fused exact stores no maybe IDs, so its logical counter can cover
         * every materialized candidate without consuming candidate-scale
         * memory.  The separate path retains the configured bounded channel. */
        maybe_candidate_capacity =
            fused_symbolic_admission
                ? candidate_count
                : std::min(candidate_count,
                           schedule.maybe_candidate_capacity);
        const std::size_t record_bytes =
            static_cast<std::size_t>(tail_record_capacity) *
            sizeof(DmmaTailRecord);
        const std::size_t maybe_bytes =
            fused_symbolic_admission
                ? 0
                : static_cast<std::size_t>(maybe_candidate_capacity) *
                      sizeof(int);
        if (!dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(&d_tail_records),
                              record_bytes),
                          "allocate sparse symbolic tail records") ||
            !dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(
                                  &d_tail_append_state),
                              sizeof(DmmaTailAppendState)),
                          "allocate sparse tail append state") ||
            !dmma_cuda_ok(cudaMemset(d_tail_append_state, 0,
                                     sizeof(DmmaTailAppendState)),
                          "clear sparse tail append state") ||
            !dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(
                                  &d_maybe_append_state),
                              sizeof(DmmaTailAppendState)),
                          "allocate tail maybe append state") ||
            !dmma_cuda_ok(cudaMemset(d_maybe_append_state, 0,
                                     sizeof(DmmaTailAppendState)),
                          "clear tail maybe append state"))
            goto symbolic_failure;
        if (!fused_symbolic_admission &&
            !dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(
                                  &d_maybe_candidate_ids),
                              maybe_bytes),
                          "allocate bounded tail maybe IDs"))
            goto symbolic_failure;
        stats->tail_record_capacity = tail_record_capacity;
        stats->maybe_candidate_capacity = maybe_candidate_capacity;
        stats->symbolic_task_count_bytes =
            record_bytes + maybe_bytes + 2 * sizeof(DmmaTailAppendState);
    }
    {
        unsigned int exact_blocks = 0;
        if (!dmma_launch_blocks(
                static_cast<std::size_t>(candidate_count),
                DMMA_WARPS_PER_BLOCK, &exact_blocks, "exact C masks"))
            goto symbolic_failure;
        if (collect_sparse_tail_records && !fused_symbolic_admission &&
            !unified_joint_replay)
        {
            unsigned int filter_blocks = 0;
            unsigned int full_filter_blocks = 0;
            int filter_device = 0;
            int filter_sm_count = 0;
            if (!dmma_cuda_ok(cudaGetDevice(&filter_device),
                              "read tail filter device") ||
                !dmma_cuda_ok(cudaDeviceGetAttribute(
                                  &filter_sm_count,
                                  cudaDevAttrMultiProcessorCount,
                                  filter_device),
                              "read tail filter SM count") ||
                filter_sm_count <= 0 ||
                !dmma_launch_blocks(
                    static_cast<std::size_t>(candidate_count),
                    DMMA_THREADS_PER_BLOCK, &full_filter_blocks,
                    "upper-bound tail maybe filter"))
                goto symbolic_failure;
            filter_blocks = std::min(
                full_filter_blocks,
                static_cast<unsigned int>(filter_sm_count * 8));
            if (
                (admission_events_ready &&
                 !dmma_cuda_ok(cudaEventRecord(
                                   split_async.admission_filter_start, 0),
                               "record admission filter start")))
                goto symbolic_failure;
            dmma_filter_tail_maybe_kernel
                <<<filter_blocks, DMMA_THREADS_PER_BLOCK>>>(
                    a, b, candidate_count, d_candidate_rows,
                    d_candidate_cols,
                    static_cast<float>(schedule.cost.scan),
                    static_cast<float>(schedule.cost.match),
                    static_cast<float>(schedule.split_threshold),
                    d_maybe_candidate_ids, maybe_candidate_capacity,
                    d_maybe_append_state);
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "launch upper-bound tail maybe filter") ||
                (admission_events_ready &&
                 !dmma_cuda_ok(cudaEventRecord(
                                   split_async.admission_filter_stop, 0),
                               "record admission filter stop")))
                goto symbolic_failure;
        }
        if (fused_symbolic_admission)
        {
            /* Preserve the audited tile-tail choice exactly, then extend the
             * same no-loose-counter rule to the isolated early mode. */
            int count_fused_maybe =
                tile_tail_queue_requested ? 0 : 1;
            if (tile_early_split_requested)
                count_fused_maybe = 0;
            if (admission_events_ready &&
                !dmma_cuda_ok(cudaEventRecord(
                                  split_async.admission_filter_start, 0),
                              "record fused exact admission start"))
                goto symbolic_failure;
            dmma_exact_mask_fused_tail_kernel
                <<<exact_blocks, DMMA_THREADS_PER_BLOCK>>>(
                    a, b, candidate_count, d_candidate_rows,
                    d_candidate_cols, d_candidate_masks, d_candidate_nnz,
                    d_candidate_keep,
                    static_cast<float>(schedule.cost.intercept),
                    static_cast<float>(schedule.cost.scan),
                    static_cast<float>(schedule.cost.match),
                    static_cast<float>(schedule.cost.output),
                    static_cast<float>(schedule.split_threshold),
                    static_cast<float>(resolved_chunk_target),
                    schedule.max_chunks,
                    static_cast<float>(schedule.symbolic_load_quantum),
                    collect_symbolic_load ? 1 : 0,
                    d_maybe_append_state,
                    count_fused_maybe, d_tail_records,
                    tail_record_capacity, d_tail_append_state);
            stats->admission_filter_fused_into_exact = true;
            stats->admission_count_fused_into_exact = true;
            stats->admission_count_executed = true;
            if (admission_events_ready &&
                !dmma_cuda_ok(cudaEventRecord(
                                  split_async.admission_filter_stop, 0),
                              "record fused exact admission stop"))
                goto symbolic_failure;
        }
        else
        {
            if (collect_symbolic_load)
            {
                dmma_exact_mask_packed_load_kernel
                    <<<exact_blocks, DMMA_THREADS_PER_BLOCK>>>(
                        a, b, candidate_count, d_candidate_rows,
                        d_candidate_cols, d_candidate_masks,
                        d_candidate_nnz, d_candidate_keep,
                        static_cast<float>(schedule.cost.intercept),
                        static_cast<float>(schedule.cost.scan),
                        static_cast<float>(schedule.cost.match),
                        static_cast<float>(schedule.cost.output),
                        static_cast<float>(schedule.symbolic_load_quantum));
            }
            else
            {
                /* Keep the original exact kernel for direct/static/manual-W
                 * paths unless workload collection is explicitly requested. */
                if (stats->exact_forward_spa_used)
                {
                    dmma_exact_mask_partitioned_kernel
                        <<<exact_blocks, DMMA_THREADS_PER_BLOCK>>>(
                            a, b, candidate_count, d_candidate_rows,
                            d_candidate_cols, d_exact_forward_flags,
                            d_candidate_masks, d_candidate_keep);
                    for (int row_begin = 0;
                         row_begin < exact_forward_row_count;
                         row_begin += exact_forward_batch_capacity)
                    {
                        const int remaining =
                            exact_forward_row_count - row_begin;
                        const int batch_rows =
                            exact_forward_batch_capacity < remaining
                                ? exact_forward_batch_capacity
                                : remaining;
                        dmma_exact_forward_spa_kernel
                            <<<batch_rows, DMMA_WIDE_BLOCK_THREADS>>>(
                                a, b, d_exact_forward_rows, row_begin,
                                batch_rows, b.tile_col_count,
                                d_exact_forward_mask_spas,
                                d_candidate_row_ptr, d_candidate_cols,
                                d_candidate_masks, d_candidate_keep,
                                d_exact_forward_summary);
                    }
                }
                else if (low_fill_ready)
                {
                    dmma_exact_mask_low_fill_kernel
                        <<<exact_blocks, DMMA_THREADS_PER_BLOCK>>>(
                            a, b, candidate_count, d_candidate_rows,
                            d_candidate_cols, schedule.low_fill_q,
                            d_candidate_masks, d_candidate_keep);
                    stats->low_fill_exact_tile_used = true;
                }
                else
                {
                    dmma_exact_mask_kernel
                        <<<exact_blocks, DMMA_THREADS_PER_BLOCK>>>(
                            a, b, candidate_count, d_candidate_rows,
                            d_candidate_cols, d_candidate_masks,
                            d_candidate_keep);
                }
            }
        }
        if (!dmma_cuda_ok(cudaGetLastError(), "launch exact C masks") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(), "exact C masks"))
            goto symbolic_failure;
        if (stats->exact_forward_spa_used)
        {
            if (!dmma_cuda_ok(cudaMemcpy(
                                  &exact_forward_summary,
                                  d_exact_forward_summary,
                                  sizeof(exact_forward_summary),
                                  cudaMemcpyDeviceToHost),
                              "read exact forward partition summary"))
                goto symbolic_failure;
            stats->exact_forward_candidates =
                exact_forward_summary.forward_candidates;
            if (stats->exact_forward_candidates >
                static_cast<unsigned long long>(candidate_count))
            {
                stats->exact_forward_partition_complete = false;
                std::fprintf(stderr,
                             "Exact forward row partition exceeds candidate "
                             "count.\n");
                goto symbolic_failure;
            }
            stats->exact_ordinary_candidates =
                static_cast<unsigned long long>(candidate_count) -
                stats->exact_forward_candidates;
            stats->exact_forward_partition_complete =
                stats->exact_ordinary_candidates +
                    stats->exact_forward_candidates ==
                static_cast<unsigned long long>(candidate_count);
        }
        cudaFree(d_exact_forward_flags);
        cudaFree(d_exact_forward_rows);
        cudaFree(d_exact_forward_mask_spas);
        cudaFree(d_exact_forward_summary);
        d_exact_forward_flags = nullptr;
        d_exact_forward_rows = nullptr;
        d_exact_forward_mask_spas = nullptr;
        d_exact_forward_summary = nullptr;
        if (collect_sparse_tail_records && !unified_joint_replay)
        {
            DmmaTailAppendState maybe_state{};
            if (!dmma_cuda_ok(cudaMemcpy(
                                  &maybe_state, d_maybe_append_state,
                                  sizeof(maybe_state), cudaMemcpyDeviceToHost),
                              "read tail maybe append state"))
                goto symbolic_failure;
            maybe_candidate_count = maybe_state.count;
            if (maybe_candidate_count < 0 ||
                maybe_candidate_count > candidate_count)
                goto symbolic_failure;
            maybe_candidate_overflow = maybe_state.overflow != 0 ||
                                       maybe_candidate_count >
                                           maybe_candidate_capacity;
            stats->maybe_candidate_count = maybe_candidate_count;
            stats->maybe_candidate_overflow = maybe_candidate_overflow;
            stats->maybe_candidate_count_is_lower_bound =
                maybe_candidate_overflow;
            stats->maybe_candidate_fraction =
                static_cast<double>(maybe_candidate_count) / candidate_count;
            if (fused_symbolic_admission && !maybe_candidate_overflow)
            {
                DmmaTailAppendState tail_state{};
                if (!dmma_cuda_ok(cudaMemcpy(
                                      &tail_state, d_tail_append_state,
                                      sizeof(tail_state),
                                      cudaMemcpyDeviceToHost),
                                  "read fused sparse tail append state"))
                    goto symbolic_failure;
                tail_record_count = tail_state.count;
                if (tail_record_count < 0 ||
                    tail_record_count > candidate_count)
                    goto symbolic_failure;
                tail_record_overflow = tail_state.overflow != 0 ||
                                       tail_record_count >
                                           tail_record_capacity;
                stats->tail_record_count = tail_record_count;
                stats->tail_record_overflow = tail_record_overflow;
                stats->tail_record_count_is_lower_bound =
                    tail_record_overflow;
            }
        }
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

    critical_q_begin = dmma_critical_suffix_begin(
        output_tile_count, schedule.critical_q_min);
    prefix_task_count = output_tile_count > 0 ? critical_q_begin : 0;
    suffix_task_count = output_tile_count - prefix_task_count;
    unified_terminal_begin =
        schedule.suffix_fine_tasks_per_worker > 0
            ? std::max(critical_q_begin,
                       output_tile_count -
                           schedule.suffix_fine_tasks_per_worker)
            : output_tile_count;
    stats->critical_q_begin = critical_q_begin;
    stats->prefix_tasks = prefix_task_count;
    stats->suffix_tasks = suffix_task_count;
    stats->suffix_task_fraction =
        output_tile_count > 0
            ? static_cast<double>(suffix_task_count) / output_tile_count
            : 0.0;

    /* Sparse unified metadata is allocated only after exact compaction size is
     * known.  The terminal range needs no IDs; capacity therefore bounds only
     * non-terminal fine outputs emitted by the joint replay. */
    if (unified_joint_replay && collect_sparse_tail_records &&
        output_tile_count > 0)
    {
        unified_fine_capacity =
            std::min(output_tile_count, schedule.unified_fine_capacity);
        const std::size_t fine_flag_words =
            (static_cast<std::size_t>(output_tile_count) + 31u) / 32u;
        const std::size_t fine_flag_bytes =
            fine_flag_words * sizeof(uint32_t);
        const std::size_t fine_id_bytes =
            static_cast<std::size_t>(unified_fine_capacity) * sizeof(int);
        if (unified_fine_capacity < 1 ||
            !dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(
                                  &d_unified_fine_ids),
                              fine_id_bytes),
                          "allocate sparse unified fine IDs") ||
            !dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(
                                  &d_unified_fine_flags),
                              fine_flag_bytes),
                          "allocate sparse unified fine bitset") ||
            !dmma_cuda_ok(cudaMemset(d_unified_fine_flags, 0,
                                     fine_flag_bytes),
                          "clear sparse unified fine bitset") ||
            !dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(
                                  &d_unified_replay_summary),
                              sizeof(DmmaUnifiedReplaySummary)),
                          "allocate sparse unified replay summary") ||
            !dmma_cuda_ok(cudaMemset(d_unified_replay_summary, 0,
                                     sizeof(DmmaUnifiedReplaySummary)),
                          "clear sparse unified replay summary"))
            goto symbolic_failure;
        stats->unified_fine_capacity = unified_fine_capacity;
        stats->unified_page_size = schedule.unified_page_size;
        stats->unified_sparse_replay_used = true;
        stats->unified_metadata_bytes =
            fine_id_bytes + fine_flag_bytes +
            sizeof(DmmaUnifiedReplaySummary) +
            2 * sizeof(unsigned long long);

        unsigned int filter_blocks = 0;
        unsigned int full_filter_blocks = 0;
        int filter_device = 0;
        int filter_sm_count = 0;
        if (!dmma_cuda_ok(cudaGetDevice(&filter_device),
                          "read joint filter device") ||
            !dmma_cuda_ok(cudaDeviceGetAttribute(
                              &filter_sm_count,
                              cudaDevAttrMultiProcessorCount,
                              filter_device),
                          "read joint filter SM count") ||
            filter_sm_count <= 0 ||
            !dmma_launch_blocks(
                static_cast<std::size_t>(candidate_count),
                DMMA_THREADS_PER_BLOCK, &full_filter_blocks,
                "joint unified maybe filter"))
            goto symbolic_failure;
        filter_blocks = std::min(
            full_filter_blocks,
            static_cast<unsigned int>(filter_sm_count * 8));
        if (admission_events_ready &&
            !dmma_cuda_ok(cudaEventRecord(
                              split_async.admission_filter_start, 0),
                          "record joint admission filter start"))
            goto symbolic_failure;
        dmma_filter_joint_maybe_output_window_kernel
            <<<filter_blocks, DMMA_THREADS_PER_BLOCK>>>(
                a, b, candidate_count, d_candidate_rows, d_candidate_cols,
                d_candidate_masks, d_candidate_keep, critical_q_begin,
                unified_terminal_begin,
                static_cast<float>(schedule.cost.intercept),
                static_cast<float>(schedule.cost.scan),
                static_cast<float>(schedule.cost.match),
                static_cast<float>(schedule.cost.output),
                static_cast<float>(schedule.split_threshold),
                static_cast<float>(resolved_unified_fine_threshold),
                d_maybe_candidate_ids, maybe_candidate_capacity,
                d_maybe_append_state);
        if (!dmma_cuda_ok(cudaGetLastError(),
                          "launch joint unified maybe filter") ||
            (admission_events_ready &&
             !dmma_cuda_ok(cudaEventRecord(
                               split_async.admission_filter_stop, 0),
                           "record joint admission filter stop")) ||
            !dmma_cuda_ok(cudaDeviceSynchronize(),
                          "joint unified maybe filter"))
            goto symbolic_failure;
        DmmaTailAppendState maybe_state{};
        if (!dmma_cuda_ok(cudaMemcpy(
                              &maybe_state, d_maybe_append_state,
                              sizeof(maybe_state), cudaMemcpyDeviceToHost),
                          "read joint unified maybe state"))
            goto symbolic_failure;
        maybe_candidate_count = maybe_state.count;
        if (maybe_candidate_count < 0 ||
            maybe_candidate_count > output_tile_count)
            goto symbolic_failure;
        maybe_candidate_overflow =
            maybe_state.overflow != 0 ||
            maybe_candidate_count > maybe_candidate_capacity;
        stats->maybe_candidate_count = maybe_candidate_count;
        stats->maybe_candidate_overflow = maybe_candidate_overflow;
        stats->maybe_candidate_count_is_lower_bound =
            maybe_candidate_overflow;
        /* This is joint-filter density, not an estimate of heavy precision. */
        stats->maybe_candidate_fraction =
            static_cast<double>(maybe_candidate_count) / output_tile_count;

        if (!maybe_candidate_overflow && maybe_candidate_count > 0)
        {
            unsigned int replay_blocks = 0;
            if (!dmma_launch_blocks(
                    static_cast<std::size_t>(maybe_candidate_count),
                    DMMA_THREADS_PER_BLOCK, &replay_blocks,
                    "joint unified exact replay") ||
                (admission_events_ready &&
                 !dmma_cuda_ok(cudaEventRecord(
                                   split_async.admission_count_start, 0),
                               "record joint exact replay start")))
                goto symbolic_failure;
            dmma_replay_joint_maybe_kernel
                <<<replay_blocks, DMMA_THREADS_PER_BLOCK>>>(
                    a, b, maybe_candidate_count, d_maybe_candidate_ids,
                    d_candidate_rows, d_candidate_cols, d_candidate_masks,
                    d_candidate_keep, critical_q_begin,
                    unified_terminal_begin,
                    static_cast<float>(schedule.cost.intercept),
                    static_cast<float>(schedule.cost.scan),
                    static_cast<float>(schedule.cost.match),
                    static_cast<float>(schedule.cost.output),
                    static_cast<float>(schedule.split_threshold),
                    static_cast<float>(resolved_unified_fine_threshold),
                    static_cast<float>(resolved_chunk_target),
                    schedule.max_chunks,
                    static_cast<float>(schedule.symbolic_load_quantum),
                    d_unified_fine_ids, unified_fine_capacity,
                    d_unified_fine_flags, d_unified_replay_summary,
                    d_tail_records, tail_record_capacity,
                    d_tail_append_state);
            if (!dmma_cuda_ok(cudaGetLastError(),
                              "launch joint unified exact replay") ||
                (admission_events_ready &&
                 !dmma_cuda_ok(cudaEventRecord(
                                   split_async.admission_count_stop, 0),
                               "record joint exact replay stop")) ||
                !dmma_cuda_ok(cudaDeviceSynchronize(),
                              "joint unified exact replay"))
                goto symbolic_failure;
            stats->admission_count_executed = true;
            admission_count_event_recorded = admission_events_ready;
        }
        DmmaTailAppendState tail_state{};
        if (!maybe_candidate_overflow &&
            !dmma_cuda_ok(cudaMemcpy(
                              &tail_state, d_tail_append_state,
                              sizeof(tail_state), cudaMemcpyDeviceToHost),
                          "read joint unified tail state"))
            goto symbolic_failure;
        tail_record_count = maybe_candidate_overflow ? 0 : tail_state.count;
        if (tail_record_count < 0 ||
            tail_record_count > output_tile_count)
            goto symbolic_failure;
        tail_record_overflow =
            !maybe_candidate_overflow &&
            (tail_state.overflow != 0 ||
             tail_record_count > tail_record_capacity);
        stats->tail_record_count = tail_record_count;
        stats->tail_record_overflow = tail_record_overflow;
        stats->tail_record_count_is_lower_bound = tail_record_overflow;
    }

    if (collect_sparse_tail_records && !fused_symbolic_admission &&
        !unified_joint_replay && !maybe_candidate_overflow &&
        maybe_candidate_count > 0 &&
        output_tile_count > 0)
    {
        const int replay_output_begin =
            late_output_window_requested ? critical_q_begin : 0;
        unsigned int count_blocks = 0;
        if (!dmma_launch_blocks(
                static_cast<std::size_t>(maybe_candidate_count),
                DMMA_THREADS_PER_BLOCK, &count_blocks,
                "exact-count output-window maybe IDs") ||
            (admission_events_ready &&
             !dmma_cuda_ok(cudaEventRecord(
                               split_async.admission_count_start, 0),
                           "record output-window exact-count start")))
            goto symbolic_failure;
        dmma_count_tail_maybe_output_window_kernel
            <<<count_blocks, DMMA_THREADS_PER_BLOCK>>>(
                a, b, maybe_candidate_count, d_maybe_candidate_ids,
                d_candidate_rows, d_candidate_cols, d_candidate_masks,
                d_candidate_keep, replay_output_begin,
                static_cast<float>(schedule.cost.scan),
                static_cast<float>(schedule.cost.match),
                static_cast<float>(schedule.split_threshold),
                static_cast<float>(resolved_chunk_target),
                schedule.max_chunks, d_tail_records, tail_record_capacity,
                d_tail_append_state);
        if (!dmma_cuda_ok(
                cudaGetLastError(),
                "launch exact-count output-window maybe IDs") ||
            (admission_events_ready &&
             !dmma_cuda_ok(cudaEventRecord(
                               split_async.admission_count_stop, 0),
                           "record output-window exact-count stop")) ||
            !dmma_cuda_ok(cudaDeviceSynchronize(),
                          "exact-count output-window maybe IDs"))
            goto symbolic_failure;
        stats->admission_count_executed = true;
        admission_count_event_recorded = admission_events_ready;
        DmmaTailAppendState tail_state{};
        if (!dmma_cuda_ok(cudaMemcpy(
                              &tail_state, d_tail_append_state,
                              sizeof(tail_state), cudaMemcpyDeviceToHost),
                          "read output-window tail append state"))
            goto symbolic_failure;
        tail_record_count = tail_state.count;
        const int replay_output_count =
            late_output_window_requested ? suffix_task_count :
                                           output_tile_count;
        if (tail_record_count < 0 ||
            tail_record_count > replay_output_count)
            goto symbolic_failure;
        tail_record_overflow = tail_state.overflow != 0 ||
                               tail_record_count > tail_record_capacity;
        stats->tail_record_count = tail_record_count;
        stats->tail_record_overflow = tail_record_overflow;
        stats->tail_record_count_is_lower_bound = tail_record_overflow;
    }

    if (collect_sparse_tail_records)
    {
        if (tail_record_count > output_tile_count)
            goto symbolic_failure;
        stats->tail_record_fraction =
            output_tile_count > 0
                ? static_cast<double>(tail_record_count) / output_tile_count
                : 0.0;
        if (maybe_candidate_overflow)
            stats->tail_gate_reason = DMMA_TAIL_GATE_MAYBE_OVERFLOW;
        else if (tail_record_overflow)
            stats->tail_gate_reason = DMMA_TAIL_GATE_RECORD_OVERFLOW;
        else if (tail_record_count == 0)
            stats->tail_gate_reason = DMMA_TAIL_GATE_ZERO_HEAVY;
        /* Fused records were created before the keep scan and still need a
         * candidate-to-output map/filter.  Separate records already name the
         * exact output window and can take the ordinary fraction gate now. */
        else if (late_output_window_requested &&
                 fused_symbolic_admission)
        {
            stats->tail_gate_reason = DMMA_TAIL_GATE_ENABLED;
            sparse_tail_records_ready = true;
        }
        else if (!schedule.force_tail_split &&
                 stats->tail_record_fraction > schedule.max_heavy_fraction)
            stats->tail_gate_reason = DMMA_TAIL_GATE_HEAVY_FRACTION;
        else
        {
            stats->tail_gate_reason = DMMA_TAIL_GATE_ENABLED;
            sparse_tail_records_ready = true;
        }
    }
    cudaFree(d_maybe_candidate_ids);
    d_maybe_candidate_ids = nullptr;
    cudaFree(d_maybe_append_state);
    d_maybe_append_state = nullptr;

    if (output_tile_count == 0)
    {
        if (!dmma_cuda_ok(
                cudaMemset(d_candidate_row_ptr, 0,
                           (static_cast<std::size_t>(a.tile_row_count) +
                            1) * sizeof(int)),
                "clear empty exact C row pointer") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(),
                          "complete empty exact C row pointer"))
            goto symbolic_failure;
        /* The native empty-C result includes a device-ready row pointer.
         * Synchronize above before closing Symbolic/Core so the asynchronous
         * memset cannot escape the advertised endpoint. */
        gettimeofday(&end, nullptr);
        stats->symbolic_ms = dmma_elapsed_ms(begin, end);
        stats->output_tiles = 0;
        stats->output_nnz = 0;
        dmma_close_core_endpoint(total_begin, &total_end, stats);
        if (admission_events_ready &&
            !dmma_read_admission_timing(
                split_async, admission_count_event_recorded, stats))
            goto symbolic_failure;
        if (schedule.mode != DMMA_SCHEDULE_DIRECT)
            dmma_print_tail_fusion_diagnostic(
                "two-stage-empty", schedule, *stats);
        if (schedule.materialize_output)
        {
            gettimeofday(&begin, nullptr);
            if (!dmma_copy_output_to_host(
                    a.rows, b.cols, a.tile_row_count,
                    b.tile_col_count, 0, 0, d_candidate_row_ptr, nullptr,
                    d_candidate_keep, nullptr, nullptr, nullptr, output))
                goto symbolic_failure;
            gettimeofday(&end, nullptr);
            stats->output_copy_ms = dmma_elapsed_ms(begin, end);
            stats->output_materialized = true;
        }
        cudaFree(d_candidate_masks);
        cudaFree(d_candidate_nnz);
        cudaFree(d_candidate_keep);
        cudaFree(d_tail_records);
        cudaFree(d_tail_append_state);
        cudaFree(d_candidate_row_ptr);
        cudaFree(d_candidate_rows);
        cudaFree(d_candidate_cols);
        cudaFree(d_row_worker_sm_ids);
        cudaFree(d_row_dynamic_next_row);
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
                      "clear output nnz"))
        goto symbolic_failure;
    if (collect_symbolic_load)
    {
        int load_device = 0;
        int load_sm_count = 0;
        if (!dmma_cuda_ok(cudaGetDevice(&load_device),
                          "read symbolic load device") ||
            !dmma_cuda_ok(cudaDeviceGetAttribute(
                              &load_sm_count,
                              cudaDevAttrMultiProcessorCount, load_device),
                          "read symbolic load SM count") ||
            load_sm_count <= 0)
            goto symbolic_failure;
        const long long wave_capacity_wide =
            static_cast<long long>(load_sm_count) *
            schedule.symbolic_wave_ctas_per_sm * DMMA_WARPS_PER_BLOCK;
        if (wave_capacity_wide <= 0 || wave_capacity_wide > INT_MAX)
            goto symbolic_failure;
        const int wave_capacity = static_cast<int>(wave_capacity_wide);
        const int predicted_waves =
            (output_tile_count + wave_capacity - 1) / wave_capacity;
        const int critical_first_wave = std::max(
            0, predicted_waves - schedule.symbolic_critical_waves);
        symbolic_critical_task_begin = critical_first_wave * wave_capacity;
        if (unified_light_requested)
        {
            unified_fine_capacity =
                std::min(output_tile_count, schedule.unified_fine_capacity);
            const std::size_t fine_flag_words =
                (static_cast<std::size_t>(output_tile_count) + 31u) / 32u;
            const std::size_t fine_flag_bytes =
                fine_flag_words * sizeof(uint32_t);
            const std::size_t fine_id_bytes =
                static_cast<std::size_t>(unified_fine_capacity) * sizeof(int);
            unified_fine_threshold_units = dmma_quantize_symbolic_load(
                static_cast<float>(resolved_unified_fine_threshold),
                static_cast<float>(schedule.symbolic_load_quantum));
            if (unified_fine_capacity < 1 ||
                unified_fine_threshold_units == 0 ||
                !dmma_cuda_ok(cudaMalloc(
                                  reinterpret_cast<void **>(
                                      &d_unified_fine_ids),
                                  fine_id_bytes),
                              "allocate unified fine task IDs") ||
                !dmma_cuda_ok(cudaMalloc(
                                  reinterpret_cast<void **>(
                                      &d_unified_fine_flags),
                                  fine_flag_bytes),
                              "allocate unified fine task bitset") ||
                !dmma_cuda_ok(cudaMemset(d_unified_fine_flags, 0,
                                         fine_flag_bytes),
                              "clear unified fine task bitset"))
                goto symbolic_failure;
            stats->unified_fine_capacity = unified_fine_capacity;
            stats->unified_page_size = schedule.unified_page_size;
            stats->unified_metadata_bytes =
                fine_id_bytes + fine_flag_bytes +
                2 * sizeof(unsigned long long);
        }
        if (!dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(
                                  &d_symbolic_load_summary),
                              sizeof(DmmaSymbolicLoadSummary)),
                          "allocate symbolic load summary") ||
            !dmma_cuda_ok(cudaMemset(d_symbolic_load_summary, 0,
                                     sizeof(DmmaSymbolicLoadSummary)),
                          "clear symbolic load summary"))
            goto symbolic_failure;
        stats->symbolic_load_metadata = true;
        stats->symbolic_load_metadata_bytes =
            sizeof(DmmaSymbolicLoadSummary);
        stats->symbolic_load_quantum = schedule.symbolic_load_quantum;
        stats->symbolic_wave_task_capacity = wave_capacity;
        stats->symbolic_predicted_waves = predicted_waves;
    }
    {
        const int threads = 256;
        unsigned int blocks = 0;
        unsigned int row_blocks = 0;
        if (!dmma_launch_blocks(
                static_cast<std::size_t>(candidate_count), threads,
                &blocks, "compact exact C tiles") ||
            !dmma_launch_blocks(
                static_cast<std::size_t>(a.tile_row_count) + 1, threads,
                &row_blocks, "gather exact C row pointer"))
            goto symbolic_failure;
        if (collect_symbolic_load)
        {
            dmma_compact_candidates_with_load_kernel<<<blocks, threads>>>(
                candidate_count, d_candidate_rows, d_candidate_cols,
                d_candidate_masks, d_candidate_nnz, d_candidate_keep,
                critical_q_begin, symbolic_critical_task_begin,
                critical_q_begin, unified_terminal_begin,
                unified_fine_threshold_units, d_unified_fine_ids,
                unified_fine_capacity, d_unified_fine_flags,
                d_output_rows, d_output_cols, d_output_masks, d_output_nnz,
                d_symbolic_load_summary);
        }
        else
        {
            dmma_compact_candidates_kernel<<<blocks, threads>>>(
                candidate_count, d_candidate_rows, d_candidate_cols,
                d_candidate_masks, d_candidate_keep,
                d_output_rows, d_output_cols, d_output_masks,
                d_output_nnz);
        }
        const bool map_tail_records =
            fused_symbolic_admission && tail_record_count > 0 &&
            !tail_record_overflow && !maybe_candidate_overflow &&
            (sparse_tail_records_ready || collect_symbolic_load);
        if (map_tail_records)
        {
            unsigned int tail_blocks = 0;
            if (!dmma_launch_blocks(
                    static_cast<std::size_t>(tail_record_count), threads,
                    &tail_blocks, "map sparse tail candidates to output tasks"))
                goto symbolic_failure;
            dmma_map_tail_records_to_outputs_kernel
                <<<tail_blocks, threads>>>(tail_record_count,
                                           d_candidate_keep, d_tail_records);
            if (late_output_window_requested &&
                sparse_tail_records_ready)
            {
                const std::size_t filtered_bytes =
                    static_cast<std::size_t>(tail_record_count) *
                    sizeof(DmmaTailRecord);
                if (!dmma_cuda_ok(cudaMalloc(
                                      reinterpret_cast<void **>(
                                          &d_window_tail_records),
                                      filtered_bytes),
                                  "allocate output-window tail records") ||
                    !dmma_cuda_ok(cudaMemset(
                                      d_tail_append_state, 0,
                                      sizeof(DmmaTailAppendState)),
                                  "clear output-window tail append state"))
                    goto symbolic_failure;
                dmma_filter_tail_output_suffix_kernel
                    <<<tail_blocks, threads>>>(
                        tail_record_count, critical_q_begin, d_tail_records,
                        d_window_tail_records, d_tail_append_state);
                stats->symbolic_task_count_bytes += filtered_bytes;
            }
        }
        if (collect_symbolic_load && tail_record_count > 0 &&
            !tail_record_overflow && !maybe_candidate_overflow)
        {
            unsigned int tail_blocks = 0;
            if (!dmma_launch_blocks(
                    static_cast<std::size_t>(tail_record_count), threads,
                    &tail_blocks, "summarize sparse tail workload"))
                goto symbolic_failure;
            dmma_summarize_critical_tail_kernel<<<tail_blocks, threads>>>(
                tail_record_count, d_tail_records, output_tile_count,
                critical_q_begin, symbolic_critical_task_begin,
                d_output_nnz,
                static_cast<float>(schedule.cost.intercept),
                static_cast<float>(schedule.cost.scan),
                static_cast<float>(schedule.cost.match),
                static_cast<float>(schedule.cost.output),
                static_cast<float>(schedule.symbolic_load_quantum),
                d_unified_fine_flags,
                d_symbolic_load_summary);
        }
        dmma_gather_output_row_ptr_kernel<<<row_blocks, threads>>>(
            a.tile_row_count, d_candidate_row_ptr, d_candidate_keep);
        if (unified_joint_replay && d_unified_replay_summary != nullptr &&
            unified_terminal_begin < output_tile_count)
        {
            unsigned int terminal_blocks = 0;
            if (!dmma_launch_blocks(
                    static_cast<std::size_t>(output_tile_count -
                                             unified_terminal_begin),
                    DMMA_WARPS_PER_BLOCK, &terminal_blocks,
                    "summarize unified terminal tasks"))
                goto symbolic_failure;
            dmma_summarize_unified_terminal_kernel
                <<<terminal_blocks, DMMA_THREADS_PER_BLOCK>>>(
                    a, b, output_tile_count, unified_terminal_begin,
                    d_output_rows, d_output_cols, d_output_masks,
                    static_cast<float>(schedule.cost.intercept),
                    static_cast<float>(schedule.cost.scan),
                    static_cast<float>(schedule.cost.match),
                    static_cast<float>(schedule.cost.output),
                    static_cast<float>(schedule.symbolic_load_quantum),
                    d_unified_replay_summary);
        }
        if (!dmma_cuda_ok(cudaGetLastError(), "compact exact C tiles") ||
            !dmma_cuda_ok(cudaDeviceSynchronize(), "compact exact C tiles"))
            goto symbolic_failure;
        d_output_row_ptr = d_candidate_row_ptr;
        d_candidate_row_ptr = nullptr;
        if (late_output_window_requested && sparse_tail_records_ready &&
            fused_symbolic_admission)
        {
            DmmaTailAppendState window_state{};
            if (!dmma_cuda_ok(cudaMemcpy(
                                  &window_state, d_tail_append_state,
                                  sizeof(window_state),
                                  cudaMemcpyDeviceToHost),
                              "read output-window tail append state") ||
                window_state.overflow != 0 || window_state.count < 0 ||
                window_state.count > tail_record_count)
                goto symbolic_failure;
            const int filtered_capacity = tail_record_count;
            tail_record_count = window_state.count;
            cudaFree(d_tail_records);
            d_tail_records = d_window_tail_records;
            d_window_tail_records = nullptr;
            tail_record_capacity = filtered_capacity;
            stats->tail_record_count = tail_record_count;
            stats->tail_record_capacity = filtered_capacity;
            stats->tail_record_overflow = false;
            stats->tail_record_count_is_lower_bound = false;
            stats->tail_record_fraction =
                output_tile_count > 0
                    ? static_cast<double>(tail_record_count) /
                          output_tile_count
                    : 0.0;
            if (tail_record_count == 0)
            {
                stats->tail_gate_reason = DMMA_TAIL_GATE_ZERO_HEAVY;
                sparse_tail_records_ready = false;
            }
            else if (!schedule.force_tail_split &&
                     stats->tail_record_fraction >
                         schedule.max_heavy_fraction)
            {
                stats->tail_gate_reason = DMMA_TAIL_GATE_HEAVY_FRACTION;
                sparse_tail_records_ready = false;
            }
            else
            {
                stats->tail_gate_reason = DMMA_TAIL_GATE_ENABLED;
            }
        }
        if (collect_symbolic_load)
        {
            DmmaSymbolicLoadSummary summary{};
            if (!dmma_cuda_ok(cudaMemcpy(
                                  &summary, d_symbolic_load_summary,
                                  sizeof(summary), cudaMemcpyDeviceToHost),
                              "read symbolic load summary") ||
                summary.task_count != output_tile_count ||
                summary.suffix_task_count != suffix_task_count ||
                summary.suffix_units > summary.total_units ||
                summary.fine_units > summary.total_units ||
                summary.split_units > summary.total_units ||
                summary.split_suffix_units > summary.suffix_units ||
                summary.split_fine_units > summary.fine_units ||
                summary.split_fine_units > summary.split_units ||
                summary.fine_task_count < 0 ||
                summary.fine_task_count > output_tile_count ||
                summary.split_fine_task_count < 0 ||
                summary.split_fine_task_count > summary.fine_task_count)
                goto symbolic_failure;
            const double quantum = schedule.symbolic_load_quantum;
            stats->symbolic_load_tasks = summary.task_count;
            stats->symbolic_load_saturated_tasks =
                summary.saturated_task_count;
            stats->symbolic_critical_tasks = summary.critical_task_count;
            stats->symbolic_critical_tail_tasks =
                summary.critical_tail_task_count;
            stats->symbolic_total_work =
                static_cast<double>(summary.total_units) * quantum;
            stats->symbolic_suffix_work =
                static_cast<double>(summary.suffix_units) * quantum;
            stats->symbolic_split_suffix_work =
                static_cast<double>(summary.split_suffix_units) * quantum;
            if (unified_light_requested)
            {
                unified_fine_task_count = summary.fine_task_count;
                stats->unified_fine_queue_tasks =
                    unified_fine_task_count;
                stats->unified_fine_overflow =
                    summary.fine_overflow != 0 ||
                    unified_fine_task_count > unified_fine_capacity;
                stats->unified_fallback_reason =
                    dmma_unified_symbolic_fallback(
                        true, unified_fine_task_count,
                        unified_fine_capacity,
                        stats->unified_fine_overflow,
                        summary.saturated_task_count,
                        summary.total_units);
                unified_symbolic_ready =
                    stats->unified_fallback_reason ==
                    DMMA_UNIFIED_FALLBACK_NONE;

                const int admitted_heavy_tasks =
                    sparse_tail_records_ready ? tail_record_count : 0;
                const int admitted_heavy_fine_tasks =
                    sparse_tail_records_ready
                        ? summary.split_fine_task_count
                        : 0;
                const DmmaUnifiedPartitionCounts partition =
                    dmma_unified_partition_counts(
                        output_tile_count, unified_fine_task_count,
                        admitted_heavy_tasks,
                        admitted_heavy_fine_tasks);
                const unsigned long long admitted_heavy_units =
                    sparse_tail_records_ready ? summary.split_units : 0ull;
                const unsigned long long admitted_heavy_fine_units =
                    sparse_tail_records_ready
                        ? summary.split_fine_units
                        : 0ull;
                if (!partition.valid ||
                    admitted_heavy_fine_units > summary.fine_units ||
                    admitted_heavy_units - admitted_heavy_fine_units >
                        summary.total_units - summary.fine_units)
                    goto symbolic_failure;
                stats->unified_coarse_tasks = partition.coarse;
                stats->unified_fine_tasks = partition.fine;
                stats->unified_heavy_fine_tasks = partition.heavy_fine;
                stats->unified_heavy_work =
                    static_cast<double>(admitted_heavy_units) * quantum;
                stats->unified_fine_work =
                    static_cast<double>(summary.fine_units -
                                        admitted_heavy_fine_units) *
                    quantum;
                stats->unified_coarse_work =
                    static_cast<double>(
                        summary.total_units - summary.fine_units -
                        (admitted_heavy_units -
                         admitted_heavy_fine_units)) *
                    quantum;
                stats->unified_coarse_work_available = true;
            }
            if (summary.total_units > 0)
                stats->symbolic_suffix_work_fraction =
                    static_cast<double>(summary.suffix_units) /
                    static_cast<double>(summary.total_units);
            /* Keep split_suffix_units as potential admission telemetry even
             * when the global heavy-fraction gate rejects the split path.
             * Only parents that will actually run on the independent chunk
             * stream may be removed from the regular-light worker ratio. */
            const unsigned long long admitted_split_suffix_units =
                sparse_tail_records_ready ? summary.split_suffix_units : 0ull;
            const unsigned long long regular_total_units =
                summary.total_units - admitted_split_suffix_units;
            const unsigned long long regular_suffix_units =
                summary.suffix_units - admitted_split_suffix_units;
            if (regular_total_units > 0)
                stats->symbolic_regular_suffix_work_fraction =
                    static_cast<double>(regular_suffix_units) /
                    static_cast<double>(regular_total_units);
            stats->symbolic_max_task_work =
                static_cast<double>(summary.max_task_units) * quantum;
            stats->symbolic_critical_work =
                static_cast<double>(summary.critical_units) * quantum;
            stats->symbolic_critical_max_task_work =
                static_cast<double>(summary.critical_max_task_units) *
                quantum;
            stats->symbolic_critical_tail_work =
                static_cast<double>(summary.critical_tail_units) * quantum;
            const int critical_waves = std::min(
                stats->symbolic_predicted_waves,
                schedule.symbolic_critical_waves);
            if (summary.total_units > 0 && critical_waves > 0)
            {
                const double average_wave =
                    static_cast<double>(summary.total_units) /
                    stats->symbolic_predicted_waves;
                stats->symbolic_critical_work_over_average_wave =
                    (static_cast<double>(summary.critical_units) /
                     critical_waves) /
                    average_wave;
            }
            if (summary.critical_units > 0)
                stats->symbolic_critical_tail_work_fraction =
                    static_cast<double>(summary.critical_tail_units) /
                    summary.critical_units;
            cudaFree(d_symbolic_load_summary);
            d_symbolic_load_summary = nullptr;
        }
        if (unified_joint_replay)
        {
            DmmaUnifiedReplaySummary summary{};
            const bool metadata_available =
                collect_sparse_tail_records &&
                d_unified_replay_summary != nullptr;
            if (metadata_available &&
                !dmma_cuda_ok(cudaMemcpy(
                                  &summary, d_unified_replay_summary,
                                  sizeof(summary), cudaMemcpyDeviceToHost),
                              "read sparse unified replay summary"))
                goto symbolic_failure;
            const int terminal_task_count =
                output_tile_count - unified_terminal_begin;
            const long long total_fine_wide =
                static_cast<long long>(summary.sparse_fine_count) +
                terminal_task_count;
            if (summary.sparse_fine_count < 0 ||
                summary.sparse_fine_count > output_tile_count ||
                total_fine_wide < 0 ||
                total_fine_wide > output_tile_count ||
                summary.heavy_fine_count < 0 ||
                summary.heavy_fine_count > tail_record_count ||
                summary.heavy_fine_count > total_fine_wide)
                goto symbolic_failure;
            unified_sparse_fine_task_count =
                summary.sparse_fine_count;
            unified_fine_task_count =
                static_cast<int>(total_fine_wide);
            stats->unified_fine_queue_tasks = unified_fine_task_count;
            stats->unified_fine_overflow =
                summary.fine_overflow != 0 ||
                unified_sparse_fine_task_count > unified_fine_capacity;

            if (!metadata_available)
                stats->unified_fallback_reason =
                    DMMA_UNIFIED_FALLBACK_SYMBOLIC_UNAVAILABLE;
            else if (maybe_candidate_overflow || tail_record_overflow)
                stats->unified_fallback_reason =
                    DMMA_UNIFIED_FALLBACK_TAIL_METADATA;
            else if (stats->unified_fine_overflow)
                stats->unified_fallback_reason =
                    DMMA_UNIFIED_FALLBACK_FINE_OVERFLOW;
            else if (summary.saturated_task_count != 0)
                stats->unified_fallback_reason =
                    DMMA_UNIFIED_FALLBACK_SATURATED_LOAD;
            else if (!(schedule.cost.intercept > 0.0 ||
                       schedule.cost.scan > 0.0 ||
                       schedule.cost.match > 0.0 ||
                       schedule.cost.output > 0.0))
                stats->unified_fallback_reason =
                    DMMA_UNIFIED_FALLBACK_ZERO_LOAD;
            else
                stats->unified_fallback_reason =
                    DMMA_UNIFIED_FALLBACK_NONE;
            unified_symbolic_ready =
                stats->unified_fallback_reason ==
                DMMA_UNIFIED_FALLBACK_NONE;

            const int admitted_heavy_tasks =
                sparse_tail_records_ready ? tail_record_count : 0;
            const int admitted_heavy_fine_tasks =
                sparse_tail_records_ready ? summary.heavy_fine_count : 0;
            const DmmaUnifiedPartitionCounts partition =
                dmma_unified_partition_counts(
                    output_tile_count, unified_fine_task_count,
                    admitted_heavy_tasks, admitted_heavy_fine_tasks);
            const unsigned long long fine_units =
                summary.sparse_fine_units + summary.terminal_units;
            const unsigned long long admitted_heavy_units =
                sparse_tail_records_ready ? summary.heavy_units : 0ull;
            const unsigned long long admitted_heavy_fine_units =
                sparse_tail_records_ready ? summary.heavy_fine_units : 0ull;
            if (!partition.valid ||
                admitted_heavy_fine_units > fine_units ||
                admitted_heavy_units < admitted_heavy_fine_units)
                goto symbolic_failure;
            const double quantum = schedule.symbolic_load_quantum;
            stats->unified_coarse_tasks = partition.coarse;
            stats->unified_fine_tasks = partition.fine;
            stats->unified_heavy_fine_tasks = partition.heavy_fine;
            stats->unified_fine_work =
                static_cast<double>(fine_units -
                                    admitted_heavy_fine_units) *
                quantum;
            stats->unified_heavy_work =
                static_cast<double>(admitted_heavy_units) * quantum;
            /* Sparse replay deliberately does not count every coarse merge.
             * Expose that scope instead of presenting an upper bound as exact
             * work; explicit --collect-symbolic-load retains the full audit. */
            stats->unified_coarse_work = 0.0;
            stats->unified_coarse_work_available = false;
            stats->symbolic_load_saturated_tasks =
                summary.saturated_task_count;
            cudaFree(d_unified_replay_summary);
            d_unified_replay_summary = nullptr;
        }
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
    cudaFree(d_tail_append_state);
    cudaFree(d_symbolic_load_summary);
    cudaFree(d_unified_replay_summary);
    d_tail_append_state = nullptr;
    d_unified_replay_summary = nullptr;
    if (!sparse_tail_records_ready)
    {
        cudaFree(d_tail_records);
        d_tail_records = nullptr;
    }
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
    if (row_dynamic_auto_requested)
    {
        timeval gate_begin{}, gate_end{};
        DmmaRowGateDeviceSummary gate_summary{};
        gettimeofday(&gate_begin, nullptr);
        const int gate_threads =
            dmma_row_gate_reduction_threads(row_worker_count);
        if (gate_threads <= 0 || d_output_row_ptr == nullptr ||
            output_tile_count <= 0 ||
            !dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(
                                  &d_row_gate_summary),
                              sizeof(*d_row_gate_summary)),
                          "allocate exact-row-ptr gate summary"))
            goto numeric_failure;
        dmma_row_gate_exact_row_ptr_kernel<<<1, gate_threads>>>(
            d_output_row_ptr, a.tile_row_count, row_worker_count,
            d_row_gate_summary);
        if (!dmma_cuda_ok(cudaGetLastError(),
                          "launch exact-row-ptr gate reduction") ||
            !dmma_cuda_ok(cudaMemcpy(
                              &gate_summary, d_row_gate_summary,
                              sizeof(gate_summary), cudaMemcpyDeviceToHost),
                          "read exact-row-ptr gate summary"))
            goto numeric_failure;
        cudaFree(d_row_gate_summary);
        d_row_gate_summary = nullptr;
        const DmmaRowGateFeatures gate_features =
            dmma_row_gate_features(
                gate_summary, a.tile_row_count, row_worker_count,
                output_tile_count);
        /* A row worker serializes all C tiles belonging to the claimed row.
         * It is therefore only a viable granularity for short rows.  The
         * imbalance-only gate selected TSOPF (341 C tiles/row on average)
         * and changed numeric from 3.6 ms to 44.5 ms.  Fail closed to the
         * original tile-dynamic kernel when a row contains more than one
         * eight-warp scheduling window on average; long-row tails need tile
         * splitting, not whole-row ownership. */
        constexpr double kRowDynamicMaximumMeanTiles = 32.0;
        const double mean_tiles_per_row =
            gate_summary.rows > 0
                ? static_cast<double>(gate_summary.exact_tiles) /
                      static_cast<double>(gate_summary.rows)
                : 0.0;
        const bool gate_dynamic =
            mean_tiles_per_row <= kRowDynamicMaximumMeanTiles &&
            dmma_row_gate_select_dynamic(
                gate_summary, gate_features,
                schedule.row_dynamic_threshold);
        gettimeofday(&gate_end, nullptr);
        stats->row_gate_reduction_ms =
            dmma_elapsed_ms(gate_begin, gate_end);
        stats->row_gate_used = true;
        stats->row_gate_valid = gate_features.valid;
        stats->row_gate_decision_dynamic = gate_dynamic;
        stats->row_gate_rows = gate_summary.rows;
        stats->row_gate_workers = gate_summary.workers;
        stats->row_gate_zero_workers = gate_summary.zero_workers;
        stats->row_gate_exact_tiles = gate_summary.exact_tiles;
        stats->row_gate_load_sum = gate_summary.load_sum;
        stats->row_gate_load_max = gate_summary.load_max;
        stats->row_gate_load_sum_sq = gate_summary.load_sum_sq;
        stats->row_gate_static_max_over_mean =
            gate_features.static_max_over_mean;
        stats->row_gate_static_cv = gate_features.static_cv;
        if (!gate_features.valid)
        {
            std::fprintf(stderr,
                         "Invalid exact-row-ptr-v1 gate reduction.\n");
            goto numeric_failure;
        }
        row_dynamic_requested = gate_dynamic;
        /* A rejected auto gate must preserve the production tile-dynamic
         * baseline.  The previous experiment selected row-static here, which
         * changed numeric even on matrices where balancing was predicted not
         * to help. */
        row_static_requested = false;
        row_worker_requested = gate_dynamic;
        stats->direct_numeric_layout =
            gate_dynamic ? DMMA_DIRECT_NUMERIC_ROW_DYNAMIC
                         : DMMA_DIRECT_NUMERIC_TILE_DYNAMIC;
    }
    if (schedule.cost_balanced && output_tile_count > 0)
    {
        if (d_output_numeric_work == nullptr)
        {
            std::fprintf(stderr,
                         "cost-balanced numeric requires Super16 task-work "
                         "metadata.\n");
            goto numeric_failure;
        }
        gettimeofday(&begin, nullptr);
        int device = 0;
        int sm_count = 0;
        const std::size_t work_bytes =
            static_cast<std::size_t>(output_tile_count) *
            sizeof(*d_cost_sort_keys);
        const std::size_t order_bytes =
            static_cast<std::size_t>(output_tile_count) *
            sizeof(*d_cost_task_order);
        if (!dmma_cuda_ok(cudaGetDevice(&device),
                          "read cost-balanced CUDA device") ||
            !dmma_cuda_ok(cudaDeviceGetAttribute(
                              &sm_count, cudaDevAttrMultiProcessorCount,
                              device),
                          "read cost-balanced SM count") ||
            sm_count <= 0 ||
            !dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(&d_cost_sort_keys),
                              work_bytes),
                          "allocate cost-balanced sort keys") ||
            !dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(&d_cost_task_order),
                              order_bytes),
                          "allocate cost-balanced task order") ||
            !dmma_cuda_ok(cudaMalloc(
                              reinterpret_cast<void **>(&d_cost_queue_head),
                              sizeof(*d_cost_queue_head)),
                          "allocate cost-balanced queue head") ||
            !dmma_cuda_ok(cudaMemcpy(
                              d_cost_sort_keys, d_output_numeric_work,
                              work_bytes, cudaMemcpyDeviceToDevice),
                          "copy cost-balanced sort keys") ||
            !dmma_cuda_ok(cudaMemset(d_cost_queue_head, 0,
                                     sizeof(*d_cost_queue_head)),
                          "clear cost-balanced queue head"))
            goto numeric_failure;
        try
        {
            thrust::device_ptr<std::uint64_t> keys(d_cost_sort_keys);
            thrust::device_ptr<int> order(d_cost_task_order);
            thrust::sequence(order, order + output_tile_count, 0);
            thrust::sort_by_key(keys, keys + output_tile_count, order,
                                thrust::greater<std::uint64_t>());
        }
        catch (const std::exception &exception)
        {
            std::fprintf(stderr,
                         "Thrust error preparing cost-balanced order: %s\n",
                         exception.what());
            cudaGetLastError();
            goto numeric_failure;
        }
        cost_worker_blocks = sm_count * schedule.cost_workers_per_sm;
        stats->cost_balanced_used = true;
        stats->cost_worker_blocks = cost_worker_blocks;
        stats->cost_metadata_bytes = work_bytes + order_bytes +
                                     sizeof(*d_cost_queue_head);
        gettimeofday(&end, nullptr);
        stats->scheduler_ms += dmma_elapsed_ms(begin, end);
    }
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
    if (row_worker_requested)
    {
        if (row_worker_count <= 0 || d_row_worker_sm_ids == nullptr ||
            d_row_dynamic_next_row == nullptr)
            goto numeric_failure;
        if (row_static_requested)
        {
            stats->row_static_used = true;
            stats->row_static_ctas = row_worker_count;
        }
        else
        {
            stats->row_dynamic_used = true;
            stats->row_dynamic_ctas = row_worker_count;
        }
    }
    gettimeofday(&end, nullptr);
    stats->allocation_ms += dmma_elapsed_ms(begin, end);

    if (schedule.mode != DMMA_SCHEDULE_DIRECT && output_tile_count > 0)
    {
        gettimeofday(&begin, nullptr);
        if (oversized_candidate_stream)
            stats->tail_gate_reason = DMMA_TAIL_GATE_OVERSIZED_SYMBOLIC;

        /* A persistent light suffix is useful even when exact symbolic found
         * no heavy parent in that suffix.  In this one safe fallback case we
         * run prefix+suffix with an all-zero heavy bitset and leave the heavy
         * stream empty.  Overflow/high-fraction/disabled-reuse cases still
         * take the proven direct path because their heavy metadata is not a
         * complete trustworthy set. */
        const bool persistent_suffix_zero_heavy =
            persistent_suffix_requested && !sparse_tail_records_ready &&
            stats->tail_gate_reason == DMMA_TAIL_GATE_ZERO_HEAVY &&
            !tail_record_overflow && !maybe_candidate_overflow &&
            !oversized_candidate_stream;
        if (unified_light_requested && !unified_symbolic_ready &&
            stats->unified_fallback_reason ==
                DMMA_UNIFIED_FALLBACK_NONE)
            stats->unified_fallback_reason =
                DMMA_UNIFIED_FALLBACK_SYMBOLIC_UNAVAILABLE;
        /* A rejected heavy-fraction gate does not invalidate the complete
         * joint fine metadata.  Run all heavy parents as ordinary light tasks
         * while retaining the fine queue; only incomplete/overflowed tail
         * metadata rejects the whole unified schedule. */
        const bool unified_light_without_heavy =
            unified_light_requested && unified_symbolic_ready &&
            !sparse_tail_records_ready &&
            (stats->tail_gate_reason == DMMA_TAIL_GATE_ZERO_HEAVY ||
             stats->tail_gate_reason == DMMA_TAIL_GATE_HEAVY_FRACTION) &&
            !tail_record_overflow && !maybe_candidate_overflow &&
            !oversized_candidate_stream;
        const bool unified_metadata_rejected =
            unified_light_requested && !unified_symbolic_ready;
        if (unified_metadata_rejected ||
            (!sparse_tail_records_ready &&
             !persistent_suffix_zero_heavy &&
             !unified_light_without_heavy))
        {
            /* No dense scheduler-side replay: overflow, a high heavy fraction,
             * disabled reuse, and fused oversized symbolic all retain the
             * proven direct path and its branch-free numeric kernel. */
            use_direct_numeric = true;
            stats->tail_gate_fallback_to_direct = true;
            stats->zero_heavy_fallback_to_direct =
                stats->tail_gate_reason == DMMA_TAIL_GATE_ZERO_HEAVY;
            if (unified_light_requested && !unified_metadata_rejected)
                stats->unified_fallback_reason =
                    DMMA_UNIFIED_FALLBACK_TAIL_METADATA;
            cudaFree(d_tail_records);
            d_tail_records = nullptr;
        }
        else
        {
            unified_light_active =
                unified_light_requested && unified_symbolic_ready;
            stats->unified_light_used = unified_light_active;
            heavy_task_count =
                sparse_tail_records_ready ? tail_record_count : 0;
            if (split_context_borrowed)
            {
                if (!dmma_split_async_state_ready(split_async) ||
                    split_async.launch_policy != schedule.launch_policy)
                {
                    std::fprintf(
                        stderr,
                        "Borrowed split context is uninitialized or uses "
                        "a mismatched launch policy.\n");
                    goto numeric_failure;
                }
            }
            else if (!dmma_create_split_async_state(schedule.launch_policy,
                                                     &split_async))
            {
                goto numeric_failure;
            }
            if (!dmma_cuda_ok(cudaEventRecord(split_async.scheduler_start, 0),
                              "record split scheduler start"))
                goto numeric_failure;
            const int flag_task_begin =
                persistent_suffix_requested ? critical_q_begin : 0;
            const int flag_task_count =
                persistent_suffix_requested ? suffix_task_count
                                            : output_tile_count;
            const std::size_t task_count =
                static_cast<std::size_t>(flag_task_count);
            const std::size_t heavy_count =
                static_cast<std::size_t>(heavy_task_count);
            if (heavy_task_count > INT_MAX / schedule.max_chunks)
            {
                std::fprintf(stderr,
                             "Sparse tail chunk prefix sum may exceed "
                             "32-bit storage.\n");
                goto numeric_failure;
            }
            const std::size_t heavy_flag_words = (task_count + 31u) / 32u;
            const std::size_t heavy_flag_bytes =
                heavy_flag_words * sizeof(uint32_t);
            stats->heavy_flag_bytes = heavy_flag_bytes;
            if (!dmma_cuda_ok(cudaMalloc(
                                  reinterpret_cast<void **>(&d_heavy_flags),
                                  heavy_flag_bytes),
                              "allocate heavy-task bitset") ||
                !dmma_cuda_ok(cudaMemset(
                                  d_heavy_flags, 0,
                                  heavy_flag_bytes),
                              "clear heavy-task bitset") ||
                !dmma_cuda_ok(cudaMalloc(
                                  reinterpret_cast<void **>(
                                      &d_invalid_tail_records),
                                  sizeof(int)),
                              "allocate invalid tail-record flag") ||
                !dmma_cuda_ok(cudaMemset(
                                  d_invalid_tail_records, 0, sizeof(int)),
                              "clear invalid tail-record flag") ||
                !dmma_cuda_ok(cudaMalloc(
                                  reinterpret_cast<void **>(&d_chunk_offsets),
                                  (heavy_count + 1) * sizeof(int)),
                              "allocate sparse heavy chunk offsets") ||
                !dmma_cuda_ok(cudaMemset(
                                  d_chunk_offsets, 0,
                                  (heavy_count + 1) * sizeof(int)),
                              "clear sparse heavy chunk offsets"))
                goto numeric_failure;

            const int metadata_threads = 256;
            unsigned int metadata_blocks = 0;
            int invalid_tail_records = 0;
            if (heavy_task_count > 0)
            {
                if (!dmma_launch_blocks(heavy_count, metadata_threads,
                                        &metadata_blocks,
                                        "prepare sparse tail metadata"))
                    goto numeric_failure;
                dmma_prepare_sparse_tail_metadata_kernel
                    <<<metadata_blocks, metadata_threads>>>(
                        heavy_task_count, d_tail_records, flag_task_begin,
                        flag_task_count, schedule.max_chunks,
                        d_heavy_flags, d_chunk_offsets,
                        d_invalid_tail_records);
                if (!dmma_cuda_ok(cudaGetLastError(),
                                  "prepare sparse tail metadata") ||
                    !dmma_cuda_ok(cudaMemcpy(
                                      &invalid_tail_records,
                                      d_invalid_tail_records, sizeof(int),
                                      cudaMemcpyDeviceToHost),
                                  "read invalid tail-record flag") ||
                    invalid_tail_records != 0)
                {
                    std::fprintf(stderr,
                                 "Sparse tail record escaped its exact-output "
                                 "scheduling window.\n");
                    goto numeric_failure;
                }
            }
            if (!dmma_exclusive_scan_int(
                    d_chunk_offsets, heavy_count + 1,
                    "scan sparse heavy chunk counts") ||
                !dmma_cuda_ok(cudaMemcpy(
                                  &split_chunk_count,
                                  d_chunk_offsets + heavy_task_count,
                                  sizeof(int), cudaMemcpyDeviceToHost),
                              "read sparse split chunk count") ||
                split_chunk_count < 0)
                goto numeric_failure;

            if (split_chunk_count > 0 &&
                !dmma_cuda_ok(cudaMalloc(
                                  reinterpret_cast<void **>(
                                      &d_chunk_descriptors),
                                  static_cast<std::size_t>(split_chunk_count) *
                                      sizeof(DmmaChunkDescriptor)),
                              "allocate chunk descriptors"))
                goto numeric_failure;
            if (split_chunk_count > 0 && schedule.collect_task_stats &&
                !dmma_cuda_ok(cudaMalloc(
                                  reinterpret_cast<void **>(&d_chunk_sm_ids),
                                  static_cast<std::size_t>(split_chunk_count) *
                                      sizeof(uint32_t)),
                              "allocate chunk SM IDs"))
                goto numeric_failure;
            const std::size_t workspace_bytes =
                static_cast<std::size_t>(split_chunk_count) *
                DMMA_OUTPUT_ELEMS * sizeof(MAT_VAL_TYPE);
            if (effective_reduction == DMMA_REDUCTION_WORKSPACE &&
                workspace_bytes > schedule.workspace_limit_bytes)
            {
                effective_reduction = DMMA_REDUCTION_ATOMIC;
                stats->workspace_fallback_to_atomic = true;
            }
            if (effective_reduction == DMMA_REDUCTION_WORKSPACE &&
                split_chunk_count > 0)
            {
                if (!dmma_cuda_ok(cudaMalloc(
                                      reinterpret_cast<void **>(
                                          &d_partial_workspace),
                                      workspace_bytes),
                                  "allocate partial C workspace"))
                    goto numeric_failure;
                stats->partial_workspace_bytes = workspace_bytes;
            }
            if ((schedule.mode == DMMA_SCHEDULE_SPLIT_PERSISTENT ||
                 tile_tail_queue_requested ||
                 tile_early_split_requested) &&
                split_chunk_count > 0 &&
                (!dmma_cuda_ok(cudaMalloc(
                                   reinterpret_cast<void **>(
                                       &d_persistent_queue_head),
                                   sizeof(int)),
                               "allocate persistent queue head") ||
                 !dmma_cuda_ok(cudaMemset(d_persistent_queue_head, 0,
                                          sizeof(int)),
                               "clear persistent queue head")))
                goto numeric_failure;

            if (unified_light_active)
            {
                int active_light_ctas_per_sm = 0;
                int device = 0;
                cudaDeviceProp properties{};
                if (!dmma_cuda_ok(cudaGetDevice(&device),
                                  "read unified schedule device") ||
                    !dmma_cuda_ok(cudaGetDeviceProperties(&properties,
                                                          device),
                                  "read unified schedule properties") ||
                    !dmma_cuda_ok(
                        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            &active_light_ctas_per_sm,
                            dmma_numeric_unified_persistent_light_kernel<false>,
                            DMMA_THREADS_PER_BLOCK, 0),
                        "query unified light occupancy") ||
                    active_light_ctas_per_sm < 1 ||
                    schedule.unified_workers_per_sm >
                        active_light_ctas_per_sm ||
                    properties.multiProcessorCount < 1)
                    goto numeric_failure;
                /* Leave one resident-CTA slot per SM for admitted high-
                 * priority heavy chunks whenever occupancy permits it. */
                const int auto_light_ctas_per_sm =
                    heavy_task_count > 0 && active_light_ctas_per_sm > 1
                        ? active_light_ctas_per_sm - 1
                        : active_light_ctas_per_sm;
                const int light_ctas_per_sm =
                    schedule.unified_workers_per_sm > 0
                        ? schedule.unified_workers_per_sm
                        : auto_light_ctas_per_sm;
                const long long worker_blocks_wide =
                    static_cast<long long>(properties.multiProcessorCount) *
                    light_ctas_per_sm;
                if (worker_blocks_wide <= 0 || worker_blocks_wide > INT_MAX)
                    goto numeric_failure;
                unified_worker_blocks = static_cast<int>(
                    std::min<long long>(output_tile_count,
                                        worker_blocks_wide));
                if (unified_worker_blocks < 1 ||
                    !dmma_cuda_ok(cudaMalloc(
                                      reinterpret_cast<void **>(
                                          &d_unified_coarse_head),
                                      sizeof(unsigned long long)),
                                  "allocate unified coarse queue head") ||
                    !dmma_cuda_ok(cudaMalloc(
                                      reinterpret_cast<void **>(
                                          &d_unified_fine_head),
                                      sizeof(unsigned long long)),
                                  "allocate unified fine queue head") ||
                    !dmma_cuda_ok(cudaMemset(
                                      d_unified_coarse_head, 0,
                                      sizeof(unsigned long long)),
                                  "clear unified coarse queue head") ||
                    !dmma_cuda_ok(cudaMemset(
                                      d_unified_fine_head, 0,
                                      sizeof(unsigned long long)),
                                  "clear unified fine queue head"))
                    goto numeric_failure;
                stats->unified_worker_blocks = unified_worker_blocks;
                stats->unified_workers_per_sm = light_ctas_per_sm;
                stats->unified_page_size = schedule.unified_page_size;
                stats->unified_coarse_pages =
                    output_tile_count / schedule.unified_page_size +
                    (output_tile_count % schedule.unified_page_size != 0
                         ? 1
                         : 0);
            }

            if (persistent_suffix_requested)
            {
                int prefix_ctas_per_sm = 0;
                int suffix_ctas_per_sm = 0;
                int device = 0;
                cudaDeviceProp properties{};
                if (!dmma_cuda_ok(cudaGetDevice(&device),
                                  "read suffix schedule device") ||
                    !dmma_cuda_ok(cudaGetDeviceProperties(&properties,
                                                          device),
                                  "read suffix schedule properties") ||
                    !dmma_cuda_ok(
                        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            &prefix_ctas_per_sm,
                            dmma_numeric_prefix_kernel,
                            DMMA_THREADS_PER_BLOCK, 0),
                        "query prefix numeric occupancy") ||
                    !dmma_cuda_ok(
                        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                            &suffix_ctas_per_sm,
                            dmma_numeric_suffix_persistent_kernel,
                            WARP_SIZE, 0),
                        "query suffix numeric occupancy") ||
                    prefix_ctas_per_sm < 1 || suffix_ctas_per_sm < 1 ||
                    properties.multiProcessorCount < 1)
                    goto numeric_failure;
                const int full_warps =
                    prefix_ctas_per_sm * DMMA_WARPS_PER_BLOCK;
                int requested_workers = schedule.suffix_workers_per_sm;
                if (requested_workers == 0)
                {
                    const DmmaSuffixAutoSelection selection =
                        dmma_select_suffix_auto_fraction(
                            schedule.suffix_auto_basis,
                            stats->suffix_task_fraction,
                            stats->symbolic_total_work,
                            stats->symbolic_suffix_work,
                            sparse_tail_records_ready
                                ? stats->symbolic_split_suffix_work
                                : 0.0,
                            stats->symbolic_load_saturated_tasks);
                    stats->suffix_auto_fraction = selection.fraction;
                    stats->suffix_auto_used_symbolic_work =
                        selection.used_symbolic_work;
                    stats->suffix_auto_fallback_to_tasks =
                        selection.fallback_to_tasks;
                    requested_workers =
                        dmma_suffix_auto_requested_workers(
                            selection.fraction, full_warps);
                }
                suffix_workers_per_sm = std::max(
                    1, std::min(requested_workers, suffix_ctas_per_sm));
                const long long worker_blocks_wide =
                    static_cast<long long>(properties.multiProcessorCount) *
                    suffix_workers_per_sm;
                if (worker_blocks_wide <= 0 ||
                    worker_blocks_wide > INT_MAX)
                    goto numeric_failure;
                suffix_worker_blocks = static_cast<int>(
                    std::min<long long>(suffix_task_count,
                                        worker_blocks_wide));
                suffix_fine_tasks = dmma_suffix_fine_task_count(
                    suffix_task_count, suffix_worker_blocks,
                    schedule.suffix_fine_tasks_per_worker);
                suffix_bulk_tasks =
                    suffix_task_count - suffix_fine_tasks;
                if (suffix_worker_blocks < 1 ||
                    !dmma_cuda_ok(cudaMalloc(
                                      reinterpret_cast<void **>(
                                          &d_suffix_bulk_head),
                                      sizeof(unsigned long long)),
                                  "allocate suffix bulk queue head") ||
                    !dmma_cuda_ok(cudaMalloc(
                                      reinterpret_cast<void **>(
                                          &d_suffix_fine_head),
                                      sizeof(unsigned long long)),
                                  "allocate suffix fine queue head") ||
                    !dmma_cuda_ok(cudaMemset(
                                      d_suffix_bulk_head, 0,
                                      sizeof(unsigned long long)),
                                  "clear suffix bulk queue head") ||
                    !dmma_cuda_ok(cudaMemset(
                                      d_suffix_fine_head, 0,
                                      sizeof(unsigned long long)),
                                  "clear suffix fine queue head"))
                    goto numeric_failure;
                stats->suffix_workers_per_sm = suffix_workers_per_sm;
                stats->suffix_worker_blocks = suffix_worker_blocks;
                stats->suffix_queue_batch = schedule.suffix_queue_batch;
                stats->suffix_fine_tasks = suffix_fine_tasks;
            }

            if (heavy_task_count > 0)
            {
                unsigned int descriptor_blocks = 0;
                if (!dmma_launch_blocks(heavy_count, metadata_threads,
                                        &descriptor_blocks,
                                        "emit sparse heavy descriptors"))
                    goto numeric_failure;
                dmma_emit_chunk_descriptors_kernel
                    <<<descriptor_blocks, metadata_threads>>>(
                        a, b, heavy_task_count, d_tail_records,
                        d_output_rows, d_output_cols, d_chunk_offsets,
                        schedule.cost, d_chunk_descriptors);
                if (!dmma_cuda_ok(cudaGetLastError(),
                                  "emit chunk descriptors") ||
                    !dmma_launch_blocks(heavy_count, DMMA_WARPS_PER_BLOCK,
                                        &heavy_task_blocks,
                                        "prepare heavy split output"))
                    goto numeric_failure;
                dmma_prepare_heavy_output_kernel
                    <<<heavy_task_blocks, DMMA_THREADS_PER_BLOCK>>>(
                        heavy_task_count, d_tail_records, d_output_masks,
                        d_output_nnz, d_output_tile_row_ptr,
                        d_output_value_cols, d_output_values,
                        effective_reduction == DMMA_REDUCTION_ATOMIC ? 1 : 0);
                if (!dmma_cuda_ok(cudaGetLastError(),
                                  "prepare split numeric layout"))
                    goto numeric_failure;
            }

            stats->scheduler_reused_symbolic_counts = true;
            stats->scheduler_reused_count_tasks = heavy_task_count;
            stats->heavy_tasks = heavy_task_count;
            stats->split_chunks = split_chunk_count;
            stats->split_task_fraction =
                static_cast<double>(heavy_task_count) / output_tile_count;
            stats->average_chunks =
                heavy_task_count > 0
                    ? static_cast<double>(split_chunk_count) /
                          heavy_task_count
                    : 0.0;
            stats->reduction_mode = effective_reduction;
            /* Descriptor generation and heavy-output preparation above are
             * ordered on stream zero.  Non-blocking numeric streams do not
             * inherit legacy-default-stream dependencies, so publish an
             * explicit per-call readiness event instead of synchronizing the
             * entire device here. */
            if (!dmma_cuda_ok(cudaEventRecord(split_async.scheduler_stop, 0),
                              "record split scheduler stop") ||
                !dmma_cuda_ok(cudaEventRecord(split_async.metadata_ready, 0),
                              "record split scheduler metadata ready"))
                goto numeric_failure;
            stats->split_streams_created = !split_context_borrowed;
            stats->split_streams_used = true;
            stats->split_context_reused = split_context_borrowed;
            stats->light_stream_priority = split_async.light_priority;
            stats->suffix_stream_priority = split_async.suffix_priority;
            stats->heavy_stream_priority = split_async.heavy_priority;
            stats->stream_priority_range_supported =
                split_async.priority_range_supported;
        }
        gettimeofday(&end, nullptr);
        stats->scheduler_ms = dmma_elapsed_ms(begin, end);
    }
#ifdef DMMA_ENABLE_TIMELINE_TRACE
    /* The legacy trace format is one slot per dynamically scheduled output
     * warp and cannot represent persistent row workers. */
    timeline_path = row_worker_requested ? nullptr :
                                           std::getenv("DMMA_TRACE_FILE");
    if (use_direct_numeric && timeline_path != nullptr &&
        *timeline_path != '\0')
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
        if (!row_worker_requested && numeric_blocks == 0 &&
            !dmma_launch_blocks(
                static_cast<std::size_t>(output_tile_count),
                DMMA_WARPS_PER_BLOCK, &numeric_blocks,
                "uniform DMMA kernel"))
            goto numeric_failure;
        if (use_direct_numeric)
        {
            if (schedule.cost_balanced && output_tile_count > 0)
            {
                dmma_numeric_cost_queue_kernel
                    <<<static_cast<unsigned int>(cost_worker_blocks),
                       DMMA_THREADS_PER_BLOCK>>>(
                        a, b, output_tile_count, d_cost_task_order,
                        d_cost_queue_head, d_output_rows, d_output_cols,
                        d_output_masks, d_output_nnz,
                        d_output_tile_row_ptr, d_output_value_cols,
                        d_output_values);
            }
            else if (row_worker_requested)
            {
                dmma_numeric_row_worker_kernel<false>
                    <<<static_cast<unsigned int>(row_worker_count),
                       DMMA_THREADS_PER_BLOCK>>>(
                        a, b, a.tile_row_count, output_tile_count,
                        d_output_row_ptr, d_output_rows, d_output_cols,
                        d_output_masks, d_output_nnz,
                        d_output_tile_row_ptr, d_output_value_cols,
                        d_output_values, row_dynamic_requested ? 1 : 0,
                        schedule.row_queue_batch, d_row_dynamic_next_row,
                        d_row_worker_sm_ids, nullptr, nullptr);
            }
            else
            {
                if (stats->low_fill_exact_tile_used)
                    dmma_numeric_low_fill_exact_tile_kernel
                        <<<numeric_blocks, DMMA_THREADS_PER_BLOCK>>>(
                            a, b, output_tile_count, d_output_rows,
                            d_output_cols, d_output_masks, d_output_nnz,
                            schedule.low_fill_q, d_output_tile_row_ptr,
                            d_output_value_cols, d_output_values
#ifdef DMMA_ENABLE_TIMELINE_TRACE
                            , timeline
#endif
                            );
                else
                    dmma_numeric_kernel
                        <<<numeric_blocks, DMMA_THREADS_PER_BLOCK>>>(
                            a, b, output_tile_count, d_output_rows,
                            d_output_cols, d_output_masks, d_output_nnz,
                            d_output_tile_row_ptr, d_output_value_cols,
                            d_output_values
#ifdef DMMA_ENABLE_TIMELINE_TRACE
                            , timeline
#endif
                            );
            }
            if (!dmma_cuda_ok(cudaGetLastError(),
                              schedule.cost_balanced
                                  ? "launch cost-balanced DMMA kernel"
                                  : row_static_requested
                                  ? "launch row-static-block DMMA kernel"
                                  : (row_dynamic_requested
                                         ? "launch row-dynamic DMMA kernel"
                                         : "launch uniform DMMA kernel")) ||
                !dmma_cuda_ok(cudaDeviceSynchronize(),
                              row_static_requested
                                  ? "row-static-block DMMA kernel"
                                  : (row_dynamic_requested
                                         ? "row-dynamic DMMA kernel"
                                         : "uniform DMMA kernel")))
                goto numeric_failure;
            /* The synchronized native C is the direct-path Core endpoint;
             * numeric timing bookkeeping below must remain outside it. */
            dmma_close_core_endpoint(total_begin, &total_end, stats);
        }
        else
        {
            const int workspace_mode =
                effective_reduction == DMMA_REDUCTION_WORKSPACE ? 1 : 0;
            int persistent_blocks = 0;
            if (split_chunk_count > 0 &&
                (schedule.mode == DMMA_SCHEDULE_SPLIT_PERSISTENT ||
                 tile_tail_queue_requested ||
                 tile_early_split_requested))
            {
                int device = 0;
                cudaDeviceProp properties{};
                if (!dmma_cuda_ok(cudaGetDevice(&device),
                                  "read persistent schedule device") ||
                    !dmma_cuda_ok(cudaGetDeviceProperties(&properties,
                                                          device),
                                  "read persistent schedule properties"))
                    goto numeric_failure;
                const int requested_blocks = tile_early_split_requested
                    ? dmma_tile_early_split_global_worker_cap(
                          properties.multiProcessorCount)
                    : std::max(1, properties.multiProcessorCount * 2);
                if (requested_blocks <= 0)
                {
                    std::fprintf(stderr,
                                 "Invalid early-split global heavy worker "
                                 "cap.\n");
                    goto numeric_failure;
                }
                persistent_blocks =
                    std::min(split_chunk_count, requested_blocks);
                if (tile_early_split_requested)
                {
                    stats->early_split_used = true;
                    stats->early_heavy_worker_block_cap = requested_blocks;
                    stats->early_heavy_worker_blocks = persistent_blocks;
                }
                int expected_chunk_queue_head = 0;
                if (!dmma_tile_early_split_expected_queue_final_head(
                        split_chunk_count, persistent_blocks,
                        &expected_chunk_queue_head))
                {
                    std::fprintf(stderr,
                                 "Persistent heavy queue head may exceed "
                                 "32-bit storage.\n");
                    goto numeric_failure;
                }
            }

            if (!dmma_cuda_ok(cudaStreamWaitEvent(
                                  split_async.light_stream,
                                  split_async.metadata_ready, 0),
                              "wait for light scheduler metadata") ||
                !dmma_cuda_ok(cudaStreamWaitEvent(
                                  split_async.heavy_stream,
                                  split_async.metadata_ready, 0),
                              "wait for heavy scheduler metadata") ||
                (persistent_suffix_requested &&
                 !dmma_cuda_ok(cudaStreamWaitEvent(
                                   split_async.suffix_stream,
                                   split_async.metadata_ready, 0),
                               "wait for suffix scheduler metadata")) ||
                !dmma_cuda_ok(cudaEventRecord(split_async.numeric_start, 0),
                              "record split numeric start") ||
                !dmma_cuda_ok(cudaStreamWaitEvent(
                                  split_async.light_stream,
                                  split_async.numeric_start, 0),
                              "start light numeric stream") ||
                !dmma_cuda_ok(cudaStreamWaitEvent(
                                  split_async.heavy_stream,
                                  split_async.numeric_start, 0),
                              "start heavy numeric stream") ||
                (persistent_suffix_requested &&
                 !dmma_cuda_ok(cudaStreamWaitEvent(
                                   split_async.suffix_stream,
                                   split_async.numeric_start, 0),
                               "start suffix numeric stream")))
                goto numeric_failure;

            cudaEvent_t heavy_stop = nullptr;
            const auto enqueue_light = [&]() -> bool {
                if (!dmma_cuda_ok(cudaEventRecord(
                                      split_async.light_start,
                                      split_async.light_stream),
                                  "record light numeric start"))
                    return false;
                if (flat_grid_requested)
                {
                    const std::size_t flat_work_items =
                        static_cast<std::size_t>(split_chunk_count) +
                        static_cast<std::size_t>(output_tile_count);
                    const unsigned long long flat_blocks_wide =
                        dmma_flat_transpose_grid_blocks_wide(
                            split_chunk_count, output_tile_count,
                            schedule.flat_warps_per_cta);
                    unsigned int flat_blocks = 0;
                    if (!dmma_launch_blocks(
                            static_cast<std::size_t>(flat_blocks_wide), 1,
                            &flat_blocks, "flat mixed DMMA grid") ||
                        !dmma_cuda_ok(cudaEventRecord(
                                          split_async.chunk_start,
                                          split_async.light_stream),
                                      "record flat mixed numeric start"))
                        return false;
                    const auto launch_flat = [&](auto warp_tag) {
                        constexpr int warps = decltype(warp_tag)::value;
                        dmma_numeric_flat_mixed_kernel<warps, false>
                            <<<flat_blocks, warps * WARP_SIZE, 0,
                               split_async.light_stream>>>(
                                a, b, split_chunk_count,
                                d_chunk_descriptors, output_tile_count,
                                d_output_rows, d_output_cols, d_output_masks,
                                d_output_nnz, d_heavy_flags,
                                d_output_tile_row_ptr, d_output_value_cols,
                                d_output_values, d_partial_workspace,
                                d_chunk_sm_ids, workspace_mode, nullptr,
                                nullptr);
                    };
                    if (schedule.flat_warps_per_cta == 1)
                        launch_flat(std::integral_constant<int, 1>{});
                    else if (schedule.flat_warps_per_cta == 2)
                        launch_flat(std::integral_constant<int, 2>{});
                    else
                        launch_flat(std::integral_constant<int, 4>{});
                    if (!dmma_cuda_ok(cudaGetLastError(),
                                      "launch flat mixed DMMA kernel") ||
                        !dmma_cuda_ok(cudaEventRecord(
                                          split_async.chunk_stop,
                                          split_async.light_stream),
                                      "record flat mixed numeric stop") ||
                        !dmma_cuda_ok(cudaEventRecord(
                                          split_async.light_stop,
                                          split_async.light_stream),
                                      "record flat mixed light stop"))
                        return false;
                    stats->flat_grid_used = true;
                    stats->flat_warps_per_cta =
                        schedule.flat_warps_per_cta;
                    stats->flat_grid_blocks = flat_blocks;
                    stats->flat_work_items =
                        static_cast<unsigned long long>(flat_work_items);
                    heavy_stop = split_async.chunk_stop;
                    if (effective_reduction == DMMA_REDUCTION_WORKSPACE)
                    {
                        if (!dmma_cuda_ok(cudaEventRecord(
                                              split_async.reduction_start,
                                              split_async.light_stream),
                                          "record flat reduction start"))
                            return false;
                        dmma_reduce_chunk_workspace_kernel
                            <<<heavy_task_blocks, DMMA_THREADS_PER_BLOCK, 0,
                               split_async.light_stream>>>(
                                heavy_task_count, d_tail_records,
                                d_chunk_offsets, d_output_masks,
                                d_output_nnz, d_partial_workspace,
                                d_output_values);
                        if (!dmma_cuda_ok(cudaGetLastError(),
                                          "launch flat partial C reduction") ||
                            !dmma_cuda_ok(cudaEventRecord(
                                              split_async.reduction_stop,
                                              split_async.light_stream),
                                          "record flat reduction stop"))
                            return false;
                        heavy_stop = split_async.reduction_stop;
                    }
                    return true;
                }
                if (unified_light_active)
                {
                    dmma_numeric_unified_persistent_light_kernel<false>
                        <<<static_cast<unsigned int>(unified_worker_blocks),
                           DMMA_THREADS_PER_BLOCK, 0,
                            split_async.light_stream>>>(
                            a, b, output_tile_count,
                            unified_sparse_fine_task_count,
                            unified_terminal_begin,
                            schedule.unified_page_size,
                            d_unified_fine_ids, d_unified_fine_flags,
                            d_output_rows, d_output_cols, d_output_masks,
                            d_output_nnz, d_heavy_flags,
                            d_unified_coarse_head, d_unified_fine_head,
                            d_output_tile_row_ptr, d_output_value_cols,
                            d_output_values, nullptr);
                    return dmma_cuda_ok(
                               cudaGetLastError(),
                               "launch unified persistent light DMMA kernel") &&
                           dmma_cuda_ok(cudaEventRecord(
                                            split_async.light_stop,
                                            split_async.light_stream),
                                        "record unified light numeric stop");
                }
                if (persistent_suffix_requested)
                {
                    if (prefix_task_count > 0)
                    {
                        unsigned int prefix_blocks = 0;
                        if (!dmma_launch_blocks(
                                static_cast<std::size_t>(prefix_task_count),
                                DMMA_WARPS_PER_BLOCK, &prefix_blocks,
                                "prefix DMMA kernel"))
                            return false;
                        dmma_numeric_prefix_kernel
                            <<<prefix_blocks, DMMA_THREADS_PER_BLOCK, 0,
                               split_async.light_stream>>>(
                                a, b, output_tile_count, prefix_task_count,
                                d_output_rows, d_output_cols, d_output_masks,
                                d_output_nnz, d_output_tile_row_ptr,
                                d_output_value_cols, d_output_values);
                        if (!dmma_cuda_ok(cudaGetLastError(),
                                          "launch prefix DMMA kernel"))
                            return false;
                    }
                    if (!dmma_cuda_ok(cudaEventRecord(
                                          split_async.light_stop,
                                          split_async.light_stream),
                                      "record prefix numeric stop") ||
                        !dmma_cuda_ok(cudaEventRecord(
                                          split_async.suffix_start,
                                          split_async.suffix_stream),
                                      "record suffix numeric start"))
                        return false;
                    dmma_numeric_suffix_persistent_kernel
                        <<<static_cast<unsigned int>(suffix_worker_blocks),
                           WARP_SIZE, 0, split_async.suffix_stream>>>(
                            a, b, output_tile_count, critical_q_begin,
                            suffix_task_count, suffix_bulk_tasks,
                            schedule.suffix_queue_batch, d_output_rows,
                            d_output_cols, d_output_masks, d_output_nnz,
                            d_heavy_flags, d_suffix_bulk_head,
                            d_suffix_fine_head, d_output_tile_row_ptr,
                            d_output_value_cols, d_output_values);
                    return dmma_cuda_ok(cudaGetLastError(),
                                        "launch persistent suffix DMMA kernel") &&
                           dmma_cuda_ok(cudaEventRecord(
                                            split_async.suffix_stop,
                                            split_async.suffix_stream),
                                        "record suffix numeric stop");
                }

                dmma_numeric_light_kernel
                    <<<numeric_blocks, DMMA_THREADS_PER_BLOCK, 0,
                       split_async.light_stream>>>(
                        a, b, output_tile_count, d_output_rows,
                        d_output_cols, d_output_masks, d_output_nnz,
                        d_heavy_flags, d_output_tile_row_ptr,
                        d_output_value_cols, d_output_values);
                return dmma_cuda_ok(cudaGetLastError(),
                                    "launch light DMMA kernel") &&
                       dmma_cuda_ok(cudaEventRecord(
                                        split_async.light_stop,
                                        split_async.light_stream),
                                    "record light numeric stop");
            };
            const auto enqueue_heavy = [&]() -> bool {
                if (flat_grid_requested)
                    return true;
                if (split_chunk_count == 0)
                    return true;
                const cudaStream_t chunk_stream =
                    tile_tail_queue_requested ? split_async.light_stream
                                              : split_async.heavy_stream;
                if (!dmma_cuda_ok(cudaEventRecord(
                                      split_async.chunk_start, chunk_stream),
                                  "record chunk numeric start"))
                    return false;
                if (schedule.mode == DMMA_SCHEDULE_SPLIT_CTA)
                {
                    dmma_chunk_cta_kernel
                        <<<static_cast<unsigned int>(split_chunk_count),
                           WARP_SIZE, 0, chunk_stream>>>(
                            a, b, split_chunk_count, d_chunk_descriptors,
                            d_output_masks, d_output_nnz, d_output_values,
                            d_partial_workspace, d_chunk_sm_ids,
                            workspace_mode);
                }
                else
                {
                    dmma_chunk_persistent_kernel
                        <<<static_cast<unsigned int>(persistent_blocks),
                           DMMA_THREADS_PER_BLOCK, 0, chunk_stream>>>(
                            a, b, split_chunk_count, d_chunk_descriptors,
                            d_output_masks, d_output_nnz, d_output_values,
                            d_partial_workspace, d_chunk_sm_ids,
                            d_persistent_queue_head, workspace_mode);
                }
                if (!dmma_cuda_ok(cudaGetLastError(),
                                  "launch heavy chunk DMMA kernel") ||
                    !dmma_cuda_ok(cudaEventRecord(split_async.chunk_stop,
                                                  chunk_stream),
                                  "record chunk numeric stop"))
                    return false;
                heavy_stop = split_async.chunk_stop;

                if (effective_reduction == DMMA_REDUCTION_WORKSPACE)
                {
                    if (!dmma_cuda_ok(cudaEventRecord(
                                          split_async.reduction_start,
                                          chunk_stream),
                                      "record partial C reduction start"))
                        return false;
                    if (tile_early_split_requested)
                    {
                        const int reduction_blocks = std::min(
                            heavy_task_count,
                            stats->early_heavy_worker_block_cap);
                        int expected_reduction_queue_head = 0;
                        if (reduction_blocks <= 0 ||
                            !dmma_tile_early_split_expected_queue_final_head(
                                heavy_task_count, reduction_blocks,
                                &expected_reduction_queue_head) ||
                            !dmma_cuda_ok(cudaMemsetAsync(
                                              d_persistent_queue_head, 0,
                                              sizeof(int), chunk_stream),
                                          "reset early reduction queue head"))
                            return false;
                        stats->early_reduction_worker_blocks =
                            reduction_blocks;
                        dmma_reduce_chunk_workspace_persistent_kernel
                            <<<static_cast<unsigned int>(reduction_blocks),
                               DMMA_THREADS_PER_BLOCK, 0, chunk_stream>>>(
                                heavy_task_count, d_tail_records,
                                d_chunk_offsets, d_output_masks,
                                d_output_nnz, d_partial_workspace,
                                d_output_values, d_persistent_queue_head);
                    }
                    else
                    {
                        dmma_reduce_chunk_workspace_kernel
                            <<<heavy_task_blocks, DMMA_THREADS_PER_BLOCK, 0,
                               chunk_stream>>>(
                                heavy_task_count, d_tail_records,
                                d_chunk_offsets, d_output_masks,
                                d_output_nnz, d_partial_workspace,
                                d_output_values);
                    }
                    if (!dmma_cuda_ok(cudaGetLastError(),
                                      "launch partial C reduction") ||
                        !dmma_cuda_ok(cudaEventRecord(
                                          split_async.reduction_stop,
                                              chunk_stream),
                                      "record partial C reduction stop"))
                        return false;
                    heavy_stop = split_async.reduction_stop;
                }
                return true;
            };

            /* LIGHT_FIRST exactly preserves P2's host enqueue order for the
             * ablation.  Both tail-first policies enqueue the complete heavy
             * chain (including workspace reduction) before exposing the
             * saturating light grid to the device scheduler. */
            if (tile_tail_queue_requested)
            {
                /* Bulk first is the defining invariant: no chunk CTA can
                 * reserve registers/shared memory or reduce the ordinary
                 * tile grid's occupancy.  Once light_stop is reached, the
                 * same stream starts persistent queue CTAs on fully released
                 * SMs. */
                if (!enqueue_light() || !enqueue_heavy())
                    goto numeric_failure;
            }
            else if (tile_early_split_requested)
            {
                /* Equal-priority streams are a fail-closed CLI invariant.
                 * Enqueue the globally capped heavy queue first so its chunks
                 * are eligible at numeric_start, then expose the complete
                 * ordinary grid. CUDA decides which ready CTA uses each
                 * available SM resource; no affinity is assumed. */
                if (!enqueue_heavy() || !enqueue_light())
                    goto numeric_failure;
            }
            else if (schedule.launch_policy == DMMA_SPLIT_LAUNCH_LIGHT_FIRST)
            {
                if (!enqueue_light() || !enqueue_heavy())
                    goto numeric_failure;
            }
            else
            {
                if (!enqueue_heavy() || !enqueue_light())
                    goto numeric_failure;
            }

            if (!dmma_cuda_ok(cudaStreamWaitEvent(0, split_async.light_stop,
                                                  0),
                              persistent_suffix_requested
                                  ? "join prefix numeric stream"
                                  : "join light numeric stream") ||
                (persistent_suffix_requested &&
                 (!dmma_cuda_ok(cudaStreamWaitEvent(
                                    0, split_async.suffix_stop, 0),
                                "join suffix numeric stream") ||
                  !dmma_cuda_ok(cudaEventRecord(
                                    split_async.light_join_stop, 0),
                                "record joined light numeric stop"))) ||
                (heavy_stop != nullptr &&
                 !dmma_cuda_ok(cudaStreamWaitEvent(0, heavy_stop, 0),
                               "join heavy numeric stream")) ||
                !dmma_cuda_ok(cudaEventRecord(split_async.numeric_stop, 0),
                              "record split numeric stop") ||
                !dmma_cuda_ok(cudaEventSynchronize(split_async.numeric_stop),
                              "complete split numeric streams"))
                goto numeric_failure;

            /* numeric_stop is the first point at which native C is ready.
             * Close Core immediately: event queries, diagnostics, and local
             * split-context destruction below are host bookkeeping. */
            dmma_close_core_endpoint(total_begin, &total_end, stats);

            float elapsed = 0.0f;
            if (!dmma_cuda_ok(cudaEventElapsedTime(
                                  &elapsed, split_async.scheduler_start,
                                  split_async.scheduler_stop),
                              "time split scheduler device chain"))
                goto numeric_failure;
            stats->scheduler_device_ms = static_cast<double>(elapsed);
            if (!dmma_cuda_ok(cudaEventElapsedTime(
                                  &elapsed, split_async.numeric_start,
                                  split_async.numeric_stop),
                              "time complete split numeric"))
                goto numeric_failure;
            stats->numeric_ms = static_cast<double>(elapsed);
            if (persistent_suffix_requested)
            {
                if (!dmma_cuda_ok(cudaEventElapsedTime(
                                      &elapsed, split_async.light_start,
                                      split_async.light_stop),
                                  "time prefix numeric"))
                    goto numeric_failure;
                stats->prefix_numeric_ms = static_cast<double>(elapsed);
                if (!dmma_cuda_ok(cudaEventElapsedTime(
                                      &elapsed, split_async.suffix_start,
                                      split_async.suffix_stop),
                                  "time suffix numeric"))
                    goto numeric_failure;
                stats->suffix_numeric_ms = static_cast<double>(elapsed);
                if (!dmma_cuda_ok(cudaEventElapsedTime(
                                      &elapsed, split_async.numeric_start,
                                      split_async.light_join_stop),
                                  "time joined light numeric"))
                    goto numeric_failure;
                stats->light_numeric_ms = static_cast<double>(elapsed);

                unsigned long long bulk_head = 0;
                unsigned long long fine_head = 0;
                if (!dmma_cuda_ok(cudaMemcpy(
                                      &bulk_head, d_suffix_bulk_head,
                                      sizeof(bulk_head),
                                      cudaMemcpyDeviceToHost),
                                  "read suffix bulk queue head") ||
                    !dmma_cuda_ok(cudaMemcpy(
                                      &fine_head, d_suffix_fine_head,
                                      sizeof(fine_head),
                                      cudaMemcpyDeviceToHost),
                                  "read suffix fine queue head"))
                    goto numeric_failure;
                stats->suffix_queue_atomics =
                    bulk_head /
                        static_cast<unsigned long long>(
                            schedule.suffix_queue_batch) +
                    fine_head;
            }
            else
            {
                if (!dmma_cuda_ok(cudaEventElapsedTime(
                                      &elapsed, split_async.light_start,
                                      split_async.light_stop),
                                  "time light numeric"))
                    goto numeric_failure;
                stats->light_numeric_ms = static_cast<double>(elapsed);
                if (unified_light_active)
                {
                    unsigned long long coarse_head = 0;
                    unsigned long long fine_head = 0;
                    if (!dmma_cuda_ok(cudaMemcpy(
                                          &coarse_head,
                                          d_unified_coarse_head,
                                          sizeof(coarse_head),
                                          cudaMemcpyDeviceToHost),
                                      "read unified coarse queue head") ||
                        !dmma_cuda_ok(cudaMemcpy(
                                          &fine_head,
                                          d_unified_fine_head,
                                          sizeof(fine_head),
                                          cudaMemcpyDeviceToHost),
                                      "read unified fine queue head"))
                        goto numeric_failure;
                    stats->unified_coarse_page_claims =
                        coarse_head /
                        static_cast<unsigned long long>(
                            schedule.unified_page_size);
                    stats->unified_fine_ticket_claims = fine_head;
                }
            }
            if (split_chunk_count > 0)
            {
                if (!dmma_cuda_ok(cudaEventElapsedTime(
                                      &elapsed, split_async.chunk_start,
                                      split_async.chunk_stop),
                                  flat_grid_requested
                                      ? "time flat mixed numeric"
                                      : "time chunk numeric"))
                    goto numeric_failure;
                if (flat_grid_requested)
                    stats->flat_numeric_ms = static_cast<double>(elapsed);
                else
                    stats->chunk_numeric_ms = static_cast<double>(elapsed);
                if (effective_reduction == DMMA_REDUCTION_WORKSPACE)
                {
                    if (!dmma_cuda_ok(cudaEventElapsedTime(
                                          &elapsed,
                                          split_async.reduction_start,
                                          split_async.reduction_stop),
                                      "time partial C reduction"))
                        goto numeric_failure;
                    stats->reduction_ms = static_cast<double>(elapsed);
                }
            }
        }
    }
    gettimeofday(&end, nullptr);
    if (use_direct_numeric)
        stats->numeric_ms = dmma_elapsed_ms(begin, end);
    if (!split_context_borrowed)
        dmma_destroy_split_async_state(&split_async, false);

    if (!stats->core_completion_wall_valid)
        dmma_close_core_endpoint(total_begin, &total_end, stats);

    /* printf is deliberately post-endpoint.  The reported reduction_ms is
     * the in-Core wall interval covering summary allocation, the one-block
     * reduction, D2H, validation/decision, and release. */
    if (stats->row_gate_used)
        std::printf(
            "ROW_GATE_FEATURES version=exact-row-ptr-v1 "
            "source=exact-output-row-ptr-pre-numeric oracle=0 "
            "core_included=1 timing=end-to-end-reduction-d2h-dispatch "
            "rows=%d workers=%d exact_c_tiles=%llu load_sum=%llu "
            "load_sum_sq=%.17g load_max=%llu zero_workers=%d "
            "static_max_over_mean=%.17g static_cv=%.17g "
            "threshold=%.17g reduction_ms=%.6f valid=%d "
            "decision=%s selected_layout=%s queue_batch=%d "
            "cta_internal_semantics=unchanged\n",
            stats->row_gate_rows, stats->row_gate_workers,
            stats->row_gate_exact_tiles, stats->row_gate_load_sum,
            stats->row_gate_load_sum_sq, stats->row_gate_load_max,
            stats->row_gate_zero_workers,
            stats->row_gate_static_max_over_mean,
            stats->row_gate_static_cv, stats->row_gate_threshold,
            stats->row_gate_reduction_ms,
            stats->row_gate_valid ? 1 : 0,
            stats->row_gate_decision_dynamic ? "row-dynamic"
                                             : "tile-dynamic",
            dmma_direct_numeric_layout_name(
                stats->direct_numeric_layout),
            stats->row_dynamic_queue_batch);
    if (stats->cost_balanced_requested)
        std::printf(
            "DMMA_COST_BALANCE requested=1 used=%d model=hybrid-format-v1 "
            "task_order=descending-work primitive=unchanged-tensor-core "
            "worker_blocks=%d metadata_bytes=%zu scheduler_ms=%.6f "
            "core_included=1\n",
            stats->cost_balanced_used ? 1 : 0,
            stats->cost_worker_blocks, stats->cost_metadata_bytes,
            stats->scheduler_ms);

    /* These D2H audits are deliberately after both the native-output Core
     * endpoint and numeric_ms.  Placement establishes whether static and
     * dynamic samples used the same strict one-worker-per-SM shape; the
     * dynamic head additionally proves R successful plus P terminal claims. */
    if (row_worker_requested && row_worker_count > 0)
    {
        std::vector<uint32_t> worker_sms(
            static_cast<std::size_t>(row_worker_count));
        if (!dmma_cuda_ok(cudaMemcpy(
                              worker_sms.data(),
                              d_row_worker_sm_ids,
                              worker_sms.size() * sizeof(worker_sms[0]),
                              cudaMemcpyDeviceToHost),
                          "copy row-worker SM IDs"))
            goto numeric_failure;
        bool ids_valid = true;
        for (uint32_t sm : worker_sms)
            ids_valid = ids_valid &&
                        sm < static_cast<uint32_t>(
                                 row_worker_device_sm_count);
        std::sort(worker_sms.begin(), worker_sms.end());
        const int unique_sms = static_cast<int>(
            std::unique(worker_sms.begin(), worker_sms.end()) -
            worker_sms.begin());
        const bool mapping_valid =
            ids_valid && unique_sms == row_worker_count;
        if (row_static_requested)
        {
            stats->row_static_unique_sms = unique_sms;
            stats->row_static_mapping_valid = mapping_valid;
            std::printf(
                "DMMA_ROW_STATIC layout=row-static-block "
                "partition=contiguous-equal-row-count cyclic_smoothing=0 "
                "used=1 ctas=%d "
                "unique_sms=%d mapping_valid=%d no_queue=1 no_steal=1 "
                "audit_setup_untimed=1 control_head_setup_untimed=1 "
                "control_head_unused=1 smid_stores_per_cta=1 "
                "mapping_audit_core_excluded=1\n",
                row_worker_count, unique_sms,
                stats->row_static_mapping_valid ? 1 : 0);
        }
        else
        {
            int final_head = -1;
            if (!dmma_cuda_ok(cudaMemcpy(
                                  &final_head, d_row_dynamic_next_row,
                                  sizeof(final_head), cudaMemcpyDeviceToHost),
                              "copy row-dynamic queue final head"))
                goto numeric_failure;
            const int batch = schedule.row_queue_batch;
            const int atomic_claims =
                final_head >= 0 && final_head % batch == 0
                    ? final_head / batch
                    : -1;
            const int expected_claims =
                dmma_row_dynamic_expected_atomic_claims(
                    a.tile_row_count, row_worker_count, batch);
            const int expected_final_head =
                dmma_row_dynamic_expected_final_head(
                    a.tile_row_count, row_worker_count, batch);
            stats->row_dynamic_unique_sms = unique_sms;
            stats->row_dynamic_mapping_valid = mapping_valid;
            stats->row_dynamic_claims = atomic_claims;
            stats->row_dynamic_final_head = final_head;
            stats->row_dynamic_expected_claims = expected_claims;
            stats->row_dynamic_expected_final_head = expected_final_head;
            stats->row_dynamic_claims_valid =
                expected_claims >= 0 && expected_final_head >= 0 &&
                atomic_claims == expected_claims &&
                final_head == expected_final_head;
            std::printf(
                "DMMA_ROW_DYNAMIC layout=row-dynamic "
                "partition=atomic-next-row-batch used=1 ctas=%d "
                "unique_sms=%d mapping_valid=%d batch=%d "
                "atomic_claims=%d expected_atomic_claims=%d "
                "final_head=%d expected_final_head=%d "
                "claims_valid=%d cost_model=0 row_sort=0 heavy_split=0 "
                "reuse=0 last_batch_clipped=1 "
                "atomic_formula=ceil_rows_over_batch_plus_ctas "
                "head_formula=batch_times_atomic_claims "
                "queue_setup_untimed=1 smid_stores_per_cta=1 "
                "mapping_and_claim_audit_core_excluded=1\n",
                row_worker_count, unique_sms,
                stats->row_dynamic_mapping_valid ? 1 : 0, batch,
                atomic_claims, expected_claims, final_head,
                expected_final_head,
                stats->row_dynamic_claims_valid ? 1 : 0);
        }
    }

    /* Admission event queries are diagnostics and therefore occur only after
     * the native-output Core endpoint.  Their recorded filter/count kernels
     * and both count D2H reads remain inside Core. */
    if (admission_events_ready && collect_sparse_tail_records)
    {
        if (!dmma_read_admission_timing(
                split_async, admission_count_event_recorded, stats))
            goto numeric_failure;
    }

    /* Machine-readable diagnostics are emitted only after the Core endpoint;
     * stdio latency must never contribute to a performance sample. */
    if (stats->low_fill_exact_tile_requested)
        std::printf(
            "DMMA_LOW_FILL_EXACT_TILE requested=1 used=%d q=%d "
            "reason=%s metadata_bytes=%zu metadata_budget_bytes=%zu "
            "global_guard=%d symbolic_backend=%s numeric_backend=%s "
            "selected_task_count=post_core_audit_only timed_atomics=0 "
            "per_c_metadata_bytes=0 partial_workspace_bytes=0 "
            "performance_claim=0\n",
            stats->low_fill_exact_tile_used ? 1 : 0,
            stats->low_fill_q,
            dmma_low_fill_exact_tile_reason_name(stats->low_fill_reason),
            stats->low_fill_metadata_bytes,
            DMMA_LOW_FILL_METADATA_BUDGET_BYTES,
            stats->low_fill_global_guard ? 1 : 0,
            stats->low_fill_exact_tile_used ? "mixed-sparse-dense"
                                            : "original",
            stats->low_fill_exact_tile_used ? "mixed-sparse-dmma"
                                            : "original");
    if (schedule.mode != DMMA_SCHEDULE_DIRECT && output_tile_count > 0)
        dmma_print_tail_fusion_diagnostic(
            sparse_tail_records_ready
                ? "two-stage-sparse"
                : (unified_light_active
                       ? "unified-zero-heavy"
                       : (persistent_suffix_requested &&
                           stats->split_streams_used
                       ? "persistent-suffix-zero-heavy"
                       : "direct")),
            schedule, *stats);
    if (oversized_candidate_stream)
        std::printf("DMMA candidate stream=%llu uses bounded fused exact "
                    "symbolic (no candidate array).\n",
                    candidate_count_total);

#ifdef DMMA_ENABLE_TIMELINE_TRACE
    /* Timeline D2H and file I/O are diagnostics, not Core work. */
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

    /* Profiling is a separate replay after the Core stopwatch.  With the
     * default null path this is one host-side branch and does not alter the
     * production numeric kernel, launch shape, registers, or allocations. */
    if (schedule.task_trace_path != nullptr)
    {
        gettimeofday(&begin, nullptr);
        if (!dmma_profile_direct_task_trace(
                a, b, output_tile_count, d_output_rows, d_output_cols,
                d_output_masks, d_output_nnz, d_output_tile_row_ptr,
                d_output_value_cols, d_output_values, schedule, stats))
            goto numeric_failure;
        gettimeofday(&end, nullptr);
        stats->task_trace_ms = dmma_elapsed_ms(begin, end);
        std::printf("DMMA_TASK_TRACE=%s matrix=%s tasks=%d sample_shift=%u "
                    "sample_phase=%u trace_ms=%.3f\n",
                    schedule.task_trace_path, schedule.matrix_name,
                    stats->task_trace_tasks,
                    schedule.task_trace_sample_shift,
                    schedule.task_trace_sample_phase, stats->task_trace_ms);
    }

    if (schedule.collect_task_stats && split_chunk_count > 0 &&
        d_chunk_sm_ids != nullptr && d_tail_records != nullptr)
    {
        gettimeofday(&begin, nullptr);
        std::vector<DmmaChunkDescriptor> descriptors(
            static_cast<std::size_t>(split_chunk_count));
        std::vector<uint32_t> sm_ids(static_cast<std::size_t>(split_chunk_count));
        std::vector<DmmaTailRecord> tail_records(
            static_cast<std::size_t>(heavy_task_count));
        std::vector<int> chunk_offsets(
            static_cast<std::size_t>(heavy_task_count) + 1);
        if (!dmma_cuda_ok(cudaMemcpy(
                              descriptors.data(), d_chunk_descriptors,
                              descriptors.size() * sizeof(descriptors[0]),
                              cudaMemcpyDeviceToHost),
                          "copy chunk descriptors for statistics") ||
            !dmma_cuda_ok(cudaMemcpy(sm_ids.data(), d_chunk_sm_ids,
                                     sm_ids.size() * sizeof(sm_ids[0]),
                                     cudaMemcpyDeviceToHost),
                          "copy chunk SM IDs") ||
            !dmma_cuda_ok(cudaMemcpy(tail_records.data(), d_tail_records,
                                     tail_records.size() *
                                         sizeof(tail_records[0]),
                                     cudaMemcpyDeviceToHost),
                          "copy sparse tail records for statistics") ||
            !dmma_cuda_ok(cudaMemcpy(chunk_offsets.data(), d_chunk_offsets,
                                     chunk_offsets.size() *
                                         sizeof(chunk_offsets[0]),
                                     cudaMemcpyDeviceToHost),
                          "copy chunk offsets"))
            goto numeric_failure;
        int device = 0;
        cudaDeviceProp properties{};
        if (!dmma_cuda_ok(cudaGetDevice(&device),
                          "read statistics CUDA device") ||
            !dmma_cuda_ok(cudaGetDeviceProperties(&properties, device),
                          "read statistics CUDA properties"))
            goto numeric_failure;
        /* This is intentionally a heavy-chunk-only scheduling estimate.  A
         * parent's fixed intercept/output work and every light task are
         * excluded, so these values must not be reported as whole-kernel SM
         * utilization or whole-kernel imbalance. */
        std::vector<double> sm_load(
            static_cast<std::size_t>(properties.multiProcessorCount), 0.0);
        std::vector<double> chunk_weights(
            static_cast<std::size_t>(split_chunk_count), 0.0);
        for (int heavy = 0; heavy < heavy_task_count; ++heavy)
        {
            const int first = chunk_offsets[heavy];
            const int last = chunk_offsets[heavy + 1];
            const int count = last - first;
            const DmmaTailRecord &record = tail_records[heavy];
            const double weight =
                schedule.cost.scan * static_cast<double>(record.scans) +
                schedule.cost.match * static_cast<double>(record.matches);
            if (count > 0)
                for (int chunk = first; chunk < last; ++chunk)
                    chunk_weights[static_cast<std::size_t>(chunk)] =
                        weight / count;
        }
        std::vector<std::pair<int, uint32_t>> task_sm;
        task_sm.reserve(descriptors.size());
        for (std::size_t chunk = 0; chunk < descriptors.size(); ++chunk)
        {
            const int task = descriptors[chunk].task;
            const uint32_t sm = sm_ids[chunk];
            if (sm < sm_load.size())
                sm_load[sm] += chunk_weights[chunk];
            task_sm.emplace_back(task, sm);
        }
        std::sort(task_sm.begin(), task_sm.end());
        long long distinct_total = 0;
        int distinct_tasks = 0;
        for (std::size_t begin_index = 0; begin_index < task_sm.size();)
        {
            std::size_t end_index = begin_index;
            uint32_t previous_sm = UINT_MAX;
            int distinct = 0;
            while (end_index < task_sm.size() &&
                   task_sm[end_index].first == task_sm[begin_index].first)
            {
                const uint32_t sm = task_sm[end_index].second;
                if (sm != previous_sm)
                {
                    ++distinct;
                    previous_sm = sm;
                }
                ++end_index;
            }
            distinct_total += distinct;
            ++distinct_tasks;
            begin_index = end_index;
        }
        stats->average_distinct_sms =
            distinct_tasks > 0
                ? static_cast<double>(distinct_total) / distinct_tasks
                : 0.0;
        double load_sum = 0.0;
        double load_max = 0.0;
        for (double value : sm_load)
        {
            load_sum += value;
            load_max = std::max(load_max, value);
        }
        const double load_mean = sm_load.empty() ? 0.0 : load_sum / sm_load.size();
        double variance = 0.0;
        for (double value : sm_load)
            variance += (value - load_mean) * (value - load_mean);
        if (!sm_load.empty())
            variance /= sm_load.size();
        stats->sm_work_max_over_mean =
            load_mean > 0.0 ? load_max / load_mean : 1.0;
        stats->sm_work_cv =
            load_mean > 0.0 ? std::sqrt(variance) / load_mean : 0.0;
        gettimeofday(&end, nullptr);
        stats->task_stats_ms = dmma_elapsed_ms(begin, end);
    }

    if (schedule.materialize_output)
    {
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
        stats->output_materialized = true;
    }
    cudaFree(d_heavy_flags);
    cudaFree(d_unified_fine_ids);
    cudaFree(d_unified_fine_flags);
    cudaFree(d_unified_coarse_head);
    cudaFree(d_unified_fine_head);
    cudaFree(d_invalid_tail_records);
    cudaFree(d_chunk_offsets);
    cudaFree(d_tail_records);
    cudaFree(d_window_tail_records);
    cudaFree(d_chunk_descriptors);
    cudaFree(d_partial_workspace);
    cudaFree(d_chunk_sm_ids);
    cudaFree(d_row_worker_sm_ids);
    cudaFree(d_row_dynamic_next_row);
    cudaFree(d_row_gate_summary);
    cudaFree(d_cost_sort_keys);
    cudaFree(d_cost_task_order);
    cudaFree(d_cost_queue_head);
    cudaFree(d_persistent_queue_head);
    cudaFree(d_suffix_bulk_head);
    cudaFree(d_suffix_fine_head);
    cudaFree(d_output_tile_row_ptr);
    cudaFree(d_output_value_cols);
    cudaFree(d_output_values);
    cudaFree(d_candidate_masks);
    cudaFree(d_candidate_nnz);
    cudaFree(d_candidate_keep);
    cudaFree(d_exact_forward_flags);
    cudaFree(d_exact_forward_rows);
    cudaFree(d_exact_forward_mask_spas);
    cudaFree(d_exact_forward_summary);
    cudaFree(d_maybe_candidate_ids);
    cudaFree(d_maybe_append_state);
    cudaFree(d_tail_append_state);
    cudaFree(d_symbolic_load_summary);
    cudaFree(d_unified_replay_summary);
    cudaFree(d_output_rows);
    cudaFree(d_output_cols);
    cudaFree(d_output_masks);
    cudaFree(d_output_nnz);
    cudaFree(d_output_row_ptr);
    cudaFree(d_output_numeric_work);
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
    /* Any default-stream metadata kernel or borrowed numeric stream may still
     * reference per-call allocations.  Synchronize before freeing them, but
     * retain a borrowed process-level context for orderly caller cleanup. */
    (void)cudaDeviceSynchronize();
#ifdef DMMA_ENABLE_TIMELINE_TRACE
    cudaFree(timeline.warp_start);
    cudaFree(timeline.warp_end);
    cudaFree(timeline.sm_id);
#endif
    if (!split_context_borrowed)
        dmma_destroy_split_async_state(&split_async, false);
    cudaFree(d_heavy_flags);
    cudaFree(d_unified_fine_ids);
    cudaFree(d_unified_fine_flags);
    cudaFree(d_unified_coarse_head);
    cudaFree(d_unified_fine_head);
    cudaFree(d_invalid_tail_records);
    cudaFree(d_chunk_offsets);
    cudaFree(d_chunk_descriptors);
    cudaFree(d_partial_workspace);
    cudaFree(d_chunk_sm_ids);
    cudaFree(d_row_gate_summary);
    cudaFree(d_persistent_queue_head);
    cudaFree(d_suffix_bulk_head);
    cudaFree(d_suffix_fine_head);
    cudaFree(d_output_tile_row_ptr);
    cudaFree(d_output_value_cols);
    cudaFree(d_output_values);
    goto symbolic_cleanup;
symbolic_failure:
    /* Admission/event failures can leave default-stream filter/count work in
     * flight.  Do not release per-call buffers until all such users retire. */
    (void)cudaDeviceSynchronize();
symbolic_cleanup:
    cudaFree(d_candidate_masks);
    cudaFree(d_candidate_nnz);
    cudaFree(d_candidate_keep);
    cudaFree(d_exact_forward_flags);
    cudaFree(d_exact_forward_rows);
    cudaFree(d_exact_forward_mask_spas);
    cudaFree(d_exact_forward_summary);
    cudaFree(d_maybe_candidate_ids);
    cudaFree(d_maybe_append_state);
    cudaFree(d_tail_records);
    cudaFree(d_window_tail_records);
    cudaFree(d_tail_append_state);
    cudaFree(d_symbolic_load_summary);
    cudaFree(d_unified_replay_summary);
    cudaFree(d_unified_fine_ids);
    cudaFree(d_unified_fine_flags);
    cudaFree(d_unified_coarse_head);
    cudaFree(d_unified_fine_head);
    cudaFree(d_output_rows);
    cudaFree(d_output_cols);
    cudaFree(d_output_masks);
    cudaFree(d_output_nnz);
    cudaFree(d_output_row_ptr);
    cudaFree(d_output_numeric_work);
failure:
    cudaFree(d_row_worker_sm_ids);
    cudaFree(d_row_dynamic_next_row);
    cudaFree(d_row_gate_summary);
    cudaFree(d_cost_sort_keys);
    cudaFree(d_cost_task_order);
    cudaFree(d_cost_queue_head);
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
                                   SMatrix *output, DmmaSpGemmStats *stats,
                                   const DmmaNumericScheduleConfig *schedule_config = nullptr)
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
    const bool ok = dmma_tilespgemm(a.view, b.view, output, stats,
                                    schedule_config);
    destroy_device_tiles(&a);
    destroy_device_tiles(&b);
    return ok;
}

#endif // RTT_SPGEMM_DMMA_SPGEMM_H_
