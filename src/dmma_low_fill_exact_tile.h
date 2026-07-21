#ifndef RTT_SPGEMM_DMMA_LOW_FILL_EXACT_TILE_H_
#define RTT_SPGEMM_DMMA_LOW_FILL_EXACT_TILE_H_

#include <cstddef>
#include <cstdint>
#include <climits>
#include <limits>

#if defined(__CUDACC__)
#define DMMA_LF_HD __host__ __device__
#define DMMA_LF_FORCEINLINE __forceinline__
#else
#define DMMA_LF_HD
#define DMMA_LF_FORCEINLINE inline
#endif

/*
 * ExactTile-Sparse v1 host/device admission contract.
 *
 * This header contains no CUDA launch and no matrix-name/timing input.  The
 * production integration is deliberately fail closed: default construction
 * is disabled, and every non-ready reason selects the unchanged RTT exact and
 * DMMA numeric kernels for the whole call.
 */

static constexpr std::size_t DMMA_LOW_FILL_METADATA_BUDGET_BYTES =
    std::size_t(64) * 1024 * 1024;

enum DmmaLowFillExactTileReason
{
    DMMA_LOW_FILL_NOT_REQUESTED = 0,
    DMMA_LOW_FILL_ENABLED = 1,
    DMMA_LOW_FILL_INVALID_CONFIGURATION = 2,
    DMMA_LOW_FILL_UNSUPPORTED_SCHEDULE = 3,
    DMMA_LOW_FILL_METADATA_MISSING = 4,
    DMMA_LOW_FILL_METADATA_OVERFLOW = 5,
    DMMA_LOW_FILL_METADATA_BUDGET = 6,
    DMMA_LOW_FILL_GLOBAL_GATE_REJECTED = 7,
    DMMA_LOW_FILL_OVERSIZED_SYMBOLIC = 8
};

static inline const char *dmma_low_fill_exact_tile_reason_name(
    DmmaLowFillExactTileReason reason)
{
    switch (reason)
    {
    case DMMA_LOW_FILL_ENABLED:
        return "enabled";
    case DMMA_LOW_FILL_INVALID_CONFIGURATION:
        return "invalid-configuration";
    case DMMA_LOW_FILL_UNSUPPORTED_SCHEDULE:
        return "unsupported-schedule";
    case DMMA_LOW_FILL_METADATA_MISSING:
        return "metadata-missing";
    case DMMA_LOW_FILL_METADATA_OVERFLOW:
        return "metadata-overflow";
    case DMMA_LOW_FILL_METADATA_BUDGET:
        return "metadata-budget";
    case DMMA_LOW_FILL_GLOBAL_GATE_REJECTED:
        return "global-gate-rejected";
    case DMMA_LOW_FILL_OVERSIZED_SYMBOLIC:
        return "oversized-symbolic";
    default:
        return "not-requested";
    }
}

static DMMA_LF_HD DMMA_LF_FORCEINLINE bool
dmma_low_fill_q_valid(int q)
{
    return q == 4 || q == 8 || q == 12 || q == 16;
}

/* q means fill <= q/64.  A tile stores at most 32 structural entries, so
 * 2*nnz <= q*tiles is the exact integer form with no floating-point gate. */
static DMMA_LF_HD DMMA_LF_FORCEINLINE bool
dmma_low_fill_input_at_most_q(
    unsigned long long structural_nnz, unsigned long long tile_count, int q)
{
    return dmma_low_fill_q_valid(q) && tile_count > 0 &&
           tile_count <= static_cast<unsigned long long>(INT_MAX) &&
           structural_nnz <= 32ull * tile_count &&
           2ull * structural_nnz <=
               static_cast<unsigned long long>(q) * tile_count;
}

static DMMA_LF_HD DMMA_LF_FORCEINLINE bool
dmma_low_fill_global_guard(
    unsigned long long a_structural_nnz,
    unsigned long long a_tile_count,
    unsigned long long b_structural_nnz,
    unsigned long long b_tile_count, int q)
{
    return dmma_low_fill_input_at_most_q(
               a_structural_nnz, a_tile_count, q) ||
           dmma_low_fill_input_at_most_q(
               b_structural_nnz, b_tile_count, q);
}

static DMMA_LF_HD DMMA_LF_FORCEINLINE bool
dmma_low_fill_local_guard(
    std::uint32_t a_row_nnz_sum, std::uint32_t a_row_degree,
    std::uint32_t b_col_nnz_sum, std::uint32_t b_col_degree, int q)
{
    if (!dmma_low_fill_q_valid(q) || a_row_degree == 0 ||
        b_col_degree == 0 ||
        static_cast<unsigned long long>(a_row_nnz_sum) >
            32ull * a_row_degree ||
        static_cast<unsigned long long>(b_col_nnz_sum) >
            32ull * b_col_degree)
        return false;
    return 2ull * a_row_nnz_sum <=
               static_cast<unsigned long long>(q) * a_row_degree ||
           2ull * b_col_nnz_sum <=
               static_cast<unsigned long long>(q) * b_col_degree;
}

static inline bool dmma_low_fill_metadata_bytes(
    int a_tile_rows, int b_tile_cols, std::size_t *bytes)
{
    if (bytes == nullptr || a_tile_rows < 0 || b_tile_cols < 0)
        return false;
    const std::size_t a = static_cast<std::size_t>(a_tile_rows);
    const std::size_t b = static_cast<std::size_t>(b_tile_cols);
    if (a > std::numeric_limits<std::size_t>::max() - b ||
        a + b > std::numeric_limits<std::size_t>::max() /
                    sizeof(std::uint32_t))
        return false;
    *bytes = (a + b) * sizeof(std::uint32_t);
    return true;
}

#undef DMMA_LF_HD
#undef DMMA_LF_FORCEINLINE

#endif
