#ifndef RTT_SPGEMM_DMMA_TILE_EARLY_SPLIT_H_
#define RTT_SPGEMM_DMMA_TILE_EARLY_SPLIT_H_

#include "dmma_tile_tail_queue.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

/*
 * Host proof model for the default-off early-split numeric mode.
 *
 * Admission, chunk counts, descriptor storage and partial workspace are
 * intentionally inherited from the already audited exact-symbolic tail
 * planner.  The new contract changes only when the heavy queue is exposed to
 * CUDA: a globally bounded heavy grid is enqueued before the unchanged
 * four-warp light grid on an equal-priority stream.  CUDA decides CTA
 * placement.  No field in this interface denotes, predicts, or constrains an
 * individual SM assignment.
 *
 * The 1/2 cap is an engineering constant selected by a calibration-only
 * offline screen.  It means at most ceil(SM_count/2) persistent heavy CTAs
 * exist globally, not one CTA on each selected SM.  It is therefore a
 * pressure bound, not an SM-affinity or co-residency guarantee.
 */
static constexpr int DMMA_TILE_EARLY_SPLIT_CAP_NUMERATOR = 1;
static constexpr int DMMA_TILE_EARLY_SPLIT_CAP_DENOMINATOR = 2;

struct DmmaTileEarlySplitPolicy
{
    bool enabled = false;
    double admission_threshold = 684947.875;
    double chunk_target = 342473.938;
    int max_chunks = 8;
    double max_heavy_fraction = 0.001;
    int record_capacity = 131072;
    std::size_t workspace_limit_bytes =
        std::size_t(512) * 1024 * 1024;
    bool workspace_reduction = true;
};

enum DmmaTileEarlySplitGateReason
{
    DMMA_TILE_EARLY_SPLIT_DEFAULT_OFF = 0,
    DMMA_TILE_EARLY_SPLIT_ENABLED = 1,
    DMMA_TILE_EARLY_SPLIT_INVALID_TOPOLOGY = 2,
    DMMA_TILE_EARLY_SPLIT_BASE_FALLBACK = 3,
    DMMA_TILE_EARLY_SPLIT_WORKER_CAP_OVERFLOW = 4
};

static inline const char *dmma_tile_early_split_gate_reason_name(
    DmmaTileEarlySplitGateReason reason)
{
    switch (reason)
    {
    case DMMA_TILE_EARLY_SPLIT_ENABLED:
        return "enabled";
    case DMMA_TILE_EARLY_SPLIT_INVALID_TOPOLOGY:
        return "invalid-topology";
    case DMMA_TILE_EARLY_SPLIT_BASE_FALLBACK:
        return "base-fallback";
    case DMMA_TILE_EARLY_SPLIT_WORKER_CAP_OVERFLOW:
        return "worker-cap-overflow";
    default:
        return "default-off";
    }
}

struct DmmaTileEarlySplitPlan
{
    DmmaTileEarlySplitGateReason reason =
        DMMA_TILE_EARLY_SPLIT_DEFAULT_OFF;
    DmmaTileTailQueueGateReason base_reason =
        DMMA_TILE_TAIL_QUEUE_DEFAULT_OFF;
    std::size_t output_tasks = 0;
    std::size_t ordinary_tasks = 0;
    std::size_t heavy_tasks = 0;
    std::size_t chunks = 0;
    int sm_count = 0;
    int global_heavy_worker_block_cap = 0;
    int heavy_worker_blocks = 0;
    int reduction_worker_blocks = 0;
    bool exact_parent_partition = true;
    bool normal_four_warp_light_grid = true;
    bool heavy_enqueued_before_light = true;
    bool streams_have_equal_priority = true;
    bool reduction_waits_only_on_heavy_stream = true;
    bool final_endpoint_joins_light_and_heavy = true;
    bool sm_affinity_claimed = false;
    std::size_t descriptor_bytes = 0;
    std::size_t heavy_flag_bytes = 0;
    std::size_t partial_workspace_bytes = 0;
    std::vector<DmmaTileTailQueueTaskPlan> heavy;
};

static inline int dmma_tile_early_split_global_worker_cap(int sm_count)
{
    if (sm_count <= 0)
        return 0;
    const long long numerator =
        static_cast<long long>(sm_count) *
        DMMA_TILE_EARLY_SPLIT_CAP_NUMERATOR;
    const long long cap =
        (numerator + DMMA_TILE_EARLY_SPLIT_CAP_DENOMINATOR - 1) /
        DMMA_TILE_EARLY_SPLIT_CAP_DENOMINATOR;
    return cap > INT_MAX ? 0 : static_cast<int>(cap);
}

/* Every one of the four warps in every persistent CTA performs exactly one
 * terminal fetch after all queue items have been claimed.  This checked host
 * oracle is shared by the compile-time contract and the production overflow
 * guard; it does not model or require an SM placement. */
static inline bool dmma_tile_early_split_expected_queue_final_head(
    int queue_items, int worker_blocks, int *final_head)
{
    if (queue_items < 0 || worker_blocks <= 0 || final_head == nullptr)
        return false;
    const long long terminal = static_cast<long long>(worker_blocks) * 4ll;
    if (terminal > INT_MAX || queue_items > INT_MAX - terminal)
        return false;
    *final_head = queue_items + static_cast<int>(terminal);
    return true;
}

static inline bool dmma_tile_early_split_exact_parent_partition(
    std::size_t output_tasks,
    const std::vector<DmmaTileTailQueueTaskPlan> &heavy)
{
    std::vector<unsigned char> owner(output_tasks, 0);
    for (const DmmaTileTailQueueTaskPlan &task : heavy)
    {
        if (task.task < 0 ||
            static_cast<std::size_t>(task.task) >= output_tasks ||
            task.chunks < 2 || owner[static_cast<std::size_t>(task.task)] != 0)
            return false;
        owner[static_cast<std::size_t>(task.task)] = 1;
    }
    std::size_t ordinary = 0;
    std::size_t heavy_count = 0;
    for (unsigned char value : owner)
    {
        ordinary += value == 0;
        heavy_count += value == 1;
    }
    return ordinary + heavy_count == output_tasks &&
           heavy_count == heavy.size();
}

static inline DmmaTileEarlySplitPlan dmma_plan_tile_early_split(
    const std::vector<DmmaTileTailQueueExactTask> &tasks,
    const DmmaTileEarlySplitPolicy &policy,
    const DmmaTileTailQueueCostModel &model, int sm_count)
{
    DmmaTileEarlySplitPlan plan;
    plan.output_tasks = tasks.size();
    plan.ordinary_tasks = tasks.size();
    plan.sm_count = sm_count;
    if (!policy.enabled)
        return plan;
    if (sm_count <= 0)
    {
        plan.reason = DMMA_TILE_EARLY_SPLIT_INVALID_TOPOLOGY;
        return plan;
    }

    DmmaTileTailQueuePolicy base;
    base.enabled = true;
    base.admission_threshold = policy.admission_threshold;
    base.chunk_target = policy.chunk_target;
    base.max_chunks = policy.max_chunks;
    base.max_heavy_fraction = policy.max_heavy_fraction;
    base.record_capacity = policy.record_capacity;
    base.workspace_limit_bytes = policy.workspace_limit_bytes;
    base.workspace_reduction = policy.workspace_reduction;
    const DmmaTileTailQueuePlan partition =
        dmma_plan_tile_tail_queue(tasks, base, model);
    plan.base_reason = partition.reason;
    if (partition.reason != DMMA_TILE_TAIL_QUEUE_ENABLED)
    {
        plan.reason = DMMA_TILE_EARLY_SPLIT_BASE_FALLBACK;
        /* Heavy-fraction/workspace gates retain diagnostic estimates in the
         * base oracle.  They are never executable here: this returned plan
         * keeps its initialized all-ordinary partition and copies none of
         * those provisional heavy records. */
        plan.exact_parent_partition = true;
        return plan;
    }

    const int cap = dmma_tile_early_split_global_worker_cap(sm_count);
    if (cap <= 0)
    {
        plan.reason = DMMA_TILE_EARLY_SPLIT_WORKER_CAP_OVERFLOW;
        return plan;
    }
    plan.reason = DMMA_TILE_EARLY_SPLIT_ENABLED;
    plan.ordinary_tasks = partition.ordinary_tasks;
    plan.heavy_tasks = partition.heavy_tasks;
    plan.chunks = partition.chunks;
    plan.global_heavy_worker_block_cap = cap;
    plan.heavy_worker_blocks = static_cast<int>(
        std::min<std::size_t>(partition.chunks,
                              static_cast<std::size_t>(cap)));
    plan.reduction_worker_blocks = policy.workspace_reduction
        ? static_cast<int>(std::min<std::size_t>(
              partition.heavy_tasks, static_cast<std::size_t>(cap)))
        : 0;
    plan.descriptor_bytes = partition.descriptor_bytes;
    plan.heavy_flag_bytes = partition.heavy_flag_bytes;
    plan.partial_workspace_bytes = partition.partial_workspace_bytes;
    plan.heavy = partition.heavy;
    plan.exact_parent_partition =
        plan.ordinary_tasks + plan.heavy_tasks == plan.output_tasks &&
        dmma_tile_early_split_exact_parent_partition(
            plan.output_tasks, plan.heavy);
    if (!plan.exact_parent_partition || plan.heavy_worker_blocks <= 0)
    {
        plan.reason = DMMA_TILE_EARLY_SPLIT_BASE_FALLBACK;
        plan.ordinary_tasks = plan.output_tasks;
        plan.heavy_tasks = 0;
        plan.chunks = 0;
        plan.heavy_worker_blocks = 0;
        plan.reduction_worker_blocks = 0;
        plan.descriptor_bytes = 0;
        plan.heavy_flag_bytes = 0;
        plan.partial_workspace_bytes = 0;
        plan.heavy.clear();
    }
    return plan;
}

#endif // RTT_SPGEMM_DMMA_TILE_EARLY_SPLIT_H_
