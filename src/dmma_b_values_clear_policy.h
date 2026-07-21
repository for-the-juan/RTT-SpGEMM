#ifndef RTT_SPGEMM_DMMA_B_VALUES_CLEAR_POLICY_H_
#define RTT_SPGEMM_DMMA_B_VALUES_CLEAR_POLICY_H_

/*
 * Policy-only part of the dynamic-B values update.  This header deliberately
 * has no CUDA dependency so the safety gate can be exhaustively unit tested
 * on the host.  A no-clear update is legal only when every occupied payload
 * slot is overwritten exactly once and the initially-zero dense padding has
 * remained read-only.
 */
enum DmmaBValuesClearPolicy
{
    DMMA_B_VALUES_ALWAYS_CLEAR = 0,
    DMMA_B_VALUES_NOCLEAR = 1,
    DMMA_B_VALUES_SAFE_AUTO = 2
};

enum DmmaBValuesClearReason
{
    DMMA_B_VALUES_CLEAR_POLICY_ALWAYS = 0,
    DMMA_B_VALUES_NOCLEAR_ELIGIBLE = 1,
    DMMA_B_VALUES_AUTO_ELIGIBLE = 2,
    DMMA_B_VALUES_FALLBACK_STRUCTURE_UPDATE = 3,
    DMMA_B_VALUES_FALLBACK_INVALID_STATE = 4,
    DMMA_B_VALUES_FALLBACK_DUPLICATES = 5,
    DMMA_B_VALUES_FALLBACK_NONUNIQUE_MAPPING = 6,
    DMMA_B_VALUES_FALLBACK_UNINITIALIZED_PAYLOAD = 7,
    DMMA_B_VALUES_FALLBACK_NUMERIC_WRITE_CONTRACT = 8,
    DMMA_B_VALUES_FALLBACK_INVALID_POLICY = 9,
    DMMA_B_VALUES_FALLBACK_INCOMPLETE_MAPPING = 10,
    DMMA_B_VALUES_FALLBACK_DENSE_HOLES = 11
};

struct DmmaBValuesClearDecision
{
    bool clear_payload = true;
    bool skip_clear = false;
    bool fallback = false;
    DmmaBValuesClearReason reason = DMMA_B_VALUES_CLEAR_POLICY_ALWAYS;
};

static inline const char *dmma_b_values_clear_policy_name(
    DmmaBValuesClearPolicy policy)
{
    switch (policy)
    {
    case DMMA_B_VALUES_ALWAYS_CLEAR:
        return "always-clear";
    case DMMA_B_VALUES_NOCLEAR:
        return "noclear";
    case DMMA_B_VALUES_SAFE_AUTO:
        return "safe-auto";
    default:
        return "invalid";
    }
}

static inline const char *dmma_b_values_clear_reason_name(
    DmmaBValuesClearReason reason)
{
    switch (reason)
    {
    case DMMA_B_VALUES_CLEAR_POLICY_ALWAYS:
        return "policy-always-clear";
    case DMMA_B_VALUES_NOCLEAR_ELIGIBLE:
        return "eligible-explicit-noclear";
    case DMMA_B_VALUES_AUTO_ELIGIBLE:
        return "eligible-safe-auto";
    case DMMA_B_VALUES_FALLBACK_STRUCTURE_UPDATE:
        return "structure-update-legacy-rebuild";
    case DMMA_B_VALUES_FALLBACK_INVALID_STATE:
        return "invalid-dynamic-b-state";
    case DMMA_B_VALUES_FALLBACK_DUPLICATES:
        return "active-source-duplicates";
    case DMMA_B_VALUES_FALLBACK_NONUNIQUE_MAPPING:
        return "source-to-payload-not-injective";
    case DMMA_B_VALUES_FALLBACK_UNINITIALIZED_PAYLOAD:
        return "payload-not-fully-initialized";
    case DMMA_B_VALUES_FALLBACK_NUMERIC_WRITE_CONTRACT:
        return "numeric-b-not-read-only";
    case DMMA_B_VALUES_FALLBACK_INVALID_POLICY:
        return "invalid-policy";
    case DMMA_B_VALUES_FALLBACK_INCOMPLETE_MAPPING:
        return "active-source-map-incomplete";
    case DMMA_B_VALUES_FALLBACK_DENSE_HOLES:
        return "dense-holes-not-overwritten-each-update";
    default:
        return "unknown";
    }
}

static inline DmmaBValuesClearDecision dmma_choose_b_values_clear(
    DmmaBValuesClearPolicy policy, bool values_only_update,
    bool dynamic_b_valid, bool has_active_duplicates,
    bool active_source_mapping_complete,
    bool active_source_to_payload_injective,
    bool payload_fully_initialized, bool every_payload_slot_overwritten,
    bool numeric_b_read_only)
{
    DmmaBValuesClearDecision decision;
    if (policy == DMMA_B_VALUES_ALWAYS_CLEAR)
        return decision;

    decision.fallback = true;
    if (policy != DMMA_B_VALUES_NOCLEAR &&
        policy != DMMA_B_VALUES_SAFE_AUTO)
    {
        decision.reason = DMMA_B_VALUES_FALLBACK_INVALID_POLICY;
        return decision;
    }
    if (!values_only_update)
    {
        decision.reason = DMMA_B_VALUES_FALLBACK_STRUCTURE_UPDATE;
        return decision;
    }
    if (!dynamic_b_valid)
    {
        decision.reason = DMMA_B_VALUES_FALLBACK_INVALID_STATE;
        return decision;
    }
    if (has_active_duplicates)
    {
        decision.reason = DMMA_B_VALUES_FALLBACK_DUPLICATES;
        return decision;
    }
    if (!active_source_mapping_complete)
    {
        decision.reason = DMMA_B_VALUES_FALLBACK_INCOMPLETE_MAPPING;
        return decision;
    }
    if (!active_source_to_payload_injective)
    {
        decision.reason = DMMA_B_VALUES_FALLBACK_NONUNIQUE_MAPPING;
        return decision;
    }
    if (!payload_fully_initialized)
    {
        decision.reason = DMMA_B_VALUES_FALLBACK_UNINITIALIZED_PAYLOAD;
        return decision;
    }
    if (!every_payload_slot_overwritten)
    {
        decision.reason = DMMA_B_VALUES_FALLBACK_DENSE_HOLES;
        return decision;
    }
    if (!numeric_b_read_only)
    {
        decision.reason = DMMA_B_VALUES_FALLBACK_NUMERIC_WRITE_CONTRACT;
        return decision;
    }

    decision.clear_payload = false;
    decision.skip_clear = true;
    decision.fallback = false;
    decision.reason = policy == DMMA_B_VALUES_NOCLEAR
                          ? DMMA_B_VALUES_NOCLEAR_ELIGIBLE
                          : DMMA_B_VALUES_AUTO_ELIGIBLE;
    return decision;
}

#endif // RTT_SPGEMM_DMMA_B_VALUES_CLEAR_POLICY_H_
