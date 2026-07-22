#include "common.h"
#include "mmio_highlevel.h"
#include "gpu_dmma_tiles.h"
#include "tile2csr.h"
#include "spgemm_cu.h"
#include "cusparse_core_benchmark.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <utility>
#include <vector>

enum BUpdateMode
{
    B_UPDATE_NONE,
    B_UPDATE_VALUES,
    B_UPDATE_STRUCTURE
};

enum CusparseBenchmarkMode
{
    CUSPARSE_BENCHMARK_OFF,
    CUSPARSE_BENCHMARK_ORIGINAL,
    CUSPARSE_BENCHMARK_SAME_ORDER,
    CUSPARSE_BENCHMARK_BOTH
};

enum CoreTargetMode
{
    CORE_TARGET_BOTH,
    CORE_TARGET_OURS,
    CORE_TARGET_CUSPARSE
};

enum CusparseWorkspacePolicy
{
    CUSPARSE_WORKSPACE_PER_ITERATION,
    CUSPARSE_WORKSPACE_REUSABLE
};

enum OutputExportMode
{
    OUTPUT_EXPORT_EVERY,
    OUTPUT_EXPORT_LAST
};

struct Options
{
    int device = 0;
    int aat = 0;
    int dense_threshold = 24;
    int warmup_iterations = 2;
    int iterations = 10;
    bool tileflex16_symbolic = false;
    BUpdateMode b_update = B_UPDATE_VALUES;
    DmmaBValuesClearPolicy b_values_clear_policy =
        DMMA_B_VALUES_ALWAYS_CLEAR;
    CusparseBenchmarkMode cusparse_benchmark = CUSPARSE_BENCHMARK_BOTH;
    CoreTargetMode core_target = CORE_TARGET_BOTH;
    CusparseWorkspacePolicy cusparse_workspace =
        CUSPARSE_WORKSPACE_PER_ITERATION;
    OutputExportMode output_export = OUTPUT_EXPORT_EVERY;
    DmmaNumericScheduleMode numeric_schedule = DMMA_SCHEDULE_DIRECT;
    bool cost_balanced = false;
    int cost_workers_per_sm = 4;
    DmmaDirectNumericLayout direct_numeric_layout =
        DMMA_DIRECT_NUMERIC_TILE_DYNAMIC;
    int row_queue_batch = 1;
    bool row_queue_batch_explicit = false;
    int row_dynamic_auto = 0;
    double row_dynamic_threshold = 1.10;
    DmmaPartialReductionMode partial_reduction = DMMA_REDUCTION_ATOMIC;
    DmmaLightPolicy light_policy = DMMA_LIGHT_STATIC;
    DmmaSplitLaunchPolicy split_launch_policy =
        DMMA_SPLIT_LAUNCH_HEAVY_PRIORITY;
    DmmaSymbolicAdmissionMode symbolic_admission =
        DMMA_SYMBOLIC_ADMISSION_SEPARATE;
    double split_threshold_ns = 256.0;
    double split_chunk_target_ns = 0.0;
    int split_max_chunks = 8;
    int flat_warps_per_cta = 2;
    int split_workspace_mb = 1024;
    double critical_q_min = 0.75;
    int suffix_workers_per_sm = 0;
    DmmaSuffixAutoBasis suffix_auto_basis =
        DMMA_SUFFIX_AUTO_REGULAR_WORK;
    int suffix_queue_batch = 16;
    int suffix_fine_tail = 8;
    int unified_page_size = 256;
    int unified_workers_per_sm = 0;
    double unified_fine_threshold_ns = 0.0;
    int unified_fine_capacity = 1 << 20;
    int tail_record_capacity = 1 << 20;
    int tail_maybe_capacity = 1 << 20;
    double max_heavy_fraction = 0.01;
    int force_tail_split = 0;
    int benchmark_sequence_index = 0;
    int collect_task_stats = 0;
    int collect_symbolic_load = 0;
    int exact_forward_spa = 0;
    unsigned long long exact_forward_min_row_pairs = 65536ull;
    double exact_forward_min_ratio = 2.0;
    int low_fill_exact_tile = 0;
    int low_fill_q = 0;
    double symbolic_load_quantum_ns = 64.0;
    int symbolic_wave_ctas_per_sm = 4;
    int symbolic_critical_waves = 1;
    bool no_reorder = false;
    const char *row_order_filename = nullptr;
    const char *inner_order_filename = nullptr;
    const char *reorder_name = nullptr;
    const char *a_filename = nullptr;
    const char *b_filename = nullptr;
    const char *dump_prefix = nullptr;
    const char *heatmap_prefix = nullptr;
    const char *task_cost_model_filename = nullptr;
    const char *task_trace_filename = nullptr;
    const char *bgrf_variant = "full";
    int task_trace_sample_shift = 0;
    int task_trace_sample_phase = 0;
    int heatmap_bins = 256;
    bool prepare_only = false;
};

static void print_usage(const char *program)
{
    std::printf(
        "Usage: %s -d <gpu> -aat <0|1> [--b B.mtx] "
        "[--dense-threshold <1..32>] [--warmup-iterations N] "
        "[--iterations N] [--cusparse-benchmark off|original|same-order|both] "
        "[--core-target both|ours|cusparse] "
        "[--cusparse-workspace per-iteration|reusable] "
        "[--output-export every|last] "
        "[--symbolic-mode legacy|tileflex16] "
        "[--numeric-schedule direct|cost-balanced|split-cta|split-persistent|split-flat|tile-tail-queue|tile-early-split] "
        "[--cost-workers-per-sm 1..16] "
        "[--direct-numeric-layout tile-dynamic|tile-static|row-static-block|row-dynamic] "
        "[--row-queue-batch 1|2|4] "
        "[--row-dynamic-auto 0|1 --row-dynamic-threshold FLOAT] "
        "[--partial-reduction atomic|workspace] "
        "[--light-policy static|persistent-suffix|persistent-unified] "
        "[--critical-q-min FLOAT] [--suffix-workers-per-sm N] "
        "[--suffix-auto-basis task|work|regular-work] "
        "[--suffix-queue-batch N] [--suffix-fine-tail N] "
        "[--unified-page-size N] [--unified-workers-per-sm N] "
        "[--unified-fine-threshold-ns FLOAT] "
        "[--unified-fine-capacity N] "
        "[--split-launch-policy light-first|heavy-first|heavy-priority] "
        "[--symbolic-admission separate|fused-exact] "
        "[--task-cost-model FILE] [--split-threshold-ns FLOAT] "
        "[--split-chunk-target-ns FLOAT] "
        "[--flat-warps-per-cta 1|2|4] "
        "[--task-trace FILE --task-trace-sample-shift N "
        "--task-trace-sample-phase N] "
        "[--split-max-chunks N] [--split-workspace-mb N] "
        "[--tail-record-capacity N] [--tail-maybe-capacity N] "
        "[--max-heavy-fraction FLOAT] "
        "[--force-tail-split 0|1] "
        "[--benchmark-sequence-index N] "
        "[--collect-task-stats 0|1] "
        "[--exact-forward-spa 0|1 "
        "--exact-forward-min-row-pairs N "
        "--exact-forward-min-ratio FLOAT] "
        "[--low-fill-exact-tile 0|1 --low-fill-q 4|8|12|16] "
        "[--collect-symbolic-load 0|1 --symbolic-load-quantum-ns FLOAT "
        "--symbolic-wave-ctas-per-sm N --symbolic-critical-waves N] "
        "[--bgrf-variant full|coarse|fine|coarse-row|coarse-inner|unguarded] "
        "[--b-update none|values|structure] "
        "[--b-values-clear always|noclear|safe-auto] [--no-reorder] "
        "[--row-order FILE --inner-order FILE --reorder-name NAME] "
        "[--dump-reorder-prefix P] [--dump-reorder-heatmap P] "
        "[--heatmap-bins N] [--prepare-only] A.mtx\n"
        "  tile-tail-queue requires --direct-numeric-layout tile-dynamic "
        "--light-policy static --symbolic-admission fused-exact\n"
        "  tile-early-split additionally requires --split-launch-policy "
        "heavy-first; its equal-priority heavy queue is globally capped at "
        "ceil(SM_count/2) CTAs\n"
        "  low-fill-exact-tile requires direct + tile-dynamic + separate "
        "symbolic admission and is incompatible with row/split/tail modes\n",
        program);
}

static bool parse_arguments(int argc, char **argv, Options *options)
{
    if (options == nullptr)
        return false;
    *options = Options();
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc)
            options->device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-aat") == 0 && i + 1 < argc)
            options->aat = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--dense-threshold") == 0 &&
                 i + 1 < argc)
            options->dense_threshold = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iterations") == 0 &&
                 i + 1 < argc)
            options->iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--warmup-iterations") == 0 &&
                 i + 1 < argc)
            options->warmup_iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--symbolic-mode") == 0 &&
                 i + 1 < argc)
        {
            const char *mode = argv[++i];
            if (std::strcmp(mode, "legacy") == 0)
                options->tileflex16_symbolic = false;
            else if (std::strcmp(mode, "tileflex16") == 0)
                options->tileflex16_symbolic = true;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--cusparse-benchmark") == 0 &&
                 i + 1 < argc)
        {
            const char *mode = argv[++i];
            if (std::strcmp(mode, "off") == 0)
                options->cusparse_benchmark = CUSPARSE_BENCHMARK_OFF;
            else if (std::strcmp(mode, "original") == 0)
                options->cusparse_benchmark = CUSPARSE_BENCHMARK_ORIGINAL;
            else if (std::strcmp(mode, "same-order") == 0)
                options->cusparse_benchmark = CUSPARSE_BENCHMARK_SAME_ORDER;
            else if (std::strcmp(mode, "both") == 0)
                options->cusparse_benchmark = CUSPARSE_BENCHMARK_BOTH;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--core-target") == 0 &&
                 i + 1 < argc)
        {
            const char *target = argv[++i];
            if (std::strcmp(target, "both") == 0)
                options->core_target = CORE_TARGET_BOTH;
            else if (std::strcmp(target, "ours") == 0)
                options->core_target = CORE_TARGET_OURS;
            else if (std::strcmp(target, "cusparse") == 0)
                options->core_target = CORE_TARGET_CUSPARSE;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--cusparse-workspace") == 0 &&
                 i + 1 < argc)
        {
            const char *policy = argv[++i];
            if (std::strcmp(policy, "per-iteration") == 0)
                options->cusparse_workspace =
                    CUSPARSE_WORKSPACE_PER_ITERATION;
            else if (std::strcmp(policy, "reusable") == 0)
                options->cusparse_workspace = CUSPARSE_WORKSPACE_REUSABLE;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--output-export") == 0 &&
                 i + 1 < argc)
        {
            const char *mode = argv[++i];
            if (std::strcmp(mode, "every") == 0)
                options->output_export = OUTPUT_EXPORT_EVERY;
            else if (std::strcmp(mode, "last") == 0)
                options->output_export = OUTPUT_EXPORT_LAST;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--numeric-schedule") == 0 &&
                 i + 1 < argc)
        {
            const char *mode = argv[++i];
            if (std::strcmp(mode, "direct") == 0)
            {
                options->numeric_schedule = DMMA_SCHEDULE_DIRECT;
                options->cost_balanced = false;
            }
            else if (std::strcmp(mode, "cost-balanced") == 0)
            {
                options->numeric_schedule = DMMA_SCHEDULE_DIRECT;
                options->cost_balanced = true;
            }
            else if (std::strcmp(mode, "split-cta") == 0)
                options->numeric_schedule = DMMA_SCHEDULE_SPLIT_CTA;
            else if (std::strcmp(mode, "split-persistent") == 0)
                options->numeric_schedule = DMMA_SCHEDULE_SPLIT_PERSISTENT;
            else if (std::strcmp(mode, "split-flat") == 0)
                options->numeric_schedule = DMMA_SCHEDULE_SPLIT_FLAT;
            else if (std::strcmp(mode, "tile-tail-queue") == 0)
                options->numeric_schedule =
                    DMMA_SCHEDULE_TILE_TAIL_QUEUE;
            else if (std::strcmp(mode, "tile-early-split") == 0)
                options->numeric_schedule =
                    DMMA_SCHEDULE_TILE_EARLY_SPLIT;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--direct-numeric-layout") == 0 &&
                 i + 1 < argc)
        {
            const char *layout = argv[++i];
            if (std::strcmp(layout, "tile-dynamic") == 0)
                options->direct_numeric_layout =
                    DMMA_DIRECT_NUMERIC_TILE_DYNAMIC;
            else if (std::strcmp(layout, "tile-static") == 0)
                options->direct_numeric_layout =
                    DMMA_DIRECT_NUMERIC_TILE_STATIC;
            else if (std::strcmp(layout, "row-static-block") == 0)
                options->direct_numeric_layout =
                    DMMA_DIRECT_NUMERIC_ROW_STATIC;
            else if (std::strcmp(layout, "row-dynamic") == 0)
                options->direct_numeric_layout =
                    DMMA_DIRECT_NUMERIC_ROW_DYNAMIC;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--cost-workers-per-sm") == 0 &&
                 i + 1 < argc)
            options->cost_workers_per_sm = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--row-queue-batch") == 0 &&
                 i + 1 < argc)
        {
            options->row_queue_batch = std::atoi(argv[++i]);
            options->row_queue_batch_explicit = true;
        }
        else if (std::strcmp(argv[i], "--row-dynamic-auto") == 0 &&
                 i + 1 < argc)
            options->row_dynamic_auto = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--row-dynamic-threshold") == 0 &&
                 i + 1 < argc)
            options->row_dynamic_threshold = std::strtod(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--partial-reduction") == 0 &&
                 i + 1 < argc)
        {
            const char *mode = argv[++i];
            if (std::strcmp(mode, "atomic") == 0)
                options->partial_reduction = DMMA_REDUCTION_ATOMIC;
            else if (std::strcmp(mode, "workspace") == 0)
                options->partial_reduction = DMMA_REDUCTION_WORKSPACE;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--flat-warps-per-cta") == 0 &&
                 i + 1 < argc)
            options->flat_warps_per_cta = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--light-policy") == 0 &&
                 i + 1 < argc)
        {
            const char *policy = argv[++i];
            if (std::strcmp(policy, "static") == 0)
                options->light_policy = DMMA_LIGHT_STATIC;
            else if (std::strcmp(policy, "persistent-suffix") == 0)
                options->light_policy = DMMA_LIGHT_PERSISTENT_SUFFIX;
            else if (std::strcmp(policy, "persistent-unified") == 0)
                options->light_policy = DMMA_LIGHT_PERSISTENT_UNIFIED;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--critical-q-min") == 0 &&
                 i + 1 < argc)
            options->critical_q_min = std::strtod(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--suffix-workers-per-sm") == 0 &&
                 i + 1 < argc)
            options->suffix_workers_per_sm = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--suffix-auto-basis") == 0 &&
                 i + 1 < argc)
        {
            const char *basis = argv[++i];
            if (std::strcmp(basis, "task") == 0)
                options->suffix_auto_basis = DMMA_SUFFIX_AUTO_TASK;
            else if (std::strcmp(basis, "work") == 0)
                options->suffix_auto_basis = DMMA_SUFFIX_AUTO_WORK;
            else if (std::strcmp(basis, "regular-work") == 0)
                options->suffix_auto_basis =
                    DMMA_SUFFIX_AUTO_REGULAR_WORK;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--suffix-queue-batch") == 0 &&
                 i + 1 < argc)
            options->suffix_queue_batch = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--suffix-fine-tail") == 0 &&
                 i + 1 < argc)
            options->suffix_fine_tail = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--unified-page-size") == 0 &&
                 i + 1 < argc)
            options->unified_page_size = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--unified-workers-per-sm") == 0 &&
                 i + 1 < argc)
            options->unified_workers_per_sm = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--unified-fine-threshold-ns") == 0 &&
                 i + 1 < argc)
            options->unified_fine_threshold_ns =
                std::strtod(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--unified-fine-capacity") == 0 &&
                 i + 1 < argc)
            options->unified_fine_capacity = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--split-launch-policy") == 0 &&
                 i + 1 < argc)
        {
            const char *policy = argv[++i];
            if (std::strcmp(policy, "light-first") == 0)
                options->split_launch_policy =
                    DMMA_SPLIT_LAUNCH_LIGHT_FIRST;
            else if (std::strcmp(policy, "heavy-first") == 0)
                options->split_launch_policy =
                    DMMA_SPLIT_LAUNCH_HEAVY_FIRST;
            else if (std::strcmp(policy, "heavy-priority") == 0)
                options->split_launch_policy =
                    DMMA_SPLIT_LAUNCH_HEAVY_PRIORITY;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--task-cost-model") == 0 &&
                 i + 1 < argc)
            options->task_cost_model_filename = argv[++i];
        else if (std::strcmp(argv[i], "--symbolic-admission") == 0 &&
                 i + 1 < argc)
        {
            const char *mode = argv[++i];
            if (std::strcmp(mode, "separate") == 0)
                options->symbolic_admission =
                    DMMA_SYMBOLIC_ADMISSION_SEPARATE;
            else if (std::strcmp(mode, "fused-exact") == 0)
                options->symbolic_admission =
                    DMMA_SYMBOLIC_ADMISSION_FUSED_EXACT;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--task-trace") == 0 &&
                 i + 1 < argc)
            options->task_trace_filename = argv[++i];
        else if (std::strcmp(argv[i], "--task-trace-sample-shift") == 0 &&
                 i + 1 < argc)
            options->task_trace_sample_shift = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--task-trace-sample-phase") == 0 &&
                 i + 1 < argc)
            options->task_trace_sample_phase = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--split-threshold-ns") == 0 &&
                 i + 1 < argc)
            options->split_threshold_ns = std::strtod(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--split-chunk-target-ns") == 0 &&
                 i + 1 < argc)
            options->split_chunk_target_ns = std::strtod(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--split-max-chunks") == 0 &&
                 i + 1 < argc)
            options->split_max_chunks = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--split-workspace-mb") == 0 &&
                 i + 1 < argc)
            options->split_workspace_mb = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--tail-record-capacity") == 0 &&
                 i + 1 < argc)
            options->tail_record_capacity = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--tail-maybe-capacity") == 0 &&
                 i + 1 < argc)
            options->tail_maybe_capacity = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--max-heavy-fraction") == 0 &&
                 i + 1 < argc)
            options->max_heavy_fraction = std::strtod(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--force-tail-split") == 0 &&
                 i + 1 < argc)
            options->force_tail_split = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--benchmark-sequence-index") == 0 &&
                 i + 1 < argc)
            options->benchmark_sequence_index = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--collect-task-stats") == 0 &&
                 i + 1 < argc)
            options->collect_task_stats = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--exact-forward-spa") == 0 &&
                 i + 1 < argc)
            options->exact_forward_spa = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i],
                             "--exact-forward-min-row-pairs") == 0 &&
                 i + 1 < argc)
        {
            const char *text = argv[++i];
            char *end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(text, &end, 10);
            if (errno != 0 || end == text || *end != '\0')
                return false;
            options->exact_forward_min_row_pairs = value;
        }
        else if (std::strcmp(argv[i], "--exact-forward-min-ratio") == 0 &&
                 i + 1 < argc)
            options->exact_forward_min_ratio =
                std::strtod(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--low-fill-exact-tile") == 0 &&
                 i + 1 < argc)
            options->low_fill_exact_tile = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--low-fill-q") == 0 &&
                 i + 1 < argc)
            options->low_fill_q = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--collect-symbolic-load") == 0 &&
                 i + 1 < argc)
            options->collect_symbolic_load = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--symbolic-load-quantum-ns") == 0 &&
                 i + 1 < argc)
            options->symbolic_load_quantum_ns =
                std::strtod(argv[++i], nullptr);
        else if (std::strcmp(argv[i], "--symbolic-wave-ctas-per-sm") == 0 &&
                 i + 1 < argc)
            options->symbolic_wave_ctas_per_sm = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--symbolic-critical-waves") == 0 &&
                 i + 1 < argc)
            options->symbolic_critical_waves = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--bgrf-variant") == 0 &&
                 i + 1 < argc)
            options->bgrf_variant = argv[++i];
        else if (std::strcmp(argv[i], "--b") == 0 && i + 1 < argc)
            options->b_filename = argv[++i];
        else if (std::strcmp(argv[i], "--b-update") == 0 &&
                 i + 1 < argc)
        {
            const char *mode = argv[++i];
            if (std::strcmp(mode, "none") == 0)
                options->b_update = B_UPDATE_NONE;
            else if (std::strcmp(mode, "values") == 0)
                options->b_update = B_UPDATE_VALUES;
            else if (std::strcmp(mode, "structure") == 0)
                options->b_update = B_UPDATE_STRUCTURE;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--b-values-clear") == 0 &&
                 i + 1 < argc)
        {
            const char *policy = argv[++i];
            if (std::strcmp(policy, "always") == 0 ||
                std::strcmp(policy, "always-clear") == 0)
                options->b_values_clear_policy =
                    DMMA_B_VALUES_ALWAYS_CLEAR;
            else if (std::strcmp(policy, "noclear") == 0)
                options->b_values_clear_policy = DMMA_B_VALUES_NOCLEAR;
            else if (std::strcmp(policy, "safe-auto") == 0)
                options->b_values_clear_policy = DMMA_B_VALUES_SAFE_AUTO;
            else
                return false;
        }
        else if (std::strcmp(argv[i], "--no-reorder") == 0)
            options->no_reorder = true;
        else if (std::strcmp(argv[i], "--row-order") == 0 &&
                 i + 1 < argc)
            options->row_order_filename = argv[++i];
        else if (std::strcmp(argv[i], "--inner-order") == 0 &&
                 i + 1 < argc)
            options->inner_order_filename = argv[++i];
        else if (std::strcmp(argv[i], "--reorder-name") == 0 &&
                 i + 1 < argc)
            options->reorder_name = argv[++i];
        else if (std::strcmp(argv[i], "--dump-reorder-prefix") == 0 &&
                 i + 1 < argc)
            options->dump_prefix = argv[++i];
        else if (std::strcmp(argv[i], "--dump-reorder-heatmap") == 0 &&
                 i + 1 < argc)
            options->heatmap_prefix = argv[++i];
        else if (std::strcmp(argv[i], "--heatmap-bins") == 0 &&
                 i + 1 < argc)
            options->heatmap_bins = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--prepare-only") == 0)
            options->prepare_only = true;
        else if (argv[i][0] != '-' && options->a_filename == nullptr)
            options->a_filename = argv[i];
        else
            return false;
    }
    const bool any_external = options->row_order_filename != nullptr ||
                              options->inner_order_filename != nullptr ||
                              options->reorder_name != nullptr;
    const bool complete_external = options->row_order_filename != nullptr &&
                                   options->inner_order_filename != nullptr &&
                                   options->reorder_name != nullptr;
    bool valid_reorder_name = true;
    const bool valid_bgrf_variant =
        std::strcmp(options->bgrf_variant, "full") == 0 ||
        std::strcmp(options->bgrf_variant, "coarse") == 0 ||
        std::strcmp(options->bgrf_variant, "fine") == 0 ||
        std::strcmp(options->bgrf_variant, "coarse-row") == 0 ||
        std::strcmp(options->bgrf_variant, "coarse-inner") == 0 ||
        std::strcmp(options->bgrf_variant, "unguarded") == 0;
    if (options->reorder_name != nullptr)
    {
        const std::size_t length = std::strlen(options->reorder_name);
        valid_reorder_name = length > 0 && length <= 63;
        for (std::size_t index = 0; valid_reorder_name && index < length;
             ++index)
        {
            const char value = options->reorder_name[index];
            valid_reorder_name =
                (value >= 'a' && value <= 'z') ||
                (value >= 'A' && value <= 'Z') ||
                (value >= '0' && value <= '9') || value == '_' ||
                value == '-' || value == '.' || value == '+';
        }
    }
    return options->a_filename != nullptr &&
           (options->aat == 0 || options->aat == 1) &&
           options->dense_threshold >= 1 &&
           options->dense_threshold <= DMMA_INPUT_ELEMS &&
           options->warmup_iterations >= 0 && options->iterations > 0 &&
           options->cost_workers_per_sm >= 1 &&
           options->cost_workers_per_sm <= 16 &&
           std::isfinite(options->split_threshold_ns) &&
           options->split_threshold_ns > 0.0 &&
           std::isfinite(options->split_chunk_target_ns) &&
           options->split_chunk_target_ns >= 0.0 &&
           options->split_max_chunks >= 2 &&
           options->split_max_chunks <= 32 &&
           (options->flat_warps_per_cta == 1 ||
            options->flat_warps_per_cta == 2 ||
            options->flat_warps_per_cta == 4) &&
           dmma_row_queue_batch_valid(options->row_queue_batch) &&
           (!options->row_queue_batch_explicit ||
            options->row_dynamic_auto == 1 ||
            options->direct_numeric_layout ==
                DMMA_DIRECT_NUMERIC_ROW_DYNAMIC) &&
           (options->row_dynamic_auto == 0 ||
            options->row_dynamic_auto == 1) &&
           std::isfinite(options->row_dynamic_threshold) &&
           options->row_dynamic_threshold >= 1.0 &&
           (options->row_dynamic_auto == 0 ||
            options->numeric_schedule == DMMA_SCHEDULE_DIRECT) &&
           options->split_workspace_mb >= 0 &&
           std::isfinite(options->critical_q_min) &&
           options->critical_q_min >= 0.0 &&
           options->critical_q_min <= 1.0 &&
           options->suffix_workers_per_sm >= 0 &&
           options->suffix_workers_per_sm <= 32 &&
           (options->suffix_auto_basis == DMMA_SUFFIX_AUTO_TASK ||
            options->suffix_auto_basis == DMMA_SUFFIX_AUTO_WORK ||
            options->suffix_auto_basis ==
                DMMA_SUFFIX_AUTO_REGULAR_WORK) &&
           options->suffix_queue_batch >= 1 &&
           options->suffix_queue_batch <= 1024 &&
           options->suffix_fine_tail >= 0 &&
           options->suffix_fine_tail <= 1024 &&
           options->unified_page_size >= 1 &&
           options->unified_page_size <= 4096 &&
           options->unified_workers_per_sm >= 0 &&
           options->unified_workers_per_sm <= 32 &&
           std::isfinite(options->unified_fine_threshold_ns) &&
           options->unified_fine_threshold_ns >= 0.0 &&
           options->unified_fine_capacity >= 1 &&
           options->tail_record_capacity > 0 &&
           options->tail_maybe_capacity > 0 &&
           std::isfinite(options->max_heavy_fraction) &&
           options->max_heavy_fraction >= 0.0 &&
           options->max_heavy_fraction <= 1.0 &&
           (options->force_tail_split == 0 ||
            options->force_tail_split == 1) &&
           options->benchmark_sequence_index >= 0 &&
           (options->collect_task_stats == 0 ||
            options->collect_task_stats == 1) &&
           (options->exact_forward_spa == 0 ||
            options->exact_forward_spa == 1) &&
           options->exact_forward_min_row_pairs > 0 &&
           std::isfinite(options->exact_forward_min_ratio) &&
           options->exact_forward_min_ratio >= 1.0 &&
           (options->low_fill_exact_tile == 0 ||
            options->low_fill_exact_tile == 1) &&
           ((options->low_fill_exact_tile == 0 &&
             options->low_fill_q == 0) ||
            (options->low_fill_exact_tile == 1 &&
             dmma_low_fill_q_valid(options->low_fill_q) &&
             options->numeric_schedule == DMMA_SCHEDULE_DIRECT &&
             options->direct_numeric_layout ==
                 DMMA_DIRECT_NUMERIC_TILE_DYNAMIC &&
             options->row_dynamic_auto == 0 &&
             options->exact_forward_spa == 0 &&
             options->symbolic_admission ==
                 DMMA_SYMBOLIC_ADMISSION_SEPARATE &&
             options->collect_symbolic_load == 0)) &&
           (options->exact_forward_spa == 0 ||
            (options->numeric_schedule == DMMA_SCHEDULE_DIRECT &&
             options->row_dynamic_auto == 0 &&
             options->direct_numeric_layout ==
                 DMMA_DIRECT_NUMERIC_TILE_DYNAMIC &&
             options->symbolic_admission ==
                 DMMA_SYMBOLIC_ADMISSION_SEPARATE &&
             options->collect_symbolic_load == 0)) &&
           (options->collect_symbolic_load == 0 ||
            options->collect_symbolic_load == 1) &&
           std::isfinite(options->symbolic_load_quantum_ns) &&
           options->symbolic_load_quantum_ns > 0.0 &&
           options->symbolic_wave_ctas_per_sm >= 1 &&
           options->symbolic_wave_ctas_per_sm <= 32 &&
           options->symbolic_critical_waves >= 1 &&
           options->symbolic_critical_waves <= 64 &&
           (options->direct_numeric_layout ==
                DMMA_DIRECT_NUMERIC_TILE_DYNAMIC ||
            options->numeric_schedule == DMMA_SCHEDULE_DIRECT) &&
           (options->symbolic_admission ==
                DMMA_SYMBOLIC_ADMISSION_SEPARATE ||
            options->numeric_schedule != DMMA_SCHEDULE_DIRECT) &&
           (options->numeric_schedule != DMMA_SCHEDULE_TILE_TAIL_QUEUE ||
            (options->direct_numeric_layout ==
                 DMMA_DIRECT_NUMERIC_TILE_DYNAMIC &&
             options->light_policy == DMMA_LIGHT_STATIC &&
             options->symbolic_admission ==
                 DMMA_SYMBOLIC_ADMISSION_FUSED_EXACT)) &&
           (options->numeric_schedule != DMMA_SCHEDULE_TILE_EARLY_SPLIT ||
            (options->direct_numeric_layout ==
                 DMMA_DIRECT_NUMERIC_TILE_DYNAMIC &&
             options->light_policy == DMMA_LIGHT_STATIC &&
             options->symbolic_admission ==
                 DMMA_SYMBOLIC_ADMISSION_FUSED_EXACT &&
             options->split_launch_policy ==
                 DMMA_SPLIT_LAUNCH_HEAVY_FIRST)) &&
           options->task_trace_sample_shift >= 0 &&
           options->task_trace_sample_shift <= 30 &&
           options->task_trace_sample_phase >= 0 &&
           static_cast<unsigned int>(options->task_trace_sample_phase) <
               (1u << options->task_trace_sample_shift) &&
           (options->task_trace_filename == nullptr ||
            options->numeric_schedule == DMMA_SCHEDULE_DIRECT) &&
           valid_bgrf_variant && options->heatmap_bins >= 16 &&
           options->heatmap_bins <= 1024 &&
           (!any_external || complete_external) &&
           !(options->no_reorder && any_external) && valid_reorder_name &&
           (options->core_target != CORE_TARGET_CUSPARSE ||
            options->cusparse_benchmark == CUSPARSE_BENCHMARK_ORIGINAL ||
            options->cusparse_benchmark == CUSPARSE_BENCHMARK_SAME_ORDER);
}

static double elapsed_ms(const timeval &begin, const timeval &end)
{
    return (end.tv_sec - begin.tv_sec) * 1000.0 +
           (end.tv_usec - begin.tv_usec) / 1000.0;
}

static double elapsed_ms(
    const std::chrono::steady_clock::time_point &begin,
    const std::chrono::steady_clock::time_point &end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static double median(std::vector<double> values)
{
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    return values.size() % 2 != 0
               ? values[middle]
               : (values[middle - 1] + values[middle]) * 0.5;
}

static double minimum(const std::vector<double> &values)
{
    return values.empty()
               ? 0.0
               : *std::min_element(values.begin(), values.end());
}

static double maximum(const std::vector<double> &values)
{
    return values.empty()
               ? 0.0
               : *std::max_element(values.begin(), values.end());
}

static const char *schedule_name(DmmaNumericScheduleMode mode)
{
    switch (mode)
    {
    case DMMA_SCHEDULE_SPLIT_CTA:
        return "split-cta";
    case DMMA_SCHEDULE_SPLIT_PERSISTENT:
        return "split-persistent";
    case DMMA_SCHEDULE_SPLIT_FLAT:
        return "split-flat";
    case DMMA_SCHEDULE_TILE_TAIL_QUEUE:
        return "tile-tail-queue";
    case DMMA_SCHEDULE_TILE_EARLY_SPLIT:
        return "tile-early-split";
    default:
        return "direct";
    }
}

static const char *effective_schedule_name(DmmaNumericScheduleMode mode,
                                           bool cost_balanced)
{
    return cost_balanced ? "cost-balanced" : schedule_name(mode);
}

static const char *core_target_name(CoreTargetMode target)
{
    switch (target)
    {
    case CORE_TARGET_OURS:
        return "ours";
    case CORE_TARGET_CUSPARSE:
        return "cusparse";
    default:
        return "both";
    }
}

static const char *cusparse_benchmark_name(CusparseBenchmarkMode mode)
{
    switch (mode)
    {
    case CUSPARSE_BENCHMARK_ORIGINAL:
        return "original";
    case CUSPARSE_BENCHMARK_SAME_ORDER:
        return "same-order";
    case CUSPARSE_BENCHMARK_BOTH:
        return "both";
    default:
        return "off";
    }
}

static const char *cusparse_workspace_name(CusparseWorkspacePolicy policy)
{
    return policy == CUSPARSE_WORKSPACE_REUSABLE ? "reusable" :
                                                   "per-iteration";
}

static const char *reduction_name(DmmaPartialReductionMode mode)
{
    return mode == DMMA_REDUCTION_WORKSPACE ? "workspace" : "atomic";
}

static const char *launch_policy_name(DmmaSplitLaunchPolicy policy)
{
    switch (policy)
    {
    case DMMA_SPLIT_LAUNCH_LIGHT_FIRST:
        return "light-first";
    case DMMA_SPLIT_LAUNCH_HEAVY_FIRST:
        return "heavy-first";
    default:
        return "heavy-priority";
    }
}

static void print_samples(const std::vector<double> &samples)
{
    std::printf("[");
    for (std::size_t index = 0; index < samples.size(); ++index)
        std::printf("%s%.6f", index == 0 ? "" : ",", samples[index]);
    std::printf("]");
}

static bool parse_finite_nonnegative(const std::string &text,
                                     double *value)
{
    if (value == nullptr || text.empty())
        return false;
    errno = 0;
    char *end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    if (errno != 0 || end == text.c_str() || *end != '\0' ||
        !std::isfinite(parsed) || parsed < 0.0)
        return false;
    *value = parsed;
    return true;
}

static bool load_task_cost_model(const char *filename,
                                 DmmaTaskCostModel *model)
{
    if (model == nullptr)
        return false;
    if (filename == nullptr)
        return true;
    std::ifstream input(filename);
    if (!input)
    {
        std::fprintf(stderr, "Unable to open task cost model: %s\n",
                     filename);
        return false;
    }
    bool have_intercept = false;
    bool have_scan = false;
    bool have_match = false;
    bool have_output = false;
    bool valid_feature_order = false;
    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        const std::size_t separator = line.find('=');
        if (separator == std::string::npos)
            continue;
        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        double coefficient = 0.0;
        if (key == "feature_order")
        {
            valid_feature_order = value == "scan_steps,matches,output_nnz";
        }
        else if (key == "beta0_ns")
        {
            have_intercept = parse_finite_nonnegative(value, &coefficient);
            if (have_intercept)
                model->intercept = coefficient;
        }
        else if (key == "beta_scan_ns_per_step")
        {
            have_scan = parse_finite_nonnegative(value, &coefficient);
            if (have_scan)
                model->scan = coefficient;
        }
        else if (key == "beta_match_ns_per_match")
        {
            have_match = parse_finite_nonnegative(value, &coefficient);
            if (have_match)
                model->match = coefficient;
        }
        else if (key == "beta_output_ns_per_nnz")
        {
            have_output = parse_finite_nonnegative(value, &coefficient);
            if (have_output)
                model->output = coefficient;
        }
    }
    if (!valid_feature_order || !have_intercept || !have_scan ||
        !have_match || !have_output)
    {
        std::fprintf(stderr,
                     "Invalid task cost model %s: expected analyzer's four "
                     "non-negative coefficients and feature_order.\n",
                     filename);
        return false;
    }
    return true;
}

struct HostCsrBuffer
{
    int rows = 0;
    int cols = 0;
    std::vector<int> row_offsets;
    std::vector<int> column_indices;
    std::vector<MAT_VAL_TYPE> values;
    /* Destination CSR position -> entry position in transform_host_csr's
     * source CSR.  This is an untimed setup artifact for online values-only
     * updates of a materialized layout. */
    std::vector<int> destination_to_source;

    SMatrix view()
    {
        SMatrix matrix{};
        matrix.m = rows;
        matrix.n = cols;
        matrix.nnz = static_cast<int>(column_indices.size());
        matrix.rowpointer = row_offsets.data();
        matrix.columnindex = column_indices.data();
        matrix.value = values.data();
        return matrix;
    }
};

/* Build an untimed CSR layout for the same-order cuSPARSE control.  If
 * transpose is true, source (r,c) becomes (mapped-c,mapped-r). */
static bool transform_host_csr(const SMatrix &source, bool transpose,
                               const int *row_old_to_new,
                               const int *col_old_to_new,
                               HostCsrBuffer *output)
{
    if (output == nullptr || source.m < 0 || source.n < 0 || source.nnz < 0 ||
        source.rowpointer == nullptr ||
        (source.nnz > 0 &&
         (source.columnindex == nullptr || source.value == nullptr)))
        return false;
    HostCsrBuffer result;
    result.rows = transpose ? source.n : source.m;
    result.cols = transpose ? source.m : source.n;
    result.row_offsets.assign(static_cast<std::size_t>(result.rows) + 1, 0);
    result.column_indices.resize(source.nnz);
    result.values.resize(source.nnz);
    result.destination_to_source.resize(source.nnz);
    for (int old_row = 0; old_row < source.m; ++old_row)
    {
        for (int entry = source.rowpointer[old_row];
             entry < source.rowpointer[old_row + 1]; ++entry)
        {
            const int old_col = source.columnindex[entry];
            if (old_col < 0 || old_col >= source.n)
                return false;
            const int mapped_row =
                row_old_to_new == nullptr ? old_row
                                          : row_old_to_new[old_row];
            const int mapped_col =
                col_old_to_new == nullptr ? old_col
                                          : col_old_to_new[old_col];
            const int new_row = transpose ? mapped_col : mapped_row;
            if (new_row < 0 || new_row >= result.rows)
                return false;
            ++result.row_offsets[static_cast<std::size_t>(new_row) + 1];
        }
    }
    for (int row = 0; row < result.rows; ++row)
        result.row_offsets[row + 1] += result.row_offsets[row];
    std::vector<int> cursor = result.row_offsets;
    for (int old_row = 0; old_row < source.m; ++old_row)
    {
        for (int entry = source.rowpointer[old_row];
             entry < source.rowpointer[old_row + 1]; ++entry)
        {
            const int old_col = source.columnindex[entry];
            const int mapped_row =
                row_old_to_new == nullptr ? old_row
                                          : row_old_to_new[old_row];
            const int mapped_col =
                col_old_to_new == nullptr ? old_col
                                          : col_old_to_new[old_col];
            const int new_row = transpose ? mapped_col : mapped_row;
            const int new_col = transpose ? mapped_row : mapped_col;
            if (new_col < 0 || new_col >= result.cols)
                return false;
            const int position = cursor[new_row]++;
            result.column_indices[position] = new_col;
            result.values[position] = source.value[entry];
            result.destination_to_source[position] = entry;
        }
    }
    struct RowEntry
    {
        int column;
        MAT_VAL_TYPE value;
        int source_entry;
    };
    std::vector<RowEntry> row_entries;
    for (int row = 0; row < result.rows; ++row)
    {
        const int begin = result.row_offsets[row];
        const int end = result.row_offsets[row + 1];
        row_entries.clear();
        row_entries.reserve(static_cast<std::size_t>(end - begin));
        for (int entry = begin; entry < end; ++entry)
            row_entries.push_back(RowEntry{
                result.column_indices[entry], result.values[entry],
                result.destination_to_source[entry]});
        std::stable_sort(row_entries.begin(), row_entries.end(),
                         [](const RowEntry &lhs, const RowEntry &rhs) {
                             return lhs.column < rhs.column;
                         });
        for (int entry = begin; entry < end; ++entry)
        {
            const RowEntry &row_entry = row_entries[entry - begin];
            result.column_indices[entry] = row_entry.column;
            result.values[entry] = row_entry.value;
            result.destination_to_source[entry] = row_entry.source_entry;
        }
    }
    *output = std::move(result);
    return true;
}

static cusparseStatus_t create_cusparse_input_descriptor(
    const DmmaOwnedDeviceCsr &matrix, cusparseSpMatDescr_t *descriptor)
{
    if (descriptor == nullptr)
        return CUSPARSE_STATUS_INVALID_VALUE;
    return cusparseCreateCsr(
        descriptor, matrix.rows, matrix.cols, matrix.nnz,
        matrix.row_ptr, matrix.col_idx, matrix.values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
}

static void print_cusparse_stage(
    const char *input_name, const char *phase, std::size_t iteration,
    int sequence, double core_ms,
    const rtt_spgemm::cusparse_core::IterationStageStats &stage,
    std::size_t work_estimation_capacity_bytes,
    std::size_t compute_capacity_bytes)
{
    std::printf(
        "CUSPARSE_STAGE input=%s phase=%s iteration=%zu sequence=%d "
        "core_ms=%.6f b_update_submit_wall_ms=%.6f "
        "c_row_allocation_wall_ms=%.6f "
        "work_estimation_query_wall_ms=%.6f "
        "work_scratch_allocation_wall_ms=%.6f "
        "work_estimation_execute_wall_ms=%.6f "
        "compute_query_wall_ms=%.6f "
        "compute_scratch_allocation_wall_ms=%.6f "
        "compute_execute_wall_ms=%.6f c_size_query_wall_ms=%.6f "
        "c_payload_allocation_wall_ms=%.6f copy_submit_wall_ms=%.6f "
        "final_sync_wall_ms=%.6f memory_free_before_bytes=%zu "
        "memory_total_before_bytes=%zu memory_free_after_bytes=%zu "
        "memory_total_after_bytes=%zu "
        "required_work_estimation_bytes=%zu required_compute_bytes=%zu "
        "work_estimation_capacity_bytes=%zu compute_capacity_bytes=%zu "
        "pointer_stable=%d scratch_allocated_in_timed=%d\n",
        input_name, phase, iteration, sequence, core_ms,
        stage.b_update_submit_wall_ms, stage.c_row_allocation_wall_ms,
        stage.work_estimation_query_wall_ms,
        stage.work_scratch_allocation_wall_ms,
        stage.work_estimation_execute_wall_ms,
        stage.compute_query_wall_ms,
        stage.compute_scratch_allocation_wall_ms,
        stage.compute_execute_wall_ms, stage.c_size_query_wall_ms,
        stage.c_payload_allocation_wall_ms, stage.copy_submit_wall_ms,
        stage.final_sync_wall_ms, stage.memory_free_before_bytes,
        stage.memory_total_before_bytes, stage.memory_free_after_bytes,
        stage.memory_total_after_bytes,
        stage.required_work_estimation_bytes,
        stage.required_compute_bytes, work_estimation_capacity_bytes,
        compute_capacity_bytes, stage.scratch_pointer_stable ? 1 : 0,
        stage.scratch_allocated_in_timed ? 1 : 0);
}

static bool run_cusparse_core(const char *input_name,
                              cusparseHandle_t handle,
                              cusparseSpMatDescr_t matrix_a_descriptor,
                              cusparseSpMatDescr_t matrix_b_descriptor,
                              const DmmaOwnedDeviceCsr &matrix_b,
                              const Options &options,
                              int expected_nnz,
                              const char *b_update_mode,
                              const MAT_VAL_TYPE *original_b_values = nullptr,
                              const int *destination_to_source = nullptr)
{
    const char *input_boundary =
        std::strcmp(b_update_mode, "mapped-values") == 0
            ? "b-update"
            : "source-b-csr-ready";
    rtt_spgemm::cusparse_core::BenchmarkConfig config;
    config.warmup_iterations = options.warmup_iterations;
    config.timed_iterations = options.iterations;
    config.reuse_scratch_workspace =
        options.cusparse_workspace == CUSPARSE_WORKSPACE_REUSABLE;
    if (std::strcmp(b_update_mode, "mapped-values") == 0)
    {
        config.b_value_update.active = true;
        config.b_value_update.source_values = original_b_values;
        config.b_value_update.destination_values = matrix_b.values;
        config.b_value_update.destination_to_source = destination_to_source;
        config.b_value_update.count = matrix_b.nnz;
    }
    rtt_spgemm::cusparse_core::BenchmarkResult result;
    const rtt_spgemm::cusparse_core::Status benchmark_status =
        rtt_spgemm::cusparse_core::BenchmarkPrepared(
            handle, matrix_a_descriptor, matrix_b_descriptor, config,
            &result);
    if (!benchmark_status.ok())
    {
        std::fprintf(stderr,
                     "cuSPARSE Core benchmark (%s) failed at %s "
                     "(cuda=%d, cusparse=%d).\n",
                     input_name, benchmark_status.stage,
                     static_cast<int>(benchmark_status.cuda_status),
                     static_cast<int>(benchmark_status.cusparse_status));
        return false;
    }
    if (config.reuse_scratch_workspace &&
        (!result.reusable_workspace_used ||
         result.warmup_stage_samples.size() !=
             result.warmup_samples_ms.size() ||
         result.stage_samples.size() != result.samples_ms.size()))
    {
        std::fprintf(stderr,
                     "cuSPARSE reusable-workspace diagnostics (%s) are "
                     "incomplete.\n",
                     input_name);
        return false;
    }
    if (expected_nnz >= 0 && result.output.nnz != expected_nnz)
    {
        std::fprintf(stderr,
                     "cuSPARSE Core benchmark (%s) nnz mismatch: got %lld, "
                     "expected %d.\n",
                     input_name, static_cast<long long>(result.output.nnz),
                     expected_nnz);
        return false;
    }
    if (config.reuse_scratch_workspace)
        std::printf(
            "CUSPARSE_REUSABLE_WORKSPACE input=%s policy=reusable "
            "preparation=untimed work_estimation_capacity_bytes=%zu "
            "compute_capacity_bytes=%zu prepare_wall_ms=%.6f "
            "memory_free_before_bytes=%zu memory_total_before_bytes=%zu "
            "memory_free_after_bytes=%zu memory_total_after_bytes=%zu "
            "pointer_stable=%d capacity_growth=0 "
            "scratch_allocated_in_timed=%d "
            "output_allocation=per-sample-timed "
            "scratch_lifetime=benchmark-call\n",
            input_name,
            result.reusable_work_estimation_capacity_bytes,
            result.reusable_compute_capacity_bytes,
            result.reusable_workspace_prepare_wall_ms,
            result.reusable_memory_free_before_bytes,
            result.reusable_memory_total_before_bytes,
            result.reusable_memory_free_after_bytes,
            result.reusable_memory_total_after_bytes,
            result.all_scratch_pointers_stable ? 1 : 0,
            result.any_scratch_allocated_in_timed ? 1 : 0);
    for (std::size_t index = 0; index < result.warmup_samples_ms.size();
         ++index)
    {
        std::printf("CORE_BENCH method=cusparse backend=cusparse-%s "
                    "input=%s phase=warmup warmup=1 iteration=%zu "
                    "core_ms=%.6f b_update_ms=%.6f "
                    "b_update_device_ms=%.6f b_update_mode=%s "
                    "output_state=native-output-ready output_format=csr "
                    "clock=continuous-wall monotonic=1 presync=1 sequence=%d\n",
                    input_name, input_name, index + 1,
                    result.warmup_samples_ms[index],
                    result.warmup_b_update_samples_ms[index],
                    result.warmup_b_update_samples_ms[index], b_update_mode,
                    options.benchmark_sequence_index);
        if (config.reuse_scratch_workspace)
            print_cusparse_stage(
                input_name, "warmup", index + 1,
                options.benchmark_sequence_index,
                result.warmup_samples_ms[index],
                result.warmup_stage_samples[index],
                result.reusable_work_estimation_capacity_bytes,
                result.reusable_compute_capacity_bytes);
    }
    for (std::size_t index = 0; index < result.samples_ms.size(); ++index)
    {
        std::printf("CORE_BENCH method=cusparse backend=cusparse-%s "
                    "input=%s phase=timed warmup=0 iteration=%zu "
                    "core_ms=%.6f b_update_ms=%.6f "
                    "b_update_device_ms=%.6f b_update_mode=%s "
                    "output_state=native-output-ready output_format=csr "
                    "clock=continuous-wall monotonic=1 presync=1 sequence=%d\n",
                    input_name, input_name, index + 1,
                    result.samples_ms[index], result.b_update_samples_ms[index],
                    result.b_update_samples_ms[index], b_update_mode,
                    options.benchmark_sequence_index);
        if (config.reuse_scratch_workspace)
            print_cusparse_stage(
                input_name, "timed", index + 1,
                options.benchmark_sequence_index,
                result.samples_ms[index], result.stage_samples[index],
                result.reusable_work_estimation_capacity_bytes,
                result.reusable_compute_capacity_bytes);
    }
    std::printf("CORE_SUMMARY method=cusparse input=%s "
                "boundary=%s+work-estimation+symbolic+numeric+"
                "output-allocation+copy+native-output-ready "
                "clock=continuous-wall monotonic=1 presync=1 "
                "b_update_mode=%s "
                "output_state=native-output-ready "
                "output_format=csr sequence=%d warmup=%d iterations=%d "
                "samples_ms=",
                input_name, input_boundary, b_update_mode,
                options.benchmark_sequence_index,
                options.warmup_iterations, options.iterations);
    print_samples(result.samples_ms);
    std::printf(" min_ms=%.6f median_ms=%.6f max_ms=%.6f nnzC=%lld "
                "work_estimation_workspace_bytes=%zu "
                "compute_workspace_bytes=%zu\n",
                result.min_ms, result.median_ms, result.max_ms,
                static_cast<long long>(result.output.nnz),
                result.final_work_estimation_buffer_bytes,
                result.final_compute_buffer_bytes);
    return true;
}

static bool run_requested_cusparse_benchmarks(
    const Options &options, const SMatrix &host_a, const SMatrix &host_b,
    const DmmaPreparedA &prepared_a,
    const DmmaOwnedDeviceCsr &device_b, int expected_nnz)
{
    const bool general_ab = options.b_filename != nullptr;
    const bool derived_transpose = !general_ab && options.aat != 0;
    const bool run_original =
        options.cusparse_benchmark == CUSPARSE_BENCHMARK_ORIGINAL ||
        options.cusparse_benchmark == CUSPARSE_BENCHMARK_BOTH;
    const bool run_same_order =
        options.cusparse_benchmark == CUSPARSE_BENCHMARK_SAME_ORDER ||
        options.cusparse_benchmark == CUSPARSE_BENCHMARK_BOTH;
    DmmaOwnedDeviceCsr natural_transpose;
    DmmaOwnedDeviceCsr same_left_device;
    DmmaOwnedDeviceCsr same_right_device;
    HostCsrBuffer natural_transpose_host;
    HostCsrBuffer same_left;
    HostCsrBuffer same_right;
    SMatrix natural_transpose_host_view{};
    const SMatrix *original_b_host = general_ab ? &host_b : &host_a;
    const DmmaOwnedDeviceCsr *original_b_device =
        general_ab ? &device_b : &prepared_a.csr;
    int *d_same_right_destination_to_source = nullptr;
    cusparseHandle_t original_handle = nullptr;
    cusparseHandle_t same_order_handle = nullptr;
    cusparseSpMatDescr_t original_a_descriptor = nullptr;
    cusparseSpMatDescr_t original_b_descriptor = nullptr;
    cusparseSpMatDescr_t same_left_descriptor = nullptr;
    cusparseSpMatDescr_t same_right_descriptor = nullptr;
    bool ok = true;

    /* A*A^T has no separately loaded B.  Materialize its natural B=A^T once
     * as an untimed source CSR so both controls start from a device-resident
     * original-B values array.  No per-sample transpose is performed. */
    if (derived_transpose)
    {
        if (!transform_host_csr(host_a, true, nullptr, nullptr,
                                &natural_transpose_host))
        {
            std::fprintf(stderr,
                         "Unable to materialize natural A^T for cuSPARSE.\n");
            ok = false;
        }
        if (ok)
        {
            natural_transpose_host_view = natural_transpose_host.view();
            double h2d_ms = 0.0;
            double validation_ms = 0.0;
            if (!gpu_upload_csr(natural_transpose_host_view,
                                &natural_transpose, &h2d_ms,
                                &validation_ms))
            {
                std::fprintf(stderr,
                             "Unable to upload natural A^T for cuSPARSE.\n");
                ok = false;
            }
            else
            {
                std::printf("CUSPARSE_INPUT input=original-b-transpose "
                            "untimed_h2d_ms=%.6f validation_ms=%.6f "
                            "b_update_mode=none-source-csr-ready\n",
                            h2d_ms, validation_ms);
                original_b_host = &natural_transpose_host_view;
                original_b_device = &natural_transpose;
            }
        }
    }

    if (ok && run_original)
        std::printf("CUSPARSE_INPUT input=original "
                    "input_source=device-resident-original-csr "
                    "b_update_mode=none-source-csr-ready\n");

    /* Materialize and upload every requested layout and mapping before either
     * control enters its Core.  In particular, both mode must not let the
     * original control run while same-order setup is still pending. */
    if (ok && run_same_order)
    {
        if (!transform_host_csr(
                host_a, false, prepared_a.reorder.h_row_old_to_new,
                prepared_a.reorder.h_inner_old_to_new, &same_left))
        {
            std::fprintf(stderr,
                         "Unable to materialize same-order cuSPARSE A.\n");
            ok = false;
        }
        if (ok && !transform_host_csr(
                                      *original_b_host, false,
                                      prepared_a.reorder.h_inner_old_to_new,
                                      nullptr,
                                      &same_right))
        {
            std::fprintf(stderr,
                         "Unable to materialize same-order cuSPARSE B.\n");
            ok = false;
        }
        if (ok)
        {
            SMatrix left_view = same_left.view();
            SMatrix right_view = same_right.view();
            double left_h2d_ms = 0.0;
            double left_validation_ms = 0.0;
            double right_h2d_ms = 0.0;
            double right_validation_ms = 0.0;
            if (!gpu_upload_csr(left_view, &same_left_device, &left_h2d_ms,
                                &left_validation_ms) ||
                !gpu_upload_csr(right_view, &same_right_device, &right_h2d_ms,
                                &right_validation_ms))
            {
                std::fprintf(stderr,
                             "Unable to upload same-order cuSPARSE inputs.\n");
                ok = false;
            }
            else
            {
                double mapping_h2d_ms = 0.0;
                bool mapping_ok =
                    same_right.destination_to_source.size() ==
                        static_cast<std::size_t>(original_b_device->nnz) &&
                    same_right_device.nnz == original_b_device->nnz;
                for (std::size_t entry = 0;
                     mapping_ok &&
                     entry < same_right.destination_to_source.size();
                     ++entry)
                {
                    const int source_entry =
                        same_right.destination_to_source[entry];
                    mapping_ok = source_entry >= 0 &&
                                 source_entry < original_b_device->nnz;
                }
                if (mapping_ok && same_right_device.nnz > 0)
                {
                    const std::chrono::steady_clock::time_point begin =
                        std::chrono::steady_clock::now();
                    const std::size_t mapping_bytes =
                        static_cast<std::size_t>(same_right_device.nnz) *
                        sizeof(int);
                    cudaError_t mapping_status = cudaMalloc(
                        reinterpret_cast<void **>(
                            &d_same_right_destination_to_source),
                        mapping_bytes);
                    if (mapping_status == cudaSuccess)
                        mapping_status = cudaMemcpy(
                            d_same_right_destination_to_source,
                            same_right.destination_to_source.data(),
                            mapping_bytes, cudaMemcpyHostToDevice);
                    mapping_h2d_ms =
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - begin).count();
                    if (mapping_status != cudaSuccess)
                    {
                        std::fprintf(stderr,
                                     "Unable to upload same-order B value "
                                     "mapping: %s.\n",
                                     cudaGetErrorString(mapping_status));
                        mapping_ok = false;
                    }
                }
                if (!mapping_ok)
                {
                    std::fprintf(stderr,
                                 "Invalid same-order B value mapping.\n");
                    ok = false;
                }
                std::printf("CUSPARSE_INPUT input=same-order "
                            "untimed_h2d_ms=%.6f validation_ms=%.6f "
                            "untimed_mapping_h2d_ms=%.6f mapping_bytes=%zu "
                            "b_update_mode=mapped-values\n",
                            left_h2d_ms + right_h2d_ms,
                            left_validation_ms + right_validation_ms,
                            mapping_h2d_ms,
                            same_right.destination_to_source.size() *
                                sizeof(int));
            }
        }
    }

    /* The shared handle and every requested A/B input descriptor are setup
     * artifacts too.  Create all of them before the first control Core, then
     * call BenchmarkPrepared so no input descriptor creation can be ordered
     * after the other control's samples. */
    cusparseStatus_t descriptor_status = CUSPARSE_STATUS_SUCCESS;
    if (ok && run_original)
        descriptor_status = cusparseCreate(&original_handle);
    if (ok && descriptor_status == CUSPARSE_STATUS_SUCCESS && run_original)
        descriptor_status = cusparseSetStream(original_handle, 0);
    if (ok && descriptor_status == CUSPARSE_STATUS_SUCCESS && run_original)
        descriptor_status = create_cusparse_input_descriptor(
            prepared_a.csr, &original_a_descriptor);
    if (ok && descriptor_status == CUSPARSE_STATUS_SUCCESS && run_original)
        descriptor_status = create_cusparse_input_descriptor(
            *original_b_device, &original_b_descriptor);
    if (ok && descriptor_status == CUSPARSE_STATUS_SUCCESS && run_same_order)
        descriptor_status = cusparseCreate(&same_order_handle);
    if (ok && descriptor_status == CUSPARSE_STATUS_SUCCESS && run_same_order)
        descriptor_status = cusparseSetStream(same_order_handle, 0);
    if (ok && descriptor_status == CUSPARSE_STATUS_SUCCESS && run_same_order)
        descriptor_status = create_cusparse_input_descriptor(
            same_left_device, &same_left_descriptor);
    if (ok && descriptor_status == CUSPARSE_STATUS_SUCCESS && run_same_order)
        descriptor_status = create_cusparse_input_descriptor(
            same_right_device, &same_right_descriptor);
    if (ok && descriptor_status != CUSPARSE_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "Unable to prepare all requested cuSPARSE input "
                     "descriptors (cusparse=%d).\n",
                     static_cast<int>(descriptor_status));
        ok = false;
    }

    const bool same_order_first =
        options.cusparse_benchmark == CUSPARSE_BENCHMARK_BOTH &&
        (options.benchmark_sequence_index & 1) != 0;
    const char *control_mode =
        options.cusparse_benchmark == CUSPARSE_BENCHMARK_ORIGINAL
            ? "original"
            : (options.cusparse_benchmark == CUSPARSE_BENCHMARK_SAME_ORDER
                   ? "same-order"
                   : "both");
    const char *first_control =
        run_original && run_same_order
            ? (same_order_first ? "same-order" : "original")
            : (run_original ? "original" : "same-order");
    const char *second_control =
        run_original && run_same_order
            ? (same_order_first ? "original" : "same-order")
            : "none";
    if (ok)
        std::printf("CUSPARSE_CONTROL_ORDER mode=%s sequence=%d parity=%s "
                    "first=%s second=%s inputs_prepared=1 "
                    "descriptors_prepared=1 handles_prepared=1 "
                    "policy=sequence-parity\n",
                    control_mode, options.benchmark_sequence_index,
                    (options.benchmark_sequence_index & 1) != 0 ? "odd" :
                                                                  "even",
                    first_control, second_control);

    const auto run_original_control = [&]() {
        return run_cusparse_core(
            "original", original_handle, original_a_descriptor,
            original_b_descriptor, *original_b_device, options,
            expected_nnz, "none-source-csr-ready");
    };
    const auto run_same_order_control = [&]() {
        return run_cusparse_core(
            "same-order", same_order_handle, same_left_descriptor,
            same_right_descriptor, same_right_device, options, expected_nnz,
            "mapped-values", original_b_device->values,
            d_same_right_destination_to_source);
    };
    if (ok && run_original && run_same_order)
    {
        if (same_order_first)
        {
            ok = run_same_order_control();
            if (ok)
                ok = run_original_control();
        }
        else
        {
            ok = run_original_control();
            if (ok)
                ok = run_same_order_control();
        }
    }
    else if (ok && run_original)
    {
        ok = run_original_control();
    }
    else if (ok && run_same_order)
    {
        ok = run_same_order_control();
    }

    if (same_right_descriptor != nullptr)
        (void)cusparseDestroySpMat(same_right_descriptor);
    if (same_left_descriptor != nullptr)
        (void)cusparseDestroySpMat(same_left_descriptor);
    if (original_b_descriptor != nullptr)
        (void)cusparseDestroySpMat(original_b_descriptor);
    if (original_a_descriptor != nullptr)
        (void)cusparseDestroySpMat(original_a_descriptor);
    if (same_order_handle != nullptr)
        (void)cusparseDestroy(same_order_handle);
    if (original_handle != nullptr)
        (void)cusparseDestroy(original_handle);
    cudaFree(d_same_right_destination_to_source);
    destroy_device_csr(&same_right_device);
    destroy_device_csr(&same_left_device);
    destroy_device_csr(&natural_transpose);
    return ok;
}

static void free_host_csr(SMatrix *matrix)
{
    if (matrix == nullptr)
        return;
    std::free(matrix->rowpointer);
    std::free(matrix->columnindex);
    std::free(matrix->value);
    matrix->rowpointer = nullptr;
    matrix->columnindex = nullptr;
    matrix->value = nullptr;
}

static void destroy_output_matrix(SMatrix *matrix)
{
    if (matrix == nullptr)
        return;
    std::free(matrix->tile_ptr);
    std::free(matrix->tile_columnidx);
    std::free(matrix->tile_rowidx);
    std::free(matrix->tile_nnz);
    std::free(matrix->tile_csr_Value);
    std::free(matrix->tile_csr_Col);
    std::free(matrix->tile_csr_Ptr);
    std::free(matrix->mask);
    std::free(matrix->csc_tile_ptr);
    std::free(matrix->csc_tile_rowidx);
    std::free(matrix->rowpointer);
    std::free(matrix->columnindex);
    std::free(matrix->value);
    std::memset(matrix, 0, sizeof(*matrix));
}

static bool load_matrix(const char *filename, SMatrix *matrix,
                        double *load_ms)
{
    timeval begin{}, end{};
    gettimeofday(&begin, nullptr);
    const int status = mmio_allinone(
        &matrix->m, &matrix->n, &matrix->nnz, &matrix->isSymmetric,
        &matrix->rowpointer, &matrix->columnindex, &matrix->value,
        const_cast<char *>(filename));
    gettimeofday(&end, nullptr);
    if (load_ms != nullptr)
        *load_ms = elapsed_ms(begin, end);
    if (status != 0)
    {
        std::fprintf(stderr, "Unable to read %s (error %d).\n", filename,
                     status);
        return false;
    }
    for (int i = 0; i < matrix->nnz; ++i)
        matrix->value[i] = static_cast<MAT_VAL_TYPE>(i % 10);
    return true;
}

static int heatmap_bin(int index, int extent, int bins)
{
    if (extent <= 0)
        return 0;
    const std::uint64_t scaled =
        static_cast<std::uint64_t>(index) * static_cast<std::uint64_t>(bins);
    return std::min(bins - 1, static_cast<int>(scaled / extent));
}

/*
 * Export a compact, exact scalar-NNZ density summary of A before and after
 * the unified row/K permutation.  This scans the already expanded host CSR
 * once, so symmetric Matrix Market inputs are visualized exactly as consumed
 * by the GPU without materializing another full reordered matrix.
 */
static bool dump_reorder_heatmap(const SMatrix &matrix,
                                 const DmmaReorderPlan &plan,
                                 const char *prefix, int bins)
{
    if (prefix == nullptr || bins < 1 || matrix.m != plan.rows ||
        matrix.n != plan.cols || matrix.nnz != plan.nnz ||
        matrix.rowpointer == nullptr ||
        (matrix.nnz > 0 && matrix.columnindex == nullptr))
        return false;

    const std::size_t cells =
        static_cast<std::size_t>(bins) * static_cast<std::size_t>(bins);
    std::vector<std::uint64_t> original(cells, 0);
    std::vector<std::uint64_t> reordered(cells, 0);
    const int *row_map = plan.h_row_old_to_new;
    const int *inner_map = plan.h_inner_old_to_new;

    for (int old_row = 0; old_row < matrix.m; ++old_row)
    {
        const int new_row = row_map == nullptr ? old_row : row_map[old_row];
        const int old_bin_row = heatmap_bin(old_row, matrix.m, bins);
        const int new_bin_row = heatmap_bin(new_row, matrix.m, bins);
        for (int entry = matrix.rowpointer[old_row];
             entry < matrix.rowpointer[old_row + 1]; ++entry)
        {
            const int old_col = matrix.columnindex[entry];
            const int new_col = inner_map == nullptr ? old_col
                                                     : inner_map[old_col];
            const int old_bin_col = heatmap_bin(old_col, matrix.n, bins);
            const int new_bin_col = heatmap_bin(new_col, matrix.n, bins);
            ++original[static_cast<std::size_t>(old_bin_row) * bins +
                       old_bin_col];
            ++reordered[static_cast<std::size_t>(new_bin_row) * bins +
                        new_bin_col];
        }
    }

    const std::string stem(prefix);
    const std::string heatmap_path = stem + "_heatmap.csv";
    const std::string metadata_path = stem + "_heatmap_meta.csv";
    FILE *file = std::fopen(heatmap_path.c_str(), "w");
    if (file == nullptr)
        return false;
    std::fprintf(file, "bin_row,bin_col,original_nnz,reordered_nnz\n");
    for (int row = 0; row < bins; ++row)
        for (int col = 0; col < bins; ++col)
        {
            const std::size_t cell =
                static_cast<std::size_t>(row) * bins + col;
            std::fprintf(file, "%d,%d,%llu,%llu\n", row, col,
                         static_cast<unsigned long long>(original[cell]),
                         static_cast<unsigned long long>(reordered[cell]));
        }
    if (std::fclose(file) != 0)
        return false;

    file = std::fopen(metadata_path.c_str(), "w");
    if (file == nullptr)
        return false;
    std::uint64_t moved_rows = 0;
    std::uint64_t moved_inner = 0;
    std::uint64_t row_displacement = 0;
    std::uint64_t inner_displacement = 0;
    for (int old_row = 0; old_row < matrix.m; ++old_row)
    {
        const int new_row = row_map == nullptr ? old_row : row_map[old_row];
        moved_rows += new_row != old_row;
        row_displacement += static_cast<std::uint64_t>(
            new_row >= old_row ? new_row - old_row : old_row - new_row);
    }
    for (int old_col = 0; old_col < matrix.n; ++old_col)
    {
        const int new_col = inner_map == nullptr ? old_col
                                                 : inner_map[old_col];
        moved_inner += new_col != old_col;
        inner_displacement += static_cast<std::uint64_t>(
            new_col >= old_col ? new_col - old_col : old_col - new_col);
    }
    std::fprintf(file, "key,value\n");
    std::fprintf(file, "rows,%d\ncols,%d\nnnz,%d\nbins,%d\n", matrix.m,
                 matrix.n, matrix.nnz, bins);
    std::fprintf(file,
                 "algorithm,%s\nsweeps,%d\nrow_window,%d\n"
                 "inner_window,%d\nactive_rows,%d\nactive_inner,%d\n",
                 plan.kind == DMMA_REORDER_IDENTITY
                     ? "identity_baseline"
                     : plan.algorithm,
                 plan.sweeps, plan.row_window, plan.inner_window,
                 plan.active_rows, plan.active_inner);
    std::fprintf(file, "moved_rows,%llu\nmoved_inner,%llu\n"
                       "row_displacement_sum,%llu\n"
                       "inner_displacement_sum,%llu\n",
                 static_cast<unsigned long long>(moved_rows),
                 static_cast<unsigned long long>(moved_inner),
                 static_cast<unsigned long long>(row_displacement),
                 static_cast<unsigned long long>(inner_displacement));
    std::fprintf(file,
                 "unified_tiles,%lld\nunified_active_row_tiles,%d\n"
                 "unified_active_k_tiles,%d\nunified_sparse_tiles,%lld\n"
                 "unified_dense_tiles,%lld\nunified_payload,%llu\n"
                 "accepted_row_windows,%llu\n"
                 "accepted_inner_windows,%llu\n"
                 "exact_row_tile_reduction,%llu\n"
                 "exact_inner_tile_reduction,%llu\n"
                 "row_fanout_before,%llu\nrow_fanout_after,%llu\n"
                 "inner_fanout_before,%llu\ninner_fanout_after,%llu\n"
                 "coarse_components,%d\ncoarse_levels,%d\n"
                 "coarse_level_budget,%d\n"
                 "coarse_candidate_accepted,%d\n"
                 "coarse_tile_reduction,%llu\n"
                 "coarse_ms,%.6f\nfine_ms,%.6f\n"
                 "reorder_peak_workspace_bytes,%zu\n",
                 plan.num_tiles, plan.active_row_tiles,
                 plan.active_k_tiles, plan.sparse_tiles,
                 plan.dense_tiles, plan.payload,
                 plan.accepted_row_windows, plan.accepted_inner_windows,
                 plan.row_tile_reduction, plan.inner_tile_reduction,
                 plan.row_fanout_before, plan.row_fanout_after,
                 plan.inner_fanout_before, plan.inner_fanout_after,
                 plan.coarse_components, plan.coarse_levels,
                 plan.coarse_level_budget,
                 plan.coarse_candidate_accepted ? 1 : 0,
                 plan.coarse_tile_reduction,
                 plan.coarse_ms, plan.fine_ms,
                 plan.reorder_peak_workspace_bytes);
    return std::fclose(file) == 0;
}

static void print_device_tile_stats(const char *name,
                                    const DmmaDeviceTiles &matrix)
{
    const double payload_mb =
        static_cast<double>(matrix.payload_size) * sizeof(MAT_VAL_TYPE) /
        (1024.0 * 1024.0);
    double metadata_bytes =
        static_cast<double>(matrix.num_tiles + 1) * sizeof(MAT_PTR_TYPE) +
        static_cast<double>(matrix.num_tiles) *
            (sizeof(int) + sizeof(uint32_t)) +
        static_cast<double>(matrix.tile_row_count + 1) * sizeof(MAT_PTR_TYPE);
    if (matrix.tile_col_ptr != nullptr)
        metadata_bytes +=
            static_cast<double>(matrix.tile_col_count + 1) *
                sizeof(MAT_PTR_TYPE) +
            static_cast<double>(matrix.num_tiles) * (sizeof(int) + sizeof(int));
    std::printf(
        "%s GPU tiles: total=%d dense=%d bitmask=%d payload=%.2f MB "
        "metadata=%.2f MB\n",
        name, matrix.num_tiles, matrix.dense_tiles, matrix.sparse_tiles,
        payload_mb, metadata_bytes / (1024.0 * 1024.0));
}

static void print_a_stats(const DmmaPreparedA &prepared,
                          const DmmaOfflineAStats &stats)
{
    const DmmaReorderPlan &plan = prepared.reorder;
    std::printf("A CSR H2D = %.3f ms; validation = %.3f ms\n",
                stats.h2d_ms, stats.validation_ms);
    std::printf("A reorder = %.3f ms; key sort/reduce = %.3f ms; "
                "tile build = %.3f ms; C16 view = %.3f ms\n",
                stats.reorder_ms, stats.key_sort_reduce_ms,
                stats.tile_build_ms, stats.super16_build_ms);
    std::printf("A offline ready-on-device = %.3f ms; "
                "peak workspace = %.2f MB\n",
                stats.total_ms,
                static_cast<double>(stats.peak_workspace_bytes) /
                    (1024.0 * 1024.0));
    std::printf("A reorder algorithm=%s; active rows=%d/%d; "
                "active inner=%d/%d\n",
                plan.algorithm,
                plan.active_rows, plan.rows, plan.active_inner, plan.cols);
    std::printf("A permutation: sweeps=%d row-window=%d inner-window=%d "
                "moved-rows=%llu moved-inner=%llu\n",
                plan.sweeps, plan.row_window, plan.inner_window,
                plan.moved_rows, plan.moved_inner);
    std::printf("A balanced fine windows: row-accepted=%llu "
                "inner-accepted=%llu row-tile-reduction=%llu "
                "inner-tile-reduction=%llu\n",
                plan.accepted_row_windows, plan.accepted_inner_windows,
                plan.row_tile_reduction, plan.inner_tile_reduction);
    std::printf("A BGRF stages: coarse=%.3f ms fine=%.3f ms "
                "components=%d levels=%d/%d reorder-peak=%.2f MB\n",
                plan.coarse_ms, plan.fine_ms, plan.coarse_components,
                plan.coarse_levels, plan.coarse_level_budget,
                static_cast<double>(plan.reorder_peak_workspace_bytes) /
                    (1024.0 * 1024.0));
    std::printf("A BGRF joint coarse: accepted=%d "
                "A-tile-reduction=%llu\n",
                plan.coarse_candidate_accepted ? 1 : 0,
                plan.coarse_tile_reduction);
    std::printf("A fanout/span proxy: row=%llu->%llu inner=%llu->%llu\n",
                plan.row_fanout_before, plan.row_fanout_after,
                plan.inner_fanout_before, plan.inner_fanout_after);
    std::printf("A %s layout: tiles=%lld active-row-tiles=%d "
                "active-k-tiles=%d sparse=%lld dense=%lld payload=%llu\n",
                plan.kind == DMMA_REORDER_IDENTITY
                    ? "identity"
                    : plan.algorithm,
                plan.num_tiles,
                plan.active_row_tiles, plan.active_k_tiles,
                plan.sparse_tiles, plan.dense_tiles, plan.payload);
    print_device_tile_stats("A", prepared.tiles.view);
}

static void print_b_update_stats(const char *label,
                                 const DmmaBUpdateStats &stats)
{
    if (stats.structure_rebuilt)
    {
        std::printf(
            "%s B structure: total=%.3f ms validation=%.3f "
            "sort/reduce=%.3f tile-build=%.3f CSC=%.3f mapping=%.3f "
            "low-fill-metadata=%.3f C16-view=%.3f; "
            "entries source=%d active=%d unique=%d; peak=%.2f MB\n",
            label, stats.total_ms, stats.validation_ms,
            stats.key_sort_reduce_ms, stats.tile_build_ms, stats.csc_ms,
            stats.mapping_ms, stats.low_fill_metadata_ms,
            stats.super16_build_ms,
            stats.source_entries, stats.active_entries, stats.unique_entries,
            static_cast<double>(stats.peak_workspace_bytes) /
                (1024.0 * 1024.0));
    }
    else
    {
        std::printf("%s B values: total=%.3f ms value-update=%.3f ms; "
                    "source entries=%d\n",
                    label, stats.total_ms, stats.value_update_ms,
                    stats.source_entries);
    }
}

static const char *b_update_mode_name(BUpdateMode mode)
{
    if (mode == B_UPDATE_NONE)
        return "none";
    return mode == B_UPDATE_VALUES ? "values" : "structure";
}

static void print_b_values_clear_marker(
    const char *phase, int iteration, const Options &options,
    const DmmaDynamicB &matrix, const DmmaBValuesClearDecision &decision)
{
    const bool values_only = options.b_update == B_UPDATE_VALUES;
    const bool no_update = options.b_update == B_UPDATE_NONE;
    const std::size_t payload_bytes =
        static_cast<std::size_t>(matrix.tiles.view.payload_size) *
        sizeof(MAT_VAL_TYPE);
    std::printf(
        "B_VALUES_CLEAR phase=%s iteration=%d requested=%s "
        "effective=%s values_only=%d clear_executed=%d "
        "clear_skipped=%d fallback=%d reason=%s payload_bytes=%zu "
        "active_entries=%d active_duplicates=%d "
        "active_mapping_complete=%d active_mapping_injective=%d "
        "payload_initialized=%d dense_tiles=%d "
        "every_payload_slot_overwritten=%d numeric_b_read_only=1 "
        "source_aliases_a=%d structure_legacy_rebuild=%d\n",
        phase, iteration,
        dmma_b_values_clear_policy_name(options.b_values_clear_policy),
        no_update ? "none-static-AA"
                  : (values_only
                         ? (decision.skip_clear ? "noclear" : "clear")
                         : "structure-rebuild"),
        values_only ? 1 : 0,
        values_only && decision.clear_payload && payload_bytes > 0 ? 1 : 0,
        values_only && decision.skip_clear ? 1 : 0,
        decision.fallback ? 1 : 0,
        dmma_b_values_clear_reason_name(decision.reason), payload_bytes,
        matrix.active_entries, matrix.has_duplicates ? 1 : 0,
        matrix.active_source_mapping_complete ? 1 : 0,
        matrix.active_source_to_payload_injective ? 1 : 0,
        matrix.payload_fully_initialized ? 1 : 0,
        matrix.tiles.view.dense_tiles,
        matrix.tiles.view.dense_tiles == 0 ? 1 : 0,
        options.b_filename == nullptr ? 1 : 0,
        options.b_update == B_UPDATE_STRUCTURE ? 1 : 0);
}

static bool rebuild_b(const Options &options, const DmmaPreparedA &a,
                      const DmmaOwnedDeviceCsr &independent_b,
                      DmmaDynamicB *b, DmmaBUpdateStats *stats)
{
    const int *inner_old_to_new = a.reorder.d_inner_old_to_new;
    const int active_inner = a.reorder.active_inner;
    bool rebuilt = false;
    if (options.b_filename != nullptr)
        rebuilt = gpu_rebuild_dynamic_b(
            device_csr_view(independent_b), inner_old_to_new, active_inner,
            options.dense_threshold, b, stats);
    else if (options.aat != 0)
        rebuilt = gpu_rebuild_dynamic_b_transpose(
            device_csr_view(a.csr), inner_old_to_new, active_inner,
            options.dense_threshold, b, stats);
    else
        rebuilt = gpu_rebuild_dynamic_b(
            device_csr_view(a.csr), inner_old_to_new, active_inner,
            options.dense_threshold, b, stats);
    if (rebuilt && options.low_fill_exact_tile != 0)
    {
        /* A failed optional sum build is a whole-component fallback, not a B
         * rebuild failure.  For structure updates this synchronized work is
         * intentionally inside the existing B-update/Core boundary. */
        const std::chrono::steady_clock::time_point metadata_begin =
            std::chrono::steady_clock::now();
        gpu_prepare_low_fill_exact_tile_metadata(&b->tiles, false, true);
        stats->low_fill_metadata_ms = elapsed_ms(
            metadata_begin, std::chrono::steady_clock::now());
        stats->total_ms += stats->low_fill_metadata_ms;
    }
    if (rebuilt)
    {
        const auto begin = std::chrono::steady_clock::now();
        b->super16.reset();
        rebuilt = rtt::super16::build_device_index(
            b->tiles.view, rtt::super16::OperandRole::B4x8, nullptr,
            &b->super16);
        stats->super16_build_ms = elapsed_ms(
            begin, std::chrono::steady_clock::now());
        stats->total_ms += stats->super16_build_ms;
    }
    return rebuilt;
}

int main(int argc, char **argv)
{
    Options options;
    if (!parse_arguments(argc, argv, &options))
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    DmmaNumericScheduleConfig schedule_config;
    schedule_config.tileflex16_symbolic = options.tileflex16_symbolic;
    schedule_config.cost_balanced = options.cost_balanced;
    schedule_config.cost_workers_per_sm = options.cost_workers_per_sm;
    DmmaReorderConfig reorder_config;
    if (!parse_dmma_reorder_variant(options.bgrf_variant,
                                    &reorder_config.variant))
        return EXIT_FAILURE;
    schedule_config.mode = options.numeric_schedule;
    schedule_config.direct_numeric_layout = options.direct_numeric_layout;
    schedule_config.row_queue_batch = options.row_queue_batch;
    schedule_config.row_dynamic_auto = options.row_dynamic_auto != 0;
    schedule_config.row_dynamic_threshold = options.row_dynamic_threshold;
    schedule_config.reduction = options.partial_reduction;
    schedule_config.light_policy = options.light_policy;
    schedule_config.launch_policy = options.split_launch_policy;
    schedule_config.symbolic_admission = options.symbolic_admission;
    schedule_config.split_threshold = options.split_threshold_ns;
    schedule_config.chunk_target = options.split_chunk_target_ns;
    schedule_config.max_chunks = options.split_max_chunks;
    schedule_config.flat_warps_per_cta = options.flat_warps_per_cta;
    schedule_config.critical_q_min = options.critical_q_min;
    schedule_config.suffix_workers_per_sm = options.suffix_workers_per_sm;
    schedule_config.suffix_auto_basis = options.suffix_auto_basis;
    schedule_config.suffix_queue_batch = options.suffix_queue_batch;
    schedule_config.suffix_fine_tasks_per_worker =
        options.suffix_fine_tail;
    schedule_config.unified_page_size = options.unified_page_size;
    schedule_config.unified_workers_per_sm =
        options.unified_workers_per_sm;
    schedule_config.unified_fine_threshold =
        options.unified_fine_threshold_ns;
    schedule_config.unified_fine_capacity = options.unified_fine_capacity;
    schedule_config.workspace_limit_bytes =
        static_cast<std::size_t>(options.split_workspace_mb) *
        static_cast<std::size_t>(1024) * static_cast<std::size_t>(1024);
    schedule_config.tail_record_capacity = options.tail_record_capacity;
    schedule_config.maybe_candidate_capacity = options.tail_maybe_capacity;
    schedule_config.max_heavy_fraction = options.max_heavy_fraction;
    schedule_config.force_tail_split = options.force_tail_split != 0;
    schedule_config.collect_task_stats = options.collect_task_stats != 0;
    schedule_config.exact_forward_spa = options.exact_forward_spa != 0;
    schedule_config.exact_forward_min_row_pairs =
        options.exact_forward_min_row_pairs;
    schedule_config.exact_forward_min_ratio =
        options.exact_forward_min_ratio;
    schedule_config.low_fill_exact_tile =
        options.low_fill_exact_tile != 0;
    schedule_config.low_fill_q = options.low_fill_q;
    schedule_config.collect_symbolic_load =
        options.collect_symbolic_load != 0;
    schedule_config.symbolic_load_quantum =
        options.symbolic_load_quantum_ns;
    schedule_config.symbolic_wave_ctas_per_sm =
        options.symbolic_wave_ctas_per_sm;
    schedule_config.symbolic_critical_waves =
        options.symbolic_critical_waves;
    schedule_config.task_trace_sample_shift =
        static_cast<unsigned int>(options.task_trace_sample_shift);
    schedule_config.task_trace_sample_phase =
        static_cast<unsigned int>(options.task_trace_sample_phase);
    const char *matrix_label = std::strrchr(options.a_filename, '/');
    schedule_config.matrix_name =
        matrix_label == nullptr ? options.a_filename : matrix_label + 1;
    if (!load_task_cost_model(options.task_cost_model_filename,
                              &schedule_config.cost))
        return EXIT_FAILURE;

    if (!dmma_cuda_ok(cudaSetDevice(options.device), "select GPU"))
        return EXIT_FAILURE;
    cudaDeviceProp properties{};
    if (!dmma_cuda_ok(
            cudaGetDeviceProperties(&properties, options.device),
            "query GPU"))
        return EXIT_FAILURE;
    if (properties.major < 8)
    {
        std::fprintf(stderr,
                     "FP64 DMMA requires compute capability 8.0 or newer; "
                     "device %d is %d.%d.\n",
                     options.device, properties.major, properties.minor);
        return EXIT_FAILURE;
    }
    const size_t persisting_l2 =
        static_cast<size_t>(properties.l2CacheSize * 0.8) <
                properties.persistingL2CacheMaxSize
            ? static_cast<size_t>(properties.l2CacheSize * 0.8)
            : properties.persistingL2CacheMaxSize;
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persisting_l2);

    SMatrix host_a{};
    SMatrix host_b{};
    double a_load_ms = 0.0;
    double b_load_ms = 0.0;
    if (!load_matrix(options.a_filename, &host_a, &a_load_ms))
        return EXIT_FAILURE;
    if (options.b_filename != nullptr &&
        !load_matrix(options.b_filename, &host_b, &b_load_ms))
    {
        free_host_csr(&host_a);
        return EXIT_FAILURE;
    }

    const bool general_ab = options.b_filename != nullptr;
    const bool external_reorder = options.row_order_filename != nullptr;
    if ((general_ab && host_a.n != host_b.m) ||
        (!general_ab && options.aat == 0 && host_a.m != host_a.n))
    {
        if (general_ab)
            std::fprintf(stderr,
                         "A*B requires A.n == B.m, got %d and %d.\n",
                         host_a.n, host_b.m);
        else
            std::fprintf(stderr,
                         "AA requires a square A; use -aat 1 for A*A^T.\n");
        free_host_csr(&host_b);
        free_host_csr(&host_a);
        return EXIT_FAILURE;
    }

    std::printf("---------------------------------------------------------------\n");
    std::printf("Device [ %d ] %s @ %.2f MHz, compute capability %d.%d\n",
                options.device, properties.name,
                properties.clockRate * 1e-3, properties.major,
                properties.minor);
    std::printf("A: %s, shape=(%d,%d), nnz=%d, load=%.5f sec\n",
                options.a_filename, host_a.m, host_a.n, host_a.nnz,
                a_load_ms / 1000.0);
    if (general_ab)
        std::printf("B: %s, shape=(%d,%d), nnz=%d, load=%.5f sec\n",
                    options.b_filename, host_b.m, host_b.n, host_b.nnz,
                    b_load_ms / 1000.0);
    const std::string reorder_algorithm =
        options.no_reorder
            ? "identity-baseline"
            : (external_reorder
                   ? std::string("external:") + options.reorder_name
                   : std::string("bgrf-v1-") + options.bgrf_variant);
    std::printf("mode=%s, reorder=%s, B-update=%s, warmup=%d, "
                "iterations=%d, sequence=%d\n",
                general_ab ? "AB" : (options.aat ? "AAT" : "AA"),
                reorder_algorithm.c_str(),
                b_update_mode_name(options.b_update),
                options.warmup_iterations, options.iterations,
                options.benchmark_sequence_index);
    std::printf("DMMA tiles: A=8x4, B=4x8, C=8x8; "
                "dense threshold=%d/32\n",
                options.dense_threshold);
    std::printf(
        "B_VALUES_CLEAR_CONFIG requested=%s default=always-clear "
        "default_off=1 legacy_entrypoint_unchanged=1 "
        "eligibility=values-only+valid+no-active-duplicates+"
        "complete-injective-active-map+initialized-payload+sparse-only+"
        "read-only-numeric "
        "unsafe_action=clear-fallback timing_boundary=unchanged\n",
        dmma_b_values_clear_policy_name(options.b_values_clear_policy));
    std::printf("NUMERIC_SCHEDULE mode=%s direct_layout=%s "
                "row_queue_batch=%d "
                "row_dynamic_auto=%d row_dynamic_threshold=%.9f "
                "row_gate_version=exact-row-ptr-v1 "
                "reduction=%s launch_policy=%s "
                "light_policy=%s critical_q_min=%.9f "
                "suffix_workers_per_sm=%d suffix_auto_basis=%s "
                "suffix_queue_batch=%d "
                "suffix_fine_tail=%d "
                "unified_page_size=%d unified_workers_per_sm=%d "
                "unified_fine_threshold=%.6f "
                "unified_fine_capacity=%d "
                "threshold=%.6f chunk_target=%.6f "
                "flat_warps_per_cta=%d "
                "admission=two-stage-upper admission_path=%s "
                "admission_precision=fp32 max_chunks=%d workspace_mb=%d "
                "tail_capacity=%d maybe_capacity=%d "
                "max_heavy_fraction=%.9f force_tail_split=%d model=%s "
                "symbolic_load_explicit=%d symbolic_load_auto=%d "
                "symbolic_load_effective=%d "
                "unified_sparse_replay_auto=%d load_quantum=%.6f "
                "wave_ctas_per_sm=%d critical_waves=%d "
                "beta0=%.9g beta_scan=%.9g beta_match=%.9g "
                "beta_output=%.9g\n",
                effective_schedule_name(schedule_config.mode,
                                        schedule_config.cost_balanced),
                dmma_direct_numeric_layout_name(
                    schedule_config.direct_numeric_layout),
                schedule_config.row_queue_batch,
                schedule_config.row_dynamic_auto ? 1 : 0,
                schedule_config.row_dynamic_threshold,
                reduction_name(schedule_config.reduction),
                launch_policy_name(schedule_config.launch_policy),
                dmma_light_policy_name(schedule_config.light_policy),
                schedule_config.critical_q_min,
                schedule_config.suffix_workers_per_sm,
                dmma_suffix_auto_basis_name(
                    schedule_config.suffix_auto_basis),
                schedule_config.suffix_queue_batch,
                schedule_config.suffix_fine_tasks_per_worker,
                schedule_config.unified_page_size,
                schedule_config.unified_workers_per_sm,
                schedule_config.unified_fine_threshold > 0.0
                    ? schedule_config.unified_fine_threshold
                    : schedule_config.split_threshold * 0.5,
                schedule_config.unified_fine_capacity,
                schedule_config.split_threshold,
                schedule_config.chunk_target > 0.0
                    ? schedule_config.chunk_target
                    : schedule_config.split_threshold,
                schedule_config.flat_warps_per_cta,
                dmma_symbolic_admission_name(
                    schedule_config.symbolic_admission),
                schedule_config.max_chunks,
                options.split_workspace_mb,
                schedule_config.tail_record_capacity,
                schedule_config.maybe_candidate_capacity,
                schedule_config.max_heavy_fraction,
                schedule_config.force_tail_split ? 1 : 0,
                options.task_cost_model_filename == nullptr
                    ? "builtin"
                    : options.task_cost_model_filename,
                schedule_config.collect_symbolic_load ? 1 : 0,
                schedule_config.mode != DMMA_SCHEDULE_DIRECT &&
                        schedule_config.light_policy ==
                            DMMA_LIGHT_PERSISTENT_SUFFIX &&
                        schedule_config.suffix_workers_per_sm == 0 &&
                        schedule_config.suffix_auto_basis !=
                            DMMA_SUFFIX_AUTO_TASK
                    ? 1
                    : 0,
                schedule_config.collect_symbolic_load ||
                        (schedule_config.mode != DMMA_SCHEDULE_DIRECT &&
                         schedule_config.light_policy ==
                             DMMA_LIGHT_PERSISTENT_SUFFIX &&
                         schedule_config.suffix_workers_per_sm == 0 &&
                         schedule_config.suffix_auto_basis !=
                             DMMA_SUFFIX_AUTO_TASK)
                    ? 1
                    : 0,
                schedule_config.mode != DMMA_SCHEDULE_DIRECT &&
                        schedule_config.light_policy ==
                            DMMA_LIGHT_PERSISTENT_UNIFIED &&
                        !schedule_config.collect_symbolic_load
                    ? 1
                    : 0,
                schedule_config.symbolic_load_quantum,
                schedule_config.symbolic_wave_ctas_per_sm,
                schedule_config.symbolic_critical_waves,
                schedule_config.cost.intercept, schedule_config.cost.scan,
                schedule_config.cost.match, schedule_config.cost.output);
    std::printf(
        "TILE_TAIL_QUEUE_CONFIG requested=%d default_off=1 "
        "ordinary_kernel=four-warp-tile-dynamic "
        "ordinary_task=one-C-tile-per-warp heavy_source=exact-symbolic "
        "loose_maybe_counter=disabled bulk_first=1 "
        "bulk_chunk_overlap=0 queue=persistent-global-atomic "
        "row_worker_reused=0 reduction=%s "
        "gate_fallback=direct-tile-dynamic "
        "workspace_limit_fallback=atomic "
        "allocation_error=fail-closed\n",
        schedule_config.mode == DMMA_SCHEDULE_TILE_TAIL_QUEUE ? 1 : 0,
        reduction_name(schedule_config.reduction));
    std::printf(
        "TILE_EARLY_SPLIT_CONFIG requested=%d default_off=1 "
        "ordinary_kernel=four-warp-tile-dynamic "
        "ordinary_task=one-C-tile-per-warp heavy_source=exact-symbolic "
        "loose_maybe_counter=disabled heavy_parent_skipped_by_light=1 "
        "heavy_enqueue_before_light=1 bulk_chunk_overlap=1 "
        "stream_priority=equal queue=persistent-global-atomic "
        "global_heavy_worker_cap=ceil(sm_count*%d/%d) "
        "sm_affinity_claimed=0 row_worker_reused=0 reduction=%s "
        "reduction_dependency=heavy-stream-only "
        "core_endpoint=join-light-and-heavy-reduction "
        "gate_fallback=direct-tile-dynamic "
        "workspace_limit_fallback=atomic allocation_error=fail-closed\n",
        schedule_config.mode == DMMA_SCHEDULE_TILE_EARLY_SPLIT ? 1 : 0,
        DMMA_TILE_EARLY_SPLIT_CAP_NUMERATOR,
        DMMA_TILE_EARLY_SPLIT_CAP_DENOMINATOR,
        reduction_name(schedule_config.reduction));
    std::printf(
        "DMMA_EXACT_FORWARD_CONFIG requested=%d default_off=1 "
        "gate_source=pre-candidate-structure min_row_pairs=%llu "
        "min_reverse_over_forward=%.9f scratch_limit_bytes=%zu "
        "scratch_hard_cap=1\n",
        schedule_config.exact_forward_spa ? 1 : 0,
        schedule_config.exact_forward_min_row_pairs,
        schedule_config.exact_forward_min_ratio,
        DMMA_EXACT_FORWARD_SPA_SCRATCH_BYTES);
    std::printf("TASK_STATS enabled=%d core_timing_excludes_d2h=1\n",
                schedule_config.collect_task_stats ? 1 : 0);
    std::printf("CORE_COMPARABILITY b_update=%s comparable=%d "
                "reason=%s\n",
                b_update_mode_name(options.b_update),
                options.b_update != B_UPDATE_STRUCTURE ? 1 : 0,
                options.b_update == B_UPDATE_NONE
                    ? "static-AA-no-update"
                    : (options.b_update == B_UPDATE_VALUES
                           ? "matched-online-values-contract"
                           : "cusparse-controls-do-not-rebuild-structure"));
    const bool core_target_runs_ours =
        options.core_target != CORE_TARGET_CUSPARSE;
    const bool core_target_runs_cusparse =
        options.core_target != CORE_TARGET_OURS &&
        options.cusparse_benchmark != CUSPARSE_BENCHMARK_OFF;
    const char *effective_cusparse =
        options.core_target == CORE_TARGET_OURS
            ? "off"
            : cusparse_benchmark_name(options.cusparse_benchmark);
#if CHECK_RESULT
    const int compiled_check_result = 1;
#else
    const int compiled_check_result = 0;
#endif
    std::printf(
        "CORE_TARGET mode=%s process_backend=%s requested_cusparse=%s "
        "effective_cusparse=%s cusparse_workspace=%s "
        "ours_core=%d cusparse_core=%d "
        "process_backend_count=%d cuda_context_shared_between_methods=%d "
        "setup_core_excluded=1 compiled_check_result=%d "
        "legacy_validation=%d correctness_validation=out-of-band "
        "warmup=%d iterations=%d selection=min\n",
        core_target_name(options.core_target),
        options.core_target == CORE_TARGET_CUSPARSE
            ? "cusparse"
            : (options.core_target == CORE_TARGET_OURS ? "ours" : "both"),
        cusparse_benchmark_name(options.cusparse_benchmark),
        effective_cusparse,
        cusparse_workspace_name(options.cusparse_workspace),
        core_target_runs_ours ? 1 : 0,
        core_target_runs_cusparse ? 1 : 0,
        (core_target_runs_ours ? 1 : 0) +
            (core_target_runs_cusparse ? 1 : 0),
        core_target_runs_ours && core_target_runs_cusparse ? 1 : 0,
        compiled_check_result,
        options.core_target == CORE_TARGET_CUSPARSE
            ? 0
            : compiled_check_result,
        options.warmup_iterations, options.iterations);
    std::printf("OUTPUT_EXPORT policy=%s warmup=never core_excluded=1\n",
                options.output_export == OUTPUT_EXPORT_LAST ? "last" :
                                                              "every");
    if (options.task_trace_filename != nullptr)
        std::printf("TASK_TRACE path=%s sample_shift=%d sample_phase=%d "
                    "profiled_timed_iteration=1 core_excluded=1\n",
                    options.task_trace_filename,
                    options.task_trace_sample_shift,
                    options.task_trace_sample_phase);
    if (general_ab && options.aat != 0)
        std::printf("Note: --b selects general A*B; -aat is ignored.\n");

    const bool cusparse_runs_first =
        options.core_target == CORE_TARGET_BOTH &&
        core_target_runs_cusparse &&
        (options.benchmark_sequence_index & 1) != 0;
    std::printf("CORE_ORDER scope=process sequence=%d first=%s second=%s\n",
                options.benchmark_sequence_index,
                options.core_target == CORE_TARGET_CUSPARSE
                    ? "cusparse-controls"
                    : (cusparse_runs_first ? "cusparse-controls" : "ours"),
                options.core_target != CORE_TARGET_BOTH ||
                        !core_target_runs_cusparse
                    ? "none"
                    : (cusparse_runs_first ? "ours" : "cusparse-controls"));

    DmmaPreparedA prepared_a;
    DmmaOfflineAStats a_stats;
    DmmaOwnedDeviceCsr device_b;
    DmmaDynamicB dynamic_b;
    DmmaSplitAsyncState split_context;
    bool split_context_ready = false;
    double split_context_create_ms_untimed = 0.0;
    SMatrix matrix_c{};
    int status = EXIT_FAILURE;
    unsigned long long nnz_cub = 0;
    double nnz_cub_ms = 0.0;
    std::vector<double> b_update_times;
    std::vector<double> dmma_times;
    std::vector<double> combined_times;
    std::vector<double> single_shot_all_cost_times;
    std::vector<double> export_times;
    std::vector<double> restore_times;
    DmmaPartialReductionMode summary_effective_reduction =
        schedule_config.reduction;
    bool summary_workspace_fallback = false;
    bool summary_context_reused = false;
    bool summary_row_static_mapping_valid = true;
    int summary_row_static_ctas = 0;
    int summary_row_static_min_unique_sms = 0;
    bool summary_row_dynamic_mapping_valid = true;
    bool summary_row_dynamic_claims_valid = true;
    int summary_row_dynamic_ctas = 0;
    int summary_row_dynamic_min_unique_sms = 0;
    DmmaDirectNumericLayout summary_direct_numeric_layout =
        schedule_config.direct_numeric_layout;
    bool summary_direct_numeric_layout_initialized = false;
    int b_values_clear_timed_samples = 0;
    int b_values_noclear_timed_samples = 0;
    int b_values_clear_fallback_timed_samples = 0;

    bool a_ready = false;
    if (options.no_reorder)
        a_ready = gpu_prepare_identity_a(
            host_a, options.dense_threshold, &prepared_a, &a_stats);
    else if (external_reorder)
        a_ready = gpu_prepare_external_a(
            host_a, options.row_order_filename,
            options.inner_order_filename, options.reorder_name,
            options.dense_threshold, &prepared_a, &a_stats);
    else
        a_ready = gpu_prepare_reordered_a(
            host_a, options.dense_threshold, &prepared_a, &a_stats, 0,
            &reorder_config);
    if (!a_ready)
    {
        std::fprintf(stderr, "GPU A preparation failed.\n");
        goto cleanup;
    }
    if (options.tileflex16_symbolic)
    {
        const auto begin = std::chrono::steady_clock::now();
        if (!rtt::super16::build_device_index(
                prepared_a.tiles.view,
                rtt::super16::OperandRole::A8x4, nullptr,
                &prepared_a.super16))
        {
            std::fprintf(stderr,
                         "GPU A C16-view construction failed.\n");
            goto cleanup;
        }
        a_stats.super16_build_ms = elapsed_ms(
            begin, std::chrono::steady_clock::now());
        a_stats.total_ms += a_stats.super16_build_ms;
    }
    if (options.low_fill_exact_tile != 0)
    {
        const std::chrono::steady_clock::time_point metadata_begin =
            std::chrono::steady_clock::now();
        const bool metadata_ok =
            gpu_prepare_low_fill_exact_tile_metadata(
                &prepared_a.tiles, true, false);
        const double metadata_ms = elapsed_ms(
            metadata_begin, std::chrono::steady_clock::now());
        std::printf(
            "DMMA_LOW_FILL_METADATA operand=A phase=offline requested=1 "
            "ready=%d entries=%d bytes=%zu build_ms=%.6f "
            "core_included=0 overflow=%d performance_claim=0\n",
            metadata_ok ? 1 : 0, prepared_a.tiles.view.tile_row_count,
            static_cast<std::size_t>(
                prepared_a.tiles.view.tile_row_count) * sizeof(uint32_t),
            metadata_ms,
            prepared_a.tiles.view.low_fill_metadata_overflow ? 1 : 0);
    }
    print_a_stats(prepared_a, a_stats);

    if (options.dump_prefix != nullptr)
    {
        if (options.no_reorder)
        {
            std::printf("Reorder dump skipped with --no-reorder.\n");
        }
        else if (!dump_dmma_reorder_plan(
                     prepared_a.reorder, options.dump_prefix,
                     prepared_a.csr.row_ptr, prepared_a.csr.col_idx, true))
        {
            std::fprintf(stderr, "Unable to dump reorder plan with prefix %s.\n",
                         options.dump_prefix);
            goto cleanup;
        }
        else
        {
            std::printf("Reorder diagnostics dumped with prefix %s.\n",
                        options.dump_prefix);
        }
    }

    if (options.heatmap_prefix != nullptr)
    {
        if (!dump_reorder_heatmap(host_a, prepared_a.reorder,
                                  options.heatmap_prefix,
                                  options.heatmap_bins))
        {
            std::fprintf(stderr,
                         "Unable to dump reorder heatmap with prefix %s.\n",
                         options.heatmap_prefix);
            goto cleanup;
        }
        std::printf("Reorder heatmap (%dx%d) dumped with prefix %s.\n",
                    options.heatmap_bins, options.heatmap_bins,
                    options.heatmap_prefix);
    }

    if (options.prepare_only)
    {
        status = EXIT_SUCCESS;
        goto cleanup;
    }

    if (general_ab)
    {
        double b_h2d_ms = 0.0;
        double b_validation_ms = 0.0;
        if (!gpu_upload_csr(host_b, &device_b, &b_h2d_ms,
                            &b_validation_ms))
        {
            std::fprintf(stderr, "GPU B upload failed.\n");
            goto cleanup;
        }
        std::printf("B CSR H2D = %.3f ms; validation = %.3f ms\n",
                    b_h2d_ms, b_validation_ms);
    }
    else
    {
        std::printf("B source reuses A device CSR; additional CSR H2D = 0 ms\n");
    }

    /* Isolated cuSPARSE mode exits before any RTT dynamic-B construction,
     * output allocation, Core sample, or legacy validation.  The selected
     * control performs its same-order input/mapping/descriptor setup inside
     * run_requested_cusparse_benchmarks(), before BenchmarkPrepared starts
     * the 2+10 Core timing protocol. */
    if (options.core_target == CORE_TARGET_CUSPARSE)
    {
        const bool cusparse_ok = run_requested_cusparse_benchmarks(
            options, host_a, host_b, prepared_a, device_b, -1);
        status = cusparse_ok ? EXIT_SUCCESS : EXIT_FAILURE;
        std::printf(
            "CORE_TARGET_EXIT mode=cusparse status=%s "
            "ours_dynamic_b_prepared=0 ours_core_samples=0 "
            "ours_output_allocated=0 legacy_validation_ran=0\n",
            cusparse_ok ? "SUCCESS" : "FAILED");
        goto cleanup;
    }

    {
        DmmaBUpdateStats initial_b_stats;
        if (!rebuild_b(options, prepared_a, device_b, &dynamic_b,
                       &initial_b_stats))
        {
            std::fprintf(stderr, "Initial GPU B rebuild failed.\n");
            goto cleanup;
        }
        print_b_update_stats("Initial", initial_b_stats);
        print_device_tile_stats("B", dynamic_b.tiles.view);
        if (options.low_fill_exact_tile != 0)
            std::printf(
                "DMMA_LOW_FILL_METADATA operand=B phase=initial requested=1 "
                "ready=%d entries=%d bytes=%zu build_in_b_rebuild=1 "
                "initial_core_sample=0 overflow=%d performance_claim=0\n",
                dynamic_b.tiles.view.col_tile_nnz_sum_valid ? 1 : 0,
                dynamic_b.tiles.view.tile_col_count,
                static_cast<std::size_t>(
                    dynamic_b.tiles.view.tile_col_count) * sizeof(uint32_t),
                dynamic_b.tiles.view.low_fill_metadata_overflow ? 1 : 0);
        std::printf(
            "B_VALUES_CLEAR_INVARIANTS active_entries=%d "
            "active_duplicates=%d active_mapping_complete=%d "
            "active_mapping_injective=%d "
            "payload_initialized_by_pack=%d dense_tiles=%d "
            "every_payload_slot_overwritten_each_update=%d "
            "numeric_b_read_only=1 source_aliases_a=%d "
            "dense_padding_initialized_zero=1 sparse_payload_padding=0\n",
            dynamic_b.active_entries,
            dynamic_b.has_duplicates ? 1 : 0,
            dynamic_b.active_source_mapping_complete ? 1 : 0,
            dynamic_b.active_source_to_payload_injective ? 1 : 0,
            dynamic_b.payload_fully_initialized ? 1 : 0,
            dynamic_b.tiles.view.dense_tiles,
            dynamic_b.tiles.view.dense_tiles == 0 ? 1 : 0,
            general_ab ? 0 : 1);
    }

    if (options.tileflex16_symbolic)
    {
        schedule_config.super16_a = prepared_a.super16.view();
        schedule_config.super16_b = dynamic_b.super16.view();
    }

    if (options.b_update == B_UPDATE_VALUES)
    {
        release_dynamic_b_rebuild_workspace(&dynamic_b);
        release_dmma_reorder_device_maps(&prepared_a.reorder);
    }

    if (general_ab)
    {
        if (!gpu_compute_nnz_cub_ab(
                device_csr_view(prepared_a.csr), device_csr_view(device_b),
                &nnz_cub, &nnz_cub_ms))
        {
            std::fprintf(stderr, "GPU AB nnzCub failed.\n");
            goto cleanup;
        }
    }
    else if (!gpu_compute_nnz_cub_derived(
                 prepared_a.csr, options.aat != 0, &nnz_cub,
                 &nnz_cub_ms))
    {
        std::fprintf(stderr, "GPU derived nnzCub failed.\n");
        goto cleanup;
    }
    std::printf("SpGEMM nnzCub = %llu; GPU nnzCub time = %.3f ms\n",
                nnz_cub, nnz_cub_ms);

    if (cusparse_runs_first &&
        !run_requested_cusparse_benchmarks(
            options, host_a, host_b, prepared_a, device_b, -1))
    {
        std::printf("CUSPARSE_CORE_STATUS=FAILED sequence=%d "
                    "ours_will_continue=1\n",
                    options.benchmark_sequence_index);
        (void)cudaGetLastError();
    }

    if (schedule_config.mode != DMMA_SCHEDULE_DIRECT)
    {
        timeval context_begin{}, context_end{};
        gettimeofday(&context_begin, nullptr);
        split_context_ready = dmma_create_split_async_state(
            schedule_config.launch_policy, &split_context);
        gettimeofday(&context_end, nullptr);
        split_context_create_ms_untimed =
            elapsed_ms(context_begin, context_end);
        if (!split_context_ready)
        {
            std::fprintf(stderr,
                         "Unable to create reusable split execution context.\n");
            goto cleanup;
        }
        schedule_config.split_context = &split_context;
    }
    std::printf(
        "SPLIT_CONTEXT requested=%d created=%d context_reused=0 "
        "context_create_ms_untimed=%.6f launch_policy=%s light_policy=%s\n",
        schedule_config.mode != DMMA_SCHEDULE_DIRECT ? 1 : 0,
        split_context_ready ? 1 : 0, split_context_create_ms_untimed,
        launch_policy_name(schedule_config.launch_policy),
        dmma_light_policy_name(schedule_config.light_policy));

    b_update_times.reserve(options.iterations);
    dmma_times.reserve(options.iterations);
    combined_times.reserve(options.iterations);
    single_shot_all_cost_times.reserve(options.iterations);
    export_times.reserve(options.iterations);
    restore_times.reserve(options.iterations);

    for (int warmup = 0; warmup < options.warmup_iterations; ++warmup)
    {
        DmmaBUpdateStats update_stats{};
        DmmaSpGemmStats warmup_stats;
        DmmaNumericScheduleConfig warmup_schedule = schedule_config;
        warmup_schedule.materialize_output = false;
        bool update_ok = false;
        DmmaBValuesClearDecision update_clear_decision =
            dmma_choose_b_values_clear(
                options.b_values_clear_policy,
                options.b_update == B_UPDATE_VALUES, dynamic_b.valid,
                dynamic_b.has_duplicates,
                dynamic_b.active_source_mapping_complete,
                dynamic_b.active_source_to_payload_injective,
                dynamic_b.payload_fully_initialized,
                dynamic_b.tiles.view.dense_tiles == 0, true);
        if (!dmma_cuda_ok(cudaDeviceSynchronize(),
                          "presynchronize ours warmup Core sample"))
            goto cleanup;
        const std::chrono::steady_clock::time_point core_begin =
            std::chrono::steady_clock::now();
        if (options.b_update == B_UPDATE_NONE)
        {
            update_ok = true;
        }
        else if (options.b_update == B_UPDATE_STRUCTURE)
        {
            update_ok = rebuild_b(options, prepared_a, device_b, &dynamic_b,
                                  &update_stats);
        }
        else
        {
            const DmmaOwnedDeviceCsr &source =
                general_ab ? device_b : prepared_a.csr;
            if (options.b_values_clear_policy ==
                DMMA_B_VALUES_ALWAYS_CLEAR)
                update_ok = gpu_update_dynamic_b_values(
                    source.values, source.nnz, &dynamic_b, &update_stats);
            else
                update_ok = gpu_update_dynamic_b_values_with_policy(
                    source.values, source.nnz, &dynamic_b,
                    options.b_values_clear_policy, &update_clear_decision,
                    &update_stats);
        }
        if (update_ok && options.tileflex16_symbolic)
            warmup_schedule.super16_b = dynamic_b.super16.view();
        if (!update_ok ||
            !dmma_tilespgemm(prepared_a.tiles.view, dynamic_b.tiles.view,
                             &matrix_c, &warmup_stats, &warmup_schedule))
        {
            std::fprintf(stderr, "Warmup failed at iteration %d.\n",
                         warmup + 1);
            goto cleanup;
        }
        if (!warmup_stats.core_completion_wall_valid)
        {
            std::fprintf(stderr, "Warmup Core endpoint is unavailable.\n");
            goto cleanup;
        }
        const double warmup_core_ms =
            elapsed_ms(core_begin, warmup_stats.core_completion_wall);
        print_b_values_clear_marker("warmup", warmup + 1, options,
                                    dynamic_b, update_clear_decision);
        std::printf("CORE_BENCH method=ours backend=rtt input=same-order "
                    "phase=warmup warmup=1 iteration=%d core_ms=%.6f "
                    "b_update_ms=%.6f b_update_wall_ms=%.6f "
                    "b_update_mode=%s dmma_ms=%.6f "
                    "schedule=%s direct_layout=%s "
                    "row_static_used=%d row_static_ctas=%d "
                    "row_static_unique_sms=%d row_static_mapping_valid=%d "
                    "row_dynamic_used=%d row_dynamic_ctas=%d "
                    "row_dynamic_unique_sms=%d "
                    "row_dynamic_mapping_valid=%d "
                    "row_dynamic_queue_batch=%d "
                    "row_dynamic_atomic_claims=%d "
                    "row_dynamic_final_head=%d "
                    "row_dynamic_expected_atomic_claims=%d "
                    "row_dynamic_expected_final_head=%d "
                    "row_dynamic_claims=%d row_dynamic_claims_valid=%d "
                    "launch_policy=%s light_policy=%s "
                    "context_reused=%d context_create_ms_untimed=%.6f "
                    "output_state=native-output-ready output_format=tile "
                    "output_export=skipped clock=continuous-wall "
                    "monotonic=1 presync=1 "
                    "sequence=%d\n",
                    warmup + 1,
                    warmup_core_ms,
                    update_stats.total_ms,
                    update_stats.total_ms,
                    b_update_mode_name(options.b_update),
                    warmup_stats.total_ms,
                    effective_schedule_name(
                        warmup_stats.schedule_mode,
                        warmup_stats.cost_balanced_requested),
                    dmma_direct_numeric_layout_name(
                        warmup_stats.direct_numeric_layout),
                    warmup_stats.row_static_used ? 1 : 0,
                    warmup_stats.row_static_ctas,
                    warmup_stats.row_static_unique_sms,
                    warmup_stats.row_static_mapping_valid ? 1 : 0,
                    warmup_stats.row_dynamic_used ? 1 : 0,
                    warmup_stats.row_dynamic_ctas,
                    warmup_stats.row_dynamic_unique_sms,
                    warmup_stats.row_dynamic_mapping_valid ? 1 : 0,
                    warmup_stats.row_dynamic_queue_batch,
                    warmup_stats.row_dynamic_claims,
                    warmup_stats.row_dynamic_final_head,
                    warmup_stats.row_dynamic_expected_claims,
                    warmup_stats.row_dynamic_expected_final_head,
                    warmup_stats.row_dynamic_claims,
                    warmup_stats.row_dynamic_claims_valid ? 1 : 0,
                    launch_policy_name(warmup_stats.launch_policy),
                    dmma_light_policy_name(warmup_stats.light_policy),
                    warmup_stats.split_context_reused ? 1 : 0,
                    split_context_create_ms_untimed,
                    options.benchmark_sequence_index);
        destroy_output_matrix(&matrix_c);
    }

    for (int iteration = 0; iteration < options.iterations; ++iteration)
    {
        DmmaBUpdateStats update_stats{};
        DmmaSpGemmStats dmma_stats;
        DmmaNumericScheduleConfig iteration_schedule = schedule_config;
        const bool materialize_output =
            options.output_export == OUTPUT_EXPORT_EVERY ||
            iteration + 1 == options.iterations;
        iteration_schedule.materialize_output = materialize_output;
        iteration_schedule.task_trace_path =
            options.task_trace_filename != nullptr && iteration == 0
                ? options.task_trace_filename
                : nullptr;
        bool update_ok = false;
        DmmaBValuesClearDecision update_clear_decision =
            dmma_choose_b_values_clear(
                options.b_values_clear_policy,
                options.b_update == B_UPDATE_VALUES, dynamic_b.valid,
                dynamic_b.has_duplicates,
                dynamic_b.active_source_mapping_complete,
                dynamic_b.active_source_to_payload_injective,
                dynamic_b.payload_fully_initialized,
                dynamic_b.tiles.view.dense_tiles == 0, true);
        if (!dmma_cuda_ok(cudaDeviceSynchronize(),
                          "presynchronize ours timed Core sample"))
            goto cleanup;
        const std::chrono::steady_clock::time_point core_begin =
            std::chrono::steady_clock::now();
        if (options.b_update == B_UPDATE_NONE)
        {
            update_ok = true;
        }
        else if (options.b_update == B_UPDATE_STRUCTURE)
        {
            update_ok = rebuild_b(options, prepared_a, device_b, &dynamic_b,
                                  &update_stats);
        }
        else
        {
            const DmmaOwnedDeviceCsr &source =
                general_ab ? device_b : prepared_a.csr;
            if (options.b_values_clear_policy ==
                DMMA_B_VALUES_ALWAYS_CLEAR)
                update_ok = gpu_update_dynamic_b_values(
                    source.values, source.nnz, &dynamic_b, &update_stats);
            else
                update_ok = gpu_update_dynamic_b_values_with_policy(
                    source.values, source.nnz, &dynamic_b,
                    options.b_values_clear_policy, &update_clear_decision,
                    &update_stats);
        }
        if (!update_ok)
        {
            std::fprintf(stderr, "B update failed at iteration %d.\n",
                         iteration + 1);
            goto cleanup;
        }
        if (options.tileflex16_symbolic)
            iteration_schedule.super16_b = dynamic_b.super16.view();

        if (!dmma_tilespgemm(prepared_a.tiles.view, dynamic_b.tiles.view,
                             &matrix_c, &dmma_stats, &iteration_schedule))
        {
            if (dmma_stats.wide_output_unrepresentable)
            {
                std::printf(
                    "DMMA_STATUS=WIDE_OUTPUT_UNREPRESENTABLE "
                    "candidate_tiles=%llu exact_output_tiles=%llu "
                    "nnzC=%llu candidate_ms=%.3f exact_mask_ms=%.3f "
                    "dmma_total_ms=%.3f b_update_ms=%.3f iteration=%d\n",
                    dmma_stats.candidate_tiles,
                    dmma_stats.wide_output_tiles,
                    dmma_stats.wide_output_nnz,
                    dmma_stats.candidate_ms, dmma_stats.symbolic_ms,
                    dmma_stats.total_ms, update_stats.total_ms,
                    iteration + 1);
            }
            else
            {
                std::fprintf(stderr, "DMMA failed at iteration %d.\n",
                             iteration + 1);
            }
            goto cleanup;
        }
        if (!dmma_stats.core_completion_wall_valid)
        {
            std::fprintf(stderr, "Timed Core endpoint is unavailable.\n");
            goto cleanup;
        }
        const double single_shot_all_cost_ms =
            elapsed_ms(core_begin, dmma_stats.core_completion_wall);
        /* TileSpGEMM/FlexSpGEMM construct operand-side tile metadata before
         * their repeated total-runtime loop.  Report the same boundary as
         * core_ms, while retaining the honest one-shot boundary separately. */
        const double combined_ms = std::max(
            0.0, single_shot_all_cost_ms -
                     dmma_stats.super16_index_prepare_ms);
        const double tile_compatible_dmma_ms = dmma_stats.total_ms;
        print_b_values_clear_marker("timed", iteration + 1, options,
                                    dynamic_b, update_clear_decision);
        if (options.b_update == B_UPDATE_VALUES)
        {
            if (update_clear_decision.skip_clear)
                ++b_values_noclear_timed_samples;
            else
                ++b_values_clear_timed_samples;
            if (update_clear_decision.fallback)
                ++b_values_clear_fallback_timed_samples;
        }

        double restore_ms = 0.0;
        if (materialize_output)
        {
            if (!dmma_stats.output_materialized)
            {
                std::fprintf(stderr,
                             "Requested output export was not materialized at "
                             "iteration %d.\n",
                             iteration + 1);
                goto cleanup;
            }
            timeval restore_begin{}, restore_end{};
            gettimeofday(&restore_begin, nullptr);
            const bool restored =
                prepared_a.reorder.kind == DMMA_REORDER_IDENTITY
                    ? tile2csr(&matrix_c, true)
                    : tile2csr_restore_rows(
                          &matrix_c,
                          prepared_a.reorder.h_row_new_to_old,
                          prepared_a.reorder.active_rows, true);
            gettimeofday(&restore_end, nullptr);
            restore_ms = elapsed_ms(restore_begin, restore_end);
            if (!restored)
            {
                std::fprintf(
                    stderr,
                    "Tile-to-CSR row restoration failed at iteration %d.\n",
                    iteration + 1);
                goto cleanup;
            }
        }

        summary_effective_reduction = dmma_stats.reduction_mode;
        summary_workspace_fallback =
            summary_workspace_fallback ||
            dmma_stats.workspace_fallback_to_atomic;
        summary_context_reused =
            summary_context_reused || dmma_stats.split_context_reused;
        if (!summary_direct_numeric_layout_initialized)
        {
            summary_direct_numeric_layout =
                dmma_stats.direct_numeric_layout;
            summary_direct_numeric_layout_initialized = true;
        }
        else if (summary_direct_numeric_layout !=
                 dmma_stats.direct_numeric_layout)
        {
            std::fprintf(stderr,
                         "Numeric layout changed across timed iterations.\n");
            goto cleanup;
        }
        if (dmma_stats.row_static_used)
        {
            summary_row_static_mapping_valid =
                summary_row_static_mapping_valid &&
                dmma_stats.row_static_mapping_valid;
            summary_row_static_ctas = dmma_stats.row_static_ctas;
            if (summary_row_static_min_unique_sms == 0)
                summary_row_static_min_unique_sms =
                    dmma_stats.row_static_unique_sms;
            else
                summary_row_static_min_unique_sms = std::min(
                    summary_row_static_min_unique_sms,
                    dmma_stats.row_static_unique_sms);
        }
        if (dmma_stats.row_dynamic_used)
        {
            summary_row_dynamic_mapping_valid =
                summary_row_dynamic_mapping_valid &&
                dmma_stats.row_dynamic_mapping_valid;
            summary_row_dynamic_claims_valid =
                summary_row_dynamic_claims_valid &&
                dmma_stats.row_dynamic_claims_valid;
            summary_row_dynamic_ctas = dmma_stats.row_dynamic_ctas;
            if (summary_row_dynamic_min_unique_sms == 0)
                summary_row_dynamic_min_unique_sms =
                    dmma_stats.row_dynamic_unique_sms;
            else
                summary_row_dynamic_min_unique_sms = std::min(
                    summary_row_dynamic_min_unique_sms,
                    dmma_stats.row_dynamic_unique_sms);
        }
        b_update_times.push_back(update_stats.total_ms);
        dmma_times.push_back(tile_compatible_dmma_ms);
        combined_times.push_back(combined_ms);
        single_shot_all_cost_times.push_back(single_shot_all_cost_ms);
        if (materialize_output)
        {
            export_times.push_back(dmma_stats.output_copy_ms);
            restore_times.push_back(restore_ms);
        }

        std::printf("---------------- iteration %d/%d ----------------\n",
                    iteration + 1, options.iterations);
        if (options.b_update == B_UPDATE_NONE)
            std::printf("Iteration B update: none (static A*A)\n");
        else
            print_b_update_stats("Iteration", update_stats);
        std::printf("candidate C tiles=%llu; exact non-empty C tiles=%d; "
                    "nnzC=%d\n",
                    dmma_stats.candidate_tiles, dmma_stats.output_tiles,
                    dmma_stats.output_nnz);
        std::printf("step1 candidate=%.3f ms; step2 exact-mask=%.3f ms; "
                    "step3 numeric=%.4f ms; output payload allocation=%.3f ms; "
                    "scheduler submit/device=%.4f/%.4f ms\n",
                    dmma_stats.candidate_ms, dmma_stats.symbolic_ms,
                    dmma_stats.numeric_ms, dmma_stats.allocation_ms,
                    dmma_stats.scheduler_ms,
                    dmma_stats.scheduler_device_ms);
        std::printf(
            "DMMA_EXACT_FORWARD iteration=%d requested=%d used=%d "
            "reason=%s gate_source=pre-candidate-structure "
            "forward_rows=%d forward_pairs=%llu "
            "estimated_candidates=%llu estimated_reverse_work=%llu "
            "estimated_forward_work=%llu forward_candidates=%llu "
            "ordinary_candidates=%llu partition_complete=%d "
            "spa_batches=%d batch_capacity=%d scratch_bytes=%zu "
            "scratch_limit_bytes=%zu estimate_overflow=%d\n",
            iteration + 1,
            dmma_stats.exact_forward_spa_requested ? 1 : 0,
            dmma_stats.exact_forward_spa_used ? 1 : 0,
            dmma_exact_forward_spa_reason_name(
                dmma_stats.exact_forward_spa_reason),
            dmma_stats.exact_forward_rows,
            dmma_stats.exact_forward_pairs,
            dmma_stats.exact_forward_estimated_candidates,
            dmma_stats.exact_forward_estimated_reverse_work,
            dmma_stats.exact_forward_estimated_forward_work,
            dmma_stats.exact_forward_candidates,
            dmma_stats.exact_ordinary_candidates,
            dmma_stats.exact_forward_partition_complete ? 1 : 0,
            dmma_stats.exact_forward_batches,
            dmma_stats.exact_forward_batch_capacity,
            dmma_stats.exact_forward_scratch_bytes,
            DMMA_EXACT_FORWARD_SPA_SCRATCH_BYTES,
            dmma_stats.exact_forward_estimate_overflow ? 1 : 0);
        const double reduction_fraction =
            combined_ms > 0.0
                ? dmma_stats.reduction_ms / combined_ms
                : 0.0;
        std::printf(
            "TASK_SPLIT iteration=%d schedule=%s direct_layout=%s "
            "row_static_used=%d row_static_ctas=%d "
            "row_static_unique_sms=%d row_static_mapping_valid=%d "
            "row_static_mapping_audit_core_excluded=1 "
            "row_dynamic_used=%d row_dynamic_ctas=%d "
            "row_dynamic_unique_sms=%d row_dynamic_mapping_valid=%d "
            "row_dynamic_queue_batch=%d "
            "row_dynamic_atomic_claims=%d row_dynamic_final_head=%d "
            "row_dynamic_expected_atomic_claims=%d "
            "row_dynamic_expected_final_head=%d "
            "row_dynamic_claims=%d row_dynamic_claims_valid=%d "
            "row_dynamic_audit_core_excluded=1 "
            "row_dynamic_queue_setup_internal_core_excluded=1 reduction=%s "
            "requested_reduction=%s heavy_tasks=%d "
            "early_split_requested=%d early_split_used=%d "
            "early_heavy_worker_cap=%d early_heavy_worker_blocks=%d "
            "early_reduction_worker_blocks=%d "
            "early_heavy_cap_numerator=%d "
            "early_heavy_cap_denominator=%d "
            "early_sm_affinity_claimed=%d "
            "light_policy=%s critical_q_min=%.9f q_begin=%d "
            "prefix_tasks=%d suffix_tasks=%d suffix_fraction=%.9f "
            "suffix_auto_basis=%s suffix_auto_fraction=%.9f "
            "suffix_auto_used_work=%d suffix_auto_fallback_tasks=%d "
            "suffix_workers_per_sm=%d suffix_worker_blocks=%d "
            "suffix_queue_batch=%d suffix_fine_tasks=%d "
            "suffix_queue_atomics=%llu "
            "unified_used=%d unified_fallback=%s unified_workers=%d "
            "unified_workers_per_sm=%d "
            "unified_sparse_replay=%d "
            "unified_coarse_tasks=%d unified_fine_tasks=%d "
            "unified_fine_queue_tasks=%d unified_heavy_fine_tasks=%d "
            "unified_coarse_work=%.6f unified_coarse_work_available=%d "
            "unified_fine_work=%.6f "
            "unified_heavy_work=%.6f unified_page_size=%d "
            "unified_coarse_pages=%d unified_coarse_page_claims=%llu "
            "unified_fine_capacity=%d unified_fine_overflow=%d "
            "unified_fine_ticket_claims=%llu unified_metadata_bytes=%zu "
            "launch_policy=%s streams_created=%d streams_used=%d "
            "context_reused=%d context_create_ms_untimed=%.6f "
            "stream_priority_supported=%d light_stream_priority=%d "
            "suffix_stream_priority=%d heavy_stream_priority=%d "
            "split_fraction=%.9f chunks=%d avg_chunks=%.6f "
            "flat_used=%d flat_warps_per_cta=%d flat_blocks=%u "
            "flat_items=%llu flat_ms=%.6f "
            "tail_count=%d tail_capacity=%d tail_overflow=%d "
            "tail_count_lower_bound=%d tail_fraction=%.9f "
            "maybe_scope=%s maybe_count=%d maybe_capacity=%d "
            "maybe_overflow=%d "
            "maybe_count_lower_bound=%d maybe_fraction=%.9f "
            "admission_precision=fp32 admission_filter_ms=%.6f "
            "admission_count_ms=%.6f admission_device_ms=%.6f "
            "admission_timing_valid=%d admission_count_executed=%d "
            "tail_gate=%s tail_force=%d "
            "tail_symbolic_bytes=%zu tail_fallback_direct=%d "
            "avg_distinct_sms=%.6f sm_max_over_mean=%.6f sm_cv=%.6f "
            "scheduler_ms=%.6f scheduler_submit_ms=%.6f "
            "scheduler_device_ms=%.6f light_ms=%.6f prefix_ms=%.6f "
            "suffix_ms=%.6f chunk_ms=%.6f "
            "reduction_ms=%.6f reduction_timing=%s "
            "reduction_fraction=%.9f reduction_fraction_valid=%d "
            "workspace_bytes=%zu "
            "heavy_flag_bytes=%zu workspace_fallback=%d task_stats_ms=%.6f\n",
            iteration + 1,
            effective_schedule_name(dmma_stats.schedule_mode,
                                    dmma_stats.cost_balanced_requested),
            dmma_direct_numeric_layout_name(
                dmma_stats.direct_numeric_layout),
            dmma_stats.row_static_used ? 1 : 0,
            dmma_stats.row_static_ctas,
            dmma_stats.row_static_unique_sms,
            dmma_stats.row_static_mapping_valid ? 1 : 0,
            dmma_stats.row_dynamic_used ? 1 : 0,
            dmma_stats.row_dynamic_ctas,
            dmma_stats.row_dynamic_unique_sms,
            dmma_stats.row_dynamic_mapping_valid ? 1 : 0,
            dmma_stats.row_dynamic_queue_batch,
            dmma_stats.row_dynamic_claims,
            dmma_stats.row_dynamic_final_head,
            dmma_stats.row_dynamic_expected_claims,
            dmma_stats.row_dynamic_expected_final_head,
            dmma_stats.row_dynamic_claims,
            dmma_stats.row_dynamic_claims_valid ? 1 : 0,
            reduction_name(dmma_stats.reduction_mode),
            reduction_name(schedule_config.reduction),
            dmma_stats.heavy_tasks,
            dmma_stats.early_split_requested ? 1 : 0,
            dmma_stats.early_split_used ? 1 : 0,
            dmma_stats.early_heavy_worker_block_cap,
            dmma_stats.early_heavy_worker_blocks,
            dmma_stats.early_reduction_worker_blocks,
            dmma_stats.early_heavy_cap_numerator,
            dmma_stats.early_heavy_cap_denominator,
            dmma_stats.early_split_sm_affinity_claimed ? 1 : 0,
            dmma_light_policy_name(dmma_stats.light_policy),
            schedule_config.critical_q_min,
            dmma_stats.critical_q_begin,
            dmma_stats.prefix_tasks,
            dmma_stats.suffix_tasks,
            dmma_stats.suffix_task_fraction,
            dmma_suffix_auto_basis_name(dmma_stats.suffix_auto_basis),
            dmma_stats.suffix_auto_fraction,
            dmma_stats.suffix_auto_used_symbolic_work ? 1 : 0,
            dmma_stats.suffix_auto_fallback_to_tasks ? 1 : 0,
            dmma_stats.suffix_workers_per_sm,
            dmma_stats.suffix_worker_blocks,
            dmma_stats.suffix_queue_batch,
            dmma_stats.suffix_fine_tasks,
            dmma_stats.suffix_queue_atomics,
            dmma_stats.unified_light_used ? 1 : 0,
            dmma_unified_fallback_reason_name(
                dmma_stats.unified_fallback_reason),
            dmma_stats.unified_worker_blocks,
            dmma_stats.unified_workers_per_sm,
            dmma_stats.unified_sparse_replay_used ? 1 : 0,
            dmma_stats.unified_coarse_tasks,
            dmma_stats.unified_fine_tasks,
            dmma_stats.unified_fine_queue_tasks,
            dmma_stats.unified_heavy_fine_tasks,
            dmma_stats.unified_coarse_work,
            dmma_stats.unified_coarse_work_available ? 1 : 0,
            dmma_stats.unified_fine_work,
            dmma_stats.unified_heavy_work,
            dmma_stats.unified_page_size,
            dmma_stats.unified_coarse_pages,
            dmma_stats.unified_coarse_page_claims,
            dmma_stats.unified_fine_capacity,
            dmma_stats.unified_fine_overflow ? 1 : 0,
            dmma_stats.unified_fine_ticket_claims,
            dmma_stats.unified_metadata_bytes,
            launch_policy_name(dmma_stats.launch_policy),
            dmma_stats.split_streams_created ? 1 : 0,
            dmma_stats.split_streams_used ? 1 : 0,
            dmma_stats.split_context_reused ? 1 : 0,
            split_context_create_ms_untimed,
            dmma_stats.stream_priority_range_supported ? 1 : 0,
            dmma_stats.light_stream_priority,
            dmma_stats.suffix_stream_priority,
            dmma_stats.heavy_stream_priority,
            dmma_stats.split_task_fraction,
            dmma_stats.split_chunks, dmma_stats.average_chunks,
            dmma_stats.flat_grid_used ? 1 : 0,
            dmma_stats.flat_warps_per_cta,
            dmma_stats.flat_grid_blocks,
            dmma_stats.flat_work_items,
            dmma_stats.flat_numeric_ms,
            dmma_stats.tail_record_count,
            dmma_stats.tail_record_capacity,
            dmma_stats.tail_record_overflow ? 1 : 0,
            dmma_stats.tail_record_count_is_lower_bound ? 1 : 0,
            dmma_stats.tail_record_fraction,
            dmma_stats.maybe_scope_joint ? "joint" : "heavy",
            dmma_stats.maybe_candidate_count,
            dmma_stats.maybe_candidate_capacity,
            dmma_stats.maybe_candidate_overflow ? 1 : 0,
            dmma_stats.maybe_candidate_count_is_lower_bound ? 1 : 0,
            dmma_stats.maybe_candidate_fraction,
            dmma_stats.admission_filter_ms,
            dmma_stats.admission_count_ms,
            dmma_stats.admission_device_ms,
            dmma_stats.admission_timing_valid ? 1 : 0,
            dmma_stats.admission_count_executed ? 1 : 0,
            dmma_tail_gate_reason_name(dmma_stats.tail_gate_reason),
            dmma_stats.tail_split_forced ? 1 : 0,
            dmma_stats.symbolic_task_count_bytes,
            dmma_stats.tail_gate_fallback_to_direct ? 1 : 0,
            dmma_stats.average_distinct_sms,
            dmma_stats.sm_work_max_over_mean, dmma_stats.sm_work_cv,
            dmma_stats.scheduler_ms, dmma_stats.scheduler_ms,
            dmma_stats.scheduler_device_ms, dmma_stats.light_numeric_ms,
            dmma_stats.prefix_numeric_ms,
            dmma_stats.suffix_numeric_ms,
            dmma_stats.chunk_numeric_ms, dmma_stats.reduction_ms,
            dmma_stats.reduction_mode == DMMA_REDUCTION_WORKSPACE
                ? "separate-kernel"
                : "fused-into-chunk",
            reduction_fraction,
            dmma_stats.reduction_mode == DMMA_REDUCTION_WORKSPACE ? 1 : 0,
            dmma_stats.partial_workspace_bytes,
            dmma_stats.heavy_flag_bytes,
            dmma_stats.workspace_fallback_to_atomic ? 1 : 0,
            dmma_stats.task_stats_ms);
        const char *tail_record_id_space =
            dmma_stats.tail_record_count == 0
                ? "none"
                : (dmma_stats.maybe_scope_joint ||
                           dmma_stats.symbolic_admission ==
                               DMMA_SYMBOLIC_ADMISSION_SEPARATE ||
                           dmma_stats.scheduler_reused_symbolic_counts
                       ? "output"
                       : "candidate-unmapped-fallback");
        std::printf(
            "SYMBOLIC_ADMISSION iteration=%d path=%s "
            "admission_threshold=%.6f chunk_target=%.6f "
            "filter_fused=%d count_fused=%d count_scope=%s "
            "tail_record_id_space=%s\n",
            iteration + 1,
            dmma_symbolic_admission_name(dmma_stats.symbolic_admission),
            schedule_config.split_threshold,
            schedule_config.chunk_target > 0.0
                ? schedule_config.chunk_target
                : schedule_config.split_threshold,
            dmma_stats.admission_filter_fused_into_exact ? 1 : 0,
            dmma_stats.admission_count_fused_into_exact ? 1 : 0,
            dmma_stats.maybe_scope_joint
                ? "joint-exact-output-window"
                : dmma_stats.symbolic_admission ==
                    DMMA_SYMBOLIC_ADMISSION_FUSED_EXACT
                ? "fused-all-candidates-then-map"
                : "exact-output-window",
            tail_record_id_space);
        if (dmma_stats.symbolic_load_metadata)
            std::printf(
                "SYMBOLIC_LOAD iteration=%d tasks=%d saturated=%d "
                "metadata_bytes=%zu quantum=%.6f wave_task_capacity=%d "
                "predicted_waves=%d critical_tasks=%d "
                "critical_tail_tasks=%d total_work=%.6f "
                "suffix_work=%.6f potential_split_suffix_work=%.6f "
                "suffix_work_fraction=%.9f "
                "regular_suffix_work_fraction=%.9f "
                "max_task_work=%.6f critical_work=%.6f "
                "critical_max_task_work=%.6f critical_tail_work=%.6f "
                "critical_over_average_wave=%.9f "
                "critical_tail_fraction=%.9f\n",
                iteration + 1, dmma_stats.symbolic_load_tasks,
                dmma_stats.symbolic_load_saturated_tasks,
                dmma_stats.symbolic_load_metadata_bytes,
                dmma_stats.symbolic_load_quantum,
                dmma_stats.symbolic_wave_task_capacity,
                dmma_stats.symbolic_predicted_waves,
                dmma_stats.symbolic_critical_tasks,
                dmma_stats.symbolic_critical_tail_tasks,
                dmma_stats.symbolic_total_work,
                dmma_stats.symbolic_suffix_work,
                dmma_stats.symbolic_split_suffix_work,
                dmma_stats.symbolic_suffix_work_fraction,
                dmma_stats.symbolic_regular_suffix_work_fraction,
                dmma_stats.symbolic_max_task_work,
                dmma_stats.symbolic_critical_work,
                dmma_stats.symbolic_critical_max_task_work,
                dmma_stats.symbolic_critical_tail_work,
                dmma_stats.symbolic_critical_work_over_average_wave,
                dmma_stats.symbolic_critical_tail_work_fraction);
        const double numeric_gflops =
            dmma_stats.numeric_ms > 0.0
                ? 2.0 * static_cast<double>(nnz_cub) /
                      (dmma_stats.numeric_ms * 1.0e6)
                : 0.0;
        std::printf("SUPER16_BREAKDOWN iteration=%d index_prepare_ms=%.6f "
                    "index_cache_hit=%d parent_symbolic_wall_ms=%.6f\n",
                    iteration + 1, dmma_stats.super16_index_prepare_ms,
                    dmma_stats.super16_index_cache_hit ? 1 : 0,
                    dmma_stats.super16_parent_symbolic_wall_ms);
        std::printf("CUDA  TileSpGEMM-compatible runtime is %.3f ms; "
                    "single-shot all-cost is %.3f ms; numeric gflops=%.2f\n",
                    tile_compatible_dmma_ms, single_shot_all_cost_ms,
                    numeric_gflops);
        if (materialize_output)
            std::printf("iteration B-update+DMMA=%.3f ms; "
                        "C tile export D2H=%.3f ms; row restore/CSR=%.3f ms\n",
                        combined_ms, dmma_stats.output_copy_ms, restore_ms);
        else
            std::printf("iteration B-update+DMMA=%.3f ms; "
                        "C tile export D2H=SKIPPED; row restore/CSR=SKIPPED\n",
                        combined_ms);
        std::printf(
            "CORE_BENCH method=ours backend=rtt input=same-order "
            "phase=timed warmup=0 iteration=%d core_ms=%.6f "
            "single_shot_all_cost_ms=%.6f "
            "operand_index_prepare_ms=%.6f timing_scope=tilespgemm-compatible "
            "b_update_ms=%.6f b_update_wall_ms=%.6f "
            "b_update_mode=%s dmma_ms=%.6f "
            "schedule=%s direct_layout=%s row_static_used=%d "
            "row_static_ctas=%d row_static_unique_sms=%d "
            "row_static_mapping_valid=%d "
            "row_static_mapping_audit_core_excluded=1 "
            "row_dynamic_used=%d row_dynamic_ctas=%d "
            "row_dynamic_unique_sms=%d row_dynamic_mapping_valid=%d "
            "row_dynamic_queue_batch=%d "
            "row_dynamic_atomic_claims=%d row_dynamic_final_head=%d "
            "row_dynamic_expected_atomic_claims=%d "
            "row_dynamic_expected_final_head=%d "
            "row_dynamic_claims=%d row_dynamic_claims_valid=%d "
            "row_dynamic_audit_core_excluded=1 "
            "row_dynamic_queue_setup_internal_core_excluded=1 "
            "candidate_ms=%.6f exact_mask_ms=%.6f "
            "exact_kernel_ms=%.6f symbolic_finalize_ms=%.6f "
            "output_payload_allocation_ms=%.6f "
            "output_allocation_ms=%.6f allocation_ms=%.6f "
            "numeric_ms=%.6f scheduler_ms=%.6f "
            "scheduler_submit_ms=%.6f scheduler_device_ms=%.6f reduction=%s "
            "reduction_timing=%s "
            "requested_reduction=%s launch_policy=%s light_policy=%s "
            "critical_q_min=%.9f q_begin=%d prefix_tasks=%d "
            "suffix_tasks=%d suffix_fraction=%.9f "
            "suffix_auto_basis=%s suffix_auto_fraction=%.9f "
            "suffix_auto_used_work=%d suffix_auto_fallback_tasks=%d "
            "prefix_ms=%.6f suffix_ms=%.6f "
            "context_reused=%d context_create_ms_untimed=%.6f "
            "stream_priority_supported=%d light_stream_priority=%d "
            "suffix_stream_priority=%d heavy_stream_priority=%d "
            "workspace_fallback=%d "
            "exact_forward_requested=%d exact_forward_used=%d "
            "exact_forward_reason=%s exact_forward_rows=%d "
            "exact_forward_candidates=%llu "
            "exact_ordinary_candidates=%llu "
            "exact_forward_partition_complete=%d "
            "output_state=native-output-ready output_format=tile "
            "output_export=%s clock=continuous-wall monotonic=1 presync=1 "
            "sequence=%d\n",
            iteration + 1, combined_ms, single_shot_all_cost_ms,
            dmma_stats.super16_index_prepare_ms, update_stats.total_ms,
            update_stats.total_ms,
            b_update_mode_name(options.b_update),
            tile_compatible_dmma_ms,
            effective_schedule_name(dmma_stats.schedule_mode,
                                    dmma_stats.cost_balanced_requested),
            dmma_direct_numeric_layout_name(
                dmma_stats.direct_numeric_layout),
            dmma_stats.row_static_used ? 1 : 0,
            dmma_stats.row_static_ctas,
            dmma_stats.row_static_unique_sms,
            dmma_stats.row_static_mapping_valid ? 1 : 0,
            dmma_stats.row_dynamic_used ? 1 : 0,
            dmma_stats.row_dynamic_ctas,
            dmma_stats.row_dynamic_unique_sms,
            dmma_stats.row_dynamic_mapping_valid ? 1 : 0,
            dmma_stats.row_dynamic_queue_batch,
            dmma_stats.row_dynamic_claims,
            dmma_stats.row_dynamic_final_head,
            dmma_stats.row_dynamic_expected_claims,
            dmma_stats.row_dynamic_expected_final_head,
            dmma_stats.row_dynamic_claims,
            dmma_stats.row_dynamic_claims_valid ? 1 : 0,
            dmma_stats.candidate_ms,
            dmma_stats.symbolic_ms, dmma_stats.exact_kernel_ms,
            dmma_stats.symbolic_finalize_ms, dmma_stats.allocation_ms,
            dmma_stats.allocation_ms, dmma_stats.allocation_ms,
            dmma_stats.numeric_ms, dmma_stats.scheduler_ms,
            dmma_stats.scheduler_ms, dmma_stats.scheduler_device_ms,
            reduction_name(dmma_stats.reduction_mode),
            dmma_stats.reduction_mode == DMMA_REDUCTION_WORKSPACE
                ? "separate-kernel"
                : "fused-into-chunk",
            reduction_name(schedule_config.reduction),
            launch_policy_name(dmma_stats.launch_policy),
            dmma_light_policy_name(dmma_stats.light_policy),
            schedule_config.critical_q_min,
            dmma_stats.critical_q_begin,
            dmma_stats.prefix_tasks,
            dmma_stats.suffix_tasks,
            dmma_stats.suffix_task_fraction,
            dmma_suffix_auto_basis_name(dmma_stats.suffix_auto_basis),
            dmma_stats.suffix_auto_fraction,
            dmma_stats.suffix_auto_used_symbolic_work ? 1 : 0,
            dmma_stats.suffix_auto_fallback_to_tasks ? 1 : 0,
            dmma_stats.prefix_numeric_ms,
            dmma_stats.suffix_numeric_ms,
            dmma_stats.split_context_reused ? 1 : 0,
            split_context_create_ms_untimed,
            dmma_stats.stream_priority_range_supported ? 1 : 0,
            dmma_stats.light_stream_priority,
            dmma_stats.suffix_stream_priority,
            dmma_stats.heavy_stream_priority,
            dmma_stats.workspace_fallback_to_atomic ? 1 : 0,
            dmma_stats.exact_forward_spa_requested ? 1 : 0,
            dmma_stats.exact_forward_spa_used ? 1 : 0,
            dmma_exact_forward_spa_reason_name(
                dmma_stats.exact_forward_spa_reason),
            dmma_stats.exact_forward_rows,
            dmma_stats.exact_forward_candidates,
            dmma_stats.exact_ordinary_candidates,
            dmma_stats.exact_forward_partition_complete ? 1 : 0,
            materialize_output ? "materialized" : "skipped",
            options.benchmark_sequence_index);
        if (dmma_stats.task_trace_tasks > 0)
            std::printf("TASK_TRACE_RESULT iteration=%d tasks=%d "
                        "trace_ms=%.6f core_excluded=1\n",
                        iteration + 1, dmma_stats.task_trace_tasks,
                        dmma_stats.task_trace_ms);

        if (iteration + 1 < options.iterations)
            destroy_output_matrix(&matrix_c);
    }

    std::printf("---------------- median over %d iterations ----------------\n",
                options.iterations);
    std::printf("median B-update=%.3f ms; DMMA=%.3f ms; "
                "Tile-compatible B-update+DMMA=%.3f ms; "
                "single-shot all-cost=%.3f ms\n",
                median(b_update_times), median(dmma_times),
                median(combined_times), median(single_shot_all_cost_times));
    std::printf("median C tile export D2H=%.3f ms; "
                "row restore/CSR=%.3f ms; materialized_samples=%zu/%d\n",
                median(export_times), median(restore_times),
                export_times.size(), options.iterations);
    std::printf("CORE_SUMMARY method=ours input=same-order "
                "boundary=tilespgemm-compatible-operand-metadata-excluded+"
                "b-update+candidate+exact-symbolic+scans+"
                "output-allocation+schedule+reduction+numeric+"
                "native-output-ready clock=continuous-wall "
                "monotonic=1 presync=1 "
                "b_update_mode=%s "
                "output_state=native-output-ready output_format=tile "
                "sequence=%d warmup=%d iterations=%d samples_ms=",
                b_update_mode_name(options.b_update),
                options.benchmark_sequence_index,
                options.warmup_iterations, options.iterations);
    print_samples(combined_times);
    std::printf(" min_ms=%.6f median_ms=%.6f max_ms=%.6f "
                "schedule=%s direct_layout=%s row_queue_batch=%d "
                "row_dynamic_auto=%d row_dynamic_threshold=%.9f "
                "row_static_ctas=%d "
                "row_static_min_unique_sms=%d "
                "row_static_all_mapping_valid=%d "
                "row_static_mapping_audit_core_excluded=1 "
                "row_dynamic_ctas=%d row_dynamic_min_unique_sms=%d "
                "row_dynamic_all_mapping_valid=%d "
                "row_dynamic_all_claims_valid=%d "
                "row_dynamic_audit_core_excluded=1 "
                "row_dynamic_queue_setup_internal_core_excluded=1 "
                "reduction=%s requested_reduction=%s "
                "launch_policy=%s light_policy=%s critical_q_min=%.9f "
                "suffix_auto_basis=%s "
                "context_reused=%d context_create_ms_untimed=%.6f "
                "workspace_fallback=%d output_export_mode=%s "
                "materialized_samples=%zu\n",
                minimum(combined_times), median(combined_times),
                maximum(combined_times),
                effective_schedule_name(schedule_config.mode,
                                        schedule_config.cost_balanced),
                dmma_direct_numeric_layout_name(
                    summary_direct_numeric_layout),
                schedule_config.row_queue_batch,
                schedule_config.row_dynamic_auto ? 1 : 0,
                schedule_config.row_dynamic_threshold,
                summary_row_static_ctas,
                summary_row_static_min_unique_sms,
                summary_direct_numeric_layout ==
                        DMMA_DIRECT_NUMERIC_ROW_STATIC &&
                    summary_row_static_mapping_valid
                    ? 1
                    : 0,
                summary_row_dynamic_ctas,
                summary_row_dynamic_min_unique_sms,
                summary_direct_numeric_layout ==
                        DMMA_DIRECT_NUMERIC_ROW_DYNAMIC &&
                    summary_row_dynamic_mapping_valid
                    ? 1
                    : 0,
                summary_direct_numeric_layout ==
                        DMMA_DIRECT_NUMERIC_ROW_DYNAMIC &&
                    summary_row_dynamic_claims_valid
                    ? 1
                    : 0,
                reduction_name(summary_effective_reduction),
                reduction_name(schedule_config.reduction),
                launch_policy_name(schedule_config.launch_policy),
                dmma_light_policy_name(schedule_config.light_policy),
                schedule_config.critical_q_min,
                dmma_suffix_auto_basis_name(
                    schedule_config.suffix_auto_basis),
                summary_context_reused ? 1 : 0,
                split_context_create_ms_untimed,
                summary_workspace_fallback ? 1 : 0,
                options.output_export == OUTPUT_EXPORT_LAST ? "last" :
                                                              "every",
                export_times.size());
    std::printf("CORE_ALL_COST_SUMMARY method=ours "
                "boundary=single-shot-including-super16-operand-index "
                "samples_ms=");
    print_samples(single_shot_all_cost_times);
    std::printf(" min_ms=%.6f median_ms=%.6f max_ms=%.6f\n",
                minimum(single_shot_all_cost_times),
                median(single_shot_all_cost_times),
                maximum(single_shot_all_cost_times));
    std::printf(
        "B_VALUES_CLEAR_SUMMARY requested=%s b_update=%s timed_samples=%d "
        "clear_samples=%d noclear_samples=%d fallback_samples=%d "
        "legacy_always_clear_entrypoint=%d timing_boundary=unchanged\n",
        dmma_b_values_clear_policy_name(options.b_values_clear_policy),
        b_update_mode_name(options.b_update),
        options.b_update == B_UPDATE_VALUES ? options.iterations : 0,
        b_values_clear_timed_samples, b_values_noclear_timed_samples,
        b_values_clear_fallback_timed_samples,
        options.b_values_clear_policy == DMMA_B_VALUES_ALWAYS_CLEAR ? 1 : 0);

    if (!cusparse_runs_first && core_target_runs_cusparse &&
        !run_requested_cusparse_benchmarks(
            options, host_a, host_b, prepared_a, device_b, matrix_c.nnz))
    {
        std::printf("CUSPARSE_CORE_STATUS=FAILED sequence=%d "
                    "ours_samples_retained=1\n",
                    options.benchmark_sequence_index);
        (void)cudaGetLastError();
    }

    {
        int validation_status = 0;
#if CHECK_RESULT
        std::printf("-------------------------------check"
                    "----------------------------------------\n");
        unsigned long long cusparse_nnz_c = 0;
        double cusparse_compression = 0.0;
        double cusparse_time = 0.0;
        double cusparse_gflops = 0.0;
        if (general_ab)
        {
            validation_status = spgemm_cu_device_ab(
                prepared_a.csr.rows, prepared_a.csr.cols,
                prepared_a.csr.nnz, prepared_a.csr.row_ptr,
                prepared_a.csr.col_idx, prepared_a.csr.values,
                device_b.rows, device_b.cols, device_b.nnz,
                device_b.row_ptr, device_b.col_idx, device_b.values,
                matrix_c.m, matrix_c.n, matrix_c.nnz,
                matrix_c.rowpointer, matrix_c.columnindex, true, nnz_cub,
                &cusparse_nnz_c, &cusparse_compression, &cusparse_time,
                &cusparse_gflops);
        }
        else
        {
            validation_status = spgemm_cu_device(
                prepared_a.csr.rows, prepared_a.csr.cols,
                prepared_a.csr.nnz, prepared_a.csr.row_ptr,
                prepared_a.csr.col_idx, prepared_a.csr.values,
                options.aat != 0, matrix_c.m, matrix_c.n, matrix_c.nnz,
                matrix_c.rowpointer, matrix_c.columnindex, true, nnz_cub,
                &cusparse_nnz_c, &cusparse_compression, &cusparse_time,
                &cusparse_gflops);
        }
#endif
        status = validation_status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }

cleanup:
    if (split_context_ready)
    {
        dmma_destroy_split_async_state(&split_context, true);
        split_context_ready = false;
        schedule_config.split_context = nullptr;
    }
    destroy_output_matrix(&matrix_c);
    destroy_dynamic_b(&dynamic_b);
    destroy_device_csr(&device_b);
    destroy_prepared_a(&prepared_a);
    free_host_csr(&host_b);
    free_host_csr(&host_a);
    std::printf("---------------------------------------------------------------\n");
    return status;
}
