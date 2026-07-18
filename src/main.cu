#include "common.h"
#include "mmio_highlevel.h"
#include "gpu_dmma_tiles.h"
#include "tile2csr.h"
#include "spgemm_cu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <sys/time.h>
#include <vector>

enum BUpdateMode
{
    B_UPDATE_VALUES,
    B_UPDATE_STRUCTURE
};

struct Options
{
    int device = 0;
    int aat = 0;
    int dense_threshold = 24;
    int iterations = 1;
    BUpdateMode b_update = B_UPDATE_VALUES;
    bool no_reorder = false;
    const char *row_order_filename = nullptr;
    const char *inner_order_filename = nullptr;
    const char *reorder_name = nullptr;
    const char *a_filename = nullptr;
    const char *b_filename = nullptr;
    const char *dump_prefix = nullptr;
    const char *heatmap_prefix = nullptr;
    int heatmap_bins = 256;
    bool prepare_only = false;
};

static void print_usage(const char *program)
{
    std::printf(
        "Usage: %s -d <gpu> -aat <0|1> [--b B.mtx] "
        "[--dense-threshold <1..32>] [--iterations N] "
        "[--b-update values|structure] [--no-reorder] "
        "[--row-order FILE --inner-order FILE --reorder-name NAME] "
        "[--dump-reorder-prefix P] [--dump-reorder-heatmap P] "
        "[--heatmap-bins N] [--prepare-only] A.mtx\n",
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
        else if (std::strcmp(argv[i], "--b") == 0 && i + 1 < argc)
            options->b_filename = argv[++i];
        else if (std::strcmp(argv[i], "--b-update") == 0 &&
                 i + 1 < argc)
        {
            const char *mode = argv[++i];
            if (std::strcmp(mode, "values") == 0)
                options->b_update = B_UPDATE_VALUES;
            else if (std::strcmp(mode, "structure") == 0)
                options->b_update = B_UPDATE_STRUCTURE;
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
           options->iterations > 0 && options->heatmap_bins >= 16 &&
           options->heatmap_bins <= 1024 &&
           (!any_external || complete_external) &&
           !(options->no_reorder && any_external) && valid_reorder_name;
}

static double elapsed_ms(const timeval &begin, const timeval &end)
{
    return (end.tv_sec - begin.tv_sec) * 1000.0 +
           (end.tv_usec - begin.tv_usec) / 1000.0;
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
                "tile build = %.3f ms\n",
                stats.reorder_ms, stats.key_sort_reduce_ms,
                stats.tile_build_ms);
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
            "sort/reduce=%.3f tile-build=%.3f CSC=%.3f mapping=%.3f; "
            "entries source=%d active=%d unique=%d; peak=%.2f MB\n",
            label, stats.total_ms, stats.validation_ms,
            stats.key_sort_reduce_ms, stats.tile_build_ms, stats.csc_ms,
            stats.mapping_ms, stats.source_entries, stats.active_entries,
            stats.unique_entries,
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

static bool rebuild_b(const Options &options, const DmmaPreparedA &a,
                      const DmmaOwnedDeviceCsr &independent_b,
                      DmmaDynamicB *b, DmmaBUpdateStats *stats)
{
    const int *inner_old_to_new = a.reorder.d_inner_old_to_new;
    const int active_inner = a.reorder.active_inner;
    if (options.b_filename != nullptr)
        return gpu_rebuild_dynamic_b(
            device_csr_view(independent_b), inner_old_to_new, active_inner,
            options.dense_threshold, b, stats);
    if (options.aat != 0)
        return gpu_rebuild_dynamic_b_transpose(
            device_csr_view(a.csr), inner_old_to_new, active_inner,
            options.dense_threshold, b, stats);
    return gpu_rebuild_dynamic_b(
        device_csr_view(a.csr), inner_old_to_new, active_inner,
        options.dense_threshold, b, stats);
}

int main(int argc, char **argv)
{
    Options options;
    if (!parse_arguments(argc, argv, &options))
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

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
                   : "bgrf-v1");
    std::printf("mode=%s, reorder=%s, B-update=%s, iterations=%d\n",
                general_ab ? "AB" : (options.aat ? "AAT" : "AA"),
                reorder_algorithm.c_str(),
                options.b_update == B_UPDATE_VALUES ? "values" : "structure",
                options.iterations);
    std::printf("DMMA tiles: A=8x4, B=4x8, C=8x8; "
                "dense threshold=%d/32\n",
                options.dense_threshold);
    if (general_ab && options.aat != 0)
        std::printf("Note: --b selects general A*B; -aat is ignored.\n");

    DmmaPreparedA prepared_a;
    DmmaOfflineAStats a_stats;
    DmmaOwnedDeviceCsr device_b;
    DmmaDynamicB dynamic_b;
    SMatrix matrix_c{};
    int status = EXIT_FAILURE;
    unsigned long long nnz_cub = 0;
    double nnz_cub_ms = 0.0;
    std::vector<double> b_update_times;
    std::vector<double> dmma_times;
    std::vector<double> combined_times;
    std::vector<double> export_times;
    std::vector<double> restore_times;

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
            host_a, options.dense_threshold, &prepared_a, &a_stats);
    if (!a_ready)
    {
        std::fprintf(stderr, "GPU A preparation failed.\n");
        goto cleanup;
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

    b_update_times.reserve(options.iterations);
    dmma_times.reserve(options.iterations);
    combined_times.reserve(options.iterations);
    export_times.reserve(options.iterations);
    restore_times.reserve(options.iterations);

    for (int iteration = 0; iteration < options.iterations; ++iteration)
    {
        DmmaBUpdateStats update_stats;
        bool update_ok = false;
        if (options.b_update == B_UPDATE_STRUCTURE)
        {
            update_ok = rebuild_b(options, prepared_a, device_b, &dynamic_b,
                                  &update_stats);
        }
        else
        {
            const DmmaOwnedDeviceCsr &source =
                general_ab ? device_b : prepared_a.csr;
            update_ok = gpu_update_dynamic_b_values(
                source.values, source.nnz, &dynamic_b, &update_stats);
        }
        if (!update_ok)
        {
            std::fprintf(stderr, "B update failed at iteration %d.\n",
                         iteration + 1);
            goto cleanup;
        }

        DmmaSpGemmStats dmma_stats;
        if (!dmma_tilespgemm(prepared_a.tiles.view, dynamic_b.tiles.view,
                             &matrix_c, &dmma_stats))
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
        const double restore_ms = elapsed_ms(restore_begin, restore_end);
        if (!restored)
        {
            std::fprintf(stderr,
                         "Tile-to-CSR row restoration failed at iteration %d.\n",
                         iteration + 1);
            goto cleanup;
        }

        const double combined_ms = update_stats.total_ms + dmma_stats.total_ms;
        b_update_times.push_back(update_stats.total_ms);
        dmma_times.push_back(dmma_stats.total_ms);
        combined_times.push_back(combined_ms);
        export_times.push_back(dmma_stats.output_copy_ms);
        restore_times.push_back(restore_ms);

        std::printf("---------------- iteration %d/%d ----------------\n",
                    iteration + 1, options.iterations);
        print_b_update_stats("Iteration", update_stats);
        std::printf("candidate C tiles=%llu; exact non-empty C tiles=%d; "
                    "nnzC=%d\n",
                    dmma_stats.candidate_tiles, dmma_stats.output_tiles,
                    dmma_stats.output_nnz);
        std::printf("step1 candidate=%.3f ms; step2 exact-mask=%.3f ms; "
                    "step3 uniform DMMA=%.4f ms; allocation=%.3f ms\n",
                    dmma_stats.candidate_ms, dmma_stats.symbolic_ms,
                    dmma_stats.numeric_ms, dmma_stats.allocation_ms);
        const double numeric_gflops =
            dmma_stats.numeric_ms > 0.0
                ? 2.0 * static_cast<double>(nnz_cub) /
                      (dmma_stats.numeric_ms * 1.0e6)
                : 0.0;
        std::printf("CUDA  TileSpGEMM runtime is %.3f ms; numeric gflops=%.2f\n",
                    dmma_stats.total_ms, numeric_gflops);
        std::printf("iteration B-update+DMMA=%.3f ms; "
                    "C tile export D2H=%.3f ms; row restore/CSR=%.3f ms\n",
                    combined_ms, dmma_stats.output_copy_ms, restore_ms);

        if (iteration + 1 < options.iterations)
            destroy_output_matrix(&matrix_c);
    }

    std::printf("---------------- median over %d iterations ----------------\n",
                options.iterations);
    std::printf("median B-update=%.3f ms; DMMA=%.3f ms; "
                "B-update+DMMA=%.3f ms\n",
                median(b_update_times), median(dmma_times),
                median(combined_times));
    std::printf("median C tile export D2H=%.3f ms; "
                "row restore/CSR=%.3f ms\n",
                median(export_times), median(restore_times));

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
    destroy_output_matrix(&matrix_c);
    destroy_dynamic_b(&dynamic_b);
    destroy_device_csr(&device_b);
    destroy_prepared_a(&prepared_a);
    free_host_csr(&host_b);
    free_host_csr(&host_a);
    std::printf("---------------------------------------------------------------\n");
    return status;
}
