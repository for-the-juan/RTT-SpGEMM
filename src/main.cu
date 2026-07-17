#include "common.h"
#include "mmio_highlevel.h"
#include "gpu_dmma_tiles.h"
#include "tile2csr.h"
#include "spgemm_cu.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

static void print_usage(const char *program)
{
    std::printf(
        "Usage: %s -d <gpu> -aat <0|1> [--dense-threshold <1..32>] "
        "matrix.mtx\n",
        program);
}

static bool parse_arguments(int argc, char **argv, int *device, int *aat,
                            int *dense_threshold, const char **filename)
{
    *device = 0;
    *aat = 0;
    *dense_threshold = 24;
    *filename = nullptr;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "-d") == 0 && i + 1 < argc)
        {
            *device = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "-aat") == 0 && i + 1 < argc)
        {
            *aat = std::atoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--dense-threshold") == 0 &&
                 i + 1 < argc)
        {
            *dense_threshold = std::atoi(argv[++i]);
        }
        else if (argv[i][0] != '-' && *filename == nullptr)
        {
            *filename = argv[i];
        }
        else
        {
            return false;
        }
    }
    return *filename != nullptr && (*aat == 0 || *aat == 1) &&
           *dense_threshold >= 1 && *dense_threshold <= DMMA_INPUT_ELEMS;
}

static double elapsed_ms(const timeval &begin, const timeval &end)
{
    return (end.tv_sec - begin.tv_sec) * 1000.0 +
           (end.tv_usec - begin.tv_usec) / 1000.0;
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
    const double metadata_mb = metadata_bytes /
        (1024.0 * 1024.0);
    std::printf(
        "%s GPU hybrid tiles: total=%d dense=%d bitmask=%d payload=%.2f MB "
        "metadata=%.2f MB\n",
        name, static_cast<int>(matrix.num_tiles),
        static_cast<int>(matrix.dense_tiles),
        static_cast<int>(matrix.sparse_tiles), payload_mb, metadata_mb);
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

int main(int argc, char **argv)
{
    int device_id = 0;
    int aat = 0;
    int dense_threshold = 24;
    const char *filename = nullptr;
    if (!parse_arguments(argc, argv, &device_id, &aat, &dense_threshold,
                         &filename))
    {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (!dmma_cuda_ok(cudaSetDevice(device_id), "select GPU"))
        return EXIT_FAILURE;
    cudaDeviceProp device_properties{};
    if (!dmma_cuda_ok(cudaGetDeviceProperties(&device_properties, device_id),
                      "query GPU"))
        return EXIT_FAILURE;
    if (device_properties.major < 8)
    {
        std::fprintf(stderr,
                     "FP64 DMMA requires compute capability 8.0 or newer; "
                     "device %d is %d.%d.\n",
                     device_id, device_properties.major,
                     device_properties.minor);
        return EXIT_FAILURE;
    }

    const size_t persisting_l2 =
        static_cast<size_t>(device_properties.l2CacheSize * 0.8) <
                device_properties.persistingL2CacheMaxSize
            ? static_cast<size_t>(device_properties.l2CacheSize * 0.8)
            : device_properties.persistingL2CacheMaxSize;
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persisting_l2);

    std::printf("---------------------------------------------------------------\n");
    std::printf("Device [ %d ] %s @ %.2f MHz, compute capability %d.%d\n",
                device_id, device_properties.name,
                device_properties.clockRate * 1e-3,
                device_properties.major, device_properties.minor);
    std::printf("MAT: -------------- %s --------------\n", filename);
    std::printf("DMMA tiles: A=8x4, B=4x8, C=8x8; dense threshold=%d/32\n",
                dense_threshold);

    SMatrix matrix_a{};
    SMatrix matrix_c{};
    timeval begin{}, end{};
    gettimeofday(&begin, nullptr);
    const int read_status = mmio_allinone(
        &matrix_a.m, &matrix_a.n, &matrix_a.nnz, &matrix_a.isSymmetric,
        &matrix_a.rowpointer, &matrix_a.columnindex, &matrix_a.value,
        const_cast<char *>(filename));
    gettimeofday(&end, nullptr);
    if (read_status != 0)
    {
        std::fprintf(stderr, "Unable to read Matrix Market input (error %d).\n",
                     read_status);
        return EXIT_FAILURE;
    }
    std::printf("input matrix A: ( %d, %d ) nnz = %d\n",
                matrix_a.m, matrix_a.n, matrix_a.nnz);
    std::printf("loadfile time = %.5f sec\n", elapsed_ms(begin, end) / 1000.0);

    if (!aat && matrix_a.m != matrix_a.n)
    {
        std::fprintf(stderr,
                     "matrix squaring requires rowA == colA; use -aat 1 "
                     "for A*A^T.\n");
        std::free(matrix_a.rowpointer);
        std::free(matrix_a.columnindex);
        std::free(matrix_a.value);
        return EXIT_FAILURE;
    }

    /* Preserve the benchmark's existing deterministic value policy. */
    for (int i = 0; i < matrix_a.nnz; ++i)
        matrix_a.value[i] = static_cast<MAT_VAL_TYPE>(i % 10);

    DmmaPreparedOperands prepared;
    DmmaPreprocessStats preprocess;
    if (!gpu_prepare_dmma_operands(matrix_a, aat != 0, dense_threshold,
                                   &prepared, &preprocess))
    {
        std::fprintf(stderr, "GPU DMMA preprocessing failed.\n");
        destroy_prepared_operands(&prepared);
        std::free(matrix_a.rowpointer);
        std::free(matrix_a.columnindex);
        std::free(matrix_a.value);
        return EXIT_FAILURE;
    }
    const unsigned long long nnz_cub = prepared.nnz_cub;
    std::printf("SpGEMM nnzCub = %llu\n", nnz_cub);
    std::printf("GPU CSR H2D (one copy) = %.3f ms; validation = %.3f ms\n",
                preprocess.h2d_ms, preprocess.validation_ms);
    std::printf("GPU A key sort/reduce = %.3f ms; tile build = %.3f ms\n",
                preprocess.a_key_sort_reduce_ms,
                preprocess.a_tile_build_ms);
    std::printf("GPU B key sort/reduce = %.3f ms; tile build = %.3f ms; "
                "CSC = %.3f ms\n",
                preprocess.b_key_sort_reduce_ms,
                preprocess.b_tile_build_ms, preprocess.b_csc_ms);
    std::printf("GPU nnzCub = %.3f ms\n", preprocess.nnz_cub_ms);
    std::printf("GPU ready-on-device preprocessing = %.3f ms; "
                "estimated explicit peak workspace = %.2f MB\n",
                preprocess.total_ms,
                static_cast<double>(preprocess.peak_workspace_bytes) /
                    (1024.0 * 1024.0));
    print_device_tile_stats("A", prepared.a.view);
    print_device_tile_stats("B", prepared.b.view);

    DmmaSpGemmStats stats;
    if (!dmma_tilespgemm(prepared.a.view, prepared.b.view, &matrix_c, &stats))
    {
        std::fprintf(stderr, "DMMA TileSpGEMM failed.\n");
        destroy_prepared_operands(&prepared);
        std::free(matrix_a.rowpointer);
        std::free(matrix_a.columnindex);
        std::free(matrix_a.value);
        return EXIT_FAILURE;
    }
    const double numeric_gflops =
        stats.numeric_ms > 0.0
            ? 2.0 * static_cast<double>(nnz_cub) /
                  (stats.numeric_ms * 1.0e6)
            : 0.0;
    const double total_gflops =
        stats.total_ms > 0.0
            ? 2.0 * static_cast<double>(nnz_cub) /
                  (stats.total_ms * 1.0e6)
            : 0.0;
    const double input_to_c_ready_ms = preprocess.total_ms + stats.total_ms;
    const double input_to_c_ready_gflops =
        input_to_c_ready_ms > 0.0
            ? 2.0 * static_cast<double>(nnz_cub) /
                  (input_to_c_ready_ms * 1.0e6)
            : 0.0;
    const double pipeline_ms = input_to_c_ready_ms + stats.output_copy_ms;
    const double pipeline_gflops =
        pipeline_ms > 0.0
            ? 2.0 * static_cast<double>(nnz_cub) /
                  (pipeline_ms * 1.0e6)
            : 0.0;
    std::printf("candidate C tiles = %d; exact non-empty C tiles = %d\n",
                stats.candidate_tiles, stats.output_tiles);
    std::printf("Non-empty tiles of C = %d\n", stats.output_tiles);
    std::printf("step1 candidate discovery = %.2f ms\n", stats.candidate_ms);
    std::printf("step2 exact 8x8 masks/compaction = %.2f ms\n",
                stats.symbolic_ms);
    std::printf("step3 uniform FP64 DMMA = %.4f ms, gflops = %.2f\n",
                stats.numeric_ms, numeric_gflops);
    std::printf("step3 C payload allocation = %.2f ms\n",
                stats.allocation_ms);
    std::printf("nnzC = %d\n", stats.output_nnz);
    std::printf("CUDA  TileSpGEMM runtime is %.2f ms, gflops = %.2f\n",
                stats.total_ms, total_gflops);
    std::printf("GPU input-to-C-ready runtime is %.3f ms, gflops = %.2f\n",
                input_to_c_ready_ms, input_to_c_ready_gflops);
    std::printf("C output D2H (export/validation, excluded above) = %.3f ms\n",
                stats.output_copy_ms);
    std::printf("GPU full pipeline through host tile output = %.3f ms, "
                "gflops = %.2f\n",
                pipeline_ms, pipeline_gflops);

    int validation_status = 0;
#if CHECK_RESULT
    std::printf("-------------------------------check----------------------------------------\n");
    if (!tile2csr(&matrix_c))
    {
        std::fprintf(stderr, "Tile-to-CSR conversion failed.\n");
        validation_status = -1;
    }
    else
    {
        std::printf("tile to CSR conversion complete!\n");
        unsigned long long cusparse_nnz_c = 0;
        double cusparse_compression = 0.0;
        double cusparse_time = 0.0;
        double cusparse_gflops = 0.0;
        validation_status = spgemm_cu_device(
            prepared.csr.rows, prepared.csr.cols, prepared.csr.nnz,
            prepared.csr.row_ptr, prepared.csr.col_idx, prepared.csr.values,
            aat != 0, matrix_c.m, matrix_c.n, matrix_c.nnz,
            matrix_c.rowpointer, matrix_c.columnindex, true, nnz_cub,
            &cusparse_nnz_c, &cusparse_compression, &cusparse_time,
            &cusparse_gflops);
    }
#endif

    destroy_prepared_operands(&prepared);
    destroy_output_matrix(&matrix_c);
    std::free(matrix_a.rowpointer);
    std::free(matrix_a.columnindex);
    std::free(matrix_a.value);
    std::printf("---------------------------------------------------------------\n");
    return validation_status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
