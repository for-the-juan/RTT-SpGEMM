#ifndef RTT_SPGEMM_CUSPARSE_CORE_BENCHMARK_H_
#define RTT_SPGEMM_CUSPARSE_CORE_BENCHMARK_H_

/*
 * Device-resident cuSPARSE SpGEMM Core benchmark.
 *
 * Timing boundary for every sample:
 *   start: A and the original-B values are device resident, the native input
 *          CSR layouts/descriptors are ready, and C owns no storage.  When a
 *          DeviceValueUpdate is configured, the first timed operation gathers
 *          those original-B values into the prepared B layout.
 *   stop:  the optional B-value gather, workEstimation, compute, native CSR
 *          output allocation, copy, and stream synchronization have completed;
 *          C is ready in cuSPARSE's native device-CSR output format.
 *
 * Input upload, input/handle/descriptor creation, validation downloads, and
 * post-sample cleanup are deliberately outside the measured interval.  The
 * final timed C is retained in BenchmarkResult::output for validation.
 *
 * BenchmarkConfig::reuse_scratch_workspace enables the publication
 * steady-state protocol.  One extra, unreported preparation pass queries and
 * allocates the work-estimation and compute scratch buffers.  Every reported
 * warmup/timed sample still creates fresh C/SpGEMM descriptors and executes
 * the complete B-update, workEstimation, compute, C allocation, and copy
 * sequence, but it reuses those two caller-owned scratch buffers.  A later
 * query that exceeds either frozen capacity is a hard failure; scratch never
 * grows or frees inside a reported Core sample.
 *
 * This header targets the CUDA 11.8 generic SpGEMM API.  It accepts only
 * zero-based FP64 CSR matrices with int32 row offsets and column indices.
 */

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace rtt_spgemm {
namespace cusparse_core {

struct Status {
    enum Code {
        kSuccess = 0,
        kInvalidArgument,
        kCudaError,
        kCusparseError
    };

    Code code;
    cudaError_t cuda_status;
    cusparseStatus_t cusparse_status;
    const char *stage;

    Status()
        : code(kSuccess), cuda_status(cudaSuccess),
          cusparse_status(CUSPARSE_STATUS_SUCCESS), stage("success") {}

    bool ok() const { return code == kSuccess; }

    static Status Invalid(const char *where) {
        Status status;
        status.code = kInvalidArgument;
        status.stage = where;
        return status;
    }

    static Status Cuda(cudaError_t error, const char *where) {
        Status status;
        status.code = kCudaError;
        status.cuda_status = error;
        status.stage = where;
        return status;
    }

    static Status Cusparse(cusparseStatus_t error, const char *where) {
        Status status;
        status.code = kCusparseError;
        status.cusparse_status = error;
        status.stage = where;
        return status;
    }
};

struct DeviceCsrView {
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    const int *row_offsets;
    const int *column_indices;
    const double *values;

    DeviceCsrView()
        : rows(0), cols(0), nnz(0), row_offsets(NULL),
          column_indices(NULL), values(NULL) {}

    DeviceCsrView(int64_t rows_in, int64_t cols_in, int64_t nnz_in,
                  const int *row_offsets_in,
                  const int *column_indices_in,
                  const double *values_in)
        : rows(rows_in), cols(cols_in), nnz(nnz_in),
          row_offsets(row_offsets_in), column_indices(column_indices_in),
          values(values_in) {}
};

/*
 * Optional online B-values update used by the same-order control.  The
 * destination CSR structure and destination_to_source mapping are prepared
 * outside Core.  Every sample executes
 *
 *   destination_values[p] = source_values[destination_to_source[p]].
 *
 * The source is the device-resident original-B CSR values array.  Keeping the
 * mapping destination-indexed makes the update a coalesced write into P_k B.
 */
struct DeviceValueUpdate {
    bool active;
    const double *source_values;
    double *destination_values;
    const int *destination_to_source;
    int64_t count;

    DeviceValueUpdate()
        : active(false), source_values(NULL), destination_values(NULL),
          destination_to_source(NULL), count(0) {}

    bool enabled() const { return active; }
};

/* Move-only owner for the retained device CSR C. */
class DeviceCsrOutput {
public:
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    int *row_offsets;
    int *column_indices;
    double *values;

    DeviceCsrOutput()
        : rows(0), cols(0), nnz(0), row_offsets(NULL),
          column_indices(NULL), values(NULL) {}

    ~DeviceCsrOutput() { reset(); }

    DeviceCsrOutput(const DeviceCsrOutput &) = delete;
    DeviceCsrOutput &operator=(const DeviceCsrOutput &) = delete;

    DeviceCsrOutput(DeviceCsrOutput &&other) noexcept
        : rows(other.rows), cols(other.cols), nnz(other.nnz),
          row_offsets(other.row_offsets),
          column_indices(other.column_indices), values(other.values) {
        other.disarm();
    }

    DeviceCsrOutput &operator=(DeviceCsrOutput &&other) noexcept {
        if (this != &other) {
            reset();
            rows = other.rows;
            cols = other.cols;
            nnz = other.nnz;
            row_offsets = other.row_offsets;
            column_indices = other.column_indices;
            values = other.values;
            other.disarm();
        }
        return *this;
    }

    void reset() {
        if (row_offsets != NULL)
            (void)cudaFree(row_offsets);
        if (column_indices != NULL)
            (void)cudaFree(column_indices);
        if (values != NULL)
            (void)cudaFree(values);
        disarm();
    }

    DeviceCsrView view() const {
        return DeviceCsrView(rows, cols, nnz, row_offsets, column_indices,
                             values);
    }

private:
    void disarm() {
        rows = 0;
        cols = 0;
        nnz = 0;
        row_offsets = NULL;
        column_indices = NULL;
        values = NULL;
    }
};

struct BenchmarkConfig {
    int warmup_iterations;
    int timed_iterations;
    cusparseSpGEMMAlg_t algorithm;
    DeviceValueUpdate b_value_update;
    bool reuse_scratch_workspace;

    BenchmarkConfig()
        : warmup_iterations(2), timed_iterations(10),
          algorithm(CUSPARSE_SPGEMM_DEFAULT),
          reuse_scratch_workspace(false) {}
};

/* Host-wall decomposition of one sample.  API-call fields measure the wall
 * time spent in that host API region.  Because cuSPARSE work is asynchronous,
 * final_sync_wall_ms intentionally absorbs device work not completed by an
 * earlier synchronous allocation/API call.  Their sum is diagnostic rather
 * than an alternative Core stopwatch. */
struct IterationStageStats {
    double b_update_submit_wall_ms;
    double c_row_allocation_wall_ms;
    double work_estimation_query_wall_ms;
    double work_scratch_allocation_wall_ms;
    double work_estimation_execute_wall_ms;
    double compute_query_wall_ms;
    double compute_scratch_allocation_wall_ms;
    double compute_execute_wall_ms;
    double c_size_query_wall_ms;
    double c_payload_allocation_wall_ms;
    double copy_submit_wall_ms;
    double final_sync_wall_ms;
    size_t memory_free_before_bytes;
    size_t memory_total_before_bytes;
    size_t memory_free_after_bytes;
    size_t memory_total_after_bytes;
    size_t required_work_estimation_bytes;
    size_t required_compute_bytes;
    bool scratch_pointer_stable;
    bool scratch_allocated_in_timed;

    IterationStageStats()
        : b_update_submit_wall_ms(0.0), c_row_allocation_wall_ms(0.0),
          work_estimation_query_wall_ms(0.0),
          work_scratch_allocation_wall_ms(0.0),
          work_estimation_execute_wall_ms(0.0), compute_query_wall_ms(0.0),
          compute_scratch_allocation_wall_ms(0.0),
          compute_execute_wall_ms(0.0), c_size_query_wall_ms(0.0),
          c_payload_allocation_wall_ms(0.0), copy_submit_wall_ms(0.0),
          final_sync_wall_ms(0.0), memory_free_before_bytes(0),
          memory_total_before_bytes(0), memory_free_after_bytes(0),
          memory_total_after_bytes(0), required_work_estimation_bytes(0),
          required_compute_bytes(0), scratch_pointer_stable(true),
          scratch_allocated_in_timed(false) {}
};

struct BenchmarkResult {
    /* Warmup samples are retained for diagnostics but not summarized. */
    std::vector<double> warmup_samples_ms;
    std::vector<double> samples_ms;
    std::vector<double> warmup_b_update_samples_ms;
    std::vector<double> b_update_samples_ms;
    std::vector<IterationStageStats> warmup_stage_samples;
    std::vector<IterationStageStats> stage_samples;
    double min_ms;
    double median_ms;
    double max_ms;
    size_t final_work_estimation_buffer_bytes;
    size_t final_compute_buffer_bytes;
    bool reusable_workspace_used;
    size_t reusable_work_estimation_capacity_bytes;
    size_t reusable_compute_capacity_bytes;
    double reusable_workspace_prepare_wall_ms;
    size_t reusable_memory_free_before_bytes;
    size_t reusable_memory_total_before_bytes;
    size_t reusable_memory_free_after_bytes;
    size_t reusable_memory_total_after_bytes;
    bool all_scratch_pointers_stable;
    bool any_scratch_allocated_in_timed;
    DeviceCsrOutput output;

    BenchmarkResult()
        : min_ms(0.0), median_ms(0.0), max_ms(0.0),
          final_work_estimation_buffer_bytes(0),
          final_compute_buffer_bytes(0), reusable_workspace_used(false),
          reusable_work_estimation_capacity_bytes(0),
          reusable_compute_capacity_bytes(0),
          reusable_workspace_prepare_wall_ms(0.0),
          reusable_memory_free_before_bytes(0),
          reusable_memory_total_before_bytes(0),
          reusable_memory_free_after_bytes(0),
          reusable_memory_total_after_bytes(0),
          all_scratch_pointers_stable(true),
          any_scratch_allocated_in_timed(false) {}

    BenchmarkResult(const BenchmarkResult &) = delete;
    BenchmarkResult &operator=(const BenchmarkResult &) = delete;
    BenchmarkResult(BenchmarkResult &&) noexcept = default;
    BenchmarkResult &operator=(BenchmarkResult &&) noexcept = default;

    void reset() {
        warmup_samples_ms.clear();
        samples_ms.clear();
        warmup_b_update_samples_ms.clear();
        b_update_samples_ms.clear();
        warmup_stage_samples.clear();
        stage_samples.clear();
        min_ms = 0.0;
        median_ms = 0.0;
        max_ms = 0.0;
        final_work_estimation_buffer_bytes = 0;
        final_compute_buffer_bytes = 0;
        reusable_workspace_used = false;
        reusable_work_estimation_capacity_bytes = 0;
        reusable_compute_capacity_bytes = 0;
        reusable_workspace_prepare_wall_ms = 0.0;
        reusable_memory_free_before_bytes = 0;
        reusable_memory_total_before_bytes = 0;
        reusable_memory_free_after_bytes = 0;
        reusable_memory_total_after_bytes = 0;
        all_scratch_pointers_stable = true;
        any_scratch_allocated_in_timed = false;
        output.reset();
    }
};

namespace detail {

struct CsrMetadata {
    int64_t rows;
    int64_t cols;
    int64_t nnz;
    void *row_offsets;
    void *column_indices;
    void *values;
    cusparseIndexType_t row_offset_type;
    cusparseIndexType_t column_index_type;
    cusparseIndexBase_t index_base;
    cudaDataType value_type;

    CsrMetadata()
        : rows(0), cols(0), nnz(0), row_offsets(NULL),
          column_indices(NULL), values(NULL),
          row_offset_type(CUSPARSE_INDEX_32I),
          column_index_type(CUSPARSE_INDEX_32I),
          index_base(CUSPARSE_INDEX_BASE_ZERO), value_type(CUDA_R_64F) {}
};

inline Status InspectCsr(cusparseSpMatDescr_t matrix, CsrMetadata *metadata,
                         const char *stage) {
    if (matrix == NULL || metadata == NULL)
        return Status::Invalid(stage);

    const cusparseStatus_t error = cusparseCsrGet(
        matrix, &metadata->rows, &metadata->cols, &metadata->nnz,
        &metadata->row_offsets, &metadata->column_indices, &metadata->values,
        &metadata->row_offset_type, &metadata->column_index_type,
        &metadata->index_base, &metadata->value_type);
    if (error != CUSPARSE_STATUS_SUCCESS)
        return Status::Cusparse(error, stage);

    if (metadata->rows < 0 || metadata->cols < 0 || metadata->nnz < 0 ||
        metadata->rows > std::numeric_limits<int>::max() ||
        metadata->cols > std::numeric_limits<int>::max() ||
        metadata->nnz > std::numeric_limits<int>::max() ||
        metadata->row_offsets == NULL ||
        (metadata->nnz > 0 &&
         (metadata->column_indices == NULL || metadata->values == NULL)) ||
        metadata->row_offset_type != CUSPARSE_INDEX_32I ||
        metadata->column_index_type != CUSPARSE_INDEX_32I ||
        metadata->index_base != CUSPARSE_INDEX_BASE_ZERO ||
        metadata->value_type != CUDA_R_64F)
        return Status::Invalid(stage);

    return Status();
}

struct IterationState {
    cusparseSpMatDescr_t matrix_c;
    cusparseSpGEMMDescr_t spgemm_descriptor;
    void *work_estimation_buffer;
    void *compute_buffer;
    size_t work_estimation_buffer_bytes;
    size_t compute_buffer_bytes;
    cudaEvent_t b_update_begin;
    cudaEvent_t b_update_end;
    DeviceCsrOutput output;

    IterationState()
        : matrix_c(NULL), spgemm_descriptor(NULL),
          work_estimation_buffer(NULL), compute_buffer(NULL),
          work_estimation_buffer_bytes(0), compute_buffer_bytes(0),
          b_update_begin(NULL), b_update_end(NULL) {}

    ~IterationState() { cleanup(); }

    IterationState(const IterationState &) = delete;
    IterationState &operator=(const IterationState &) = delete;

    void cleanup() {
        if (work_estimation_buffer != NULL)
            (void)cudaFree(work_estimation_buffer);
        if (compute_buffer != NULL)
            (void)cudaFree(compute_buffer);
        if (spgemm_descriptor != NULL)
            (void)cusparseSpGEMM_destroyDescr(spgemm_descriptor);
        if (matrix_c != NULL)
            (void)cusparseDestroySpMat(matrix_c);
        if (b_update_end != NULL)
            (void)cudaEventDestroy(b_update_end);
        if (b_update_begin != NULL)
            (void)cudaEventDestroy(b_update_begin);
        work_estimation_buffer = NULL;
        compute_buffer = NULL;
        spgemm_descriptor = NULL;
        matrix_c = NULL;
        b_update_begin = NULL;
        b_update_end = NULL;
    }
};

/* Exclusive, caller-owned scratch for sequential SpGEMM samples.  cuSPARSE's
 * externalBuffer arguments are ordinary caller storage; reuse is safe here
 * because samples share one stream, are presynchronized, and never overlap.
 * Input/output descriptors remain per-sample objects. */
struct ReusableWorkspace {
    void *work_estimation_buffer;
    void *compute_buffer;
    size_t work_estimation_capacity_bytes;
    size_t compute_capacity_bytes;
    const void *frozen_work_estimation_pointer;
    const void *frozen_compute_pointer;

    ReusableWorkspace()
        : work_estimation_buffer(NULL), compute_buffer(NULL),
          work_estimation_capacity_bytes(0), compute_capacity_bytes(0),
          frozen_work_estimation_pointer(NULL),
          frozen_compute_pointer(NULL) {}

    ~ReusableWorkspace() { reset(); }

    ReusableWorkspace(const ReusableWorkspace &) = delete;
    ReusableWorkspace &operator=(const ReusableWorkspace &) = delete;

    void reset() {
        if (compute_buffer != NULL)
            (void)cudaFree(compute_buffer);
        if (work_estimation_buffer != NULL)
            (void)cudaFree(work_estimation_buffer);
        work_estimation_buffer = NULL;
        compute_buffer = NULL;
        work_estimation_capacity_bytes = 0;
        compute_capacity_bytes = 0;
        frozen_work_estimation_pointer = NULL;
        frozen_compute_pointer = NULL;
    }

    bool pointers_stable() const {
        return work_estimation_buffer == frozen_work_estimation_pointer &&
               compute_buffer == frozen_compute_pointer;
    }
};

inline bool ReusableCapacityFits(size_t required, size_t capacity) {
    return required <= capacity;
}

inline double WallMilliseconds(
    const std::chrono::steady_clock::time_point &begin,
    const std::chrono::steady_clock::time_point &end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

/* Descriptor creation belongs to setup and is intentionally untimed. */
inline Status PrepareIteration(int64_t rows, int64_t cols,
                               bool measure_b_update,
                               IterationState *state) {
    if (state == NULL)
        return Status::Invalid("prepare iteration arguments");

    cusparseStatus_t sparse_error = cusparseCreateCsr(
        &state->matrix_c, rows, cols, 0, NULL, NULL, NULL,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    if (sparse_error != CUSPARSE_STATUS_SUCCESS)
        return Status::Cusparse(sparse_error,
                                "create untimed C descriptor");

    sparse_error = cusparseSpGEMM_createDescr(&state->spgemm_descriptor);
    if (sparse_error != CUSPARSE_STATUS_SUCCESS)
        return Status::Cusparse(sparse_error,
                                "create untimed SpGEMM descriptor");
    if (measure_b_update) {
        cudaError_t cuda_error = cudaEventCreate(&state->b_update_begin);
        if (cuda_error == cudaSuccess)
            cuda_error = cudaEventCreate(&state->b_update_end);
        if (cuda_error != cudaSuccess)
            return Status::Cuda(cuda_error,
                                "create untimed B-update timing events");
    }
    return Status();
}

static __global__ void GatherBValuesKernel(
    const double *source_values, const int *destination_to_source,
    int count, double *destination_values) {
    for (int destination = blockIdx.x * blockDim.x + threadIdx.x;
         destination < count;
         destination += blockDim.x * gridDim.x) {
        destination_values[destination] =
            source_values[destination_to_source[destination]];
    }
}

/* Query and allocate reusable scratch once, outside every reported sample.
 * This executes workEstimation because the CUDA 11.8 API requires that state
 * before the compute-buffer query.  It deliberately does not execute compute
 * or copy and its temporary C descriptor/row pointer are destroyed on return. */
inline Status PrepareReusableWorkspace(
    cusparseHandle_t handle, cusparseSpMatDescr_t matrix_a,
    cusparseSpMatDescr_t matrix_b, const CsrMetadata &metadata_a,
    const CsrMetadata &metadata_b, const BenchmarkConfig &config,
    cudaStream_t stream, ReusableWorkspace *workspace,
    BenchmarkResult *result) {
    if (workspace == NULL || result == NULL)
        return Status::Invalid("prepare reusable workspace arguments");

    cudaError_t cuda_error = cudaStreamSynchronize(stream);
    if (cuda_error != cudaSuccess)
        return Status::Cuda(cuda_error,
                            "presynchronize reusable workspace preparation");
    cuda_error = cudaMemGetInfo(&result->reusable_memory_free_before_bytes,
                                &result->reusable_memory_total_before_bytes);
    if (cuda_error != cudaSuccess)
        return Status::Cuda(cuda_error,
                            "read memory before reusable workspace preparation");

    const std::chrono::steady_clock::time_point preparation_begin =
        std::chrono::steady_clock::now();
    IterationState state;
    Status status = PrepareIteration(metadata_a.rows, metadata_b.cols, false,
                                     &state);
    if (!status.ok())
        return status;

    state.output.rows = metadata_a.rows;
    state.output.cols = metadata_b.cols;
    cuda_error = cudaMalloc(
        reinterpret_cast<void **>(&state.output.row_offsets),
        static_cast<size_t>(metadata_a.rows + 1) * sizeof(int));
    if (cuda_error != cudaSuccess)
        return Status::Cuda(cuda_error,
                            "allocate preparation C row offsets");
    cusparseStatus_t sparse_error = cusparseCsrSetPointers(
        state.matrix_c, state.output.row_offsets, NULL, NULL);
    if (sparse_error != CUSPARSE_STATUS_SUCCESS)
        return Status::Cusparse(sparse_error,
                                "set preparation C row-offset pointer");

    const double alpha = 1.0;
    const double beta = 0.0;
    const cusparseOperation_t operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
    size_t required_work_estimation_bytes = 0;
    sparse_error = cusparseSpGEMM_workEstimation(
        handle, operation, operation, &alpha, matrix_a, matrix_b, &beta,
        state.matrix_c, CUDA_R_64F, config.algorithm,
        state.spgemm_descriptor, &required_work_estimation_bytes, NULL);
    if (sparse_error != CUSPARSE_STATUS_SUCCESS)
        return Status::Cusparse(
            sparse_error, "query reusable work-estimation capacity");
    if (required_work_estimation_bytes != 0) {
        cuda_error = cudaMalloc(&workspace->work_estimation_buffer,
                                required_work_estimation_bytes);
        if (cuda_error != cudaSuccess)
            return Status::Cuda(
                cuda_error, "allocate reusable work-estimation scratch");
    }
    workspace->work_estimation_capacity_bytes =
        required_work_estimation_bytes;
    sparse_error = cusparseSpGEMM_workEstimation(
        handle, operation, operation, &alpha, matrix_a, matrix_b, &beta,
        state.matrix_c, CUDA_R_64F, config.algorithm,
        state.spgemm_descriptor, &required_work_estimation_bytes,
        workspace->work_estimation_buffer);
    if (sparse_error != CUSPARSE_STATUS_SUCCESS)
        return Status::Cusparse(
            sparse_error, "execute reusable workspace preparation");
    if (!ReusableCapacityFits(
            required_work_estimation_bytes,
            workspace->work_estimation_capacity_bytes))
        return Status::Invalid(
            "work-estimation capacity grew during preparation");

    size_t required_compute_bytes = 0;
    sparse_error = cusparseSpGEMM_compute(
        handle, operation, operation, &alpha, matrix_a, matrix_b, &beta,
        state.matrix_c, CUDA_R_64F, config.algorithm,
        state.spgemm_descriptor, &required_compute_bytes, NULL);
    if (sparse_error != CUSPARSE_STATUS_SUCCESS)
        return Status::Cusparse(sparse_error,
                                "query reusable compute capacity");
    if (required_compute_bytes != 0) {
        cuda_error = cudaMalloc(&workspace->compute_buffer,
                                required_compute_bytes);
        if (cuda_error != cudaSuccess)
            return Status::Cuda(cuda_error,
                                "allocate reusable compute scratch");
    }
    workspace->compute_capacity_bytes = required_compute_bytes;
    workspace->frozen_work_estimation_pointer =
        workspace->work_estimation_buffer;
    workspace->frozen_compute_pointer = workspace->compute_buffer;

    cuda_error = cudaStreamSynchronize(stream);
    if (cuda_error != cudaSuccess)
        return Status::Cuda(cuda_error,
                            "synchronize reusable workspace preparation");
    if (!workspace->pointers_stable())
        return Status::Invalid(
            "reusable scratch pointer changed during preparation");
    /* Leave the memory-after snapshot with only the persistent scratch live,
     * rather than also counting the temporary preparation C row pointer and
     * descriptors.  cleanup is idempotent with IterationState's destructor. */
    state.cleanup();
    result->reusable_workspace_prepare_wall_ms = WallMilliseconds(
        preparation_begin, std::chrono::steady_clock::now());
    cuda_error = cudaMemGetInfo(&result->reusable_memory_free_after_bytes,
                                &result->reusable_memory_total_after_bytes);
    if (cuda_error != cudaSuccess)
        return Status::Cuda(cuda_error,
                            "read memory after reusable workspace preparation");
    result->reusable_workspace_used = true;
    result->reusable_work_estimation_capacity_bytes =
        workspace->work_estimation_capacity_bytes;
    result->reusable_compute_capacity_bytes =
        workspace->compute_capacity_bytes;
    return Status();
}

inline Status RunIteration(cusparseHandle_t handle,
                           cusparseSpMatDescr_t matrix_a,
                           cusparseSpMatDescr_t matrix_b,
                           const CsrMetadata &metadata_a,
                           const CsrMetadata &metadata_b,
                           const BenchmarkConfig &config,
                           cudaStream_t stream,
                           ReusableWorkspace *reusable_workspace,
                           IterationState *state,
                           double *elapsed_ms,
                           double *b_update_elapsed_ms,
                           IterationStageStats *stage_stats) {
    if (state == NULL || elapsed_ms == NULL || b_update_elapsed_ms == NULL ||
        stage_stats == NULL)
        return Status::Invalid("run iteration arguments");

    *b_update_elapsed_ms = 0.0;

    cudaError_t cuda_error = cudaStreamSynchronize(stream);
    if (cuda_error != cudaSuccess)
        return Status::Cuda(cuda_error, "pre-sample stream synchronization");
    cuda_error = cudaMemGetInfo(&stage_stats->memory_free_before_bytes,
                                &stage_stats->memory_total_before_bytes);
    if (cuda_error != cudaSuccess)
        return Status::Cuda(cuda_error,
                            "read memory before cuSPARSE Core sample");
    if (config.reuse_scratch_workspace != (reusable_workspace != NULL))
        return Status::Invalid("reusable workspace policy mismatch");
    stage_stats->scratch_pointer_stable =
        reusable_workspace == NULL || reusable_workspace->pointers_stable();
    if (!stage_stats->scratch_pointer_stable)
        return Status::Invalid("reusable scratch pointer changed before sample");

    const std::chrono::steady_clock::time_point start =
        std::chrono::steady_clock::now();

#define RTT_CUSPARSE_CORE_CUDA(call, where)                                  \
    do {                                                                     \
        const cudaError_t local_error = (call);                              \
        if (local_error != cudaSuccess)                                      \
            return Status::Cuda(local_error, where);                         \
    } while (0)

#define RTT_CUSPARSE_CORE_SPARSE(call, where)                                \
    do {                                                                     \
        const cusparseStatus_t local_error = (call);                         \
        if (local_error != CUSPARSE_STATUS_SUCCESS)                          \
            return Status::Cusparse(local_error, where);                     \
    } while (0)

    std::chrono::steady_clock::time_point stage_begin =
        std::chrono::steady_clock::now();
    if (config.b_value_update.enabled()) {
        RTT_CUSPARSE_CORE_CUDA(
            cudaEventRecord(state->b_update_begin, stream),
            "record B-values update begin");
        if (config.b_value_update.count > 0) {
            const int threads = 256;
            const int blocks = static_cast<int>(std::min<int64_t>(
                65535, (config.b_value_update.count + threads - 1) /
                           threads));
            GatherBValuesKernel<<<blocks, threads, 0, stream>>>(
                config.b_value_update.source_values,
                config.b_value_update.destination_to_source,
                static_cast<int>(config.b_value_update.count),
                config.b_value_update.destination_values);
            RTT_CUSPARSE_CORE_CUDA(cudaGetLastError(),
                                   "launch B-values update");
        }
        RTT_CUSPARSE_CORE_CUDA(
            cudaEventRecord(state->b_update_end, stream),
            "record B-values update end");
    }
    stage_stats->b_update_submit_wall_ms = WallMilliseconds(
        stage_begin, std::chrono::steady_clock::now());

    state->output.rows = metadata_a.rows;
    state->output.cols = metadata_b.cols;
    stage_begin = std::chrono::steady_clock::now();
    RTT_CUSPARSE_CORE_CUDA(
        cudaMalloc(reinterpret_cast<void **>(&state->output.row_offsets),
                   static_cast<size_t>(metadata_a.rows + 1) * sizeof(int)),
        "allocate C row offsets");
    RTT_CUSPARSE_CORE_SPARSE(
        cusparseCsrSetPointers(state->matrix_c, state->output.row_offsets,
                              NULL, NULL),
        "set initial C row-offset pointer");
    stage_stats->c_row_allocation_wall_ms = WallMilliseconds(
        stage_begin, std::chrono::steady_clock::now());

    const double alpha = 1.0;
    const double beta = 0.0;
    const cusparseOperation_t operation = CUSPARSE_OPERATION_NON_TRANSPOSE;

    stage_begin = std::chrono::steady_clock::now();
    RTT_CUSPARSE_CORE_SPARSE(
        cusparseSpGEMM_workEstimation(
            handle, operation, operation, &alpha, matrix_a, matrix_b, &beta,
            state->matrix_c, CUDA_R_64F, config.algorithm,
            state->spgemm_descriptor,
            &state->work_estimation_buffer_bytes, NULL),
        "query work-estimation buffer size");
    stage_stats->work_estimation_query_wall_ms = WallMilliseconds(
        stage_begin, std::chrono::steady_clock::now());
    stage_stats->required_work_estimation_bytes =
        state->work_estimation_buffer_bytes;
    void *work_estimation_buffer = NULL;
    stage_begin = std::chrono::steady_clock::now();
    if (reusable_workspace != NULL) {
        if (!ReusableCapacityFits(
                state->work_estimation_buffer_bytes,
                reusable_workspace->work_estimation_capacity_bytes))
            return Status::Invalid(
                "reusable work-estimation capacity exceeded");
        work_estimation_buffer = reusable_workspace->work_estimation_buffer;
    } else if (state->work_estimation_buffer_bytes != 0) {
        RTT_CUSPARSE_CORE_CUDA(
            cudaMalloc(&state->work_estimation_buffer,
                       state->work_estimation_buffer_bytes),
            "allocate work-estimation buffer");
        work_estimation_buffer = state->work_estimation_buffer;
        stage_stats->scratch_allocated_in_timed = true;
    }
    stage_stats->work_scratch_allocation_wall_ms = WallMilliseconds(
        stage_begin, std::chrono::steady_clock::now());
    stage_begin = std::chrono::steady_clock::now();
    RTT_CUSPARSE_CORE_SPARSE(
        cusparseSpGEMM_workEstimation(
            handle, operation, operation, &alpha, matrix_a, matrix_b, &beta,
            state->matrix_c, CUDA_R_64F, config.algorithm,
            state->spgemm_descriptor,
            &state->work_estimation_buffer_bytes,
            work_estimation_buffer),
        "execute work estimation");
    stage_stats->work_estimation_execute_wall_ms = WallMilliseconds(
        stage_begin, std::chrono::steady_clock::now());
    if (reusable_workspace != NULL &&
        !ReusableCapacityFits(
            state->work_estimation_buffer_bytes,
            reusable_workspace->work_estimation_capacity_bytes))
        return Status::Invalid(
            "reusable work-estimation capacity grew during execute");

    stage_begin = std::chrono::steady_clock::now();
    RTT_CUSPARSE_CORE_SPARSE(
        cusparseSpGEMM_compute(
            handle, operation, operation, &alpha, matrix_a, matrix_b, &beta,
            state->matrix_c, CUDA_R_64F, config.algorithm,
            state->spgemm_descriptor, &state->compute_buffer_bytes, NULL),
        "query compute buffer size");
    stage_stats->compute_query_wall_ms = WallMilliseconds(
        stage_begin, std::chrono::steady_clock::now());
    stage_stats->required_compute_bytes = state->compute_buffer_bytes;
    void *compute_buffer = NULL;
    stage_begin = std::chrono::steady_clock::now();
    if (reusable_workspace != NULL) {
        if (!ReusableCapacityFits(
                state->compute_buffer_bytes,
                reusable_workspace->compute_capacity_bytes))
            return Status::Invalid("reusable compute capacity exceeded");
        compute_buffer = reusable_workspace->compute_buffer;
    } else if (state->compute_buffer_bytes != 0) {
        RTT_CUSPARSE_CORE_CUDA(
            cudaMalloc(&state->compute_buffer, state->compute_buffer_bytes),
            "allocate compute buffer");
        compute_buffer = state->compute_buffer;
        stage_stats->scratch_allocated_in_timed = true;
    }
    stage_stats->compute_scratch_allocation_wall_ms = WallMilliseconds(
        stage_begin, std::chrono::steady_clock::now());
    stage_begin = std::chrono::steady_clock::now();
    RTT_CUSPARSE_CORE_SPARSE(
        cusparseSpGEMM_compute(
            handle, operation, operation, &alpha, matrix_a, matrix_b, &beta,
            state->matrix_c, CUDA_R_64F, config.algorithm,
            state->spgemm_descriptor, &state->compute_buffer_bytes,
            compute_buffer),
        "execute symbolic and numeric compute");
    stage_stats->compute_execute_wall_ms = WallMilliseconds(
        stage_begin, std::chrono::steady_clock::now());
    if (reusable_workspace != NULL &&
        !ReusableCapacityFits(state->compute_buffer_bytes,
                              reusable_workspace->compute_capacity_bytes))
        return Status::Invalid(
            "reusable compute capacity grew during execute");

    int64_t output_rows = 0;
    int64_t output_cols = 0;
    int64_t output_nnz = 0;
    stage_begin = std::chrono::steady_clock::now();
    RTT_CUSPARSE_CORE_SPARSE(
        cusparseSpMatGetSize(state->matrix_c, &output_rows, &output_cols,
                            &output_nnz),
        "query C size");
    stage_stats->c_size_query_wall_ms = WallMilliseconds(
        stage_begin, std::chrono::steady_clock::now());
    if (output_rows != metadata_a.rows || output_cols != metadata_b.cols ||
        output_nnz < 0 || output_nnz > std::numeric_limits<int>::max())
        return Status::Invalid("invalid C dimensions returned by cuSPARSE");
    state->output.nnz = output_nnz;

    stage_begin = std::chrono::steady_clock::now();
    if (output_nnz != 0) {
        RTT_CUSPARSE_CORE_CUDA(
            cudaMalloc(
                reinterpret_cast<void **>(&state->output.column_indices),
                static_cast<size_t>(output_nnz) * sizeof(int)),
            "allocate C column indices");
        RTT_CUSPARSE_CORE_CUDA(
            cudaMalloc(reinterpret_cast<void **>(&state->output.values),
                       static_cast<size_t>(output_nnz) * sizeof(double)),
            "allocate C values");
    }
    RTT_CUSPARSE_CORE_SPARSE(
        cusparseCsrSetPointers(state->matrix_c, state->output.row_offsets,
                              state->output.column_indices,
                              state->output.values),
        "set final C pointers");
    stage_stats->c_payload_allocation_wall_ms = WallMilliseconds(
        stage_begin, std::chrono::steady_clock::now());
    stage_begin = std::chrono::steady_clock::now();
    RTT_CUSPARSE_CORE_SPARSE(
        cusparseSpGEMM_copy(
            handle, operation, operation, &alpha, matrix_a, matrix_b, &beta,
            state->matrix_c, CUDA_R_64F, config.algorithm,
            state->spgemm_descriptor),
        "copy final C");
    stage_stats->copy_submit_wall_ms = WallMilliseconds(
        stage_begin, std::chrono::steady_clock::now());

    stage_begin = std::chrono::steady_clock::now();
    RTT_CUSPARSE_CORE_CUDA(cudaStreamSynchronize(stream),
                           "post-copy stream synchronization");
    stage_stats->final_sync_wall_ms = WallMilliseconds(
        stage_begin, std::chrono::steady_clock::now());
    const std::chrono::steady_clock::time_point stop =
        std::chrono::steady_clock::now();
    *elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    if (config.b_value_update.enabled()) {
        float update_ms = 0.0f;
        RTT_CUSPARSE_CORE_CUDA(
            cudaEventElapsedTime(&update_ms, state->b_update_begin,
                                 state->b_update_end),
            "measure B-values update");
        *b_update_elapsed_ms = static_cast<double>(update_ms);
    }
    stage_stats->scratch_pointer_stable =
        reusable_workspace == NULL || reusable_workspace->pointers_stable();
    if (!stage_stats->scratch_pointer_stable)
        return Status::Invalid("reusable scratch pointer changed during sample");
    cuda_error = cudaMemGetInfo(&stage_stats->memory_free_after_bytes,
                                &stage_stats->memory_total_after_bytes);
    if (cuda_error != cudaSuccess)
        return Status::Cuda(cuda_error,
                            "read memory after cuSPARSE Core sample");

#undef RTT_CUSPARSE_CORE_SPARSE
#undef RTT_CUSPARSE_CORE_CUDA

    return Status();
}

inline double Median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0)
        return values[middle];
    return (values[middle - 1] + values[middle]) * 0.5;
}

inline Status ValidateView(const DeviceCsrView &matrix,
                           const char *stage) {
    if (matrix.rows < 0 || matrix.cols < 0 || matrix.nnz < 0 ||
        matrix.rows > std::numeric_limits<int>::max() ||
        matrix.cols > std::numeric_limits<int>::max() ||
        matrix.nnz > std::numeric_limits<int>::max() ||
        matrix.row_offsets == NULL ||
        (matrix.nnz > 0 &&
         (matrix.column_indices == NULL || matrix.values == NULL)))
        return Status::Invalid(stage);
    return Status();
}

}  // namespace detail

/*
 * Main entry point.  handle, matrix_a, and matrix_b are borrowed and remain
 * owned by the caller.  Their creation is therefore outside every sample.
 * The function uses the stream already associated with handle.
 */
inline Status BenchmarkPrepared(cusparseHandle_t handle,
                                cusparseSpMatDescr_t matrix_a,
                                cusparseSpMatDescr_t matrix_b,
                                const BenchmarkConfig &config,
                                BenchmarkResult *result) {
    if (handle == NULL || matrix_a == NULL || matrix_b == NULL ||
        result == NULL || config.warmup_iterations < 0 ||
        config.timed_iterations <= 0)
        return Status::Invalid("benchmark arguments");

    result->reset();

    detail::CsrMetadata metadata_a;
    detail::CsrMetadata metadata_b;
    Status status =
        detail::InspectCsr(matrix_a, &metadata_a, "inspect CSR A");
    if (!status.ok())
        return status;
    status = detail::InspectCsr(matrix_b, &metadata_b, "inspect CSR B");
    if (!status.ok())
        return status;
    if (metadata_a.cols != metadata_b.rows)
        return Status::Invalid("incompatible A and B dimensions");
    if (config.b_value_update.enabled() &&
        (config.b_value_update.count != metadata_b.nnz ||
         config.b_value_update.destination_values != metadata_b.values ||
         (metadata_b.nnz > 0 &&
          (config.b_value_update.source_values == NULL ||
           config.b_value_update.destination_to_source == NULL))))
        return Status::Invalid("invalid B-values update mapping");

    cudaStream_t stream = NULL;
    const cusparseStatus_t stream_status = cusparseGetStream(handle, &stream);
    if (stream_status != CUSPARSE_STATUS_SUCCESS)
        return Status::Cusparse(stream_status, "query cuSPARSE stream");

    result->warmup_samples_ms.reserve(
        static_cast<size_t>(config.warmup_iterations));
    result->samples_ms.reserve(static_cast<size_t>(config.timed_iterations));
    result->warmup_b_update_samples_ms.reserve(
        static_cast<size_t>(config.warmup_iterations));
    result->b_update_samples_ms.reserve(
        static_cast<size_t>(config.timed_iterations));
    result->warmup_stage_samples.reserve(
        static_cast<size_t>(config.warmup_iterations));
    result->stage_samples.reserve(
        static_cast<size_t>(config.timed_iterations));

    detail::ReusableWorkspace reusable_workspace;
    if (config.reuse_scratch_workspace) {
        status = detail::PrepareReusableWorkspace(
            handle, matrix_a, matrix_b, metadata_a, metadata_b, config,
            stream, &reusable_workspace, result);
        if (!status.ok()) {
            result->reset();
            return status;
        }
    }

    const int total_iterations =
        config.warmup_iterations + config.timed_iterations;
    for (int iteration = 0; iteration < total_iterations; ++iteration) {
        detail::IterationState state;
        status = detail::PrepareIteration(
            metadata_a.rows, metadata_b.cols,
            config.b_value_update.enabled(), &state);
        if (!status.ok()) {
            result->reset();
            return status;
        }

        double elapsed_ms = 0.0;
        double b_update_elapsed_ms = 0.0;
        IterationStageStats stage_stats;
        status = detail::RunIteration(handle, matrix_a, matrix_b, metadata_a,
                                      metadata_b, config, stream,
                                      config.reuse_scratch_workspace
                                          ? &reusable_workspace
                                          : NULL,
                                      &state, &elapsed_ms,
                                      &b_update_elapsed_ms, &stage_stats);
        if (!status.ok()) {
            result->reset();
            return status;
        }
        result->all_scratch_pointers_stable =
            result->all_scratch_pointers_stable &&
            stage_stats.scratch_pointer_stable;
        result->any_scratch_allocated_in_timed =
            result->any_scratch_allocated_in_timed ||
            stage_stats.scratch_allocated_in_timed;

        if (iteration < config.warmup_iterations) {
            result->warmup_samples_ms.push_back(elapsed_ms);
            result->warmup_b_update_samples_ms.push_back(
                b_update_elapsed_ms);
            result->warmup_stage_samples.push_back(stage_stats);
        } else {
            result->samples_ms.push_back(elapsed_ms);
            result->b_update_samples_ms.push_back(b_update_elapsed_ms);
            result->stage_samples.push_back(stage_stats);
            if (iteration == total_iterations - 1) {
                result->final_work_estimation_buffer_bytes =
                    state.work_estimation_buffer_bytes;
                result->final_compute_buffer_bytes =
                    state.compute_buffer_bytes;
                result->output = std::move(state.output);
            }
        }
        /* state cleanup runs here, after the sample stopwatch has stopped. */
    }

    result->min_ms =
        *std::min_element(result->samples_ms.begin(), result->samples_ms.end());
    result->median_ms = detail::Median(result->samples_ms);
    result->max_ms =
        *std::max_element(result->samples_ms.begin(), result->samples_ms.end());
    return Status();
}

/*
 * Convenience entry point for raw device CSR.  It creates the handle and A/B
 * descriptors before calling BenchmarkPrepared, so those setup operations are
 * still excluded from every sample.  The caller retains ownership of A and B.
 */
inline Status BenchmarkDeviceCsr(const DeviceCsrView &matrix_a,
                                 const DeviceCsrView &matrix_b,
                                 cudaStream_t stream,
                                 const BenchmarkConfig &config,
                                 BenchmarkResult *result) {
    if (result == NULL)
        return Status::Invalid("benchmark result pointer");
    Status status = detail::ValidateView(matrix_a, "validate device CSR A");
    if (!status.ok())
        return status;
    status = detail::ValidateView(matrix_b, "validate device CSR B");
    if (!status.ok())
        return status;
    if (matrix_a.cols != matrix_b.rows)
        return Status::Invalid("incompatible device CSR dimensions");

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t descriptor_a = NULL;
    cusparseSpMatDescr_t descriptor_b = NULL;

    cusparseStatus_t sparse_error = cusparseCreate(&handle);
    if (sparse_error != CUSPARSE_STATUS_SUCCESS)
        return Status::Cusparse(sparse_error, "create untimed handle");

    sparse_error = cusparseSetStream(handle, stream);
    if (sparse_error == CUSPARSE_STATUS_SUCCESS) {
        sparse_error = cusparseCreateCsr(
            &descriptor_a, matrix_a.rows, matrix_a.cols, matrix_a.nnz,
            const_cast<int *>(matrix_a.row_offsets),
            const_cast<int *>(matrix_a.column_indices),
            const_cast<double *>(matrix_a.values), CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    }
    if (sparse_error == CUSPARSE_STATUS_SUCCESS) {
        sparse_error = cusparseCreateCsr(
            &descriptor_b, matrix_b.rows, matrix_b.cols, matrix_b.nnz,
            const_cast<int *>(matrix_b.row_offsets),
            const_cast<int *>(matrix_b.column_indices),
            const_cast<double *>(matrix_b.values), CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    }

    if (sparse_error == CUSPARSE_STATUS_SUCCESS) {
        status = BenchmarkPrepared(handle, descriptor_a, descriptor_b, config,
                                   result);
    } else {
        status = Status::Cusparse(sparse_error,
                                  "create untimed input descriptors");
    }

    if (descriptor_b != NULL)
        (void)cusparseDestroySpMat(descriptor_b);
    if (descriptor_a != NULL)
        (void)cusparseDestroySpMat(descriptor_a);
    if (handle != NULL)
        (void)cusparseDestroy(handle);
    return status;
}

}  // namespace cusparse_core
}  // namespace rtt_spgemm

#endif  // RTT_SPGEMM_CUSPARSE_CORE_BENCHMARK_H_
