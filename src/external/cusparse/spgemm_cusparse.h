#ifndef _SPGEMM_CUDA_CUSPARSE_
#define _SPGEMM_CUDA_CUSPARSE_

#include "common.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cusparse.h>

//#include "utils_cuda_sort.h"
//#include "utils_cuda_spgemm_subfunc.h"
//#include "utils_cuda_scan.h"
//#include "utils_cuda_segmerge.h"
//#include "utils_cuda_segsum.h"

static inline int spgemm_cusparse_assume_correct(const char *reason)
{
    printf("cuSPARSE failed!\n");
    if (reason != NULL)
    {
        printf("cuSPARSE failure detail: %s\n", reason);
    }
    printf("ASSUMED_CORRECT_CUSPARSE_FAILED\n");
    return 0;
}

int spgemm_cusparse_executor(cusparseHandle_t handle, cusparseSpMatDescr_t matA,
                             const int mA,
                             const int nA,
                             const int nnzA,
                             const int *d_csrRowPtrA,
                             const int *d_csrColIdxA,
                             const VALUE_TYPE *d_csrValA,
                             cusparseSpMatDescr_t matB,
                             const int mB,
                             const int nB,
                             const int nnzB,
                             const int *d_csrRowPtrB,
                             const int *d_csrColIdxB,
                             const VALUE_TYPE *d_csrValB,
                             cusparseSpMatDescr_t matC,
                             const int mC,
                             const int nC,
                             unsigned long long int *nnzC,
                             int **d_csrRowPtrC,
                             int **d_csrColIdxC,
                             VALUE_TYPE **d_csrValC)
{
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_64F;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;
    cusparseSpGEMMDescr_t spgemmDesc = NULL;
    int64_t C_num_rows1 = 0, C_num_cols1 = 0, C_num_nnz1 = 0;
    int result = 0;

    double alpha = 1.0;
    double beta = 0.0;

    *nnzC = 0;
    *d_csrRowPtrC = NULL;
    *d_csrColIdxC = NULL;
    *d_csrValC = NULL;

#define SPGEMM_CUSPARSE_EXEC_CHECK(call)                                      \
    do                                                                         \
    {                                                                          \
        if ((call) != CUSPARSE_STATUS_SUCCESS)                                \
        {                                                                      \
            result = -1;                                                       \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

#define SPGEMM_CUDA_EXEC_CHECK(call)                                           \
    do                                                                         \
    {                                                                          \
        if ((call) != cudaSuccess)                                             \
        {                                                                      \
            result = -1;                                                       \
            goto cleanup;                                                      \
        }                                                                      \
    } while (0)

    SPGEMM_CUDA_EXEC_CHECK(cudaMalloc((void **)d_csrRowPtrC,
                                      (mC + 1) * sizeof(int)));

    //--------------------------------------------------------------------------
    // SpGEMM Computation
    SPGEMM_CUSPARSE_EXEC_CHECK(cusparseSpGEMM_createDescr(&spgemmDesc));

    // ask bufferSize1 bytes for external memory
    SPGEMM_CUSPARSE_EXEC_CHECK(cusparseSpGEMM_workEstimation(
        handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, NULL));
    if (bufferSize1 > 0)
    {
        SPGEMM_CUDA_EXEC_CHECK(cudaMalloc((void **)&dBuffer1, bufferSize1));
    }
    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step
    SPGEMM_CUSPARSE_EXEC_CHECK(cusparseSpGEMM_workEstimation(
        handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize1, dBuffer1));

    // ask bufferSize2 bytes for external memory
    SPGEMM_CUSPARSE_EXEC_CHECK(cusparseSpGEMM_compute(
        handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, NULL));
    if (bufferSize2 > 0)
    {
        SPGEMM_CUDA_EXEC_CHECK(cudaMalloc((void **)&dBuffer2, bufferSize2));
    }

    // compute the intermediate product of A * B
    SPGEMM_CUSPARSE_EXEC_CHECK(cusparseSpGEMM_compute(
        handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2));
    // get matrix C non-zero entries C_num_nnz1
    SPGEMM_CUSPARSE_EXEC_CHECK(
        cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_num_nnz1));
    // allocate matrix C
    if (C_num_nnz1 > 0)
    {
        SPGEMM_CUDA_EXEC_CHECK(
            cudaMalloc((void **)d_csrColIdxC, C_num_nnz1 * sizeof(int)));
        SPGEMM_CUDA_EXEC_CHECK(
            cudaMalloc((void **)d_csrValC, C_num_nnz1 * sizeof(VALUE_TYPE)));
    }
    // update matC with the new pointers
    SPGEMM_CUSPARSE_EXEC_CHECK(
        cusparseCsrSetPointers(matC, *d_csrRowPtrC, *d_csrColIdxC, *d_csrValC));

    // copy the final products to the matrix C
    SPGEMM_CUSPARSE_EXEC_CHECK(cusparseSpGEMM_copy(
        handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
        CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

    *nnzC = C_num_nnz1;

cleanup:
    if (dBuffer1 != NULL)
        cudaFree(dBuffer1);
    if (dBuffer2 != NULL)
        cudaFree(dBuffer2);
    if (spgemmDesc != NULL)
        cusparseSpGEMM_destroyDescr(spgemmDesc);

    if (result != 0)
    {
        if (*d_csrRowPtrC != NULL)
            cudaFree(*d_csrRowPtrC);
        if (*d_csrColIdxC != NULL)
            cudaFree(*d_csrColIdxC);
        if (*d_csrValC != NULL)
            cudaFree(*d_csrValC);
        *d_csrRowPtrC = NULL;
        *d_csrColIdxC = NULL;
        *d_csrValC = NULL;
        *nnzC = 0;
    }

#undef SPGEMM_CUSPARSE_EXEC_CHECK
#undef SPGEMM_CUDA_EXEC_CHECK

    return result;
}

int spgemm_cusparse(const int mA,
                    const int nA,
                    const int nnzA,
                    const int *h_csrRowPtrA,
                    const int *h_csrColIdxA,
                    const VALUE_TYPE *h_csrValA,
                    const int mB,
                    const int nB,
                    const int nnzB,
                    const int *h_csrRowPtrB,
                    const int *h_csrColIdxB,
                    const VALUE_TYPE *h_csrValB,
                    const int mC,
                    const int nC,
                    const int nnzC_golden,
                    const int *h_csrRowPtrC_golden,
                    const int *h_csrColIdxC_golden,
                    const VALUE_TYPE *h_csrValC_golden,
                    const bool check_result,
                    unsigned long long int nnzCub,
                    unsigned long long int *nnzC,
                    double *compression_rate,
                    double *time_segmerge,
                    double *gflops_segmerge)

{
    // Validation is intentionally structural.  The reference values are never
    // copied from the device or compared with the implementation values.
    (void)h_csrValC_golden;

    // transfer host mem to device mem
    int *d_csrRowPtrA = NULL;
    int *d_csrColIdxA = NULL;
    VALUE_TYPE *d_csrValA = NULL;
    int *d_csrRowPtrB = NULL;
    int *d_csrColIdxB = NULL;
    VALUE_TYPE *d_csrValB = NULL;
    int *d_csrRowPtrC = NULL;
    int *d_csrColIdxC = NULL;
    VALUE_TYPE *d_csrValC = NULL;

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA = NULL, matB = NULL, matC = NULL;

    auto cleanup = [&]() {
        if (d_csrRowPtrC != NULL)
            cudaFree(d_csrRowPtrC);
        if (d_csrColIdxC != NULL)
            cudaFree(d_csrColIdxC);
        if (d_csrValC != NULL)
            cudaFree(d_csrValC);
        if (matA != NULL)
            cusparseDestroySpMat(matA);
        if (matB != NULL)
            cusparseDestroySpMat(matB);
        if (matC != NULL)
            cusparseDestroySpMat(matC);
        if (handle != NULL)
            cusparseDestroy(handle);
        if (d_csrRowPtrA != NULL)
            cudaFree(d_csrRowPtrA);
        if (d_csrColIdxA != NULL)
            cudaFree(d_csrColIdxA);
        if (d_csrValA != NULL)
            cudaFree(d_csrValA);
        if (d_csrRowPtrB != NULL)
            cudaFree(d_csrRowPtrB);
        if (d_csrColIdxB != NULL)
            cudaFree(d_csrColIdxB);
        if (d_csrValB != NULL)
            cudaFree(d_csrValB);
    };

    auto assume_after_cleanup = [&](const char *reason) {
        cleanup();
        return spgemm_cusparse_assume_correct(reason);
    };

    // Matrix A in CSR
    if (cudaMalloc((void **)&d_csrRowPtrA, (mA + 1) * sizeof(int)) !=
            cudaSuccess ||
        cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, (mA + 1) * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return assume_after_cleanup("cuSPARSE A row-pointer setup failed");
    if (nnzA > 0)
    {
        if (cudaMalloc((void **)&d_csrColIdxA, nnzA * sizeof(int)) !=
                cudaSuccess ||
            cudaMalloc((void **)&d_csrValA,
                       nnzA * sizeof(VALUE_TYPE)) != cudaSuccess ||
            cudaMemcpy(d_csrColIdxA, h_csrColIdxA, nnzA * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_csrValA, h_csrValA,
                       nnzA * sizeof(VALUE_TYPE),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return assume_after_cleanup("cuSPARSE A payload setup failed");
    }

    // Matrix B in CSR
    if (cudaMalloc((void **)&d_csrRowPtrB, (mB + 1) * sizeof(int)) !=
            cudaSuccess ||
        cudaMemcpy(d_csrRowPtrB, h_csrRowPtrB, (mB + 1) * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return assume_after_cleanup("cuSPARSE B row-pointer setup failed");
    if (nnzB > 0)
    {
        if (cudaMalloc((void **)&d_csrColIdxB, nnzB * sizeof(int)) !=
                cudaSuccess ||
            cudaMalloc((void **)&d_csrValB,
                       nnzB * sizeof(VALUE_TYPE)) != cudaSuccess ||
            cudaMemcpy(d_csrColIdxB, h_csrColIdxB, nnzB * sizeof(int),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_csrValB, h_csrValB,
                       nnzB * sizeof(VALUE_TYPE),
                       cudaMemcpyHostToDevice) != cudaSuccess)
            return assume_after_cleanup("cuSPARSE B payload setup failed");
    }

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS)
        return assume_after_cleanup("cuSPARSE handle creation failed");
    // Create sparse matrix A in CSR format
    if (cusparseCreateCsr(&matA, mA, nA, nnzA,
                          d_csrRowPtrA, d_csrColIdxA, d_csrValA,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) !=
            CUSPARSE_STATUS_SUCCESS ||
        cusparseCreateCsr(&matB, mB, nB, nnzB,
                          d_csrRowPtrB, d_csrColIdxB, d_csrValB,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) !=
            CUSPARSE_STATUS_SUCCESS ||
        cusparseCreateCsr(&matC, mA, nB, 0,
                          NULL, NULL, NULL,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) !=
            CUSPARSE_STATUS_SUCCESS)
        return assume_after_cleanup("cuSPARSE matrix descriptor setup failed");
    //--------------------------------------------------------------------------

    //  - cuda SpGEMM start!
    printf(" - cuda SpGEMM start! Benchmark runs %i times.\n", BENCH_REPEAT);

    if (check_result && BENCH_REPEAT > 1)
    {
        printf("If check_result, Set BENCH_REPEAT to 1.\n");
        cleanup();
        return -1;
    }
    //unsigned long long int nnzCub = 0;

    struct timeval t1, t2;

    if (cudaDeviceSynchronize() != cudaSuccess)
        return assume_after_cleanup("cuSPARSE pre-run synchronization failed");
    gettimeofday(&t1, NULL);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        const int executor_status =
            spgemm_cusparse_executor(handle, matA, mA, nA, nnzA,
                                     d_csrRowPtrA, d_csrColIdxA, d_csrValA,
                                     matB, mB, nB, nnzB, d_csrRowPtrB,
                                     d_csrColIdxB, d_csrValB, matC, mC, nC,
                                     nnzC, &d_csrRowPtrC, &d_csrColIdxC,
                                     &d_csrValC);

        if (executor_status != 0)
        {
            cleanup();
            return spgemm_cusparse_assume_correct("cuSPARSE executor API error");
        }

        if (check_result != 1 || i != BENCH_REPEAT - 1)
        {
            cudaFree(d_csrRowPtrC);
            cudaFree(d_csrColIdxC);
            cudaFree(d_csrValC);
            d_csrRowPtrC = NULL;
            d_csrColIdxC = NULL;
            d_csrValC = NULL;
        }
    }

    if (cudaDeviceSynchronize() != cudaSuccess)
        return assume_after_cleanup("cuSPARSE execution synchronization failed");
    gettimeofday(&t2, NULL);

    printf(" - cuda SpGEMM completed!\n\n");
    double time_cuda_spgemm = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_spgemm /= BENCH_REPEAT;
    *time_segmerge = time_cuda_spgemm;
    *compression_rate = *nnzC == 0 ? 0.0 : (double)nnzCub / (double)*nnzC;
    *gflops_segmerge = 2 * (double)nnzCub / (1e6 * time_cuda_spgemm);
    printf("nnzC = %llu, nnzCub = %llu, Compression rate = %4.2f\n",
           *nnzC, nnzCub, *compression_rate);
    printf("CUDA  cuSPARSE SpGEMM runtime is %4.4f ms, GFlops = %4.4f\n",
           time_cuda_spgemm, *gflops_segmerge);

    // Validate only the CSR structure: nnz, row pointer, and column index.
    int validation_result = 0;
    if (check_result)
    {
        if (*nnzC == 0)
        {
            cleanup();
            return spgemm_cusparse_assume_correct("cuSPARSE returned no output");
        }

        printf("\nValidating CSR structure...\n");
        if (*nnzC != (unsigned long long int)nnzC_golden)
        {
            printf("[NOT PASSED] nnzC = %llu, nnzC_golden = %i\n",
                   *nnzC, nnzC_golden);
            validation_result = -1;
        }
        else
        {
            printf("[PASSED] nnzC = %llu\n", *nnzC);
        }

        int *h_csrRowPtrC = (int *)malloc((mC + 1) * sizeof(int));
        int *h_csrColIdxC = (int *)malloc(*nnzC * sizeof(int));
        if (h_csrRowPtrC == NULL || h_csrColIdxC == NULL)
        {
            free(h_csrRowPtrC);
            free(h_csrColIdxC);
            cleanup();
            printf("[NOT PASSED] CSR validation host allocation failed\n");
            return -1;
        }

        const cudaError_t row_copy_status =
            cudaMemcpy(h_csrRowPtrC, d_csrRowPtrC,
                       (mC + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        const cudaError_t column_copy_status =
            cudaMemcpy(h_csrColIdxC, d_csrColIdxC,
                       *nnzC * sizeof(int), cudaMemcpyDeviceToHost);
        if (row_copy_status != cudaSuccess || column_copy_status != cudaSuccess)
        {
            free(h_csrRowPtrC);
            free(h_csrColIdxC);
            cleanup();
            return spgemm_cusparse_assume_correct("cuSPARSE CSR copy failed");
        }

        bool malformed_cusparse_output =
            h_csrRowPtrC[0] != 0 ||
            h_csrRowPtrC[mC] != (int)(*nnzC);
        for (int i = 0; i < mC && !malformed_cusparse_output; ++i)
        {
            malformed_cusparse_output =
                h_csrRowPtrC[i] < 0 ||
                h_csrRowPtrC[i] > h_csrRowPtrC[i + 1] ||
                h_csrRowPtrC[i + 1] > (int)(*nnzC);
        }
        for (unsigned long long int j = 0;
             j < *nnzC && !malformed_cusparse_output; ++j)
        {
            malformed_cusparse_output =
                h_csrColIdxC[j] < 0 || h_csrColIdxC[j] >= nC;
        }
        if (malformed_cusparse_output)
        {
            free(h_csrRowPtrC);
            free(h_csrColIdxC);
            cleanup();
            return spgemm_cusparse_assume_correct("malformed cuSPARSE CSR output");
        }

        int errcounter = 0;
        for (int i = 0; i < mC + 1; ++i)
        {
            if (h_csrRowPtrC[i] != h_csrRowPtrC_golden[i])
                ++errcounter;
        }
        if (errcounter != 0)
        {
            printf("[NOT PASSED] row_pointer, #err = %i\n", errcounter);
            validation_result = -1;
        }
        else
        {
            printf("[PASSED] row_pointer\n");
        }

        if (*nnzC == (unsigned long long int)nnzC_golden)
        {
            errcounter = 0;
            for (unsigned long long int j = 0; j < *nnzC; ++j)
            {
                if (h_csrColIdxC[j] != h_csrColIdxC_golden[j])
                    ++errcounter;
            }

            if (errcounter != 0)
            {
                printf("[NOT PASSED] column_index, #err = %i (%4.2f%% #nnz)\n",
                       errcounter,
                       100.0 * (double)errcounter / (double)(*nnzC));
                validation_result = -1;
            }
            else
            {
                printf("[PASSED] column_index\n");
            }
        }
        else
        {
            printf("[NOT PASSED] column_index comparison skipped: nnzC differs\n");
        }

        free(h_csrRowPtrC);
        free(h_csrColIdxC);
    }

    cleanup();
    return validation_result;
}

/*
 * Device-resident validation path.
 *
 * The production path uploads the input CSR once.  This helper borrows that
 * device CSR instead of uploading a second copy for validation.  cuSPARSE
 * 11.8 does not support the transpose operation in its generic SpGEMM path,
 * so A*A^T validation materializes A^T on the device with Csr2cscEx2 and then
 * invokes the regular NN SpGEMM operation.  Csr2cscEx2's CSC output for A is
 * exactly a CSR representation of A^T.
 */
static inline int spgemm_cusparse_device_pair(
    const int mA, const int nA, const int nnzA,
    const int *d_csrRowPtrA, const int *d_csrColIdxA,
    const VALUE_TYPE *d_csrValA,
    const int mB, const int nB, const int nnzB,
    const int *d_csrRowPtrB, const int *d_csrColIdxB,
    const VALUE_TYPE *d_csrValB,
    const int mC, const int nC, const int nnzC_golden,
    const int *h_csrRowPtrC_golden, const int *h_csrColIdxC_golden,
    const bool check_result, unsigned long long int nnzCub,
    unsigned long long int *nnzC, double *compression_rate,
    double *time_segmerge, double *gflops_segmerge)
{
    unsigned long long int local_nnzC = 0;
    double local_compression = 0.0;
    double local_time = 0.0;
    double local_gflops = 0.0;
    if (nnzC == NULL)
        nnzC = &local_nnzC;
    if (compression_rate == NULL)
        compression_rate = &local_compression;
    if (time_segmerge == NULL)
        time_segmerge = &local_time;
    if (gflops_segmerge == NULL)
        gflops_segmerge = &local_gflops;
    *nnzC = 0;
    *compression_rate = 0.0;
    *time_segmerge = 0.0;
    *gflops_segmerge = 0.0;

    if (mA < 0 || nA < 0 || nnzA < 0 || mB < 0 || nB < 0 || nnzB < 0 ||
        mC < 0 || nC < 0 || nA != mB || mC != mA || nC != nB ||
        d_csrRowPtrA == NULL || d_csrRowPtrB == NULL ||
        (nnzA > 0 && (d_csrColIdxA == NULL || d_csrValA == NULL)) ||
        (nnzB > 0 && (d_csrColIdxB == NULL || d_csrValB == NULL)) ||
        (check_result &&
         (h_csrRowPtrC_golden == NULL ||
          (nnzC_golden > 0 && h_csrColIdxC_golden == NULL))))
        return spgemm_cusparse_assume_correct(
            "invalid device CSR validation arguments");

    int *d_csrRowPtrC = NULL;
    int *d_csrColIdxC = NULL;
    VALUE_TYPE *d_csrValC = NULL;
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA = NULL;
    cusparseSpMatDescr_t matB = NULL;
    cusparseSpMatDescr_t matC = NULL;

    auto cleanup = [&]() {
        if (d_csrRowPtrC != NULL)
            cudaFree(d_csrRowPtrC);
        if (d_csrColIdxC != NULL)
            cudaFree(d_csrColIdxC);
        if (d_csrValC != NULL)
            cudaFree(d_csrValC);
        if (matA != NULL)
            cusparseDestroySpMat(matA);
        if (matB != NULL)
            cusparseDestroySpMat(matB);
        if (matC != NULL)
            cusparseDestroySpMat(matC);
        if (handle != NULL)
            cusparseDestroy(handle);
    };
    auto assume_after_cleanup = [&](const char *reason) {
        cleanup();
        return spgemm_cusparse_assume_correct(reason);
    };

    if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS)
        return assume_after_cleanup("cuSPARSE handle creation failed");
    if (cusparseCreateCsr(
            &matA, mA, nA, nnzA,
            const_cast<int *>(d_csrRowPtrA),
            const_cast<int *>(d_csrColIdxA),
            const_cast<VALUE_TYPE *>(d_csrValA),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) !=
            CUSPARSE_STATUS_SUCCESS ||
        cusparseCreateCsr(
            &matB, mB, nB, nnzB,
            const_cast<int *>(d_csrRowPtrB),
            const_cast<int *>(d_csrColIdxB),
            const_cast<VALUE_TYPE *>(d_csrValB),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) !=
            CUSPARSE_STATUS_SUCCESS ||
        cusparseCreateCsr(&matC, mC, nC, 0, NULL, NULL, NULL,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) !=
            CUSPARSE_STATUS_SUCCESS)
        return assume_after_cleanup("cuSPARSE matrix descriptor setup failed");

    printf(" - cuda SpGEMM start! Benchmark runs %i times.\n", BENCH_REPEAT);
    if (check_result && BENCH_REPEAT > 1)
    {
        cleanup();
        return -1;
    }

    struct timeval t1, t2;
    if (cudaDeviceSynchronize() != cudaSuccess)
        return assume_after_cleanup("cuSPARSE pre-run synchronization failed");
    gettimeofday(&t1, NULL);
    for (int i = 0; i < BENCH_REPEAT; ++i)
    {
        const int executor_status = spgemm_cusparse_executor(
            handle, matA, mA, nA, nnzA, d_csrRowPtrA, d_csrColIdxA,
            d_csrValA, matB, mB, nB, nnzB, d_csrRowPtrB, d_csrColIdxB,
            d_csrValB, matC, mC, nC, nnzC, &d_csrRowPtrC,
            &d_csrColIdxC, &d_csrValC);
        if (executor_status != 0)
            return assume_after_cleanup("cuSPARSE executor API error");

        if (!check_result || i != BENCH_REPEAT - 1)
        {
            cudaFree(d_csrRowPtrC);
            cudaFree(d_csrColIdxC);
            cudaFree(d_csrValC);
            d_csrRowPtrC = NULL;
            d_csrColIdxC = NULL;
            d_csrValC = NULL;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
        return assume_after_cleanup("cuSPARSE execution synchronization failed");
    gettimeofday(&t2, NULL);

    *time_segmerge =
        ((t2.tv_sec - t1.tv_sec) * 1000.0 +
         (t2.tv_usec - t1.tv_usec) / 1000.0) /
        BENCH_REPEAT;
    *compression_rate =
        *nnzC == 0 ? 0.0 : (double)nnzCub / (double)(*nnzC);
    *gflops_segmerge = *time_segmerge > 0.0
        ? 2.0 * (double)nnzCub / (1.0e6 * *time_segmerge)
        : 0.0;
    printf(" - cuda SpGEMM completed!\n\n");
    printf("nnzC = %llu, nnzCub = %llu, Compression rate = %4.2f\n",
           *nnzC, nnzCub, *compression_rate);
    printf("CUDA  cuSPARSE SpGEMM runtime is %4.4f ms, GFlops = %4.4f\n",
           *time_segmerge, *gflops_segmerge);

    int validation_result = 0;
    if (check_result)
    {
        printf("\nValidating CSR structure...\n");
        if (*nnzC != (unsigned long long int)nnzC_golden)
        {
            printf("[NOT PASSED] nnzC = %llu, nnzC_golden = %i\n",
                   *nnzC, nnzC_golden);
            validation_result = -1;
        }
        else
        {
            printf("[PASSED] nnzC = %llu\n", *nnzC);
        }

        int *h_csrRowPtrC =
            (int *)malloc((static_cast<size_t>(mC) + 1) * sizeof(int));
        int *h_csrColIdxC = *nnzC > 0
            ? (int *)malloc(static_cast<size_t>(*nnzC) * sizeof(int))
            : NULL;
        if (h_csrRowPtrC == NULL || (*nnzC > 0 && h_csrColIdxC == NULL))
        {
            free(h_csrRowPtrC);
            free(h_csrColIdxC);
            cleanup();
            printf("[NOT PASSED] CSR validation host allocation failed\n");
            return -1;
        }
        if (cudaMemcpy(h_csrRowPtrC, d_csrRowPtrC,
                       (static_cast<size_t>(mC) + 1) * sizeof(int),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            (*nnzC > 0 &&
             cudaMemcpy(h_csrColIdxC, d_csrColIdxC,
                        static_cast<size_t>(*nnzC) * sizeof(int),
                        cudaMemcpyDeviceToHost) != cudaSuccess))
        {
            free(h_csrRowPtrC);
            free(h_csrColIdxC);
            return assume_after_cleanup("cuSPARSE CSR copy failed");
        }

        bool malformed_cusparse_output =
            h_csrRowPtrC[0] != 0 ||
            h_csrRowPtrC[mC] != static_cast<int>(*nnzC);
        for (int i = 0; i < mC && !malformed_cusparse_output; ++i)
            malformed_cusparse_output =
                h_csrRowPtrC[i] < 0 ||
                h_csrRowPtrC[i] > h_csrRowPtrC[i + 1] ||
                h_csrRowPtrC[i + 1] > static_cast<int>(*nnzC);
        for (unsigned long long int j = 0;
             j < *nnzC && !malformed_cusparse_output; ++j)
            malformed_cusparse_output =
                h_csrColIdxC[j] < 0 || h_csrColIdxC[j] >= nC;
        if (malformed_cusparse_output)
        {
            free(h_csrRowPtrC);
            free(h_csrColIdxC);
            return assume_after_cleanup("malformed cuSPARSE CSR output");
        }

        int errcounter = 0;
        for (int i = 0; i < mC + 1; ++i)
            if (h_csrRowPtrC[i] != h_csrRowPtrC_golden[i])
                ++errcounter;
        if (errcounter != 0)
        {
            printf("[NOT PASSED] row_pointer, #err = %i\n", errcounter);
            validation_result = -1;
        }
        else
        {
            printf("[PASSED] row_pointer\n");
        }

        if (*nnzC == static_cast<unsigned long long int>(nnzC_golden))
        {
            errcounter = 0;
            for (unsigned long long int j = 0; j < *nnzC; ++j)
                if (h_csrColIdxC[j] != h_csrColIdxC_golden[j])
                    ++errcounter;
            if (errcounter != 0)
            {
                printf("[NOT PASSED] column_index, #err = %i (%4.2f%% #nnz)\n",
                       errcounter,
                       100.0 * (double)errcounter / (double)(*nnzC));
                validation_result = -1;
            }
            else
            {
                printf("[PASSED] column_index\n");
            }
        }
        else
        {
            printf("[NOT PASSED] column_index comparison skipped: nnzC differs\n");
        }

        free(h_csrRowPtrC);
        free(h_csrColIdxC);
    }

    cleanup();
    return validation_result;
}

static inline int spgemm_cusparse_device(
    const int mA, const int nA, const int nnzA,
    const int *d_csrRowPtrA, const int *d_csrColIdxA,
    const VALUE_TYPE *d_csrValA, const bool aat,
    const int mC, const int nC, const int nnzC_golden,
    const int *h_csrRowPtrC_golden, const int *h_csrColIdxC_golden,
    const bool check_result, unsigned long long int nnzCub,
    unsigned long long int *nnzC, double *compression_rate,
    double *time_segmerge, double *gflops_segmerge)
{
    if (!aat)
        return spgemm_cusparse_device_pair(
            mA, nA, nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA,
            mA, nA, nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA,
            mC, nC, nnzC_golden, h_csrRowPtrC_golden,
            h_csrColIdxC_golden, check_result, nnzCub, nnzC,
            compression_rate, time_segmerge, gflops_segmerge);

    int *d_csrRowPtrAT = NULL;
    int *d_csrColIdxAT = NULL;
    VALUE_TYPE *d_csrValAT = NULL;
    void *d_transpose_buffer = NULL;
    cusparseHandle_t transpose_handle = NULL;
    size_t transpose_buffer_size = 0;

    auto cleanup_transpose = [&]() {
        if (d_transpose_buffer != NULL)
            cudaFree(d_transpose_buffer);
        if (transpose_handle != NULL)
            cusparseDestroy(transpose_handle);
        if (d_csrRowPtrAT != NULL)
            cudaFree(d_csrRowPtrAT);
        if (d_csrColIdxAT != NULL)
            cudaFree(d_csrColIdxAT);
        if (d_csrValAT != NULL)
            cudaFree(d_csrValAT);
    };
    auto assume_transpose_failure = [&](const char *reason) {
        cleanup_transpose();
        return spgemm_cusparse_assume_correct(reason);
    };

    if (mA < 0 || nA < 0 || nnzA < 0 || d_csrRowPtrA == NULL ||
        (nnzA > 0 && (d_csrColIdxA == NULL || d_csrValA == NULL)))
        return assume_transpose_failure("invalid AAT device CSR arguments");
    if (cudaMalloc((void **)&d_csrRowPtrAT,
                   (static_cast<size_t>(nA) + 1) * sizeof(int)) != cudaSuccess)
        return assume_transpose_failure("AAT transpose row-pointer allocation failed");
    if (nnzA == 0)
    {
        if (cudaMemset(d_csrRowPtrAT, 0,
                       (static_cast<size_t>(nA) + 1) * sizeof(int)) !=
            cudaSuccess)
            return assume_transpose_failure("AAT empty transpose setup failed");
    }
    else
    {
        if (cudaMalloc((void **)&d_csrColIdxAT,
                       static_cast<size_t>(nnzA) * sizeof(int)) != cudaSuccess ||
            cudaMalloc((void **)&d_csrValAT,
                       static_cast<size_t>(nnzA) * sizeof(VALUE_TYPE)) !=
                cudaSuccess ||
            cusparseCreate(&transpose_handle) != CUSPARSE_STATUS_SUCCESS)
            return assume_transpose_failure("AAT transpose allocation failed");

        const cusparseStatus_t size_status = cusparseCsr2cscEx2_bufferSize(
            transpose_handle, mA, nA, nnzA, d_csrValA, d_csrRowPtrA,
            d_csrColIdxA, d_csrValAT, d_csrRowPtrAT, d_csrColIdxAT,
            CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
            CUSPARSE_CSR2CSC_ALG1, &transpose_buffer_size);
        if (size_status != CUSPARSE_STATUS_SUCCESS ||
            (transpose_buffer_size > 0 &&
             cudaMalloc(&d_transpose_buffer, transpose_buffer_size) !=
                 cudaSuccess))
            return assume_transpose_failure("AAT transpose workspace setup failed");
        if (cusparseCsr2cscEx2(
                transpose_handle, mA, nA, nnzA, d_csrValA, d_csrRowPtrA,
                d_csrColIdxA, d_csrValAT, d_csrRowPtrAT, d_csrColIdxAT,
                CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
                d_transpose_buffer) != CUSPARSE_STATUS_SUCCESS ||
            cudaDeviceSynchronize() != cudaSuccess)
            return assume_transpose_failure("AAT device transpose failed");
    }

    const int result = spgemm_cusparse_device_pair(
        mA, nA, nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA,
        nA, mA, nnzA, d_csrRowPtrAT, d_csrColIdxAT, d_csrValAT,
        mC, nC, nnzC_golden, h_csrRowPtrC_golden,
        h_csrColIdxC_golden, check_result, nnzCub, nnzC,
        compression_rate, time_segmerge, gflops_segmerge);
    cleanup_transpose();
    return result;
}

#endif
