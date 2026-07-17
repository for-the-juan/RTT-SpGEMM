#include "common.h"
#include <cuda_runtime.h>
#include "external/cusparse/spgemm_cusparse.h"

int spgemm_cu (         const int             mA,
                        const int             nA,
                        const int             nnzA,
                        const MAT_PTR_TYPE   *csrRowPtrA,
                        const int            *csrColIdxA,
                        const MAT_VAL_TYPE   *csrValA,
                        const int             mB,
                        const int             nB,
                        const int             nnzB,
                        const MAT_PTR_TYPE   *csrRowPtrB,
                        const int            *csrColIdxB,
                        const MAT_VAL_TYPE   *csrValB,
                        const int             mC,
                        const int             nC,
                        const MAT_PTR_TYPE    nnzC_golden,
                        const MAT_PTR_TYPE   *csrRowPtrC_golden,
                        const int            *csrColIdxC_golden,
                        const MAT_VAL_TYPE   *csrValC_golden,
                        const bool           check_result,
                        unsigned long long int nnzCub,
                        unsigned long long int *nnzC,
                        double        *compression_rate,
                        double        *time_segmerge,
                        double        *gflops_segmerge )
{
    // run cuda SpGEMM (using cuSPARSE)
    printf("\n--------------- SpGEMM (using cuSPARSE) ---------------\n");
    double compression_rate1 = 0;
    double time_cusparse = 0;
    double gflops_cusparse = 0;
    const int validation_status = spgemm_cusparse(
        mA, nA, nnzA, csrRowPtrA, csrColIdxA, csrValA,
        mB, nB, nnzB, csrRowPtrB, csrColIdxB, csrValB,
        mC, nC, nnzC_golden, csrRowPtrC_golden, csrColIdxC_golden,
        csrValC_golden, check_result, nnzCub, nnzC, &compression_rate1,
        &time_cusparse, &gflops_cusparse);
    if (compression_rate != nullptr)
        *compression_rate = compression_rate1;
    if (time_segmerge != nullptr)
        *time_segmerge = time_cusparse;
    if (gflops_segmerge != nullptr)
        *gflops_segmerge = gflops_cusparse;
    printf("---------------------------------------------------------------\n");





    return validation_status;
}

/* Validate against cuSPARSE while borrowing the production device CSR. */
static inline int spgemm_cu_device(
    const int mA, const int nA, const int nnzA,
    const MAT_PTR_TYPE *d_csrRowPtrA, const int *d_csrColIdxA,
    const MAT_VAL_TYPE *d_csrValA, const bool aat,
    const int mC, const int nC, const MAT_PTR_TYPE nnzC_golden,
    const MAT_PTR_TYPE *csrRowPtrC_golden,
    const int *csrColIdxC_golden, const bool check_result,
    unsigned long long int nnzCub, unsigned long long int *nnzC,
    double *compression_rate, double *time_segmerge,
    double *gflops_segmerge)
{
    printf("\n--------------- SpGEMM (using cuSPARSE) ---------------\n");
    const int validation_status = spgemm_cusparse_device(
        mA, nA, nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA, aat,
        mC, nC, nnzC_golden, csrRowPtrC_golden, csrColIdxC_golden,
        check_result, nnzCub, nnzC, compression_rate, time_segmerge,
        gflops_segmerge);
    printf("---------------------------------------------------------------\n");
    return validation_status;
}

