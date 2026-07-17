#ifndef RTT_SPGEMM_COMMON_H_
#define RTT_SPGEMM_COMMON_H_

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mm_malloc.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <nmmintrin.h>


#include <omp.h>

#include <sys/time.h>
#include "cuda_fp16.h"

#ifndef MAT_VAL_TYPE
#define MAT_VAL_TYPE double
#endif

#ifndef MAT_PTR_TYPE
#define MAT_PTR_TYPE int
#endif

#define WARP_SIZE 32

/* Retained only for source compatibility with the legacy CPU reference. */
#ifndef BLOCK_SIZE
#define BLOCK_SIZE  16
#endif

#ifndef TIMING
#define TIMING 1
#endif

#ifndef SPACE
#define SPACE 1
#endif


#ifndef CHECK_RESULT
#define CHECK_RESULT 1
#endif

#ifndef SMATRIX
#define SMATRIX
typedef struct 
{
    int m;
    int n;
    int nnz;
    int isSymmetric;
	MAT_VAL_TYPE *value;
	int *columnindex;
	MAT_PTR_TYPE *rowpointer;
    int tilem;
    int tilen;
    MAT_PTR_TYPE *tile_ptr;
    int *tile_columnidx;
    int *tile_rowidx;
    int *tile_nnz;
    int numtile;
    MAT_VAL_TYPE *tile_csr_Value;
    unsigned char *tile_csr_Col;
    unsigned char *tile_csr_Ptr;
    unsigned short *mask;
    int *csc_tile_ptr;
    int *csc_tile_rowidx;
}SMatrix;
#endif

/* Utility routines depend on MAT_PTR_TYPE and SMatrix above. */
#include "utils.h"

#endif // RTT_SPGEMM_COMMON_H_

