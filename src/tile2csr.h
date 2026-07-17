#ifndef _TILETOCSR_
#define _TILETOCSR_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

/*
 * The DMMA output fragment is 8x8.  Keep this header usable on its own while
 * allowing the shared tile geometry header to be introduced independently.
 */
#if defined(__has_include)
#if __has_include("dmma_tiles.h")
#include "dmma_tiles.h"
#define RTT_TILE2CSR_HAS_DMMA_TILES 1
#endif
#endif

#if defined(RTT_TILE2CSR_HAS_DMMA_TILES)
static const int OUTPUT_TILE_ROWS = DMMA_TILE_M;
static const int OUTPUT_TILE_COLS = DMMA_TILE_N;
#else
static const int OUTPUT_TILE_ROWS = 8;
static const int OUTPUT_TILE_COLS = 8;
#endif

static inline int tile2csr_row_count(int tile_row, int m)
{
    const long long row_begin = (long long)tile_row * OUTPUT_TILE_ROWS;
    const long long remaining = (long long)m - row_begin;

    if (remaining <= 0)
        return 0;
    return remaining < OUTPUT_TILE_ROWS ? (int)remaining : OUTPUT_TILE_ROWS;
}

static inline void tile2csr_row_bounds(const unsigned char *tile_csr_ptr,
                                       int tile_csrptr_offset,
                                       int local_row,
                                       int row_count,
                                       int tilennz,
                                       int *start,
                                       int *end)
{
    int row_start = tile_csr_ptr[tile_csrptr_offset + local_row];
    int row_end = local_row + 1 == row_count
                      ? tilennz
                      : tile_csr_ptr[tile_csrptr_offset + local_row + 1];

    /* Do not let malformed or truncated boundary metadata escape the tile. */
    if (row_start < 0)
        row_start = 0;
    if (row_start > tilennz)
        row_start = tilennz;
    if (row_end < row_start)
        row_end = row_start;
    if (row_end > tilennz)
        row_end = tilennz;

    *start = row_start;
    *end = row_end;
}

static inline int tile2csr_valid_global_col(int tile_col,
                                            unsigned char encoded_col,
                                            int n,
                                            int *global_col)
{
    const int local_col = (int)encoded_col;
    if (local_col >= OUTPUT_TILE_COLS || tile_col < 0)
        return 0;

    const long long col = (long long)tile_col * OUTPUT_TILE_COLS + local_col;
    if (col < 0 || col >= n)
        return 0;

    *global_col = (int)col;
    return 1;
}

static void Tile_csr_to_csr_PTR(const unsigned char *tile_csr_ptr,
                                const unsigned char *tile_csr_col,
                                int tilennz,
                                int m,
                                int n,
                                int tile_row,
                                int tile_col,
                                MAT_PTR_TYPE *csr_row_ptr,
                                int tile_csrptr_offset,
                                int tile_csr_index_offset)
{
    const int row_count = tile2csr_row_count(tile_row, m);
    const int csr_ptr_offset = tile_row * OUTPUT_TILE_ROWS;

    for (int local_row = 0; local_row < row_count; ++local_row)
    {
        int start;
        int end;
        MAT_PTR_TYPE valid_count = 0;
        tile2csr_row_bounds(tile_csr_ptr, tile_csrptr_offset, local_row,
                            row_count, tilennz, &start, &end);

        for (int entry = start; entry < end; ++entry)
        {
            int global_col;
            if (tile2csr_valid_global_col(
                    tile_col, tile_csr_col[tile_csr_index_offset + entry], n,
                    &global_col))
                ++valid_count;
        }
        csr_row_ptr[csr_ptr_offset + local_row] += valid_count;
    }
}

static void Tile_csr_to_csr(const unsigned char *tile_csr_ptr,
                            const unsigned char *tile_csr_col,
                            const MAT_VAL_TYPE *tile_csr_val,
                            int tilennz,
                            int m,
                            int n,
                            int tile_row,
                            int tile_col,
                            const MAT_PTR_TYPE *csr_row_ptr,
                            int *csr_col_idx,
                            MAT_VAL_TYPE *csr_val,
                            int tile_csrptr_offset,
                            int tile_csr_index_offset,
                            MAT_PTR_TYPE *row_nnz_offset)
{
    const int row_count = tile2csr_row_count(tile_row, m);
    const int csr_ptr_offset = tile_row * OUTPUT_TILE_ROWS;

    for (int local_row = 0; local_row < row_count; ++local_row)
    {
        const int global_row = csr_ptr_offset + local_row;
        int start;
        int end;
        tile2csr_row_bounds(tile_csr_ptr, tile_csrptr_offset, local_row,
                            row_count, tilennz, &start, &end);

        for (int entry = start; entry < end; ++entry)
        {
            int global_col;
            if (!tile2csr_valid_global_col(
                    tile_col, tile_csr_col[tile_csr_index_offset + entry], n,
                    &global_col))
                continue;

            const MAT_PTR_TYPE output =
                csr_row_ptr[global_row] + row_nnz_offset[global_row]++;
            csr_col_idx[output] = global_col;
            csr_val[output] = tile_csr_val[tile_csr_index_offset + entry];
        }
    }
}

static inline void tile2csr_swap_entries(SMatrix *matrix,
                                         MAT_PTR_TYPE lhs,
                                         MAT_PTR_TYPE rhs)
{
    const int column = matrix->columnindex[lhs];
    const MAT_VAL_TYPE value = matrix->value[lhs];
    matrix->columnindex[lhs] = matrix->columnindex[rhs];
    matrix->value[lhs] = matrix->value[rhs];
    matrix->columnindex[rhs] = column;
    matrix->value[rhs] = value;
}

static inline void tile2csr_sift_down(SMatrix *matrix,
                                      MAT_PTR_TYPE begin,
                                      MAT_PTR_TYPE root,
                                      MAT_PTR_TYPE count)
{
    for (;;)
    {
        MAT_PTR_TYPE child = root * 2 + 1;
        if (child >= count)
            return;

        if (child + 1 < count &&
            matrix->columnindex[begin + child] <
                matrix->columnindex[begin + child + 1])
            ++child;

        if (matrix->columnindex[begin + root] >=
            matrix->columnindex[begin + child])
            return;

        tile2csr_swap_entries(matrix, begin + root, begin + child);
        root = child;
    }
}

/* Sort columns deterministically and carry values along without comparing them. */
static void tile2csr_sort_rows(SMatrix *matrix)
{
#pragma omp parallel for
    for (int row = 0; row < matrix->m; ++row)
    {
        const MAT_PTR_TYPE begin = matrix->rowpointer[row];
        const MAT_PTR_TYPE end = matrix->rowpointer[row + 1];
        const MAT_PTR_TYPE count = end - begin;

        for (MAT_PTR_TYPE root = count / 2; root > 0; --root)
            tile2csr_sift_down(matrix, begin, root - 1, count);

        for (MAT_PTR_TYPE remaining = count; remaining > 1; --remaining)
        {
            tile2csr_swap_entries(matrix, begin, begin + remaining - 1);
            tile2csr_sift_down(matrix, begin, 0, remaining - 1);
        }
    }
}

bool tile2csr(SMatrix *matrix)
{
    if (matrix == NULL || matrix->m < 0 || matrix->n < 0 ||
        matrix->tilem < 0 || matrix->numtile < 0 ||
        matrix->tile_ptr == NULL || matrix->tile_nnz == NULL ||
        (matrix->numtile > 0 &&
         (matrix->tile_columnidx == NULL || matrix->tile_csr_Ptr == NULL)) ||
        (matrix->nnz > 0 &&
         (matrix->tile_csr_Col == NULL || matrix->tile_csr_Value == NULL)))
        return false;

    matrix->rowpointer =
        (MAT_PTR_TYPE *)malloc((matrix->m + 1) * sizeof(MAT_PTR_TYPE));
    if (matrix->rowpointer == NULL)
        return false;
    MAT_PTR_TYPE *csr_row_ptr = matrix->rowpointer;
    memset(csr_row_ptr, 0, (matrix->m + 1) * sizeof(MAT_PTR_TYPE));

#pragma omp parallel for
    for (int tile_row = 0; tile_row < matrix->tilem; ++tile_row)
    {
        for (MAT_PTR_TYPE tile_id = matrix->tile_ptr[tile_row];
             tile_id < matrix->tile_ptr[tile_row + 1]; ++tile_id)
        {
            const int tilennz = matrix->tile_nnz[tile_id + 1] -
                                matrix->tile_nnz[tile_id];
            const int tile_csr_index_offset = matrix->tile_nnz[tile_id];
            const int tile_csrptr_offset = tile_id * OUTPUT_TILE_ROWS;
            const int tile_col = matrix->tile_columnidx[tile_id];

            Tile_csr_to_csr_PTR(
                matrix->tile_csr_Ptr, matrix->tile_csr_Col, tilennz,
                matrix->m, matrix->n, tile_row, tile_col, csr_row_ptr,
                tile_csrptr_offset, tile_csr_index_offset);
        }
    }
    exclusive_scan(csr_row_ptr, matrix->m + 1);

    matrix->nnz = csr_row_ptr[matrix->m];
    if (matrix->nnz > 0)
    {
        matrix->value =
            (MAT_VAL_TYPE *)malloc(matrix->nnz * sizeof(MAT_VAL_TYPE));
        matrix->columnindex = (int *)malloc(matrix->nnz * sizeof(int));
        if (matrix->value == NULL || matrix->columnindex == NULL)
        {
            free(matrix->value);
            free(matrix->columnindex);
            free(matrix->rowpointer);
            matrix->value = NULL;
            matrix->columnindex = NULL;
            matrix->rowpointer = NULL;
            return false;
        }
        memset(matrix->value, 0, matrix->nnz * sizeof(MAT_VAL_TYPE));
        memset(matrix->columnindex, 0, matrix->nnz * sizeof(int));
    }
    else
    {
        matrix->value = NULL;
        matrix->columnindex = NULL;
    }

    MAT_PTR_TYPE *row_nnz_offset =
        (MAT_PTR_TYPE *)malloc(matrix->m * sizeof(MAT_PTR_TYPE));
    if (matrix->m > 0 && row_nnz_offset == NULL)
    {
        free(matrix->value);
        free(matrix->columnindex);
        free(matrix->rowpointer);
        matrix->value = NULL;
        matrix->columnindex = NULL;
        matrix->rowpointer = NULL;
        return false;
    }
    if (matrix->m > 0)
        memset(row_nnz_offset, 0, matrix->m * sizeof(MAT_PTR_TYPE));

#pragma omp parallel for
    for (int tile_row = 0; tile_row < matrix->tilem; ++tile_row)
    {
        for (MAT_PTR_TYPE tile_id = matrix->tile_ptr[tile_row];
             tile_id < matrix->tile_ptr[tile_row + 1]; ++tile_id)
        {
            const int tilennz = matrix->tile_nnz[tile_id + 1] -
                                matrix->tile_nnz[tile_id];
            const int tile_csr_index_offset = matrix->tile_nnz[tile_id];
            const int tile_csrptr_offset = tile_id * OUTPUT_TILE_ROWS;
            const int tile_col = matrix->tile_columnidx[tile_id];

            Tile_csr_to_csr(
                matrix->tile_csr_Ptr, matrix->tile_csr_Col,
                matrix->tile_csr_Value, tilennz, matrix->m, matrix->n,
                tile_row, tile_col, csr_row_ptr, matrix->columnindex,
                matrix->value, tile_csrptr_offset, tile_csr_index_offset,
                row_nnz_offset);
        }
    }

    free(row_nnz_offset);
    tile2csr_sort_rows(matrix);
    return true;
}

#undef RTT_TILE2CSR_HAS_DMMA_TILES

#endif
