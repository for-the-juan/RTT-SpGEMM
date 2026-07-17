#ifndef RTT_SPGEMM_DMMA_TILES_H_
#define RTT_SPGEMM_DMMA_TILES_H_

#include "common.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

static constexpr int DMMA_TILE_M = 8;
static constexpr int DMMA_TILE_K = 4;
static constexpr int DMMA_TILE_N = 8;
static constexpr int DMMA_INPUT_ELEMS = 32;
static constexpr int DMMA_OUTPUT_ELEMS = 64;

/*
 * Tile CSR with a mixed payload representation.  A tile is dense exactly
 * when value_offsets[tile + 1] - value_offsets[tile] is 32.  Otherwise its
 * values are packed in increasing physical mask-bit order.
 */
struct HybridTileMatrix
{
    int rows;
    int cols;
    int tile_rows;
    int tile_cols;
    int tile_row_count;
    int tile_col_count;

    MAT_PTR_TYPE num_tiles;
    MAT_PTR_TYPE payload_size;
    MAT_PTR_TYPE dense_tiles;
    MAT_PTR_TYPE sparse_tiles;

    MAT_PTR_TYPE *tile_row_ptr;
    int *tile_col_idx;
    MAT_PTR_TYPE *value_offsets;
    uint32_t *masks;
    MAT_VAL_TYPE *values;

    /* Optional tile CSC.  csc_tile_ids refer to IDs in the tile CSR above. */
    MAT_PTR_TYPE *tile_col_ptr;
    int *tile_row_idx;
    int *csc_tile_ids;

    HybridTileMatrix() noexcept
        : rows(0), cols(0), tile_rows(0), tile_cols(0), tile_row_count(0),
          tile_col_count(0), num_tiles(0), payload_size(0), dense_tiles(0),
          sparse_tiles(0), tile_row_ptr(nullptr), tile_col_idx(nullptr),
          value_offsets(nullptr), masks(nullptr), values(nullptr),
          tile_col_ptr(nullptr), tile_row_idx(nullptr), csc_tile_ids(nullptr)
    {
    }
};

static inline void destroy_hybrid_tile_matrix(HybridTileMatrix *matrix)
{
    if (matrix == nullptr)
        return;

    std::free(matrix->tile_row_ptr);
    std::free(matrix->tile_col_idx);
    std::free(matrix->value_offsets);
    std::free(matrix->masks);
    std::free(matrix->values);
    std::free(matrix->tile_col_ptr);
    std::free(matrix->tile_row_idx);
    std::free(matrix->csc_tile_ids);
    *matrix = HybridTileMatrix();
}

namespace dmma_tiles_detail
{

static_assert(std::numeric_limits<MAT_PTR_TYPE>::is_integer,
              "MAT_PTR_TYPE must be an integer type");

struct RawTileEntry
{
    int tile_col;
    uint8_t physical_index;
    MAT_VAL_TYPE value;
};

template <typename T>
static inline T *allocate_array(std::size_t count)
{
    if (count == 0)
        return nullptr;
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        return nullptr;
    return static_cast<T *>(std::malloc(count * sizeof(T)));
}

static inline bool fits_mat_ptr(std::size_t value)
{
    return static_cast<uintmax_t>(value) <=
           static_cast<uintmax_t>(std::numeric_limits<MAT_PTR_TYPE>::max());
}

static inline bool mat_ptr_to_size(MAT_PTR_TYPE value, std::size_t upper_bound,
                                   std::size_t *converted)
{
    if constexpr (std::numeric_limits<MAT_PTR_TYPE>::is_signed)
    {
        if (value < 0)
            return false;
    }

    const uintmax_t unsigned_value = static_cast<uintmax_t>(value);
    if (unsigned_value > static_cast<uintmax_t>(upper_bound))
        return false;

    *converted = static_cast<std::size_t>(unsigned_value);
    return true;
}

static inline bool validate_csr(const SMatrix *matrix)
{
    if (matrix == nullptr || matrix->m < 0 || matrix->n < 0 || matrix->nnz < 0)
        return false;

    const std::size_t nnz = static_cast<std::size_t>(matrix->nnz);
    if (matrix->m == 0)
        return matrix->nnz == 0;
    if (matrix->rowpointer == nullptr)
        return false;
    if (nnz != 0 && (matrix->columnindex == nullptr || matrix->value == nullptr))
        return false;

    std::size_t previous = 0;
    if (!mat_ptr_to_size(matrix->rowpointer[0], nnz, &previous) || previous != 0)
        return false;

    for (int row = 0; row < matrix->m; ++row)
    {
        std::size_t next = 0;
        if (!mat_ptr_to_size(matrix->rowpointer[row + 1], nnz, &next) ||
            next < previous)
            return false;
        previous = next;
    }
    if (previous != nnz)
        return false;

    for (std::size_t i = 0; i < nnz; ++i)
    {
        const int col = matrix->columnindex[i];
        if (col < 0 || col >= matrix->n)
            return false;
    }
    return true;
}

template <typename Consumer>
static inline void for_each_tile_in_row(const SMatrix *matrix, int tile_rows,
                                        int tile_cols, bool payload_col_major,
                                        int tile_row, Consumer &&consume)
{
    const int first_row = tile_row * tile_rows;
    const int rows_remaining = matrix->m - first_row;
    const int last_row = first_row + std::min(tile_rows, rows_remaining);

    std::size_t entry_count = 0;
    for (int row = first_row; row < last_row; ++row)
    {
        const std::size_t begin = static_cast<std::size_t>(matrix->rowpointer[row]);
        const std::size_t end = static_cast<std::size_t>(matrix->rowpointer[row + 1]);
        entry_count += end - begin;
    }

    std::vector<RawTileEntry> entries;
    entries.reserve(entry_count);
    for (int row = first_row; row < last_row; ++row)
    {
        const int local_row = row - first_row;
        const std::size_t begin = static_cast<std::size_t>(matrix->rowpointer[row]);
        const std::size_t end = static_cast<std::size_t>(matrix->rowpointer[row + 1]);
        for (std::size_t i = begin; i < end; ++i)
        {
            const int col = matrix->columnindex[i];
            const int local_col = col % tile_cols;
            const int physical = payload_col_major
                                     ? local_col * tile_rows + local_row
                                     : local_row * tile_cols + local_col;
            entries.push_back(
                {col / tile_cols, static_cast<uint8_t>(physical), matrix->value[i]});
        }
    }

    std::sort(entries.begin(), entries.end(),
              [](const RawTileEntry &lhs, const RawTileEntry &rhs) {
                  if (lhs.tile_col != rhs.tile_col)
                      return lhs.tile_col < rhs.tile_col;
                  return lhs.physical_index < rhs.physical_index;
              });

    std::size_t cursor = 0;
    while (cursor < entries.size())
    {
        const int current_tile_col = entries[cursor].tile_col;
        MAT_VAL_TYPE physical_values[DMMA_INPUT_ELEMS] = {};
        uint32_t mask = 0;
        int unique_nnz = 0;

        while (cursor < entries.size() &&
               entries[cursor].tile_col == current_tile_col)
        {
            const uint8_t physical = entries[cursor].physical_index;
            MAT_VAL_TYPE sum = MAT_VAL_TYPE(0);
            do
            {
                sum += entries[cursor].value;
                ++cursor;
            } while (cursor < entries.size() &&
                     entries[cursor].tile_col == current_tile_col &&
                     entries[cursor].physical_index == physical);

            physical_values[physical] = sum;
            mask |= uint32_t(1) << physical;
            ++unique_nnz;
        }

        consume(current_tile_col, mask, physical_values, unique_nnz);
    }
}

} // namespace dmma_tiles_detail

/*
 * Convert scalar CSR to 32-element input tiles.  For A use (8, 4, false),
 * and for B use (4, 8, true).  The conversion is transactional: on failure
 * `out` is left unchanged.  `out` must be a constructed HybridTileMatrix.
 */
static inline bool csr_to_hybrid_tiles(const SMatrix *matrix, int tile_rows,
                                       int tile_cols, bool payload_col_major,
                                       int dense_threshold, bool build_csc,
                                       HybridTileMatrix *out)
{
    using namespace dmma_tiles_detail;

    if (out == nullptr || !validate_csr(matrix) || tile_rows <= 0 ||
        tile_cols <= 0 || tile_rows > DMMA_INPUT_ELEMS ||
        tile_cols > DMMA_INPUT_ELEMS ||
        tile_rows * tile_cols != DMMA_INPUT_ELEMS || dense_threshold < 1 ||
        dense_threshold > DMMA_INPUT_ELEMS)
        return false;

    HybridTileMatrix result;
    try
    {
        result.rows = matrix->m;
        result.cols = matrix->n;
        result.tile_rows = tile_rows;
        result.tile_cols = tile_cols;
        result.tile_row_count = matrix->m / tile_rows +
                                (matrix->m % tile_rows != 0 ? 1 : 0);
        result.tile_col_count = matrix->n / tile_cols +
                                (matrix->n % tile_cols != 0 ? 1 : 0);

        const std::size_t tile_row_pointer_count =
            static_cast<std::size_t>(result.tile_row_count) + 1;
        result.tile_row_ptr =
            allocate_array<MAT_PTR_TYPE>(tile_row_pointer_count);
        if (result.tile_row_ptr == nullptr)
            return false;
        result.tile_row_ptr[0] = MAT_PTR_TYPE(0);

        std::size_t total_tiles = 0;
        std::size_t total_payload = 0;
        std::size_t total_dense = 0;
        std::size_t total_sparse = 0;

        for (int tile_row = 0; tile_row < result.tile_row_count; ++tile_row)
        {
            std::size_t row_tiles = 0;
            for_each_tile_in_row(
                matrix, tile_rows, tile_cols, payload_col_major, tile_row,
                [&](int, uint32_t, const MAT_VAL_TYPE *, int unique_nnz) {
                    ++row_tiles;
                    if (unique_nnz >= dense_threshold)
                    {
                        total_payload += DMMA_INPUT_ELEMS;
                        ++total_dense;
                    }
                    else
                    {
                        total_payload += static_cast<std::size_t>(unique_nnz);
                        ++total_sparse;
                    }
                });

            if (row_tiles > std::numeric_limits<std::size_t>::max() - total_tiles)
            {
                destroy_hybrid_tile_matrix(&result);
                return false;
            }
            total_tiles += row_tiles;
            if (!fits_mat_ptr(total_tiles) || !fits_mat_ptr(total_payload))
            {
                destroy_hybrid_tile_matrix(&result);
                return false;
            }
            result.tile_row_ptr[tile_row + 1] =
                static_cast<MAT_PTR_TYPE>(total_tiles);
        }

        if (build_csc && total_tiles > static_cast<std::size_t>(INT_MAX))
        {
            destroy_hybrid_tile_matrix(&result);
            return false;
        }

        result.num_tiles = static_cast<MAT_PTR_TYPE>(total_tiles);
        result.payload_size = static_cast<MAT_PTR_TYPE>(total_payload);
        result.dense_tiles = static_cast<MAT_PTR_TYPE>(total_dense);
        result.sparse_tiles = static_cast<MAT_PTR_TYPE>(total_sparse);

        result.tile_col_idx = allocate_array<int>(total_tiles);
        result.value_offsets =
            allocate_array<MAT_PTR_TYPE>(total_tiles + std::size_t(1));
        result.masks = allocate_array<uint32_t>(total_tiles);
        result.values = allocate_array<MAT_VAL_TYPE>(total_payload);
        if ((total_tiles != 0 &&
             (result.tile_col_idx == nullptr || result.masks == nullptr)) ||
            result.value_offsets == nullptr ||
            (total_payload != 0 && result.values == nullptr))
        {
            destroy_hybrid_tile_matrix(&result);
            return false;
        }

        std::size_t tile_cursor = 0;
        std::size_t payload_cursor = 0;
        result.value_offsets[0] = MAT_PTR_TYPE(0);
        for (int tile_row = 0; tile_row < result.tile_row_count; ++tile_row)
        {
            for_each_tile_in_row(
                matrix, tile_rows, tile_cols, payload_col_major, tile_row,
                [&](int tile_col, uint32_t mask,
                    const MAT_VAL_TYPE *physical_values, int unique_nnz) {
                    result.tile_col_idx[tile_cursor] = tile_col;
                    result.masks[tile_cursor] = mask;

                    if (unique_nnz >= dense_threshold)
                    {
                        for (int physical = 0; physical < DMMA_INPUT_ELEMS;
                             ++physical)
                            result.values[payload_cursor++] =
                                physical_values[physical];
                    }
                    else
                    {
                        for (int physical = 0; physical < DMMA_INPUT_ELEMS;
                             ++physical)
                        {
                            if ((mask & (uint32_t(1) << physical)) != 0)
                                result.values[payload_cursor++] =
                                    physical_values[physical];
                        }
                    }

                    ++tile_cursor;
                    result.value_offsets[tile_cursor] =
                        static_cast<MAT_PTR_TYPE>(payload_cursor);
                });
        }

        if (tile_cursor != total_tiles || payload_cursor != total_payload)
        {
            destroy_hybrid_tile_matrix(&result);
            return false;
        }

        if (build_csc)
        {
            const std::size_t tile_col_pointer_count =
                static_cast<std::size_t>(result.tile_col_count) + 1;
            result.tile_col_ptr =
                allocate_array<MAT_PTR_TYPE>(tile_col_pointer_count);
            result.tile_row_idx = allocate_array<int>(total_tiles);
            result.csc_tile_ids = allocate_array<int>(total_tiles);
            if (result.tile_col_ptr == nullptr ||
                (total_tiles != 0 &&
                 (result.tile_row_idx == nullptr || result.csc_tile_ids == nullptr)))
            {
                destroy_hybrid_tile_matrix(&result);
                return false;
            }

            std::memset(result.tile_col_ptr, 0,
                        tile_col_pointer_count * sizeof(MAT_PTR_TYPE));
            for (std::size_t tile = 0; tile < total_tiles; ++tile)
                ++result.tile_col_ptr[result.tile_col_idx[tile] + 1];
            for (int col = 0; col < result.tile_col_count; ++col)
                result.tile_col_ptr[col + 1] += result.tile_col_ptr[col];

            std::vector<MAT_PTR_TYPE> next(
                result.tile_col_ptr,
                result.tile_col_ptr + static_cast<std::size_t>(result.tile_col_count));
            for (int tile_row = 0; tile_row < result.tile_row_count; ++tile_row)
            {
                const std::size_t begin =
                    static_cast<std::size_t>(result.tile_row_ptr[tile_row]);
                const std::size_t end =
                    static_cast<std::size_t>(result.tile_row_ptr[tile_row + 1]);
                for (std::size_t tile = begin; tile < end; ++tile)
                {
                    const int tile_col = result.tile_col_idx[tile];
                    const std::size_t destination =
                        static_cast<std::size_t>(next[tile_col]++);
                    result.tile_row_idx[destination] = tile_row;
                    result.csc_tile_ids[destination] = static_cast<int>(tile);
                }
            }
        }
    }
    catch (const std::bad_alloc &)
    {
        destroy_hybrid_tile_matrix(&result);
        return false;
    }
    catch (...)
    {
        destroy_hybrid_tile_matrix(&result);
        return false;
    }

    destroy_hybrid_tile_matrix(out);
    *out = result;
    return true;
}

#endif // RTT_SPGEMM_DMMA_TILES_H_
