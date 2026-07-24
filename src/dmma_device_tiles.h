#ifndef RTT_SPGEMM_DMMA_DEVICE_TILES_H_
#define RTT_SPGEMM_DMMA_DEVICE_TILES_H_

#include "common.h"

#include <cstdint>

struct DmmaDeviceTiles
{
    int rows;
    int cols;
    int tile_rows;
    int tile_cols;
    int tile_row_count;
    int tile_col_count;
    int num_tiles;
    int payload_size;
    int dense_tiles;
    int sparse_tiles;
    const int *tile_row_ptr;
    const int *tile_col_idx;
    const int *value_offsets;
    const std::uint32_t *masks;
    const MAT_VAL_TYPE *values;
    const int *tile_col_ptr;
    const int *tile_row_idx;
    const int *csc_tile_ids;
};

struct DmmaOwnedDeviceTiles
{
    DmmaDeviceTiles view{};
    int *tile_row_ptr = nullptr;
    void *metadata_storage = nullptr;
    int *tile_col_idx = nullptr;
    int *value_offsets = nullptr;
    std::uint32_t *masks = nullptr;
    MAT_VAL_TYPE *values = nullptr;
    int *tile_col_ptr = nullptr;
    int *tile_row_idx = nullptr;
    int *csc_tile_ids = nullptr;
};

#endif
