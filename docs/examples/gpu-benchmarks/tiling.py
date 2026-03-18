"""Tile generation and zarr reading utilities."""

from collections.abc import Iterator

import numpy as np
import xarray as xr


def tile_positions(
    fov_y: int, fov_x: int, tile_size: int,
) -> Iterator[tuple[int, int, int, int]]:
    """Yield (y_start, y_end, x_start, x_end) for non-overlapping tiles.

    Skips edge tiles that don't match tile_size exactly.
    """
    for y in range(0, fov_y, tile_size):
        for x in range(0, fov_x, tile_size):
            y_end = y + tile_size
            x_end = x + tile_size
            if y_end <= fov_y and x_end <= fov_x:
                yield y, y_end, x, x_end


def count_tiles(fov_y: int, fov_x: int, tile_size: int) -> int:
    """Return the number of full tiles that fit in the FOV."""
    return (fov_y // tile_size) * (fov_x // tile_size)


def read_tiles_from_zarr(
    position, tile_size: int, batch_size: int,
) -> list[xr.DataArray]:
    """Read tiles directly from zarr via oindex.

    Each tile is a separate zarr read: T=0, C=0 (Brightfield), all Z,
    then sliced spatially. Returns a list of CZYX xr.DataArrays.
    """
    shape = position.data.shape  # (T, C, Z, Y, X)
    _, _, _, fov_y, fov_x = shape

    tiles = []
    for y0, y1, x0, x1 in tile_positions(fov_y, fov_x, tile_size):
        tile_np = position.data.oindex[0, 0, :, y0:y1, x0:x1]  # (Z, tile, tile)
        da = xr.DataArray(
            tile_np[None].astype("float32"),  # (1, Z, tile, tile) = CZYX
            dims=("c", "z", "y", "x"),
        )
        tiles.append(da)
        if len(tiles) >= batch_size:
            break
    return tiles


def preload_fov(position) -> np.ndarray:
    """Read full FOV once into memory: T=0, C=0 → (Z, Y, X) float32."""
    return np.array(position.data.oindex[0, 0], dtype="float32")  # (Z, Y, X)


def slice_tiles_from_numpy(
    fov: np.ndarray, tile_size: int, batch_size: int,
) -> list[xr.DataArray]:
    """Slice tiles from a pre-loaded numpy FOV (no zarr I/O).

    fov is (Z, Y, X) float32. Returns list of CZYX xr.DataArrays.
    """
    _, fov_y, fov_x = fov.shape

    tiles = []
    for y0, y1, x0, x1 in tile_positions(fov_y, fov_x, tile_size):
        tile = fov[:, y0:y1, x0:x1].copy()  # (Z, tile, tile)
        da = xr.DataArray(
            tile[None],  # (1, Z, tile, tile) = CZYX
            dims=("c", "z", "y", "x"),
        )
        tiles.append(da)
        if len(tiles) >= batch_size:
            break
    return tiles
