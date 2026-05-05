"""Input tiles + output partitions.

``generate_tiles`` splits a spatial volume into overlapping input tiles
(squeeze mode: last tile snaps to the edge and overlap is redistributed).

``generate_output_tiles`` builds a non-overlapping grid covering the
output volume — each output tile maps 1:1 to one zarr chunk so writes
don't contend.

``input_tiles_for_output`` returns the input tile ids whose bboxes
overlap a given output tile. Stage B reads each input tile once per
output tile, accumulates a weighted contribution, divides by total
weight, and writes the output tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import xarray as xr


@dataclass(frozen=True, slots=True)
class InputTile:
    """A single input tile: pixel-space slices into the source array."""

    tile_id: int
    slices: dict[str, slice]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(s.stop - s.start for s in self.slices.values())

    def bbox(self) -> dict[str, tuple[int, int]]:
        return {d: (s.start, s.stop) for d, s in self.slices.items()}


@dataclass(frozen=True, slots=True)
class OutputTile:
    """A non-overlapping output tile — partitions output volume.

    ``slices`` are pixel-space slices into the output array. One Stage B
    task = one output tile = exactly one output zarr chunk write.
    """

    tile_id: int
    slices: dict[str, slice]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(s.stop - s.start for s in self.slices.values())

    def bbox(self) -> dict[str, tuple[int, int]]:
        return {d: (s.start, s.stop) for d, s in self.slices.items()}


def _squeeze_positions(dim_size: int, tile: int, overlap: int) -> list[int]:
    if tile >= dim_size:
        return [0]
    if overlap >= tile:
        raise ValueError(f"overlap ({overlap}) must be < tile ({tile})")
    stride = tile - overlap
    starts = list(range(0, dim_size - tile + 1, stride))
    max_start = dim_size - tile
    last_end = starts[-1] + tile
    if last_end < dim_size:
        n = len(starts) + 1
        starts = [round(i * max_start / (n - 1)) for i in range(n)] if n > 1 else [0]
    elif last_end > dim_size and len(starts) > 1:
        n = len(starts)
        starts = [round(i * max_start / (n - 1)) for i in range(n)]
    return starts


def generate_tiles(
    data: xr.DataArray,
    tile_size: dict[str, int],
    overlap: dict[str, int] | None = None,
) -> tuple[list[InputTile], tuple[str, ...]]:
    """Generate overlapping input tiles over the spatial dims in ``tile_size``.

    Returns
    -------
    (tiles, tile_dims) where ``tile_dims`` is the ordered tuple of tiled
    dimension names (in the same order they appear on ``data``).
    """
    overlap = overlap or {}
    for d in tile_size:
        if d not in data.dims:
            raise ValueError(f"tile_size dim {d!r} not found in data.dims={data.dims}")
    tile_dims = tuple(str(d) for d in data.dims if d in tile_size)

    positions = {
        d: _squeeze_positions(int(data.sizes[d]), tile_size[d], overlap.get(d, 0))
        for d in tile_dims
    }

    tiles: list[InputTile] = []
    for tile_id, coord in enumerate(product(*(positions[d] for d in tile_dims))):
        slices = {
            d: slice(start, start + tile_size[d])
            for d, start in zip(tile_dims, coord, strict=True)
        }
        tiles.append(InputTile(tile_id=tile_id, slices=slices))
    return tiles, tile_dims


def generate_output_tiles(
    full_shape: dict[str, int],
    tile_size: dict[str, int],
    tile_dims: tuple[str, ...],
) -> list[OutputTile]:
    """Non-overlapping output tile grid covering the volume.

    Each output tile is ``tile_size`` along each dim except possibly the
    last along each axis (clamped to ``full_shape`` so we don't write
    past the volume). One output tile == one output zarr chunk when the
    output zarr is configured with the same chunk size.
    """
    if not set(tile_size).issubset(full_shape):
        missing = set(tile_size) - set(full_shape)
        raise ValueError(f"tile_size dims {missing} not present in full_shape={full_shape}")

    starts: dict[str, list[int]] = {}
    for d in tile_dims:
        size = full_shape[d]
        step = tile_size[d]
        starts[d] = list(range(0, size, step))

    tiles: list[OutputTile] = []
    for tile_id, coord in enumerate(product(*(starts[d] for d in tile_dims))):
        slices = {
            d: slice(start, min(start + tile_size[d], full_shape[d]))
            for d, start in zip(tile_dims, coord, strict=True)
        }
        tiles.append(OutputTile(tile_id=tile_id, slices=slices))
    return tiles


def input_tiles_for_output(
    out_tile: OutputTile,
    input_tiles: list[InputTile],
    tile_dims: tuple[str, ...],
) -> list[int]:
    """Return ids of input tiles whose bbox overlaps ``out_tile``.

    Bbox overlap on every spatial dim — half-open intervals so adjacent
    (touching but not overlapping) tiles are excluded.
    """
    contributors: list[int] = []
    for it in input_tiles:
        if all(
            it.slices[d].start < out_tile.slices[d].stop
            and out_tile.slices[d].start < it.slices[d].stop
            for d in tile_dims
        ):
            contributors.append(it.tile_id)
    return contributors
