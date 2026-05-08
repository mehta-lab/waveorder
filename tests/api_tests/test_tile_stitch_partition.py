"""Partition tests — input tile generation, output grid, contributor lookup.

Boundary cases get explicit coverage: tile larger than dim, overlap pushing
the last tile past the volume edge (squeeze redistribution), output-grid
clamping at the high-coordinate edge.
"""

import numpy as np
import pytest
import xarray as xr

from waveorder.tile_stitch.partition import (
    InputTile,
    OutputTile,
    _squeeze_positions,
    generate_output_tiles,
    generate_tiles,
    input_tiles_for_output,
)


def _data(shape: dict[str, int]) -> xr.DataArray:
    """Helper: zero-filled DataArray with named dims."""
    return xr.DataArray(np.zeros(tuple(shape.values())), dims=tuple(shape.keys()))


# --- _squeeze_positions ---


def test_squeeze_positions_tile_equals_dim_returns_zero():
    assert _squeeze_positions(64, 64, 0) == [0]


def test_squeeze_positions_tile_larger_than_dim_returns_zero():
    assert _squeeze_positions(50, 64, 0) == [0]


def test_squeeze_positions_no_overlap_evenly_divides():
    assert _squeeze_positions(256, 64, 0) == [0, 64, 128, 192]


def test_squeeze_positions_overlap_redistributes_when_last_short():
    """When last tile would fall short, squeeze redistributes to cover the edge."""
    starts = _squeeze_positions(100, 32, 8)
    assert starts[0] == 0
    assert starts[-1] + 32 == 100  # last tile lands flush against edge


def test_squeeze_positions_overlap_redistributes_when_last_overshoots():
    """When last tile would overshoot, squeeze pulls all tiles back."""
    starts = _squeeze_positions(95, 32, 8)
    assert starts[0] == 0
    assert starts[-1] + 32 == 95


def test_squeeze_positions_rejects_overlap_geq_tile():
    with pytest.raises(ValueError, match="overlap"):
        _squeeze_positions(256, 32, 32)
    with pytest.raises(ValueError, match="overlap"):
        _squeeze_positions(256, 32, 64)


# --- generate_tiles ---


def test_generate_tiles_unknown_dim_rejected():
    data = _data({"y": 64, "x": 64})
    with pytest.raises(ValueError, match="not found in data.dims"):
        generate_tiles(data, tile_size={"z": 32})


def test_generate_tiles_simple_2d_no_overlap():
    data = _data({"y": 64, "x": 64})
    tiles, dims = generate_tiles(data, tile_size={"y": 32, "x": 32})
    assert dims == ("y", "x")
    assert len(tiles) == 4
    assert all(t.shape == (32, 32) for t in tiles)


def test_generate_tiles_with_overlap_covers_volume():
    """Last tile in each dim must reach the volume edge (squeeze guarantee)."""
    data = _data({"y": 100, "x": 100})
    tiles, dims = generate_tiles(data, tile_size={"y": 32, "x": 32}, overlap={"y": 8, "x": 8})
    assert all(t.shape == (32, 32) for t in tiles)
    max_y_stop = max(t.slices["y"].stop for t in tiles)
    max_x_stop = max(t.slices["x"].stop for t in tiles)
    assert max_y_stop == 100
    assert max_x_stop == 100


def test_generate_tiles_dims_preserve_data_order():
    """tile_dims order matches data.dims order, not tile_size dict order."""
    data = _data({"z": 4, "y": 64, "x": 64})
    _, dims = generate_tiles(data, tile_size={"x": 32, "y": 32})
    assert dims == ("y", "x")


def test_generate_tiles_returns_unique_ids():
    data = _data({"y": 128, "x": 128})
    tiles, _ = generate_tiles(data, tile_size={"y": 32, "x": 32})
    assert len({t.tile_id for t in tiles}) == len(tiles)


# --- generate_output_tiles ---


def test_generate_output_tiles_clamps_last_tile_at_edge():
    """Final tile along each axis is clamped to full_shape (may be smaller)."""
    out = generate_output_tiles(full_shape={"y": 100, "x": 100}, tile_size={"y": 32, "x": 32}, tile_dims=("y", "x"))
    last = out[-1]
    assert last.slices["y"].stop == 100
    assert last.slices["x"].stop == 100


def test_generate_output_tiles_evenly_divides_no_clamp():
    out = generate_output_tiles(full_shape={"y": 64, "x": 64}, tile_size={"y": 32, "x": 32}, tile_dims=("y", "x"))
    assert len(out) == 4
    assert all(t.shape == (32, 32) for t in out)


def test_generate_output_tiles_unknown_dim_rejected():
    with pytest.raises(ValueError, match="not present in full_shape"):
        generate_output_tiles(full_shape={"y": 64}, tile_size={"z": 32}, tile_dims=("z",))


def test_generate_output_tiles_partition_covers_full_volume():
    """Union of output tile slices == full volume; tiles do not overlap."""
    full = {"y": 100, "x": 80}
    out = generate_output_tiles(full_shape=full, tile_size={"y": 32, "x": 32}, tile_dims=("y", "x"))

    mask = np.zeros((full["y"], full["x"]), dtype=int)
    for t in out:
        mask[t.slices["y"], t.slices["x"]] += 1
    assert mask.sum() == full["y"] * full["x"]
    assert mask.max() == 1  # no overlap


# --- input_tiles_for_output ---


def test_input_tiles_for_output_overlap_only():
    """Touching-but-not-overlapping tiles are excluded (half-open intervals)."""
    inputs = [
        InputTile(tile_id=0, slices={"y": slice(0, 32), "x": slice(0, 32)}),
        InputTile(tile_id=1, slices={"y": slice(32, 64), "x": slice(0, 32)}),  # touches in y, no overlap
        InputTile(tile_id=2, slices={"y": slice(0, 32), "x": slice(32, 64)}),  # touches in x, no overlap
        InputTile(tile_id=3, slices={"y": slice(28, 60), "x": slice(28, 60)}),  # overlaps
    ]
    out = OutputTile(tile_id=0, slices={"y": slice(0, 32), "x": slice(0, 32)})
    contribs = input_tiles_for_output(out, inputs, ("y", "x"))
    assert contribs == [0, 3]


def test_input_tiles_for_output_no_contributors_returns_empty():
    inputs = [InputTile(tile_id=0, slices={"y": slice(0, 32), "x": slice(0, 32)})]
    out = OutputTile(tile_id=0, slices={"y": slice(64, 96), "x": slice(64, 96)})
    assert input_tiles_for_output(out, inputs, ("y", "x")) == []


def test_input_tiles_for_output_nested_input_inside_output_counted():
    """Input fully inside output still contributes."""
    inputs = [InputTile(tile_id=7, slices={"y": slice(8, 24), "x": slice(8, 24)})]
    out = OutputTile(tile_id=0, slices={"y": slice(0, 32), "x": slice(0, 32)})
    assert input_tiles_for_output(out, inputs, ("y", "x")) == [7]


# --- Dataclass invariants ---


def test_input_tile_is_frozen_slots():
    t = InputTile(tile_id=0, slices={"y": slice(0, 32)})
    with pytest.raises(Exception):
        t.tile_id = 1  # type: ignore[misc]


def test_output_tile_bbox_helper():
    t = OutputTile(tile_id=0, slices={"y": slice(8, 40), "x": slice(0, 32)})
    assert t.bbox() == {"y": (8, 40), "x": (0, 32)}
    assert t.shape == (32, 32)
