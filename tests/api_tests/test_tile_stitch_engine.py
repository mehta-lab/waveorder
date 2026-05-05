"""Engine tests — build_plan + blend_output_tile + tile_stitch_reconstruction.

Phantom equivalence: tile-stitch over an overlapping grid must reproduce
the non-tiled reconstruction within a tolerance derived from the
expected accumulation error.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from waveorder.api.phase import Settings as PhaseSettings
from waveorder.api.tile_stitch import (
    BlendSettings,
    TileSettings,
    TileStitchPlan,
    TileStitchSettings,
    blend_output_tile,
    build_plan,
    gaussian_mean,
    max_blend,
    min_blend,
    reconstruct_tile,
    uniform_mean,
)
from waveorder.tile_stitch.partition import InputTile, OutputTile


# --- build_plan ---


def test_build_plan_simple_2d_no_overlap():
    data = xr.DataArray(np.zeros((1, 64, 64)), dims=("c", "y", "x"))
    settings = TileStitchSettings(
        tile=TileSettings(tile_size={"y": 32, "x": 32}),
        recon=PhaseSettings(),
    )
    plan = build_plan(data, settings)
    assert isinstance(plan, TileStitchPlan)
    assert len(plan.input_tiles) == 4
    assert len(plan.output_tiles) == 4
    assert plan.tile_dims == ("y", "x")
    assert plan.full_shape == {"y": 64, "x": 64}


def test_build_plan_with_overlap_reaches_volume_edge():
    data = xr.DataArray(np.zeros((100, 100)), dims=("y", "x"))
    settings = TileStitchSettings(
        tile=TileSettings(tile_size={"y": 32, "x": 32}, overlap={"y": 8, "x": 8}),
        recon=PhaseSettings(),
    )
    plan = build_plan(data, settings)
    max_y = max(t.slices["y"].stop for t in plan.input_tiles)
    max_x = max(t.slices["x"].stop for t in plan.input_tiles)
    assert max_y == 100
    assert max_x == 100


def test_build_plan_input_order_is_a_permutation():
    data = xr.DataArray(np.zeros((1, 96, 96)), dims=("c", "y", "x"))
    settings = TileStitchSettings(
        tile=TileSettings(tile_size={"y": 32, "x": 32}),
        recon=PhaseSettings(),
    )
    plan = build_plan(data, settings)
    assert sorted(plan.input_order) == sorted(it.tile_id for it in plan.input_tiles)


def test_build_plan_no_batching_by_default():
    data = xr.DataArray(np.zeros((64, 64)), dims=("y", "x"))
    settings = TileStitchSettings(
        tile=TileSettings(tile_size={"y": 32, "x": 32}),
        recon=PhaseSettings(),
    )
    plan = build_plan(data, settings)
    assert plan.input_batches is None
    assert plan.output_to_batches is None


def test_build_plan_with_batching_partitions_inputs():
    data = xr.DataArray(np.zeros((128, 128)), dims=("y", "x"))
    settings = TileStitchSettings(
        tile=TileSettings(tile_size={"y": 32, "x": 32}, overlap={"y": 8, "x": 8}),
        recon=PhaseSettings(),
    )
    plan = build_plan(data, settings, batch_size=2)
    assert plan.input_batches is not None
    assert plan.output_to_batches is not None
    flat = sorted(t for b in plan.input_batches for t in b)
    assert flat == sorted(it.tile_id for it in plan.input_tiles)


# --- blend_output_tile ---


def _two_tile_setup():
    """Two input tiles overlapping on x∈[16,32], one output tile over x∈[0,32]."""
    in_a = InputTile(tile_id=0, slices={"y": slice(0, 32), "x": slice(0, 24)})
    in_b = InputTile(tile_id=1, slices={"y": slice(0, 32), "x": slice(16, 32)})
    out = OutputTile(tile_id=0, slices={"y": slice(0, 32), "x": slice(0, 32)})
    return in_a, in_b, out


def test_blend_output_tile_uniform_mean_two_inputs_overlap():
    in_a, in_b, out = _two_tile_setup()
    recon_a = np.full((1, 32, 24), 1.0, dtype=np.float32)  # value 1 everywhere
    recon_b = np.full((1, 32, 16), 3.0, dtype=np.float32)  # value 3 everywhere
    result = blend_output_tile(
        out,
        contributors=[(in_a, recon_a), (in_b, recon_b)],
        blend=uniform_mean(),
        leading_shape=(1,),
        tile_dims=("y", "x"),
    )
    assert result.shape == (1, 32, 32)
    # Region x∈[0,16] only A → value 1.0
    np.testing.assert_allclose(result[0, :, :16], 1.0)
    # Region x∈[16,24] both A and B with equal weight → mean = 2.0
    np.testing.assert_allclose(result[0, :, 16:24], 2.0)
    # Region x∈[24,32] only B → value 3.0
    np.testing.assert_allclose(result[0, :, 24:32], 3.0)


def test_blend_output_tile_max_blend_picks_pixelwise_max():
    in_a, in_b, out = _two_tile_setup()
    recon_a = np.full((32, 24), 1.0, dtype=np.float32)
    recon_b = np.full((32, 16), 3.0, dtype=np.float32)
    result = blend_output_tile(
        out,
        contributors=[(in_a, recon_a), (in_b, recon_b)],
        blend=max_blend(),
        tile_dims=("y", "x"),
    )
    assert result.shape == (32, 32)
    np.testing.assert_array_equal(result[:, :16], 1.0)  # A only
    np.testing.assert_array_equal(result[:, 16:24], 3.0)  # max(1, 3)
    np.testing.assert_array_equal(result[:, 24:32], 3.0)  # B only


def test_blend_output_tile_min_blend_picks_pixelwise_min():
    in_a, in_b, out = _two_tile_setup()
    recon_a = np.full((32, 24), 1.0, dtype=np.float32)
    recon_b = np.full((32, 16), 3.0, dtype=np.float32)
    result = blend_output_tile(
        out,
        contributors=[(in_a, recon_a), (in_b, recon_b)],
        blend=min_blend(),
        tile_dims=("y", "x"),
    )
    np.testing.assert_array_equal(result[:, :16], 1.0)
    np.testing.assert_array_equal(result[:, 16:24], 1.0)  # min
    np.testing.assert_array_equal(result[:, 24:32], 3.0)


def test_blend_output_tile_no_contributors_returns_fill_value():
    out = OutputTile(tile_id=0, slices={"y": slice(0, 32), "x": slice(0, 32)})
    result = blend_output_tile(out, [], uniform_mean(), tile_dims=("y", "x"))
    assert result.shape == (32, 32)
    assert np.all(np.isnan(result))


def test_blend_output_tile_zero_volume_intersection_skipped():
    """A contributor that touches but does not overlap is silently skipped."""
    inp = InputTile(tile_id=0, slices={"y": slice(0, 16), "x": slice(0, 16)})
    out = OutputTile(tile_id=0, slices={"y": slice(16, 32), "x": slice(16, 32)})
    result = blend_output_tile(
        out,
        contributors=[(inp, np.full((16, 16), 1.0, dtype=np.float32))],
        blend=uniform_mean(),
        tile_dims=("y", "x"),
    )
    # Output is fill_value (NaN) — touching contributor never landed.
    assert np.all(np.isnan(result))


def test_blend_output_tile_gaussian_mean_weights_center_pixel_higher():
    """Gaussian-weighted mean: at each contributor's kernel center, the
    output is dominated by that contributor."""
    in_a = InputTile(tile_id=0, slices={"y": slice(0, 32), "x": slice(0, 32)})
    in_b = InputTile(tile_id=1, slices={"y": slice(16, 48), "x": slice(0, 32)})
    out = OutputTile(tile_id=0, slices={"y": slice(0, 48), "x": slice(0, 32)})
    recon_a = np.full((32, 32), 0.0, dtype=np.float32)
    recon_b = np.full((32, 32), 10.0, dtype=np.float32)
    result = blend_output_tile(
        out,
        contributors=[(in_a, recon_a), (in_b, recon_b)],
        blend=gaussian_mean(sigma_fraction=0.25),
        tile_dims=("y", "x"),
    )
    # Region y∈[0,16]: only A contributes → 0.0
    np.testing.assert_allclose(result[:16, :], 0.0, atol=1e-6)
    # Region y∈[32,48]: only B contributes → 10.0
    np.testing.assert_allclose(result[32:48, :], 10.0, atol=1e-6)
    # Output row near A's center (input-local y≈15.5 → output y=15) is dominated by A's value 0
    assert result[15, 0] < 2.0
    # Output row near B's center (output y=31, B-local y=15 close to B's center) dominated by B's value 10
    assert result[31, 0] > 8.0
    # In the overlap region [16,32], values are bounded by the inputs.
    assert np.all(result[16:32, :] >= 0.0)
    assert np.all(result[16:32, :] <= 10.0)


# --- reconstruct_tile ---


def test_reconstruct_tile_calls_recon_fn_returns_ndarray():
    block = xr.DataArray(np.ones((4, 8, 8)), dims=("c", "y", "x"))

    def recon_fn(b: xr.DataArray) -> xr.DataArray:
        return b * 2.0

    result = reconstruct_tile(block, recon_fn)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, 2.0)


# --- tile_stitch_reconstruction (phantom equivalence with identity recon) ---


def test_tile_stitch_reconstruction_identity_recon_reproduces_input():
    """If recon is the identity, tile-stitching with overlap reproduces the
    input within numerical tolerance for uniform_mean."""
    rng = np.random.default_rng(42)
    data_np = rng.normal(size=(1, 96, 96)).astype(np.float32)
    data = xr.DataArray(data_np, dims=("c", "y", "x"))

    settings = TileStitchSettings(
        tile=TileSettings(tile_size={"y": 32, "x": 32}, overlap={"y": 8, "x": 8}),
        blend=BlendSettings(kind="uniform_mean"),
        recon=PhaseSettings(),
    )

    # We can't use the real phase recon (requires a TF), so we invoke the
    # primitives directly with an identity recon.
    from waveorder.tile_stitch._engine import build_plan as _build_plan
    plan = _build_plan(data, settings)
    blend_kernel = settings.blend.build()
    tiles_by_id = {it.tile_id: it for it in plan.input_tiles}

    # Identity recon: each input tile's content == input slice.
    recon_by_tile = {
        it.tile_id: np.asarray(
            data.isel({d: it.slices[d] for d in plan.tile_dims}).values, dtype=np.float32
        )
        for it in plan.input_tiles
    }

    output = np.full(data_np.shape, blend_kernel.fill_value, dtype=np.float32)
    for ot in plan.output_tiles:
        contributors = [
            (tiles_by_id[tid], recon_by_tile[tid])
            for tid in plan.output_to_inputs[ot.tile_id]
        ]
        tile_result = blend_output_tile(
            ot,
            contributors,
            blend_kernel,
            leading_shape=(1,),
            tile_dims=plan.tile_dims,
        )
        write = (slice(None),) + tuple(ot.slices[d] for d in plan.tile_dims)
        output[write] = tile_result

    # Atol from derivation: each output pixel is the mean of N
    # contributors with equal weight; accumulated f32 error scales
    # linearly with N. Worst case here is ~4 contributors at internal
    # corners.
    atol = 4 * np.finfo(np.float32).eps * float(np.max(np.abs(data_np)))
    np.testing.assert_allclose(output, data_np, atol=atol, rtol=0)


def test_tile_stitch_reconstruction_orchestrator_calls_modality_apply_inverse(monkeypatch):
    """The orchestrator dispatches to phase.apply_inverse_transfer_function
    when recon.kind == 'phase'."""
    from waveorder.api import phase as wo_phase
    from waveorder.tile_stitch import _engine as _eng

    calls: list[xr.DataArray] = []

    def fake_apply(czyx, transfer_function, recon_dim, settings, device):  # noqa: ARG001
        calls.append(czyx)
        return czyx  # identity passthrough

    monkeypatch.setattr(wo_phase, "apply_inverse_transfer_function", fake_apply)

    data = xr.DataArray(np.ones((1, 64, 64), dtype=np.float32), dims=("c", "y", "x"))
    settings = TileStitchSettings(
        tile=TileSettings(tile_size={"y": 32, "x": 32}),
        recon=PhaseSettings(),
    )
    fake_tf = xr.Dataset()  # Not used by the fake apply

    out = _eng.tile_stitch_reconstruction(
        data, settings, transfer_function=fake_tf, device="cpu"
    )

    assert len(calls) == len(_eng.build_plan(data, settings).input_tiles)
    assert out.shape == data.shape
    np.testing.assert_allclose(out.values, 1.0)
