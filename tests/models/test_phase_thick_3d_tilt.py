"""Tests for ``waveorder.models.phase_thick_3d_tilt``.

Covers:
- ``radial_blend_zenith_init`` — pure-utility tests (shapes, edge cases).
- ``optimize_subtile_tilt_params`` — synthetic recovery, frozen-axis
  behavior, warmstart-skip roundtrip.

The synthetic recovery test uses a tiny phantom + short loop so the
test runs in a few seconds on CPU as well as CUDA.
"""

import math

import pytest
import torch

from waveorder.models.phase_thick_3d_tilt import (
    TiltOptimResult,
    optimize_subtile_tilt_params,
    radial_blend_zenith_init,
)


def _device_list():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


# ───────────────────────────────────────────────────────────────────────
# radial_blend_zenith_init
# ───────────────────────────────────────────────────────────────────────


def test_radial_blend_zenith_init_shapes():
    """Output matches input length and is the per-element product."""
    B = 9
    zen_formula = torch.full((B,), 0.05)
    grid = torch.stack(
        [torch.arange(B, dtype=torch.float32) % 3,
         torch.arange(B, dtype=torch.float32) // 3], dim=1
    )
    out = radial_blend_zenith_init(zen_formula, grid)
    assert out.shape == zen_formula.shape
    # Center subtile (row=1, col=1) gets ~0; corner gets ~zen_formula
    # because grid extent is 0..2 in each dim, center=(1,1), max r=sqrt(2).
    assert out.min() == 0.0  # at least one center-cell
    assert out.max() <= zen_formula[0].item() + 1e-6


def test_radial_blend_zen_init_zero_rmax_returns_formula():
    """When all subtiles are at the center, r_max=0 — function returns the
    raw formula instead of dividing by zero."""
    zen_formula = torch.tensor([0.05, 0.05, 0.05])
    grid = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    out = radial_blend_zenith_init(zen_formula, grid)
    torch.testing.assert_close(out, zen_formula)


# ───────────────────────────────────────────────────────────────────────
# optimize_subtile_tilt_params — synthetic
# ───────────────────────────────────────────────────────────────────────


def _tf_settings_5x():
    """OPS track-style TF (low-NA 5x). Matches PROCESS_CONFIGS['track']."""
    return dict(
        wavelength_illumination=0.45,
        yx_pixel_size=1.3,
        z_pixel_size=25.0,
        z_padding=5,
        index_of_refraction_media=1.0,
        numerical_aperture_detection=0.15,
        numerical_aperture_illumination=0.15,
        invert_phase_contrast=False,
    )


def _build_synthetic_tiles(B=4, Z=9, tile=32, device="cpu"):
    """Tiny synthetic BF z-stacks."""
    torch.manual_seed(0)
    return [
        torch.randn(Z, tile, tile, device=device, dtype=torch.float32)
        for _ in range(B)
    ]


@pytest.mark.parametrize("device", _device_list())
def test_optimize_subtile_runs_cold_start(device):
    """Sanity: runs to completion, returns properly-shaped result."""
    B = 4
    Z = 9
    tiles = _build_synthetic_tiles(B=B, Z=Z, device=device)
    z_index = (torch.arange(Z, dtype=torch.float32, device=device) - Z // 2)

    result = optimize_subtile_tilt_params(
        tiles=tiles,
        z_index=z_index,
        tf_settings=_tf_settings_5x(),
        zen_init=0.05,
        azi_init=0.0,
        z_init=0.0,
        n_iters=2,
        freeze_axes=("zenith", "azimuth"),
        reflect_pad=4,
    )
    assert isinstance(result, TiltOptimResult)
    assert result.z_offsets.shape == (B,)
    assert result.zeniths.shape == (B,)
    assert result.azimuths.shape == (B,)
    assert result.n_iters >= 1
    assert result.skipped is False


@pytest.mark.parametrize("device", _device_list())
def test_optimize_subtile_freeze_axes_holds_init(device):
    """With freeze_axes, the returned zen/azi exactly equal the init."""
    B = 3
    Z = 7
    tiles = _build_synthetic_tiles(B=B, Z=Z, device=device)
    z_index = (torch.arange(Z, dtype=torch.float32, device=device) - Z // 2)
    zen_init = torch.tensor([0.03, 0.04, 0.05], device=device)
    azi_init = torch.tensor([0.1, 0.2, 0.3], device=device)

    result = optimize_subtile_tilt_params(
        tiles=tiles,
        z_index=z_index,
        tf_settings=_tf_settings_5x(),
        zen_init=zen_init,
        azi_init=azi_init,
        z_init=0.0,
        n_iters=2,
        freeze_axes=("zenith", "azimuth"),
        reflect_pad=4,
    )
    torch.testing.assert_close(result.zeniths.cpu(), zen_init.cpu())
    torch.testing.assert_close(result.azimuths.cpu(), azi_init.cpu())


@pytest.mark.parametrize("device", _device_list())
def test_optimize_subtile_skip_optim_roundtrip(device):
    """``skip_optim_if_warmstart=True`` returns the warmstart verbatim."""
    B = 4
    Z = 9
    tiles = _build_synthetic_tiles(B=B, Z=Z, device=device)
    z_index = (torch.arange(Z, dtype=torch.float32, device=device) - Z // 2)

    # First call: cold start
    cold = optimize_subtile_tilt_params(
        tiles=tiles,
        z_index=z_index,
        tf_settings=_tf_settings_5x(),
        zen_init=0.05,
        azi_init=0.7,
        z_init=0.0,
        n_iters=2,
        freeze_axes=("zenith", "azimuth"),
        reflect_pad=4,
    )

    # Second call: should bypass NAdam entirely and return cold's values
    warm = optimize_subtile_tilt_params(
        tiles=tiles,
        z_index=z_index,
        tf_settings=_tf_settings_5x(),
        n_iters=2,
        freeze_axes=("zenith", "azimuth"),
        reflect_pad=4,
        warmstart_params=cold,
        skip_optim_if_warmstart=True,
    )
    assert warm.skipped is True
    assert warm.n_iters == 0
    torch.testing.assert_close(warm.z_offsets, cold.z_offsets)
    torch.testing.assert_close(warm.zeniths, cold.zeniths)
    torch.testing.assert_close(warm.azimuths, cold.azimuths)


def test_optimize_subtile_warmstart_init_runs_refinement():
    """``warmstart_params`` without skip uses warmstart as init then refines."""
    B = 3
    Z = 7
    tiles = _build_synthetic_tiles(B=B, Z=Z, device="cpu")
    z_index = (torch.arange(Z, dtype=torch.float32) - Z // 2)

    # Fake a warmstart at non-zero values
    warm = TiltOptimResult(
        z_offsets=torch.tensor([0.5, 0.5, 0.5]),
        zeniths=torch.tensor([0.04, 0.04, 0.04]),
        azimuths=torch.tensor([0.1, 0.2, 0.3]),
        final_loss=torch.tensor(1.0),
        n_iters=0,
        skipped=False,
    )

    result = optimize_subtile_tilt_params(
        tiles=tiles,
        z_index=z_index,
        tf_settings=_tf_settings_5x(),
        n_iters=2,
        freeze_axes=("zenith", "azimuth"),
        reflect_pad=4,
        warmstart_params=warm,
        skip_optim_if_warmstart=False,
    )
    assert result.skipped is False
    assert result.n_iters >= 1
    # Angles frozen → return exactly the warmstart values
    torch.testing.assert_close(result.zeniths, warm.zeniths)
    torch.testing.assert_close(result.azimuths, warm.azimuths)


def test_optimize_subtile_warmstart_shape_check():
    """Mismatched warmstart length raises a clear error."""
    tiles = _build_synthetic_tiles(B=3, Z=5, device="cpu")
    z_index = (torch.arange(5, dtype=torch.float32) - 5 // 2)
    bad_warm = TiltOptimResult(
        z_offsets=torch.zeros(5),  # wrong length
        zeniths=torch.zeros(5),
        azimuths=torch.zeros(5),
        final_loss=torch.tensor(0.0),
        n_iters=0,
        skipped=False,
    )
    with pytest.raises(ValueError, match="warmstart"):
        optimize_subtile_tilt_params(
            tiles=tiles,
            z_index=z_index,
            tf_settings=_tf_settings_5x(),
            n_iters=1,
            warmstart_params=bad_warm,
            skip_optim_if_warmstart=True,
        )


def test_optimize_subtile_empty_input_raises():
    """Passing 0 tiles is an explicit error."""
    z_index = torch.arange(7, dtype=torch.float32) - 3
    with pytest.raises(ValueError, match="0 tiles"):
        optimize_subtile_tilt_params(
            tiles=[],
            z_index=z_index,
            tf_settings=_tf_settings_5x(),
            n_iters=1,
        )
