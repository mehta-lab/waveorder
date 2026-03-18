"""Tests for pluggable loss functions."""

import torch

from waveorder.optim.losses import (
    LaplacianVarianceLossSettings,
    MidbandPowerLossSettings,
    NormalizedVarianceLossSettings,
    SpectralFlatnessLossSettings,
    TotalVariationLossSettings,
    build_loss_fn,
)

OPTICAL_PARAMS = dict(NA_det=1.2, wavelength=0.532, pixel_size=0.1)


def test_midband_power_loss_returns_scalar():
    loss_fn = build_loss_fn(MidbandPowerLossSettings(), **OPTICAL_PARAMS)
    recon = torch.randn(64, 64)
    loss = loss_fn(recon)
    assert loss.ndim == 0


def test_midband_power_loss_3d_input():
    loss_fn = build_loss_fn(MidbandPowerLossSettings(), **OPTICAL_PARAMS)
    recon = torch.randn(5, 64, 64)
    loss = loss_fn(recon)
    assert loss.ndim == 0


def test_total_variation_loss():
    loss_fn = build_loss_fn(TotalVariationLossSettings(), **OPTICAL_PARAMS)
    recon = torch.randn(64, 64)
    loss = loss_fn(recon)
    assert loss.ndim == 0
    assert loss.item() <= 0


def test_total_variation_loss_3d():
    loss_fn = build_loss_fn(TotalVariationLossSettings(), **OPTICAL_PARAMS)
    recon = torch.randn(5, 64, 64)
    loss = loss_fn(recon)
    assert loss.ndim == 0


def test_laplacian_variance_loss():
    loss_fn = build_loss_fn(LaplacianVarianceLossSettings(), **OPTICAL_PARAMS)
    recon = torch.randn(64, 64)
    loss = loss_fn(recon)
    assert loss.ndim == 0
    assert loss.item() <= 0


def test_laplacian_variance_loss_3d():
    loss_fn = build_loss_fn(LaplacianVarianceLossSettings(), **OPTICAL_PARAMS)
    recon = torch.randn(5, 64, 64)
    loss = loss_fn(recon)
    assert loss.ndim == 0


def test_normalized_variance_loss():
    loss_fn = build_loss_fn(NormalizedVarianceLossSettings(), **OPTICAL_PARAMS)
    recon = torch.randn(64, 64) + 5.0
    loss = loss_fn(recon)
    assert loss.ndim == 0
    assert loss.item() <= 0


def test_normalized_variance_loss_3d():
    loss_fn = build_loss_fn(NormalizedVarianceLossSettings(), **OPTICAL_PARAMS)
    recon = torch.randn(5, 64, 64) + 5.0
    loss = loss_fn(recon)
    assert loss.ndim == 0


def test_spectral_flatness_loss():
    loss_fn = build_loss_fn(SpectralFlatnessLossSettings(), **OPTICAL_PARAMS)
    recon = torch.randn(64, 64).abs() + 0.1
    loss = loss_fn(recon)
    assert loss.ndim == 0
    assert loss.item() <= 0


def test_spectral_flatness_loss_3d():
    loss_fn = build_loss_fn(SpectralFlatnessLossSettings(), **OPTICAL_PARAMS)
    recon = torch.randn(5, 64, 64).abs() + 0.1
    loss = loss_fn(recon)
    assert loss.ndim == 0


def test_all_losses_are_differentiable():
    """All loss functions support backward for both 2D and 3D."""
    settings_list = [
        MidbandPowerLossSettings(),
        TotalVariationLossSettings(),
        LaplacianVarianceLossSettings(),
        NormalizedVarianceLossSettings(),
        SpectralFlatnessLossSettings(),
    ]
    for settings in settings_list:
        loss_fn = build_loss_fn(settings, **OPTICAL_PARAMS)
        for ndim, shape in [(2, (64, 64)), (3, (3, 64, 64))]:
            recon = torch.randn(*shape, requires_grad=True)
            loss = loss_fn(recon)
            loss.backward()
            assert recon.grad is not None, f"{settings.type} {ndim}D is not differentiable"


def test_sharper_image_has_higher_tv():
    """Total variation is higher for sharp images."""
    loss_fn = build_loss_fn(TotalVariationLossSettings(), **OPTICAL_PARAMS)
    smooth = torch.ones(64, 64) * 5.0
    sharp = torch.zeros(64, 64)
    sharp[::2, ::2] = 10.0
    sharp[1::2, 1::2] = 10.0

    loss_smooth = loss_fn(smooth)
    loss_sharp = loss_fn(sharp)
    assert loss_sharp < loss_smooth


def test_losses_are_tile_size_invariant():
    """Loss values should be similar for the same image at different sizes."""
    torch.manual_seed(42)
    # Generate a base image pattern and tile it
    base = torch.randn(32, 32)
    small = base  # 32x32
    large = base.repeat(2, 2)  # 64x64

    settings_list = [
        TotalVariationLossSettings(),
        LaplacianVarianceLossSettings(),
        NormalizedVarianceLossSettings(),
    ]
    for settings in settings_list:
        loss_fn = build_loss_fn(settings, **OPTICAL_PARAMS)
        loss_small = loss_fn(small).item()
        loss_large = loss_fn(large).item()
        ratio = loss_small / loss_large if loss_large != 0 else float("inf")
        assert 0.5 < ratio < 2.0, (
            f"{settings.type}: loss ratio {ratio:.2f} (small={loss_small:.4f}, large={loss_large:.4f})"
        )
