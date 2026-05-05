"""Tests for ``prepare_transfer_function`` — worker-cache contract."""

import pytest
import xarray as xr

from waveorder.api.phase import Settings as PhaseSettings
from waveorder.api.tile_stitch import (
    TileSettings,
    TileStitchSettings,
    clear_transfer_function_cache,
    prepare_transfer_function,
)


@pytest.fixture(autouse=True)
def _isolate_tf_cache():
    clear_transfer_function_cache()
    yield
    clear_transfer_function_cache()


def _phase_settings_3d() -> TileStitchSettings:
    return TileStitchSettings(
        tile=TileSettings(tile_size={"z": 4, "y": 32, "x": 32}),
        recon=PhaseSettings(),
    )


def test_prepare_tf_returns_dataset():
    settings = _phase_settings_3d()
    tf = prepare_transfer_function(settings, recon_dim=3, device="cpu")
    assert isinstance(tf, xr.Dataset)
    # 3D phase TF carries real + imaginary potential transfer functions
    assert "real_potential_transfer_function" in tf
    assert "imaginary_potential_transfer_function" in tf


def test_prepare_tf_caches_identical_calls():
    """Second call with identical args returns the same object (cache hit)."""
    settings = _phase_settings_3d()
    tf1 = prepare_transfer_function(settings, recon_dim=3, device="cpu")
    tf2 = prepare_transfer_function(settings, recon_dim=3, device="cpu")
    assert tf1 is tf2


def test_prepare_tf_different_device_misses():
    """Cache key includes device — different devices produce different entries."""
    settings = _phase_settings_3d()
    tf_cpu = prepare_transfer_function(settings, recon_dim=3, device="cpu")
    tf_none = prepare_transfer_function(settings, recon_dim=3, device=None)
    # Different cache key: identity differs
    assert tf_cpu is not tf_none


def test_prepare_tf_different_settings_misses():
    """Cache key includes settings JSON — modified settings produce different entries."""
    s1 = _phase_settings_3d()
    s2 = TileStitchSettings(
        tile=TileSettings(tile_size={"z": 4, "y": 32, "x": 32}),
        recon=PhaseSettings(
            transfer_function=PhaseSettings().transfer_function.model_copy(update={"yx_pixel_size": 0.2})
        ),
    )
    tf1 = prepare_transfer_function(s1, recon_dim=3, device="cpu")
    tf2 = prepare_transfer_function(s2, recon_dim=3, device="cpu")
    assert tf1 is not tf2


def test_clear_cache_drops_entries():
    settings = _phase_settings_3d()
    tf1 = prepare_transfer_function(settings, recon_dim=3, device="cpu")
    clear_transfer_function_cache()
    tf2 = prepare_transfer_function(settings, recon_dim=3, device="cpu")
    # Different objects after cache clear
    assert tf1 is not tf2
