"""Tests for TransferFunctionCache."""

import numpy as np
import xarray as xr

from waveorder.api._utils import _named_dataarray
from waveorder.optim._types import CacheSpec
from waveorder.optim.cache import TransferFunctionCache


def _make_compute_tf_fn():
    """A compute_tf_fn that returns a simple singular system."""

    def compute_tf_fn(scale=1.0):
        U = np.ones((2, 2, 4, 4), dtype=np.float32)
        S = np.ones((2, 4, 4), dtype=np.float32) * scale
        Vh = np.ones((2, 3, 4, 4), dtype=np.float32) * scale
        return xr.Dataset(
            {
                "singular_system_U": _named_dataarray(U, "singular_system_U"),
                "singular_system_S": _named_dataarray(S, "singular_system_S"),
                "singular_system_Vh": _named_dataarray(
                    Vh,
                    "singular_system_Vh",
                ),
            }
        )

    return compute_tf_fn


def test_build_and_lookup_nearest(tmp_path):
    """Build a cache, then look up nearest grid point."""
    cache_dir = tmp_path / "tf_cache.zarr"
    cache = TransferFunctionCache.build(
        compute_tf_fn=_make_compute_tf_fn(),
        cache_specs={"scale": CacheSpec(start=0.0, stop=2.0, step=1.0)},
        cache_dir=cache_dir,
    )

    assert cache.cache_size == 3  # 0, 1, 2

    tf = cache.lookup(scale=0.9)
    S = tf["singular_system_S"].values
    # Nearest to 0.9 is 1.0
    np.testing.assert_allclose(S, 1.0)


def test_lookup_linear(tmp_path):
    """Linear interpolation blends between cached TFs."""
    cache_dir = tmp_path / "tf_cache.zarr"
    TransferFunctionCache.build(
        compute_tf_fn=_make_compute_tf_fn(),
        cache_specs={"scale": CacheSpec(start=0.0, stop=2.0, step=1.0)},
        cache_dir=cache_dir,
    )

    cache = TransferFunctionCache(cache_dir, interpolation="linear")
    tf = cache.lookup(scale=0.5)
    S = tf["singular_system_S"].values
    # Interpolation: 0.5 * (scale=0) + 0.5 * (scale=1) = 0.5
    np.testing.assert_allclose(S, 0.5, atol=1e-6)


def test_cache_persists_on_disk(tmp_path):
    """Cache can be reopened from disk."""
    cache_dir = tmp_path / "tf_cache.zarr"
    TransferFunctionCache.build(
        compute_tf_fn=_make_compute_tf_fn(),
        cache_specs={"scale": CacheSpec(start=0.0, stop=1.0, step=0.5)},
        cache_dir=cache_dir,
    )

    # Reopen from disk
    cache2 = TransferFunctionCache(cache_dir)
    assert cache2.cache_size == 3
    tf = cache2.lookup(scale=0.5)
    S = tf["singular_system_S"].values
    np.testing.assert_allclose(S, 0.5)


def test_cache_2d_grid(tmp_path):
    """Cache works with two parameters."""

    def compute_tf_fn(a=1.0, b=0.0):
        S = np.ones((2, 4, 4), dtype=np.float32) * (a + b)
        return xr.Dataset(
            {
                "S": _named_dataarray(S, "S"),
            }
        )

    cache_dir = tmp_path / "tf_cache.zarr"
    cache = TransferFunctionCache.build(
        compute_tf_fn=compute_tf_fn,
        cache_specs={
            "a": CacheSpec(start=0.0, stop=1.0, step=1.0),
            "b": CacheSpec(start=0.0, stop=1.0, step=1.0),
        },
        cache_dir=cache_dir,
    )

    assert cache.cache_size == 4  # 2x2

    tf = cache.lookup(a=0.0, b=1.0)
    np.testing.assert_allclose(tf["S"].values, 1.0)


def test_cache_clamps_to_grid(tmp_path):
    """Values outside the grid are clamped."""
    cache_dir = tmp_path / "tf_cache.zarr"
    TransferFunctionCache.build(
        compute_tf_fn=_make_compute_tf_fn(),
        cache_specs={"scale": CacheSpec(start=0.0, stop=2.0, step=1.0)},
        cache_dir=cache_dir,
    )

    cache = TransferFunctionCache(cache_dir, interpolation="linear")
    tf = cache.lookup(scale=5.0)
    S = tf["singular_system_S"].values
    # Clamped to scale=2.0
    np.testing.assert_allclose(S, 2.0, atol=1e-6)
