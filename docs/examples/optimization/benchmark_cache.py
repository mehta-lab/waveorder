"""Benchmark TF cache: build once, apply to many tiles.

Compares:
1. Full pipeline: compute_tf + apply_inverse per tile
2. Cached nearest: load TF from zarr + apply_inverse per tile
3. Cached linear: interpolated TF + apply_inverse per tile
"""

import os
import tempfile
import time

os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import torch
import xarray as xr

from waveorder.api import phase
from waveorder.api._utils import _to_singular_system
from waveorder.models import isotropic_thin_3d
from waveorder.optim._types import CacheSpec
from waveorder.optim.cache import TransferFunctionCache

# --- Simulation ---
gt_settings = phase.Settings(
    transfer_function=phase.TransferFunctionSettings(
        z_focus_offset=2.0,
        tilt_angle_azimuth=np.pi / 4,
        tilt_angle_zenith=0.40,
    )
)
ZYX_SHAPE = (11, 178, 178)
_, data = phase.simulate(
    gt_settings,
    recon_dim=2,
    zyx_shape=ZYX_SHAPE,
)

s = gt_settings.transfer_function.resolve_floats()
zyx_data = torch.tensor(data.values[0], dtype=torch.float32)
Z, Y, X = zyx_data.shape


# --- Wrap compute_transfer_function for the cache ---
def compute_tf_for_cache(z_focus_offset=0.0):
    """Compute phase 2D singular system for given z_focus_offset."""
    from waveorder.api._utils import _position_list_from_shape_scale_offset

    z_position_list = _position_list_from_shape_scale_offset(
        shape=Z,
        scale=s.z_pixel_size,
        offset=z_focus_offset,
    )
    absorption_tf, phase_tf = isotropic_thin_3d.calculate_transfer_function(
        yx_shape=(Y, X),
        yx_pixel_size=s.yx_pixel_size,
        wavelength_illumination=s.wavelength_illumination,
        z_position_list=z_position_list,
        index_of_refraction_media=s.index_of_refraction_media,
        numerical_aperture_illumination=s.numerical_aperture_illumination,
        numerical_aperture_detection=s.numerical_aperture_detection,
        invert_phase_contrast=s.invert_phase_contrast,
    )
    U, S, Vh = isotropic_thin_3d.calculate_singular_system(
        absorption_tf,
        phase_tf,
    )

    from waveorder.api._utils import _named_dataarray

    return xr.Dataset(
        {
            "singular_system_U": _named_dataarray(
                U.cpu().numpy(),
                "singular_system_U",
            ),
            "singular_system_S": _named_dataarray(
                S.cpu().numpy(),
                "singular_system_S",
            ),
            "singular_system_Vh": _named_dataarray(
                Vh.cpu().numpy(),
                "singular_system_Vh",
            ),
        }
    )


# --- Build the cache ---
cache_spec = {"z_focus_offset": CacheSpec(start=-2.0, stop=6.0, step=0.5)}

with tempfile.TemporaryDirectory() as tmpdir:
    cache_dir = os.path.join(tmpdir, "tf_cache.zarr")

    print(f"Data shape: {zyx_data.shape}")
    print("Cache grid: z_focus_offset from -2 to 6, step 0.5")
    print()

    # Build
    t0 = time.monotonic()
    cache = TransferFunctionCache.build(
        compute_tf_fn=compute_tf_for_cache,
        cache_specs=cache_spec,
        cache_dir=cache_dir,
    )
    t_build = time.monotonic() - t0
    print(f"Cache entries: {cache.cache_size}")
    print(f"Build time:    {t_build:.2f}s")
    print()

    # Reopen as read-only (simulates a worker)
    cache_nearest = TransferFunctionCache(
        cache_dir,
        interpolation="nearest",
    )
    cache_linear = TransferFunctionCache(
        cache_dir,
        interpolation="linear",
    )

    # --- Benchmark per-tile cost ---
    N_TILES = 50
    test_z = 2.3  # not exactly on grid

    # 1. Full pipeline: compute_tf + apply_inverse
    t0 = time.monotonic()
    for _ in range(N_TILES):
        tf_ds = compute_tf_for_cache(z_focus_offset=test_z)
        singular_system = _to_singular_system(tf_ds)
        isotropic_thin_3d.apply_inverse_transfer_function(
            zyx_data,
            singular_system,
        )
    t_full = (time.monotonic() - t0) / N_TILES

    # 2. Cached nearest: lookup + apply_inverse
    t0 = time.monotonic()
    for _ in range(N_TILES):
        tf_ds = cache_nearest.lookup(z_focus_offset=test_z)
        singular_system = _to_singular_system(tf_ds)
        isotropic_thin_3d.apply_inverse_transfer_function(
            zyx_data,
            singular_system,
        )
    t_nearest = (time.monotonic() - t0) / N_TILES

    # 3. Cached linear: lookup + apply_inverse
    t0 = time.monotonic()
    for _ in range(N_TILES):
        tf_ds = cache_linear.lookup(z_focus_offset=test_z)
        singular_system = _to_singular_system(tf_ds)
        isotropic_thin_3d.apply_inverse_transfer_function(
            zyx_data,
            singular_system,
        )
    t_linear = (time.monotonic() - t0) / N_TILES

    # 4. Cached nearest: lookup only (no apply)
    t0 = time.monotonic()
    for _ in range(N_TILES):
        tf_ds = cache_nearest.lookup(z_focus_offset=test_z)
    t_lookup_nearest = (time.monotonic() - t0) / N_TILES

    # 5. Cached linear: lookup only (no apply)
    t0 = time.monotonic()
    for _ in range(N_TILES):
        tf_ds = cache_linear.lookup(z_focus_offset=test_z)
    t_lookup_linear = (time.monotonic() - t0) / N_TILES

    print(f"{'Mode':<35} {'ms/tile':>9} {'speedup':>8}")
    print("-" * 55)
    print(f"{'Full (compute_tf + apply)':<35} {t_full * 1000:>8.2f}ms {'1.0x':>8}")
    print(f"{'Cached nearest (lookup + apply)':<35} {t_nearest * 1000:>8.2f}ms {t_full / t_nearest:>7.1f}x")
    print(f"{'Cached linear (lookup + apply)':<35} {t_linear * 1000:>8.2f}ms {t_full / t_linear:>7.1f}x")
    print(f"{'Cached nearest (lookup only)':<35} {t_lookup_nearest * 1000:>8.2f}ms {t_full / t_lookup_nearest:>7.1f}x")
    print(f"{'Cached linear (lookup only)':<35} {t_lookup_linear * 1000:>8.2f}ms {t_full / t_lookup_linear:>7.1f}x")
    print()

    # Break-even analysis
    tiles_to_break_even = t_build / (t_full - t_nearest)
    print(f"Build cost: {t_build:.2f}s")
    print(f"Savings per tile: {(t_full - t_nearest) * 1000:.1f}ms")
    print(f"Break-even after: {tiles_to_break_even:.0f} tiles")
