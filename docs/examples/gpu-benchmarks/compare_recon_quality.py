"""Compare reconstruction quality: closed_form vs torch SVD on real OPS data.

Runs phase.optimize on real tiles from position 029029 with both SVD backends
and compares the output reconstructions.

Usage:
    uv run python compare_recon_quality.py --device cuda:0
    uv run python compare_recon_quality.py --device cuda:0 --n-tiles 16 --opt-iterations 50
"""

import argparse
import os
import time

os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import torch
import xarray as xr
from iohub.ngff import open_ome_zarr

from config import PHYSICS_20X, get_gpu_info, resolve_device
from tiling import preload_fov, tile_positions
from waveorder.api import phase
from waveorder.optim import OptimizableFloat
from waveorder.optim.losses import MidbandPowerLossSettings


INPUT_ZARR = "/hpc/projects/intracellular_dashboard/ops/ops0105_20260106/0-convert/live_imaging/phenotyping_transform.zarr"
POSITION = "A/1/029029"


def optimize_tile(tile, opt_iterations, device, svd_backend):
    """Run phase.optimize on a single tile with specified SVD backend."""
    p = PHYSICS_20X
    settings = phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            wavelength_illumination=p["wavelength_illumination"],
            yx_pixel_size=p["yx_pixel_size"],
            z_pixel_size=p["z_pixel_size"],
            z_focus_offset=OptimizableFloat(init=0.1, lr=0.02),
            numerical_aperture_illumination=p["numerical_aperture_illumination"],
            numerical_aperture_detection=p["numerical_aperture_detection"],
            index_of_refraction_media=p["index_of_refraction_media"],
            invert_phase_contrast=p["invert_phase_contrast"],
            tilt_angle_zenith=OptimizableFloat(init=0.1, lr=0.02),
            tilt_angle_azimuth=OptimizableFloat(init=0.1, lr=0.02),
        ),
        apply_inverse=phase.ApplyInverseSettings(
            reconstruction_algorithm="Tikhonov",
            regularization_strength=p["regularization_strength"],
        ),
    )

    # Temporarily set svd_backend in the reconstruct path
    # phase.optimize uses svd_backend="closed_form" by default now,
    # so we need to monkey-patch for torch comparison
    from waveorder.models import isotropic_thin_3d
    original_reconstruct = isotropic_thin_3d.reconstruct

    def patched_reconstruct(*args, **kwargs):
        kwargs["svd_backend"] = svd_backend
        return original_reconstruct(*args, **kwargs)

    isotropic_thin_3d.reconstruct = patched_reconstruct
    try:
        opt_settings, recon = phase.optimize(
            tile, recon_dim=2, settings=settings,
            max_iterations=opt_iterations,
            convergence_tol=None, convergence_patience=None,
            loss_settings=MidbandPowerLossSettings(midband_fractions=[0.125, 0.25]),
            device=device,
        )
    finally:
        isotropic_thin_3d.reconstruct = original_reconstruct

    return opt_settings, recon


def compare_results(recon_torch, recon_cf, settings_torch, settings_cf, tile_idx):
    """Compare two reconstructions numerically."""
    phase_torch = recon_torch.values[0, 0]  # (Y, X)
    phase_cf = recon_cf.values[0, 0]

    diff = phase_torch - phase_cf
    abs_diff = np.abs(diff)

    # Correlation
    t_flat = phase_torch.flatten()
    c_flat = phase_cf.flatten()
    corr = np.corrcoef(t_flat, c_flat)[0, 1]

    # Relative error
    scale = max(np.abs(phase_torch).max(), 1e-10)
    rel_err = abs_diff.max() / scale

    # Optimized params comparison
    st = settings_torch.transfer_function
    sc = settings_cf.transfer_function

    print(f"  Tile {tile_idx}:")
    print(f"    Phase range:  torch [{phase_torch.min():.3f}, {phase_torch.max():.3f}]  "
          f"cf [{phase_cf.min():.3f}, {phase_cf.max():.3f}]")
    print(f"    Max abs diff: {abs_diff.max():.6f}")
    print(f"    Mean abs diff: {abs_diff.mean():.6f}")
    print(f"    Relative err: {rel_err:.2e}")
    print(f"    Correlation:  {corr:.8f}")
    print(f"    Params torch: z={st.z_focus_offset:.4f} zen={st.tilt_angle_zenith:.4f} azi={st.tilt_angle_azimuth:.4f}")
    print(f"    Params cf:    z={sc.z_focus_offset:.4f} zen={sc.tilt_angle_zenith:.4f} azi={sc.tilt_angle_azimuth:.4f}")

    return {
        "max_abs_diff": abs_diff.max(),
        "mean_abs_diff": abs_diff.mean(),
        "rel_err": rel_err,
        "correlation": corr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--n-tiles", type=int, default=8, help="Number of tiles to compare")
    parser.add_argument("--opt-iterations", type=int, default=50)
    parser.add_argument("--save-npz", default=None, help="Save results to .npz file")
    args = parser.parse_args()

    args.device = resolve_device(args.device)
    gpu = get_gpu_info()
    print(f"GPU: {gpu['name']} ({gpu['vram_gb']} GB)")
    print(f"Position: {POSITION}")
    print(f"Tile: {args.tile_size}, N tiles: {args.n_tiles}, Iterations: {args.opt_iterations}")
    print()

    plate = open_ome_zarr(INPUT_ZARR, mode="r")
    position = plate[POSITION]
    print(f"Data shape: {position.data.shape}")

    # Preload FOV
    fov = preload_fov(position)
    print(f"Preloaded: {fov.shape}")
    print()

    # Extract tiles
    fov_y, fov_x = fov.shape[-2], fov.shape[-1]
    tiles = []
    for y0, y1, x0, x1 in tile_positions(fov_y, fov_x, args.tile_size):
        tile_np = fov[:, y0:y1, x0:x1].copy()
        da = xr.DataArray(tile_np[None].astype("float32"), dims=("c", "z", "y", "x"))
        tiles.append(da)
        if len(tiles) >= args.n_tiles:
            break
    print(f"Extracted {len(tiles)} tiles")

    # Warmup
    print("Warming up...")
    optimize_tile(tiles[0], max(3, args.opt_iterations // 10), args.device, "closed_form")
    optimize_tile(tiles[0], max(3, args.opt_iterations // 10), args.device, "torch")
    torch.cuda.synchronize()
    print("Done.\n")

    # Run both backends on each tile
    all_metrics = []
    all_recons_torch = []
    all_recons_cf = []

    for i, tile in enumerate(tiles):
        print(f"--- Tile {i} ---")

        # torch SVD
        t0 = time.perf_counter()
        settings_torch, recon_torch = optimize_tile(tile, args.opt_iterations, args.device, "torch")
        torch.cuda.synchronize()
        t_torch = time.perf_counter() - t0

        # closed_form SVD
        t0 = time.perf_counter()
        settings_cf, recon_cf = optimize_tile(tile, args.opt_iterations, args.device, "closed_form")
        torch.cuda.synchronize()
        t_cf = time.perf_counter() - t0

        print(f"  Time: torch={t_torch:.2f}s, cf={t_cf:.2f}s, speedup={t_torch/t_cf:.2f}×")
        metrics = compare_results(recon_torch, recon_cf, settings_torch, settings_cf, i)
        all_metrics.append(metrics)
        all_recons_torch.append(recon_torch.values[0, 0])
        all_recons_cf.append(recon_cf.values[0, 0])
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    max_diffs = [m["max_abs_diff"] for m in all_metrics]
    correlations = [m["correlation"] for m in all_metrics]
    rel_errs = [m["rel_err"] for m in all_metrics]
    print(f"Tiles compared:    {len(all_metrics)}")
    print(f"Max abs diff:      {np.max(max_diffs):.6f} (worst), {np.mean(max_diffs):.6f} (mean)")
    print(f"Relative error:    {np.max(rel_errs):.2e} (worst), {np.mean(rel_errs):.2e} (mean)")
    print(f"Correlation:       {np.min(correlations):.8f} (worst), {np.mean(correlations):.8f} (mean)")
    print(f"All correlations > 0.999: {'YES' if all(c > 0.999 for c in correlations) else 'NO'}")

    if args.save_npz:
        np.savez(
            args.save_npz,
            recons_torch=np.stack(all_recons_torch),
            recons_cf=np.stack(all_recons_cf),
            metrics=all_metrics,
            position=POSITION,
            tile_size=args.tile_size,
            opt_iterations=args.opt_iterations,
        )
        print(f"\nSaved to: {args.save_npz}")

    if torch.cuda.is_available():
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
