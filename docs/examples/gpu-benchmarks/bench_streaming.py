"""Benchmark streaming pipeline vs serial reconstruction.

Compares:
  A. Serial: read all tiles → optimize all tiles sequentially
  B. Streaming (intra-FOV): read/optimize/write tiles overlapped within 1 FOV
  C. Streaming (multi-FOV): overlaps I/O between FOVs (if multiple positions)

Usage:
    uv run python bench_streaming.py /path/to/input.zarr --tile-size 128 --batch-size 16
    uv run python bench_streaming.py /path/to/input.zarr --preload --tile-size 512 --batch-size 4
"""

import argparse
import os
import time

os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import torch
import xarray as xr
from config import PHYSICS_20X, get_gpu_info, resolve_device
from tiling import preload_fov, tile_positions

from waveorder.api import phase
from waveorder.io.streaming import (
    PipelineStats,
    StreamingReconstructor,
    make_tile_batches,
)
from waveorder.optim import OptimizableFloat
from waveorder.optim.losses import MidbandPowerLossSettings


def make_optimize_fn(opt_iterations: int, device: str, batched: bool = True, svd_backend: str = "closed_form", use_cudagraphs: bool = False):
    """Create optimize function matching real OPS workload.

    If batched=True, uses optimize_reconstruction with (B,Z,Y,X) input
    for parallel per-tile optimization. If False, optimizes tiles serially.
    """
    p = PHYSICS_20X

    if not batched:
        def optimize_fn(tiles: list[xr.DataArray]) -> list[xr.DataArray]:
            results = []
            for tile in tiles:
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
                _, recon = phase.optimize(
                    tile, recon_dim=2, settings=settings,
                    max_iterations=opt_iterations,
                    convergence_tol=None, convergence_patience=None,
                    loss_settings=MidbandPowerLossSettings(midband_fractions=[0.125, 0.25]),
                    device=device,
                )
                results.append(recon)
            return results
        return optimize_fn

    # Batched path: stack tiles → (B,Z,Y,X) → optimize_reconstruction
    from waveorder.device import resolve_device as _resolve
    from waveorder.models import isotropic_thin_3d
    from waveorder.optim.losses import build_loss_fn
    from waveorder.optim.optimize import optimize_reconstruction

    dev = _resolve(device)

    # Build reconstruct_fn and loss_fn ONCE (outside per-batch loop)
    # so cudagraphs compilation is amortized across all batches
    Z = 7  # known from data shape

    z_indices = -torch.arange(Z, device=dev) + (Z // 2)

    def reconstruct_fn(data, **tensor_params):
        z_offset = tensor_params.get("z_focus_offset", 0.1)
        tilt_zenith = tensor_params.get("tilt_angle_zenith", 0.1)
        tilt_azimuth = tensor_params.get("tilt_angle_azimuth", 0.1)

        if isinstance(z_offset, torch.Tensor) and z_offset.device.type == "cpu":
            z_offset = z_offset.to(dev)
        if isinstance(tilt_zenith, torch.Tensor) and tilt_zenith.device.type == "cpu":
            tilt_zenith = tilt_zenith.to(dev)
        if isinstance(tilt_azimuth, torch.Tensor) and tilt_azimuth.device.type == "cpu":
            tilt_azimuth = tilt_azimuth.to(dev)

        if isinstance(z_offset, torch.Tensor) and z_offset.ndim >= 1:
            z_offset = z_offset.mean()
        z_positions = (z_indices + z_offset) * p["z_pixel_size"]

        return isotropic_thin_3d.reconstruct(
            data,
            yx_pixel_size=p["yx_pixel_size"],
            z_position_list=z_positions,
            wavelength_illumination=p["wavelength_illumination"],
            index_of_refraction_media=p["index_of_refraction_media"],
            numerical_aperture_illumination=p["numerical_aperture_illumination"],
            numerical_aperture_detection=p["numerical_aperture_detection"],
            invert_phase_contrast=p["invert_phase_contrast"],
            regularization_strength=p["regularization_strength"],
            tilt_angle_zenith=tilt_zenith,
            tilt_angle_azimuth=tilt_azimuth,
            pupil_steepness=p["pupil_steepness"],
            svd_backend=svd_backend,
        )[1]

    loss_fn = build_loss_fn(
        MidbandPowerLossSettings(midband_fractions=[0.125, 0.25]),
        NA_det=p["numerical_aperture_detection"],
        wavelength=p["wavelength_illumination"],
        pixel_size=p["yx_pixel_size"],
    )

    # Compile once — reused across all batches
    recon_fn_final = reconstruct_fn
    if use_cudagraphs:
        recon_fn_final = torch.compile(reconstruct_fn, backend="cudagraphs")

    # Pre-allocate pinned buffer for H2D (reused across batches)
    _transfer_stream = torch.cuda.Stream(device=dev)
    _pinned_buf = None

    def optimize_fn_batched(tiles: list[xr.DataArray]) -> list[xr.DataArray]:
        nonlocal _pinned_buf
        B = len(tiles)

        # Stack on CPU as contiguous tensor
        cpu_batch = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(t.values[0])).float() for t in tiles]
        )  # (B, Z, Y, X) on CPU

        # Async H2D via pinned memory on transfer stream
        if _pinned_buf is None or _pinned_buf.shape != cpu_batch.shape:
            _pinned_buf = torch.empty_like(cpu_batch, pin_memory=True)
        _pinned_buf.copy_(cpu_batch)

        h2d_event = torch.cuda.Event()
        with torch.cuda.stream(_transfer_stream):
            zyx_batch = _pinned_buf.to(dev, non_blocking=True)
            h2d_event.record()

        # Wait for H2D before compute
        torch.cuda.current_stream().wait_event(h2d_event)

        optimize_reconstruction(
            data=zyx_batch,
            reconstruct_fn=recon_fn_final,
            loss_fn=loss_fn,
            optimizable_params={
                "z_focus_offset": (0.1, 0.02),
                "tilt_angle_zenith": (0.1, 0.02),
                "tilt_angle_azimuth": (0.1, 0.02),
            },
            max_iterations=opt_iterations,
            method="adam",
            use_gradients=True,
            convergence_tol=None,
            convergence_patience=None,
        )

        # Return dummy xr.DataArrays (we're benchmarking, not stitching)
        return [xr.DataArray(np.zeros((1, 1, t.shape[-2], t.shape[-1]), dtype="float32"),
                             dims=("c", "z", "y", "x")) for t in tiles]

    return optimize_fn_batched


def make_fov_write_fn(output_store, tile_size):
    """Create a write function that stitches tile results into a FOV and writes to zarr.

    For multi-FOV pipeline: write_fn(fov_results, pos_name).
    fov_results is a list of xr.DataArrays (one per tile, shape (C, Z, Y, X)).
    """
    def write_fn(fov_results, pos_name):
        if output_store is None or not fov_results:
            return
        pos = output_store[pos_name]
        Y, X = pos.data.shape[-2], pos.data.shape[-1]

        # Stitch tiles into full FOV — simple paste (no blending)
        output = np.zeros((1, Y, X), dtype="float32")  # (Z=1, Y, X)
        tile_idx = 0
        for y in range(0, Y, tile_size):
            for x in range(0, X, tile_size):
                y_end = y + tile_size
                x_end = x + tile_size
                if y_end <= Y and x_end <= X and tile_idx < len(fov_results):
                    tile_data = fov_results[tile_idx]
                    if hasattr(tile_data, 'values'):
                        tile_np = tile_data.values
                    else:
                        tile_np = np.array(tile_data)
                    # tile_np may be (C, Z, tY, tX) or (C, tY, tX) — extract phase
                    if tile_np.ndim == 4:
                        output[0, y:y_end, x:x_end] = tile_np[0, 0]
                    elif tile_np.ndim == 3:
                        output[0, y:y_end, x:x_end] = tile_np[0]
                    elif tile_np.ndim == 2:
                        output[0, y:y_end, x:x_end] = tile_np
                    tile_idx += 1

        # Write: (T=0, C=0, Z=1, Y, X)
        pos.data.oindex[0, 0] = output

    return write_fn


def run_serial(position, tile_size, batch_size, optimize_fn, preloaded_fov=None):
    """Serial: read FOV once, slice tiles, optimize batch by batch."""
    shape = position.data.shape
    fov_y, fov_x = shape[-2], shape[-1]
    batches = make_tile_batches(fov_y, fov_x, tile_size, batch_size)

    t0 = time.perf_counter()

    # Read FOV once (avoids repeated per-tile zarr decompression)
    fov = preloaded_fov if preloaded_fov is not None else np.array(
        position.data[0, 0], dtype="float32",
    )

    all_results = []
    n_tiles = 0
    for batch_bounds in batches:
        tiles = [
            xr.DataArray(
                fov[:, y0:y1, x0:x1].copy()[None].astype("float32"),
                dims=("c", "z", "y", "x"),
            )
            for y0, y1, x0, x1 in batch_bounds
        ]
        results = optimize_fn(tiles)
        all_results.extend(results)
        n_tiles += len(tiles)

    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0
    return n_tiles, t_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_zarr")
    parser.add_argument("--positions", nargs="+", default=["A/1/002026"],
                        help="Position path(s). Multiple for multi-FOV benchmark.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--opt-iterations", type=int, default=5)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--no-batch", action="store_true",
                        help="Use serial per-tile optimization instead of batched")
    parser.add_argument("--output-zarr", default=None,
                        help="Output OME-Zarr store for writing results. "
                             "Must be pre-created with create_output_store.py.")
    parser.add_argument("--compute-streams", type=int, default=1,
                        help="Number of CUDA streams for concurrent compute. "
                             "1 = sequential (default). >1 runs tile batches concurrently.")
    parser.add_argument("--svd-backend", choices=["closed_form", "torch"], default="closed_form",
                        help="SVD backend: closed_form (default) or torch (cuSOLVER)")
    parser.add_argument("--cudagraphs", action="store_true",
                        help="Wrap reconstruct_fn with torch.compile(backend='cudagraphs')")
    args = parser.parse_args()

    args.device = resolve_device(args.device)
    gpu = get_gpu_info()
    print(f"GPU: {gpu['name']} ({gpu['vram_gb']} GB)")
    print(f"Positions: {args.positions}")
    print(f"Tile: {args.tile_size}, Batch: {args.batch_size}, Iterations: {args.opt_iterations}")
    print(f"Mode: {'preloaded' if args.preload else 'zarr oindex (fastest for full FOV)'}")
    print()

    from iohub.ngff import open_ome_zarr
    plate = open_ome_zarr(args.input_zarr, mode="r")

    # Open output store if provided
    output_store = None
    if args.output_zarr:
        output_store = open_ome_zarr(args.output_zarr, mode="r+")
        print(f"Output: {args.output_zarr}")

    # Verify positions exist
    for pos in args.positions:
        _ = plate[pos]
    position = plate[args.positions[0]]
    print(f"Data shape: {position.data.shape}")
    print()

    batched = not args.no_batch
    optimize_fn = make_optimize_fn(args.opt_iterations, args.device, batched=batched,
                                   svd_backend=args.svd_backend, use_cudagraphs=args.cudagraphs)
    print(f"Optimization: {'batched' if batched else 'serial per-tile'}, SVD: {args.svd_backend}"
          f"{', cudagraphs' if args.cudagraphs else ''}")
    if args.compute_streams > 1:
        print(f"Compute streams: {args.compute_streams}")

    # Warmup
    print("Warming up...")
    warmup = xr.DataArray(
        np.random.randn(1, 7, args.tile_size, args.tile_size).astype("float32"),
        dims=("c", "z", "y", "x"),
    )
    optimize_fn([warmup])
    torch.cuda.synchronize()
    print("Done.\n")

    n_positions = len(args.positions)

    # --- Serial: process each position sequentially ---
    print(f"=== Serial ({n_positions} FOVs) ===")
    t0 = time.perf_counter()
    total_tiles = 0
    for pos_name in args.positions:
        pos = plate[pos_name]
        fov = None
        if args.preload:
            fov = preload_fov(pos)
        n_tiles, _ = run_serial(
            pos, args.tile_size, args.batch_size, optimize_fn, preloaded_fov=fov,
        )
        total_tiles += n_tiles
    torch.cuda.synchronize()
    t_serial = time.perf_counter() - t0
    print(f"  FOVs: {n_positions}, Tiles: {total_tiles}")
    print(f"  Total:      {t_serial:.2f}s")
    print(f"  Per FOV:    {t_serial / n_positions:.2f}s")
    print(f"  Throughput: {total_tiles / t_serial:.1f} tiles/s")
    print()

    # --- Streaming intra-FOV: per-position via StreamingReconstructor ---
    # Processes one FOV at a time; tile batches overlap within the FOV
    print(f"=== Streaming intra-FOV ({n_positions} FOVs) ===")
    t0 = time.perf_counter()
    intra_stats_list = []
    for pos_name in args.positions:
        rec = StreamingReconstructor(
            input_store=plate,
            tile_size=args.tile_size,
            batch_size=args.batch_size,
            reconstruct_fn=optimize_fn,
            device=args.device,
            output_store=output_store,
            n_buffers=3,
        )
        s = rec.run([pos_name], t_idx=0)
        intra_stats_list.append(s)
    torch.cuda.synchronize()
    t_intra = time.perf_counter() - t0
    fov_tiles = (2048 // args.tile_size) ** 2
    total_tiles_intra = n_positions * fov_tiles
    print(f"  FOVs: {n_positions}, Tiles: {total_tiles_intra}")
    print(f"  Total:      {t_intra:.2f}s")
    print(f"  Per FOV:    {t_intra / n_positions:.2f}s")
    print(f"  Throughput: {total_tiles_intra / t_intra:.1f} tiles/s")
    print(f"  Read:       {np.mean([s.avg_read_ms for s in intra_stats_list]):.0f}ms/FOV "
          f"({np.mean([s.read_bandwidth_mbs for s in intra_stats_list]):.0f} MB/s)")
    print(f"  Compute:    {np.mean([s.avg_compute_ms for s in intra_stats_list]):.0f}ms/FOV")
    print(f"  GPU memory: {np.mean([s.peak_gpu_memory_mb for s in intra_stats_list]):.0f} MB peak")
    print()

    # --- Streaming multi-FOV: all positions overlapped via StreamingReconstructor ---
    fov_stats = None
    if n_positions > 1:
        print(f"=== Streaming multi-FOV ({n_positions} FOVs) ===")
        rec = StreamingReconstructor(
            input_store=plate,
            tile_size=args.tile_size,
            batch_size=args.batch_size,
            reconstruct_fn=optimize_fn,
            device=args.device,
            output_store=output_store,
            n_buffers=3,
        )
        fov_stats = rec.run(args.positions, t_idx=0)
        print(fov_stats)
        print(f"  Per FOV:    {fov_stats.per_fov_s:.2f}s")
        print()

    # === Summary table ===
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Mode':<30} {'Total':>8} {'Per FOV':>10} {'Tiles/s':>8} {'Speedup':>8}")
    print("-" * 70)
    print(f"{'Serial':<30} {t_serial:>7.2f}s {t_serial/n_positions:>9.2f}s "
          f"{total_tiles/t_serial:>7.1f} {'1.00×':>8}")
    print(f"{'Streaming intra-FOV':<30} {t_intra:>7.2f}s {t_intra/n_positions:>9.2f}s "
          f"{total_tiles_intra/t_intra:>7.1f} {t_serial/t_intra:>7.2f}×")
    if fov_stats:
        print(f"{'Streaming multi-FOV':<30} {fov_stats.total_time:>7.2f}s "
              f"{fov_stats.per_fov_s:>9.2f}s "
              f"{n_positions * fov_tiles / fov_stats.total_time:>7.1f} "
              f"{t_serial/fov_stats.total_time:>7.2f}×")

    print()
    print("--- Pipeline Efficiency ---")
    print(f"Multi-FOV read bandwidth: {fov_stats.read_bandwidth_mbs:.0f} MB/s" if fov_stats else "")

    if torch.cuda.is_available():
        print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")

    if output_store is not None:
        output_store.close()
        print(f"Results written to: {args.output_zarr}")


if __name__ == "__main__":
    main()
