"""Benchmark stream_optimize_positions vs serial phase.optimize.

Uses only the public waveorder API. No local benchmark utilities.

Usage:
    uv run python benchmark_stream_optimize.py --device cuda:0
    uv run python benchmark_stream_optimize.py --device cuda:0 --skip-serial
"""

import argparse
import os
import shutil
import time

os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import torch
import xarray as xr
from iohub.ngff import open_ome_zarr

from waveorder.api import phase
from waveorder.device import resolve_device
from waveorder.io.streaming import _create_output_store, stream_optimize_positions
from waveorder.optim import OptimizableFloat
from waveorder.optim.losses import MidbandPowerLossSettings

INPUT = "/hpc/projects/intracellular_dashboard/ops/ops0105_20260106/0-convert/live_imaging/phenotyping_transform.zarr"
OUTPUT_SERIAL = "/home/sricharan.varra/mydata/data/waveorder-gpu-io/bench_serial.zarr"
OUTPUT_STREAM = "/home/sricharan.varra/mydata/data/waveorder-gpu-io/bench_stream.zarr"

DEFAULT_POSITIONS = [
    "A/1/028028", "A/1/028029", "A/1/028030",
    "A/1/029028", "A/1/029029", "A/1/029030",
    "A/1/030028", "A/1/030029", "A/1/030030",
]


def make_settings() -> phase.Settings:
    return phase.Settings(
        transfer_function=phase.TransferFunctionSettings(
            wavelength_illumination=0.45,
            yx_pixel_size=0.325,
            z_pixel_size=2.0,
            z_focus_offset=OptimizableFloat(init=0.1, lr=0.02),
            numerical_aperture_illumination=0.4,
            numerical_aperture_detection=0.55,
            index_of_refraction_media=1.0,
            invert_phase_contrast=False,
            tilt_angle_zenith=OptimizableFloat(init=0.1, lr=0.02),
            tilt_angle_azimuth=OptimizableFloat(init=0.1, lr=0.02),
        ),
        apply_inverse=phase.ApplyInverseSettings(
            reconstruction_algorithm="Tikhonov",
            regularization_strength=0.001,
        ),
    )


def run_serial(positions, opt_iterations, tile_size, device, output_zarr):
    """Serial: read FOV once, optimize tiles, write. Pure waveorder API."""
    dev = resolve_device(device)
    plate = open_ome_zarr(INPUT, mode="r")
    settings = make_settings()

    if os.path.exists(output_zarr):
        shutil.rmtree(output_zarr)
    _create_output_store(plate, output_zarr, positions)
    out_plate = open_ome_zarr(output_zarr, mode="r+")

    t_start = time.perf_counter()

    for pos_name in positions:
        pos = plate[pos_name]
        fov = np.array(pos.data.oindex[0, 0], dtype="float32")
        Z, H, W = fov.shape

        results = {}
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                if y + tile_size > H or x + tile_size > W:
                    continue
                tile = xr.DataArray(
                    fov[:, y:y+tile_size, x:x+tile_size][np.newaxis].astype("float32"),
                    dims=("c", "z", "y", "x"),
                )
                _, recon = phase.optimize(
                    tile, recon_dim=2, settings=settings,
                    max_iterations=opt_iterations,
                    convergence_tol=None, convergence_patience=None,
                    loss_settings=MidbandPowerLossSettings(midband_fractions=[0.125, 0.25]),
                    device=dev,
                )
                results[(y, x)] = recon.values[0, 0]  # (Y, X)

        output = np.zeros((1, 1, 1, H, W), dtype="float32")
        for (y, x), phase_yx in results.items():
            output[0, 0, 0, y:y+tile_size, x:x+tile_size] = phase_yx

        out_pos = out_plate[pos_name]
        out_pos.write_xarray(xr.DataArray(
            output, dims=("t", "c", "z", "y", "x"),
            coords={"c": out_pos.channel_names[:1]},
        ))

    torch.cuda.synchronize()
    total = time.perf_counter() - t_start
    plate.close()
    out_plate.close()
    return total


def verify_output(output_zarr, positions):
    with open_ome_zarr(output_zarr, mode="r") as out:
        all_pass = True
        for pos_name in positions:
            data = np.array(out[pos_name].data[0, 0, 0])
            nz = np.count_nonzero(data)
            ok = nz > data.size * 0.99
            print(f"  {pos_name}: {nz}/{data.size} nonzero {'✓' if ok else '✗ FAIL'}")
            if not ok:
                all_pass = False
    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", nargs="+", default=DEFAULT_POSITIONS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--opt-iterations", type=int, default=50)
    parser.add_argument("--skip-serial", action="store_true")
    args = parser.parse_args()

    dev = resolve_device(args.device)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"Positions: {len(args.positions)}, Tile: {args.tile_size}, "
          f"Batch: {args.batch_size}, Iters: {args.opt_iterations}")
    print()

    # --- Streaming ---
    print("=== stream_optimize_positions ===")
    if os.path.exists(OUTPUT_STREAM):
        shutil.rmtree(OUTPUT_STREAM)

    stats = stream_optimize_positions(
        input_zarr=INPUT,
        output_zarr=OUTPUT_STREAM,
        position_names=args.positions,
        settings=make_settings(),
        opt_iterations=args.opt_iterations,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(stats)
    t_stream = stats.total_time
    print()

    print("Verifying streaming output...")
    verify_output(OUTPUT_STREAM, args.positions)
    print()

    # --- Serial ---
    t_serial = None
    if not args.skip_serial:
        print("=== Serial baseline (phase.optimize per tile) ===")
        t_serial = run_serial(
            args.positions, args.opt_iterations,
            args.tile_size, args.device, OUTPUT_SERIAL,
        )
        print(f"Serial: {t_serial:.1f}s ({t_serial/len(args.positions):.1f}s/FOV)")
        print()

    # --- Summary ---
    print("=" * 50)
    print(f"Streaming: {t_stream:.1f}s ({stats.per_fov_s:.1f}s/FOV)")
    if t_serial is not None:
        print(f"Serial:    {t_serial:.1f}s ({t_serial/len(args.positions):.1f}s/FOV)")
        print(f"Speedup:   {t_serial/t_stream:.2f}×")


if __name__ == "__main__":
    main()
