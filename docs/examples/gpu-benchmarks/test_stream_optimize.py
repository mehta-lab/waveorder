"""Smoke test for stream_optimize_positions high-level API.

Runs 3 positions from the OPS dataset through the streaming pipeline
and verifies output was written correctly.

Usage:
    uv run python test_stream_optimize.py --device cuda:0
"""

import argparse
import os
import shutil

os.environ["TQDM_DISABLE"] = "1"

import numpy as np

from waveorder.api import phase
from waveorder.io.streaming import stream_optimize_positions
from waveorder.optim import OptimizableFloat

INPUT = "/hpc/projects/intracellular_dashboard/ops/ops0105_20260106/0-convert/live_imaging/phenotyping_transform.zarr"
OUTPUT = "/home/sricharan.varra/mydata/data/waveorder-gpu-io/test_stream_optimize.zarr"
POSITIONS = ["A/1/029028", "A/1/029029", "A/1/029030"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--opt-iterations", type=int, default=10)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # Clean up previous output
    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)

    print(f"Input:  {INPUT}")
    print(f"Output: {OUTPUT}")
    print(f"Positions: {POSITIONS}")
    print(f"Tile: {args.tile_size}, Batch: {args.batch_size}, Iters: {args.opt_iterations}")
    print()

    # 20x OPS physics settings
    settings = phase.Settings(
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

    print("Running stream_optimize_positions...")
    stats = stream_optimize_positions(
        input_zarr=INPUT,
        output_zarr=OUTPUT,
        position_names=POSITIONS,
        settings=settings,
        opt_iterations=args.opt_iterations,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        device=args.device,
        n_read_workers=4,
    )

    print(stats)
    print()

    # Verify outputs were written
    print("Verifying output zarr...")
    from iohub.ngff import open_ome_zarr

    with open_ome_zarr(OUTPUT, mode="r") as out:
        for pos_name in POSITIONS:
            pos = out[pos_name]
            data = pos.data[0, 0, 0]  # (Y, X)
            n_nonzero = np.count_nonzero(data)
            print(f"  {pos_name}: shape={data.shape}, "
                  f"nonzero={n_nonzero}/{data.size} "
                  f"({'PASS' if n_nonzero > 0 else 'FAIL - all zeros'})")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
