"""Create an empty OME-Zarr v0.5 output store for streaming benchmark.

Output shape per position: (T=1, C=1, Z=1, Y=2048, X=2048) float32.
Channel: Phase2D (optimized phase reconstruction).

Usage:
    uv run python create_output_store.py \
        /hpc/projects/.../phenotyping_transform.zarr \
        /home/sricharan.varra/mydata/data/waveorder-gpu-io/phase_2d_optimized.zarr \
        --positions A/1/028028 A/1/028029 ... A/1/030030
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
from iohub.ngff import open_ome_zarr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_zarr", help="Input OME-Zarr (for shape/scale)")
    parser.add_argument("output_zarr", help="Output OME-Zarr to create")
    parser.add_argument("--positions", nargs="+", required=True,
                        help="Position paths: A/1/028028 A/1/029029 ...")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output_zarr)
    if output_path.exists():
        if args.overwrite:
            shutil.rmtree(output_path)
            print(f"Removed existing: {output_path}")
        else:
            print(f"Output exists: {output_path}. Use --overwrite to replace.")
            return

    # Get input metadata
    with open_ome_zarr(args.input_zarr, mode="r") as input_store:
        pos0 = input_store[args.positions[0]]
        Y, X = pos0.data.shape[-2], pos0.data.shape[-1]
        scale = list(pos0.scale)

    # Output: (T=1, C=1, Z=1, Y, X) float32
    output_shape = (1, 1, 1, Y, X)
    output_chunks = (1, 1, 1, Y, X)

    print(f"Creating OME-Zarr v0.5 output store: {output_path}")
    print(f"  Shape per position: {output_shape}")
    print(f"  Dtype: float32")
    print(f"  Positions: {len(args.positions)}")

    with open_ome_zarr(
        output_path,
        layout="hcs",
        mode="w",
        channel_names=["Phase2D"],
        version="0.5",
    ) as output_store:
        for pos_path in args.positions:
            # v0.5 API: create_position(row, col, pos)
            parts = pos_path.split("/")
            row, col, pos_name = parts[0], parts[1], parts[2]
            pos = output_store.create_position(row, col, pos_name)
            pos.create_zeros(
                name="0",
                shape=output_shape,
                chunks=output_chunks,
                dtype=np.float32,
            )
            print(f"  Created: {pos_path}")

    print("Done.")


if __name__ == "__main__":
    main()
