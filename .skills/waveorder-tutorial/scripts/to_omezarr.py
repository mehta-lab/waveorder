"""Convert a TIFF / .npy / plain-zarr array into a viewable OME-Zarr.

Writes an HCS OME-Zarr (``.../0/0/0``) with a TCZYX array, named channels, and a
pixel-size scale transform, so it can be read by ``wo rec`` and shown by
``wo view``.

Examples
--------
    python to_omezarr.py stack.tif ./data.zarr \
        --channel-name Brightfield --axes ZYX \
        --yx-pixel-size 0.1 --z-pixel-size 0.25

    python to_omezarr.py arr.npy ./data.zarr \
        --channel-name GFP --axes CZYX --yx-pixel-size 0.1 --z-pixel-size 0.25
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
import zarr
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta


def _load(path: Path) -> np.ndarray:
    """Load a TIFF, .npy, or plain zarr array as a numpy array."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return tifffile.imread(str(path))
    if suffix == ".npy":
        return np.load(path)
    # Assume a plain zarr array directory.
    return np.asarray(zarr.open(str(path), mode="r"))


def _to_tczyx(arr: np.ndarray, axes: str) -> np.ndarray:
    """Reorder/expand an array with the given axis labels into TCZYX."""
    axes = axes.upper()
    if len(axes) != arr.ndim:
        raise ValueError(f"--axes {axes!r} has {len(axes)} labels but array has {arr.ndim} dims")
    if set(axes) - set("TCZYX"):
        raise ValueError(f"--axes may only contain T, C, Z, Y, X; got {axes!r}")

    # Move each present axis into TCZYX order.
    order = [axes.index(a) for a in "TCZYX" if a in axes]
    arr = np.transpose(arr, order)
    # Insert singleton dims for any missing axis.
    for i, a in enumerate("TCZYX"):
        if a not in axes:
            arr = np.expand_dims(arr, i)
    return arr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="input .tif/.tiff/.npy or plain zarr dir")
    p.add_argument("output", type=Path, help="output .zarr path")
    p.add_argument("--axes", default="ZYX", help="axis order of the input (subset of TCZYX)")
    p.add_argument("--channel-name", action="append", default=None, help="channel name (repeat per channel)")
    p.add_argument("--yx-pixel-size", type=float, default=0.1, help="lateral pixel size (µm)")
    p.add_argument("--z-pixel-size", type=float, default=1.0, help="axial pixel size (µm)")
    args = p.parse_args()

    arr = _to_tczyx(_load(args.input), args.axes)
    n_channels = arr.shape[1]
    names = args.channel_name or [f"Channel{i}" for i in range(n_channels)]
    if len(names) != n_channels:
        raise ValueError(f"{len(names)} channel names for {n_channels} channels")

    ds = open_ome_zarr(args.output, layout="hcs", mode="w", channel_names=names)
    pos = ds.create_position("0", "0", "0")
    pos.create_image(
        "0",
        arr.astype(np.float32),
        transform=[
            TransformationMeta(
                type="scale",
                scale=[1, 1, args.z_pixel_size, args.yx_pixel_size, args.yx_pixel_size],
            )
        ],
    )
    ds.close()
    print(f"Wrote {args.output}  shape TCZYX={arr.shape}  channels={names}")
    print(f"Inspect with:  wo view {args.output}")


if __name__ == "__main__":
    main()
