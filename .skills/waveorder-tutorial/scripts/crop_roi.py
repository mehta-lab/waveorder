"""Crop an OME-Zarr position to a small ROI for fast iteration.

Reads one position of an HCS OME-Zarr, slices the TCZYX array with the given
voxel ranges, and writes a new single-position HCS OME-Zarr that preserves
channel names and pixel scale. Omitted axes are kept whole.

Examples
--------
    python crop_roi.py -i ./data.zarr -o ./roi.zarr --y 800:1312 --x 900:1412

    python crop_roi.py -i ./data.zarr -o ./roi.zarr \
        --t 0:1 --z 5:25 --y 0:512 --x 0:512 --position 0/0/0
"""

import argparse
from pathlib import Path

from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta


def _parse_range(spec: str | None) -> slice:
    """Parse a ``start:stop`` voxel range into a slice.

    Parameters
    ----------
    spec : str or None
        Range string like ``"800:1312"``, or None to keep the whole axis.

    Returns
    -------
    slice
        Slice covering the requested range.
    """
    if spec is None:
        return slice(None)
    start, stop = spec.split(":")
    return slice(int(start) if start else None, int(stop) if stop else None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, type=Path, help="input HCS OME-Zarr")
    p.add_argument("-o", "--output", required=True, type=Path, help="output .zarr path")
    p.add_argument("--position", default=None, help="position path like 0/0/0 (default: first found)")
    p.add_argument("--t", default=None, help="time range start:stop")
    p.add_argument("--c", default=None, help="channel index range start:stop")
    p.add_argument("--z", default=None, help="z range start:stop")
    p.add_argument("--y", default=None, help="y range start:stop")
    p.add_argument("--x", default=None, help="x range start:stop")
    args = p.parse_args()

    if args.position is None:
        positions = sorted(q for q in args.input.glob("*/*/*") if q.is_dir())
        if not positions:
            raise SystemExit(f"no positions found under {args.input}/*/*/*")
        position = positions[0].relative_to(args.input).as_posix()
    else:
        position = args.position

    src = open_ome_zarr(args.input / position, mode="r")
    slices = tuple(_parse_range(s) for s in (args.t, args.c, args.z, args.y, args.x))
    cropped = src.data[slices]
    c_slice = slices[1]
    names = src.channel_names[c_slice] if args.c else src.channel_names
    scale = src.scale

    ds = open_ome_zarr(args.output, layout="hcs", mode="w", channel_names=list(names))
    pos = ds.create_position("0", "0", "0")
    pos.create_image("0", cropped, transform=[TransformationMeta(type="scale", scale=list(scale))])
    ds.close()

    print(f"Wrote {args.output}  shape TCZYX={cropped.shape}  channels={list(names)}")
    print(f"Inspect with:  wo view {args.output}")


if __name__ == "__main__":
    main()
