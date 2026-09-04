"""Sweep one or two config parameters, reconstruct each value, and view results.

Writes one config per value (or per value pair) into an output folder, runs
``wo rec`` for each, then opens the reconstructions in napari. A one-parameter
sweep opens all results side by side with ``wo view``. A two-parameter sweep
(``--param2``/``--values2``) stacks the grid of reconstructions into a single
napari layer with the two parameters on two sliders.

Examples
--------
    python sweep.py -i ./data.zarr -c ./config.yml \
        --param regularization_strength --values 1e-4 1e-3 1e-2 1e-1

    python sweep.py -i ./data.zarr -c ./config.yml \
        --param tilt_angle_azimuth --values 0 1.57 3.14 4.71 \
        --param2 tilt_angle_zenith --values2 0.05 0.1 0.2

    python sweep.py -i ./data.zarr -c ./config.yml \
        --param reconstruction_algorithm --values Tikhonov TV --no-view
"""

import argparse
import itertools
import shutil
import subprocess
from pathlib import Path

import napari
import numpy as np
import yaml
from iohub.ngff import open_ome_zarr


def _wo_cmd() -> list[str]:
    """Return the invocation prefix for the waveorder CLI."""
    if shutil.which("wo"):
        return ["wo"]
    return ["uv", "run", "wo"]


def _set_nested(cfg: dict, param: str, value) -> bool:
    """Set ``param`` anywhere it appears under an ``apply_inverse``/``transfer_function`` block."""
    found = False
    for block in ("phase", "fluorescence", "birefringence"):
        sub = cfg.get(block)
        if not isinstance(sub, dict):
            continue
        for section in ("apply_inverse", "transfer_function"):
            sec = sub.get(section)
            if isinstance(sec, dict) and param in sec:
                sec[param] = value
                found = True
    return found


def _coerce(value: str):
    """Best-effort str to number, leaving non-numeric strings as-is."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _tag(value: str) -> str:
    """Filesystem-safe tag for a parameter value."""
    return str(value).replace(".", "p").replace("-", "m")


def _load_recon(path: str) -> np.ndarray:
    """Load the first position of a reconstruction as a squeezed array."""
    positions = sorted(q for q in Path(path).glob("*/*/*") if q.is_dir())
    pos = open_ome_zarr(positions[0], mode="r")
    return np.asarray(pos.data).squeeze()


def _view_grid(outputs: list[list[str]], p1: str, v1: list[str], p2: str, v2: list[str]) -> None:
    """Open a 2-parameter sweep as one napari layer with the parameters on two sliders."""
    stack = np.stack([np.stack([_load_recon(o) for o in row]) for row in outputs])
    viewer = napari.Viewer()
    viewer.add_image(stack, name=f"{p1} x {p2} sweep")
    viewer.dims.set_axis_label(0, p1)
    viewer.dims.set_axis_label(1, p2)
    print(f"slider 0: {p1} = {v1}")
    print(f"slider 1: {p2} = {v2}")
    napari.run()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, type=Path, help="input OME-Zarr")
    p.add_argument("-c", "--config", required=True, type=Path, help="base config .yml")
    p.add_argument("--param", default="regularization_strength", help="parameter name to sweep")
    p.add_argument("--values", nargs="+", required=True, help="values to sweep over")
    p.add_argument("--param2", default=None, help="second parameter for a grid sweep")
    p.add_argument("--values2", nargs="+", default=None, help="values for the second parameter")
    p.add_argument("--outdir", type=Path, default=Path("sweep"), help="output folder")
    p.add_argument("--no-view", action="store_true", help="skip launching napari")
    args = p.parse_args()

    if (args.param2 is None) != (args.values2 is None):
        raise SystemExit("--param2 and --values2 must be given together")

    args.outdir.mkdir(parents=True, exist_ok=True)
    wo = _wo_cmd()
    # Expand HCS positions here; subprocess does not run a shell to glob.
    positions = [str(q) for q in sorted(args.input.glob("*/*/*")) if q.is_dir()]
    if not positions:
        raise SystemExit(f"no positions found under {args.input}/*/*/*")

    values2 = args.values2 or [None]
    outputs = [[None] * len(values2) for _ in args.values]
    for (i, raw1), (j, raw2) in itertools.product(enumerate(args.values), enumerate(values2)):
        cfg = yaml.safe_load(args.config.read_text())  # fresh copy
        label = f"{args.param}_{_tag(raw1)}"
        if not _set_nested(cfg, args.param, _coerce(raw1)):
            raise SystemExit(f"param {args.param!r} not found in any config block")
        if raw2 is not None:
            if not _set_nested(cfg, args.param2, _coerce(raw2)):
                raise SystemExit(f"param {args.param2!r} not found in any config block")
            label += f"_{args.param2}_{_tag(raw2)}"

        cfg_path = args.outdir / f"{label}.yml"
        out_path = args.outdir / f"recon_{label}.zarr"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        print(f"\n=== {label} -> {out_path} ===")
        subprocess.run(wo + ["rec", "-i", *positions, "-c", str(cfg_path), "-o", str(out_path)], check=True)
        outputs[i][j] = str(out_path)

    flat = [o for row in outputs for o in row]
    print("\nReconstructions:")
    for o in flat:
        print(" ", o)

    if args.no_view:
        print("\nView with:\n  " + " ".join(wo + ["view", *flat]))
    elif args.param2 is not None:
        print("\nOpening 2-parameter sweep in napari (two sliders)...")
        _view_grid(outputs, args.param, args.values, args.param2, args.values2)
    else:
        print("\nOpening in napari (grid view)...")
        subprocess.run(wo + ["view", *flat])


if __name__ == "__main__":
    main()
