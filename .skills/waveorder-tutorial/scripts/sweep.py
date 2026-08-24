"""Sweep one config parameter, reconstruct each value, and view side by side.

Writes one config per value into a ``sweep/`` folder, runs ``wo rec`` for each,
then opens all reconstructions in napari with ``wo view`` for comparison.

Examples
--------
    python sweep.py -i ./data.zarr -c ./config.yml \
        --param regularization_strength --values 1e-4 1e-3 1e-2 1e-1

    python sweep.py -i ./data.zarr -c ./config.yml \
        --param reconstruction_algorithm --values Tikhonov TV --no-view
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import yaml


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
    """Best-effort str → number, leaving non-numeric strings as-is."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, type=Path, help="input OME-Zarr")
    p.add_argument("-c", "--config", required=True, type=Path, help="base config .yml")
    p.add_argument("--param", default="regularization_strength", help="parameter name to sweep")
    p.add_argument("--values", nargs="+", required=True, help="values to sweep over")
    p.add_argument("--outdir", type=Path, default=Path("sweep"), help="output folder")
    p.add_argument("--no-view", action="store_true", help="skip launching napari")
    args = p.parse_args()

    base = yaml.safe_load(args.config.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    wo = _wo_cmd()
    # Expand HCS positions here — subprocess does not run a shell to glob.
    positions = [str(p) for p in sorted(args.input.glob("*/*/*")) if p.is_dir()]
    if not positions:
        raise SystemExit(f"no positions found under {args.input}/*/*/*")

    outputs = []
    for raw in args.values:
        value = _coerce(raw)
        cfg = yaml.safe_load(args.config.read_text())  # fresh copy
        if not _set_nested(cfg, args.param, value):
            raise SystemExit(f"param {args.param!r} not found in any config block")

        tag = str(raw).replace(".", "p").replace("-", "m")
        cfg_path = args.outdir / f"{args.param}_{tag}.yml"
        out_path = args.outdir / f"recon_{args.param}_{tag}.zarr"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        print(f"\n=== {args.param} = {value} → {out_path} ===")
        subprocess.run(wo + ["rec", "-i", *positions, "-c", str(cfg_path), "-o", str(out_path)], check=True)
        outputs.append(str(out_path))

    print("\nReconstructions:")
    for o in outputs:
        print(" ", o)

    if not args.no_view and outputs:
        print("\nOpening in napari (grid view)...")
        subprocess.run(wo + ["view", *outputs])
    else:
        print("\nView with:\n  " + " ".join(wo + ["view", *outputs]))


if __name__ == "__main__":
    main()
