# /// script
# requires-python = ">=3.12"
# dependencies = ["requests"]
# ///
"""Download the QPI-from-defocus sample dataset for the tutorial.

Fetches ``recOrder_session.zip`` from zenodo and prints the path to the
brightfield defocus stack. It is already an **OME-Zarr**, so it plugs straight
into ``wo view`` / ``wo rec`` with no conversion.

The stack is a single ``BF`` channel, 11 defocus slices (yx = 0.325 µm,
z = 2.0 µm), ideal for a **2D phase-from-defocus** reconstruction. The draft
config is in the skill's ``references/example_qpi_2d.md``; save it as
``invert_false.yml`` and ``invert_true.yml``.

Examples
--------
    uv run get_example_data.py
"""

import argparse
import io
import tempfile
import zipfile
from pathlib import Path

import requests

URL = "https://zenodo.org/record/8386856/files/recOrder_session.zip"


def main() -> None:
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()

    data_dir = Path.home() / ".waveorder_tutorial_data"
    data_dir.mkdir(exist_ok=True)
    session_dir = data_dir / "recOrder_session"

    if not session_dir.exists():
        print("Downloading QPI-defocus sample from zenodo (~24 MB)...")
        resp = requests.get(URL, timeout=(10, 300))
        resp.raise_for_status()
        # Unzip next to the target and rename when complete, so an interrupted
        # run never leaves a partial folder that later runs treat as done.
        with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
            staged = Path(tmp) / "recOrder_session"
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                z.extractall(staged)
            staged.rename(session_dir)

    raw_path = session_dir / "recOrder_session" / "phase_snap_0" / "raw_data.zarr"
    if not raw_path.exists():
        raise SystemExit(f"expected {raw_path} after unzip, not found")

    print(f"\nRaw OME-Zarr (BF defocus stack):  {raw_path}")
    print(f"\nInspect with:  wo view {raw_path}")
    print(
        "Already an OME-Zarr, so skip conversion (Stage 3, Case 1); channel 'BF', dim 2D."
    )
    print("\n2D phase-from-defocus drafts (both invert_phase_contrast copies):")
    for name in ("invert_false", "invert_true"):
        print(f"  wo rec -i {raw_path}/0/0/0 -c {name}.yml -o ./{name}.zarr")
    print(f"  wo view {raw_path} ./invert_false.zarr ./invert_true.zarr </dev/null &")


if __name__ == "__main__":
    main()
