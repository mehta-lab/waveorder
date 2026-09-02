"""Download the QPI-from-defocus sample dataset for the tutorial.

Fetches ``recOrder_session.zip`` from zenodo (the dataset used by the
``docs/examples/demos/QPI_defocus`` demo) and prints the path to the brightfield
defocus stack. It is already an **OME-Zarr**, so it plugs straight into
``wo view`` / ``wo rec`` with no conversion.

The stack is a single ``BF`` channel, 11 defocus slices (yx = 0.325 µm,
z = 2.0 µm), ideal for a **2D phase-from-defocus** reconstruction. A ready-made
config lives at ``references/example_qpi_2d.yml``.

Examples
--------
    python get_example_data.py
"""

import argparse
import io
import zipfile
from pathlib import Path

import requests

URL = "https://zenodo.org/record/8386856/files/recOrder_session.zip"


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    data_dir = Path.home() / ".waveorder_tutorial_data"
    data_dir.mkdir(exist_ok=True)
    session_dir = data_dir / "recOrder_session"

    if not session_dir.exists():
        print("Downloading QPI-defocus sample from zenodo (~10 MB)...")
        resp = requests.get(URL)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(session_dir)

    raw_path = session_dir / "recOrder_session" / "phase_snap_0" / "raw_data.zarr"
    if not raw_path.exists():
        raise SystemExit(f"expected {raw_path} after unzip, not found")

    config = Path(__file__).resolve().parent.parent / "references" / "example_qpi_2d.yml"

    print(f"\nRaw OME-Zarr (BF defocus stack):  {raw_path}")
    print(f"\nInspect with:  wo view {raw_path}")
    print("Already an OME-Zarr, so skip conversion (Stage 3, Case 1); channel 'BF', dim 2D.")
    print("\n2D phase-from-defocus reconstruction:")
    print(f"  wo rec -i {raw_path}/0/0/0 -c {config} -o ./qpi_2d_recon.zarr")
    print(f"  wo view {raw_path} ./qpi_2d_recon.zarr")


if __name__ == "__main__":
    main()
