import runpy
from pathlib import Path
import sys
import os
import subprocess
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

DOCS = Path(__file__).parent.parent / "docs"
EXAMPLES = DOCS / "examples"


@pytest.mark.parametrize(
    "example",
    [
        EXAMPLES / "maintenance" / "QLIPP_simulation/2D_QLIPP_forward.py",
        EXAMPLES / "maintenance" / "QLIPP_simulation/2D_QLIPP_recon.py",
        EXAMPLES / "maintenance" / "PTI_simulation/PTI_Simulation_Forward_2D3D.py",
        EXAMPLES / "maintenance" / "PTI_simulation/PTI_Simulation_Recon2D.py",
        EXAMPLES / "maintenance" / "PTI_simulation/PTI_Simulation_Recon3D.py",
    ],
    ids=lambda p: p.name,
)
def test_maintenance_examples(example):
    """Test maintenance examples (QLIPP, PTI) with mocked plotting"""
    with (
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.imshow"),
        patch("builtins.input"),
    ):
        try:
            runpy.run_path(example, run_name="__main__")
        except SystemExit as e:
            if e.code != 0:
                pytest.fail(f"Script {example.name} exited with code {e.code}")

    plt.close("all")


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skip on GitHub Actions, requires napari",
)
@pytest.mark.parametrize(
    "script",
    [
        "isotropic_thin_3d.py",
        "phase_thick_3d.py",
        "inplane_oriented_thick_pol3d.py",
    ],
)
def test_phase_examples(script):
    """Test phase model examples - need user input so skip on CI"""
    path = EXAMPLES / "models" / script
    # examples needs two <enters>s so send input="e\ne"
    completed_process = subprocess.run(
        [sys.executable, str(path)],
        input="e\ne",
        encoding="ascii",
        env=os.environ,
    )
    assert completed_process.returncode == 0
