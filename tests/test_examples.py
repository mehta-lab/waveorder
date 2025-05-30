import os
import subprocess
import sys

import pytest


def _run_scripts(scripts):
    for script in scripts:
        path = os.path.join(os.getcwd(), "docs/examples/maintenance/", script)
        completed_process = subprocess.run(
            [sys.executable, path], env=os.environ
        )
        assert completed_process.returncode == 0


# Currently, these tests will run in the background if no screen is available
# If you do have a screen available, plots will appear, and you'll need to
# close them to make the tests pass.
# TODO: see if we can make these run locally w/o showing matplotlib
# @patch("matplotlib.pyplot.show")
@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skip on GitHub Actions"
)
def test_qlipp_examples():
    scripts = [
        "QLIPP_simulation/2D_QLIPP_forward.py",
        "QLIPP_simulation/2D_QLIPP_recon.py",
    ]
    _run_scripts(scripts)


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true", reason="Skip on GitHub Actions"
)
def test_pti_examples():
    scripts = [
        "PTI_simulation/PTI_Simulation_Forward_2D3D.py",
        "PTI_simulation/PTI_Simulation_Recon2D.py",
        "PTI_simulation/PTI_Simulation_Recon3D.py",
    ]
    _run_scripts(scripts)


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Skip on GitHub Actions, requires napari",
)
def test_phase_examples():
    scripts = [
        "isotropic_thin_3d.py",
        "phase_thick_3d.py",
        "inplane_oriented_thick_pol3d.py",
    ]

    for script in scripts:
        path = os.path.join(os.getcwd(), "docs/examples/models/", script)
        # examples needs two <enters>s so send input="e\ne"
        completed_process = subprocess.run(
            [sys.executable, path],
            input="e\ne",
            encoding="ascii",
            env=os.environ,
        )
        assert completed_process.returncode == 0
