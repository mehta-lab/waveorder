import subprocess
import os
import runpy
import waveorder
from pathlib import Path


# Currently, these testswill run in the background if no screen is available
# If you do have a screen available, plots will appear, and you'll need to
# close them to make the tests pass.
# TODO: see if we can make these run locally w/o showing matplotlib
# @patch("matplotlib.pyplot.show")
def test_examples():
    EXAMPLE_DIR = Path(waveorder.__file__).parent.parent / "examples"
    scripts = [
        "2D_QLIPP_simulation/2D_QLIPP_forward.py",
        "2D_QLIPP_simulation/2D_QLIPP_recon.py",
        "3D_PODT_phase_simulation/3D_PODT_Phase_forward.py",
        "3D_PODT_phase_simulation/3D_PODT_Phase_recon.py",
    ]

    for script in scripts:
        path = os.path.join(os.getcwd(), "examples/", script)
        completed_process = subprocess.run(["python", path])
        assert completed_process.returncode == 0
