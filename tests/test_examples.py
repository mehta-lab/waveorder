import runpy
from pathlib import Path
import sys
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
        EXAMPLES / "models" / "isotropic_thin_3d.py",
        EXAMPLES / "models" / "phase_thick_3d.py",
        EXAMPLES / "models" / "inplane_oriented_thick_pol3d.py",
    ],
    ids=lambda p: p.name,
)
def test_examples(example):
    # run script, mocking all the show and imshow calls
    with (
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.imshow"),
        patch("builtins.input"),
    ):
        runpy.run_path(example, run_name="__main__")

    plt.close("all")
    if napari := sys.modules.get("napari"):
        if viewer := napari.current_viewer():
            viewer.close()
