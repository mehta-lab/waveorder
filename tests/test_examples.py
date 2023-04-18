import subprocess
import os
import pytest
import sys


@pytest.mark.skipif("napari" not in sys.modules, reason="requires napari")
def test_example_phase3Dto3D():
    path = os.path.join(os.getcwd(), "examples/models/phase3Dto3D.py")
    # examples needs two <enters>s so send input="e\ne"
    completed_process = subprocess.run(
        ["python", path], input="e\ne", encoding="ascii"
    )
    assert completed_process.returncode == 0
