"""Fresh-process imports, in the order the CLI triggers them.

pytest imports modules in whatever order the collected tests dictate,
which can hide circular imports that a cold ``wo`` process hits. These
tests spawn a clean interpreter per import path.
"""

import subprocess
import sys

import pytest

IMPORT_STATEMENTS = [
    # The first import `wo rec` performs when it loads a config.
    "from waveorder.config.loader import load_reconstruction_settings",
    # The reverse entry: calibration numerics before any config import.
    "from waveorder.calibrate import run_evidence_calibration, run_zernike_calibration",
    # The schema alone (imports the Zernike names from waveorder.calibrate).
    "from waveorder.config.schema import V4Config",
    # The CLI app with every registered verb.
    "import waveorder.cli.main",
]


@pytest.mark.parametrize("statement", IMPORT_STATEMENTS)
def test_fresh_process_import(statement):
    result = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"{statement!r} failed:\n{result.stderr}"
