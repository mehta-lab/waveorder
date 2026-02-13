import os
import runpy
import subprocess
import sys
from pathlib import Path
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
    os.getenv("GITHUB_ACTIONS") == "true" and sys.platform != "linux",
    reason="Skip on GitHub Actions non-Linux platforms, napari requires headless display",
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
    """Test phase model examples"""
    path = EXAMPLES / "models" / script
    # examples needs two <enters>s so send input="e\ne"
    completed_process = subprocess.run(
        [sys.executable, str(path)],
        input="e\ne",
        encoding="ascii",
        env=os.environ,
    )
    assert completed_process.returncode == 0


@pytest.mark.parametrize(
    "example",
    sorted((EXAMPLES / "api").glob("*.py")),
    ids=lambda p: p.name,
)
def test_api_examples(example):
    """Test API-level examples (no napari, no matplotlib)"""
    runpy.run_path(str(example), run_name="__main__")


@pytest.mark.parametrize(
    "example",
    sorted((EXAMPLES / "cli").glob("*.sh")),
    ids=lambda p: p.name,
)
def test_cli_examples(example, tmp_path, monkeypatch):
    """Test CLI-level shell script examples (skip 'wo view' lines)."""
    import shlex
    import shutil

    from click.testing import CliRunner

    from waveorder.cli.main import cli

    shutil.copytree(example.parent / "configs", tmp_path / "configs")
    monkeypatch.chdir(tmp_path)

    # Parse shell script into commands (join continuation lines)
    lines = example.read_text().splitlines()
    commands = []
    current = ""
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        if line.rstrip().endswith("\\"):
            current += line.rstrip()[:-1] + " "
        else:
            current += line
            commands.append(current.strip())
            current = ""

    runner = CliRunner()
    for cmd in commands:
        if cmd.startswith("wo view"):
            continue
        # Strip leading "wo " and split into args
        args = shlex.split(cmd.removeprefix("wo "))
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, f"Command '{cmd}' failed:\n{result.output}"
