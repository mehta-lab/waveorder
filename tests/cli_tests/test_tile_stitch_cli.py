"""CLI smoke tests for ``wo tile-stitch``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from waveorder.cli.main import cli


@pytest.fixture
def phantom_input(tmp_path: Path) -> Path:
    """Tiny OME-Zarr FOV with a 3D phantom volume — enough to drive a smoke run."""
    out = tmp_path / "input.zarr"
    rng = np.random.default_rng(0)
    # FOV layout: (T, C, Z, Y, X)
    data = rng.normal(size=(1, 1, 4, 32, 32)).astype(np.float32)
    ds = open_ome_zarr(out, layout="fov", mode="w", channel_names=["BF"])
    ds.create_image("0", data)
    ds.close()
    return out


@pytest.fixture
def tile_stitch_config(tmp_path: Path) -> Path:
    """Minimal phase tile-stitch settings YAML."""
    cfg = tmp_path / "tile_stitch.yml"
    settings = {
        "schema_version": "1",
        "tile": {"tile_size": {"z": 4, "y": 16, "x": 16}, "overlap": {"y": 4, "x": 4}},
        "blend": {"kind": "uniform_mean"},
        "recon": {"kind": "phase"},
    }
    cfg.write_text(yaml.safe_dump(settings))
    return cfg


def test_cli_help_lists_tile_stitch():
    """``wo --help`` lists the tile-stitch subcommand."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "tile-stitch" in result.output


def test_cli_subcommand_help_runs():
    runner = CliRunner()
    result = runner.invoke(cli, ["tile-stitch", "--help"])
    assert result.exit_code == 0
    assert "Single-process tiled reconstruction" in result.output


def test_cli_smoke_writes_output(phantom_input: Path, tile_stitch_config: Path, tmp_path: Path):
    """End-to-end: run the CLI, verify output zarr exists with expected shape."""
    output_path = tmp_path / "output.zarr"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "tile-stitch",
            "-i",
            str(phantom_input),
            "-c",
            str(tile_stitch_config),
            "-o",
            str(output_path),
        ],
    )
    if result.exit_code != 0:
        pytest.fail(f"CLI exited {result.exit_code}\n{result.output}\n{result.exception!r}")

    assert output_path.exists()
    out_ds = open_ome_zarr(output_path, layout="fov", mode="r")
    arr = out_ds["0"]
    # FOV layout: (T, C, Z, Y, X). Phantom was (1, 1, 4, 32, 32).
    assert arr.shape == (1, 1, 4, 32, 32)
