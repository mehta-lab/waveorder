"""Tests for CLI path parsing with zarr v3 structures."""

from contextlib import nullcontext

from click.testing import CliRunner

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta
from typer.main import get_command

from waveorder.cli.main import app
from waveorder.cli.parsing import _validate_and_process_paths


def test_validate_paths_filters_zarr_json_from_glob(tmp_path):
    """Test that zarr.json files are filtered when using glob pattern."""
    # Create a minimal zarr v3 plate structure with actual data
    plate_path = tmp_path / "plate.zarr"
    try:
        plate = open_ome_zarr(
            plate_path,
            layout="hcs",
            mode="w-",
            channel_names=["ch"],
            version="0.5",
        )
        pos = plate.create_position("A", "1", "0")
        # Create data so position has proper OME-NGFF metadata
        pos.create_zeros(
            "0",
            (1, 1, 2, 3, 4),
            dtype=np.uint16,
            transform=[TransformationMeta(type="scale", scale=[1, 1, 1, 1, 1])],
        )
        plate.close()
    except Exception:
        # Skip if zarr v3 not supported
        pytest.skip("Zarr v3 not supported by installed iohub version")

    # Glob pattern A/1/* returns both position dir and zarr.json
    glob_paths = list(plate_path.glob("A/1/*"))
    zarr_jsons = [p for p in glob_paths if not p.is_dir()]

    # Verify zarr.json files are in glob results
    assert len(zarr_jsons) > 0, "zarr.json files should be in glob results"

    # Call the parsing function with glob results
    result = _validate_and_process_paths([str(p) for p in glob_paths])

    # Only the position directory should remain
    assert len(result) == 1
    assert result[0].name == "0"
    assert result[0].is_dir()


@pytest.mark.parametrize("form", ["repeated", "shell-expanded"])
def test_input_option_accepts_repeated_and_greedy_paths(tmp_path, monkeypatch, form):
    input_paths = [tmp_path / "position-1", tmp_path / "position-2"]
    for path in input_paths:
        path.mkdir()
    config_path = tmp_path / "config.yml"
    config_path.touch()

    monkeypatch.setattr(
        "iohub.ngff.open_ome_zarr",
        lambda *args, **kwargs: nullcontext(object()),
    )

    command = get_command(app).commands["reconstruct"]
    received = {}
    command.callback = lambda **kwargs: received.update(kwargs)
    input_args = (
        ["-i", str(input_paths[0]), "-i", str(input_paths[1])]
        if form == "repeated"
        else ["-i", *(str(path) for path in input_paths)]
    )

    result = CliRunner().invoke(
        command,
        [*input_args, "-c", str(config_path), "-o", str(tmp_path / "output.zarr")],
    )

    assert result.exit_code == 0, result.output
    assert received["input_position_dirpaths"] == input_paths
