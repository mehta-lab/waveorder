"""Tests for CLI path parsing with zarr v3 structures."""

from pathlib import Path

import pytest
from iohub.ngff import open_ome_zarr

from waveorder.cli.parsing import _validate_and_process_paths


def test_validate_paths_filters_zarr_json_from_glob(tmp_path):
    """Test that zarr.json files are filtered when using */*/*/* glob pattern."""
    # Create a minimal zarr v3 plate structure
    plate_path = tmp_path / "plate.zarr"
    try:
        plate = open_ome_zarr(
            plate_path,
            layout="hcs",
            mode="w-",
            channel_names=["ch"],
            version="0.5",
        )
        plate.create_position("A", "1", "0")
        plate.close()
    except Exception:
        # Skip if zarr v3 not supported
        pytest.skip("Zarr v3 not supported by installed iohub version")

    # Glob pattern that would include both positions and zarr.json files
    glob_pattern = str(plate_path / "*" / "*" / "*")
    paths = [str(p) for p in Path(plate_path).glob("*/*/*")]

    # Verify zarr.json files exist in glob results
    json_files = [p for p in paths if "zarr.json" in p]
    assert len(json_files) > 0, "Test setup should include zarr.json files"

    # Call the parsing function
    result = _validate_and_process_paths(None, None, paths)

    # All results should be directories only
    assert all(p.is_dir() for p in result)
    assert len(result) == 1  # Only the position directory
    assert result[0].name == "0"
