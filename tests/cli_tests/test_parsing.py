"""Tests for CLI path parsing with zarr v3 structures."""

import numpy as np
import pytest
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta

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
            transform=[
                TransformationMeta(type="scale", scale=[1, 1, 1, 1, 1])
            ],
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
    result = _validate_and_process_paths(
        None, None, [str(p) for p in glob_paths]
    )

    # Only the position directory should remain
    assert len(result) == 1
    assert result[0].name == "0"
    assert result[0].is_dir()
