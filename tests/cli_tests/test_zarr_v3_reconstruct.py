"""Tests for Zarr v3 (OME-NGFF v0.5) reconstruction support.

Verifies that:
- Plate metadata extraction works for both v0.4 and v0.5
- Output zarr version matches input version
- The full reconstruct CLI works for v0.5 input
"""

import shutil
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pytest
from click.testing import CliRunner
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta

from waveorder.cli import settings
from waveorder.cli.apply_inverse_transfer_function import (
    get_reconstruction_output_metadata,
)
from waveorder.cli.main import cli
from waveorder.cli.utils import create_empty_hcs_zarr
from waveorder.io import utils

INPUT_SCALE = [1, 1, 2.0, 6.5, 6.5]
CHANNEL_NAMES = [f"State{x}" for x in range(4)]


def _iohub_supports_v05():
    """Check if the installed iohub actually creates v0.5 stores."""
    tmp = Path(mkdtemp()) / "v05check.zarr"
    try:
        ds = open_ome_zarr(tmp, layout="fov", mode="w-", channel_names=["ch"], version="0.5")
        supported = ds.version == "0.5"
        ds.close()
        return supported
    except Exception:
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_V05_SUPPORTED = _iohub_supports_v05()


def _create_test_plate(path, version="0.4"):
    """Create a minimal HCS plate for testing."""
    plate = open_ome_zarr(
        path,
        layout="hcs",
        mode="w-",
        channel_names=CHANNEL_NAMES,
        version=version,
    )
    pos = plate.create_position("0", "0", "0")
    pos.create_zeros(
        "0",
        (2, 4, 4, 5, 6),
        dtype=np.uint16,
        transform=[TransformationMeta(type="scale", scale=INPUT_SCALE)],
    )
    plate.zattrs["Summary"] = {"custom_key": "custom_value"}
    plate.close()


def _get_store_version(path):
    """Get the OME-NGFF version of a store using iohub's API."""
    ds = open_ome_zarr(path, mode="r")
    version = ds.version
    ds.close()
    return version


def _versions():
    versions = ["0.4"]
    if _V05_SUPPORTED:
        versions.append("0.5")
    return versions


@pytest.fixture(params=_versions())
def input_plate_path(request, tmp_path):
    """Create test input plates for supported versions."""
    version = request.param
    plate_path = tmp_path / f"input_{version}.zarr"
    _create_test_plate(plate_path, version=version)
    return plate_path, version


class TestPlateMetadataExtraction:
    """Test that get_reconstruction_output_metadata handles v0.4 and v0.5."""

    def test_metadata_extraction(self, input_plate_path, tmp_path):
        plate_path, version = input_plate_path
        pos_path = plate_path / "0" / "0" / "0"

        config = settings.ReconstructionSettings(
            input_channel_names=CHANNEL_NAMES,
            birefringence=settings.BirefringenceSettings(),
        )
        config_path = tmp_path / "config.yml"
        utils.model_to_yaml(config, config_path)

        metadata = get_reconstruction_output_metadata(pos_path, config_path)

        # Custom metadata should be preserved
        assert "Summary" in metadata["plate_metadata"]
        # OME-specific keys should be stripped
        assert "plate" not in metadata["plate_metadata"]
        assert "ome" not in metadata["plate_metadata"]

    def test_version_propagated(self, input_plate_path, tmp_path):
        plate_path, version = input_plate_path
        pos_path = plate_path / "0" / "0" / "0"

        config = settings.ReconstructionSettings(
            input_channel_names=CHANNEL_NAMES,
            birefringence=settings.BirefringenceSettings(),
        )
        config_path = tmp_path / "config.yml"
        utils.model_to_yaml(config, config_path)

        metadata = get_reconstruction_output_metadata(pos_path, config_path)
        assert metadata["version"] == version


class TestOutputVersionPreservation:
    """Test that output zarr format matches input version."""

    def test_create_empty_hcs_zarr_version(self, tmp_path):
        for version in _versions():
            out_path = tmp_path / f"output_{version}.zarr"
            create_empty_hcs_zarr(
                store_path=out_path,
                position_keys=[("A", "1", "0")],
                shape=(1, 4, 4, 5, 6),
                chunks=(1, 1, 1, 5, 6),
                scale=INPUT_SCALE,
                channel_names=["Retardance", "Orientation", "BF", "Pol"],
                dtype=np.float32,
                version=version,
            )
            assert _get_store_version(out_path) == version


class TestReconstructCLI:
    """Test the full reconstruct CLI with v0.4 and v0.5 inputs."""

    def test_reconstruct_preserves_version(self, input_plate_path, tmp_path):
        plate_path, version = input_plate_path
        pos_path = plate_path / "0" / "0" / "0"
        output_path = tmp_path / "output.zarr"

        config = settings.ReconstructionSettings(
            input_channel_names=CHANNEL_NAMES,
            birefringence=settings.BirefringenceSettings(),
            reconstruction_dimension=3,
        )
        config_path = tmp_path / "config.yml"
        utils.model_to_yaml(config, config_path)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "reconstruct",
                "-i",
                str(pos_path),
                "-c",
                str(config_path),
                "-o",
                str(output_path),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert output_path.exists()

        # Verify output version matches input
        assert _get_store_version(output_path) == version

        # Verify output is readable
        with open_ome_zarr(output_path) as dataset:
            pos = dataset["0/0/0"]
            assert pos["0"].shape[1] == 4  # Retardance, Orientation, BF, Pol
