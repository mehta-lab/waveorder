"""End-to-end tests for Micro-Manager OME-TIFF input handling.

The parsing callback in ``waveorder.cli.parsing._validate_and_process_paths``
detects MM ome-tif folders and converts them to a sibling OME-Zarr plate via
``iohub.convert.TIFFConverter``. These tests exercise that path against the
real MM OME-TIFF fixture so we catch any breakage from iohub / tifffile /
zarr version drift.
"""

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner
from iohub.ngff import open_ome_zarr

from waveorder.cli import settings
from waveorder.cli.main import cli
from waveorder.cli.parsing import _validate_and_process_paths
from waveorder.io import utils


def _copy_to_tmp(src: Path, tmp_path: Path) -> Path:
    """Copy fixture dir into ``tmp_path`` so the sibling ``_converted.zarr``
    output doesn't pollute the cached fixture directory."""
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    return dst


def test_parsing_callback_converts_ometiff(mm_ome_tiff_dir, tmp_path):
    """The CLI callback converts an MM ome-tif folder to a sibling .zarr and
    returns the expanded list of position paths."""
    src = _copy_to_tmp(mm_ome_tiff_dir, tmp_path)

    positions = _validate_and_process_paths(None, None, (str(src),))

    expected_zarr = src.parent / (src.name + "_converted.zarr")
    assert expected_zarr.exists()
    assert len(positions) >= 1
    for pos in positions:
        assert pos.is_dir()
        assert expected_zarr in pos.parents

    with open_ome_zarr(expected_zarr, mode="r") as ds:
        assert sorted(ds.channel_names) == ["Cy5", "DAPI", "FITC"]


def test_parsing_callback_idempotent(mm_ome_tiff_dir, tmp_path):
    """A second call against the same input reuses the existing conversion."""
    src = _copy_to_tmp(mm_ome_tiff_dir, tmp_path)
    first = _validate_and_process_paths(None, None, (str(src),))
    zarr_mtime = (src.parent / (src.name + "_converted.zarr")).stat().st_mtime
    second = _validate_and_process_paths(None, None, (str(src),))

    assert first == second
    assert (src.parent / (src.name + "_converted.zarr")).stat().st_mtime == zarr_mtime


def test_parsing_callback_passes_zarr_through(example_plate):
    """The callback must not touch existing zarr inputs."""
    plate_path, _ = example_plate
    positions = _validate_and_process_paths(None, None, (str(plate_path),))
    assert len(positions) == 3
    assert all(p.parent.parent.parent == plate_path for p in positions)


@pytest.mark.parametrize("recon_dim", [2, 3])
def test_compute_tf_cli_on_ometiff(mm_ome_tiff_dir, tmp_path, recon_dim):
    """``waveorder compute-tf`` runs end-to-end on an MM ome-tif folder."""
    src = _copy_to_tmp(mm_ome_tiff_dir, tmp_path)

    config_path = tmp_path / "fluorescence.yml"
    output_path = tmp_path / "transfer_function.zarr"
    recon_settings = settings.ReconstructionSettings(
        input_channel_names=["DAPI"],
        reconstruction_dimension=recon_dim,
        fluorescence=settings.FluorescenceSettings(),
    )
    utils.model_to_yaml(recon_settings, config_path)

    result = CliRunner().invoke(
        cli,
        ["compute-tf", "-i", str(src), "-c", str(config_path), "-o", str(output_path)],
    )
    assert result.exit_code == 0, result.output
    assert (src.parent / (src.name + "_converted.zarr")).exists()
    assert output_path.exists()
