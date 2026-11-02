"""migrate-config CLI, and 3.x vs migrated-4.0 end-to-end equivalence."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml
from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta
from typer.testing import CliRunner

from tests.config.repo_configs import MIGRATABLE_CONFIG_PATHS
from waveorder.cli.main import app
from waveorder.config import loader
from waveorder.config.schema import V4Config

FLUOR_3X_CONFIG = {
    "input_channel_names": ["GFP"],
    "reconstruction_dimension": 3,
    "fluorescence": {
        "transfer_function": {
            "yx_pixel_size": 0.2,
            "z_pixel_size": 0.4,
            "wavelength_emission": 0.532,
            "index_of_refraction_media": 1.3,
            "numerical_aperture_detection": 1.2,
        },
        "apply_inverse": {
            "reconstruction_algorithm": "Tikhonov",
            "regularization_strength": 1e-3,
        },
    },
}


def _write_input_zarr(input_path: Path, channel_names: list[str], tzyx_shape: tuple[int, int, int, int]) -> Path:
    """Write a tiny random single-position dataset with the given channels.

    Parameters
    ----------
    input_path : Path
        Destination zarr path.
    channel_names : list[str]
        Channel names; the channel axis has ``len(channel_names)`` planes.
    tzyx_shape : tuple[int, int, int, int]
        ``(T, Z, Y, X)`` shape of each channel.

    Returns
    -------
    Path
        ``input_path``, for chaining.
    """
    t, z, y, x = tzyx_shape
    dataset = open_ome_zarr(input_path, layout="hcs", mode="w", channel_names=channel_names)
    position = dataset.create_position("0", "0", "0")
    rng = np.random.default_rng(0)
    position.create_image(
        "0",
        rng.random((t, len(channel_names), z, y, x)).astype(np.float32),
        transform=[TransformationMeta(type="scale", scale=[1, 1, 0.4, 0.2, 0.2])],
    )
    dataset.close()
    return input_path


@pytest.fixture
def input_zarr(tmp_path):
    """A tiny single-position fluorescence dataset."""
    return _write_input_zarr(tmp_path / "input.zarr", ["GFP"], (2, 8, 16, 16))


def test_migrate_config_cli_writes_v4(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(FLUOR_3X_CONFIG))
    result = CliRunner().invoke(app, ["migrate-config", str(config_path)])
    assert result.exit_code == 0, result.output

    output_path = tmp_path / "config_v4.yml"
    assert output_path.exists()
    migrated = yaml.safe_load(output_path.read_text())
    assert migrated["version"] == 4
    V4Config(**migrated)  # validates


def test_migrate_config_cli_rejects_v4_input(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump({"version": 4}))
    result = CliRunner().invoke(app, ["migrate-config", str(config_path)])
    assert result.exit_code == 0
    assert "already a 4.0 configuration" in result.output


def _reconstruct(input_zarr: Path, config_path: Path, output_path: Path) -> np.ndarray:
    result = CliRunner().invoke(
        app,
        ["reconstruct", "-i", str(input_zarr / "0" / "0" / "0"), "-c", str(config_path), "-o", str(output_path)],
    )
    assert result.exit_code == 0, result.output
    with open_ome_zarr(output_path / "0" / "0" / "0", mode="r") as ds:
        return ds.data[:]


def test_3x_and_migrated_v4_reconstructions_match(tmp_path, input_zarr):
    """A migrated configuration produces the same output array, bit for bit."""
    config_3x = tmp_path / "config.yml"
    config_3x.write_text(yaml.safe_dump(FLUOR_3X_CONFIG))

    result = CliRunner().invoke(app, ["migrate-config", str(config_3x), "-o", str(tmp_path / "config_v4.yml")])
    assert result.exit_code == 0, result.output

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recon_3x = _reconstruct(input_zarr, config_3x, tmp_path / "out_3x.zarr")
    recon_v4 = _reconstruct(input_zarr, tmp_path / "config_v4.yml", tmp_path / "out_v4.zarr")

    np.testing.assert_array_equal(recon_3x, recon_v4)


@pytest.mark.parametrize("config_path", MIGRATABLE_CONFIG_PATHS, ids=lambda p: p.name)
def test_every_repo_config_migrates_and_matches(tmp_path, monkeypatch, config_path):
    """Design section 2: migrate every 3.x configuration in the repository
    and the published examples, run it, and assert the output matches the
    3.x output array by array."""
    monkeypatch.chdir(tmp_path)  # optimizer log_dir entries are relative paths
    raw = yaml.safe_load(config_path.read_text())
    input_path = _write_input_zarr(tmp_path / "input.zarr", list(raw["input_channel_names"]), (1, 8, 16, 16))

    config_3x = tmp_path / "config.yml"
    config_3x.write_text(config_path.read_text())
    result = CliRunner().invoke(app, ["migrate-config", str(config_3x), "-o", str(tmp_path / "config_v4.yml")])
    assert result.exit_code == 0, result.output

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recon_3x = _reconstruct(input_path, config_3x, tmp_path / "out_3x.zarr")
    recon_v4 = _reconstruct(input_path, tmp_path / "config_v4.yml", tmp_path / "out_v4.zarr")

    np.testing.assert_array_equal(recon_3x, recon_v4)


def test_3x_config_warns_and_prints_v4(tmp_path, input_zarr):
    config_3x = tmp_path / "deprecation_check.yml"
    config_3x.write_text(yaml.safe_dump(FLUOR_3X_CONFIG))

    loader._already_warned_paths.clear()
    with pytest.warns(DeprecationWarning, match="migrate-config"):
        loader.load_reconstruction_settings(config_3x)
    # Second load of the same path stays quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        loader.load_reconstruction_settings(config_3x)
