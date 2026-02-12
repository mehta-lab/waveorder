import numpy as np
import pytest
import xarray as xr
from click.testing import CliRunner

from waveorder.api import birefringence, fluorescence, phase
from waveorder.api._utils import _position_list_from_shape_scale_offset
from waveorder.cli import settings
from waveorder.cli.compute_transfer_function import (
    _write_birefringence_tf,
    _write_fluorescence_tf,
    _write_phase_tf,
)
from waveorder.cli.main import cli
from waveorder.io import utils


def _make_czyx_data(zyx_shape=(3, 4, 5), n_channels=4):
    """Create a dummy CZYX xr.DataArray for TF computation."""
    data = np.zeros((n_channels,) + zyx_shape, dtype=np.float32)
    return xr.DataArray(data, dims=("c", "z", "y", "x"))


@pytest.mark.parametrize(
    "shape, scale, offset, expected",
    [
        (5, 1.0, 0.0, [2.0, 1.0, 0.0, -1.0, -2.0]),
        (4, 0.5, 1.0, [1.5, 1.0, 0.5, 0.0]),
    ],
)
def test_position_list_from_shape_scale_offset(shape, scale, offset, expected):
    result = _position_list_from_shape_scale_offset(shape, scale, offset)
    np.testing.assert_allclose(result, expected)


def test_compute_transfer(tmp_path, example_plate):
    recon_settings = settings.ReconstructionSettings(
        input_channel_names=[f"State{i}" for i in range(4)],
        reconstruction_dimension=3,
        birefringence=settings.BirefringenceSettings(),
        phase=settings.PhaseSettings(),
    )
    config_path = tmp_path / "test.yml"
    utils.model_to_yaml(recon_settings, config_path)

    output_path = tmp_path / "output.zarr"

    plate_path, _ = example_plate
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compute-tf",
            "-i",
            str(plate_path / "A" / "1" / "0"),
            "-c",
            str(config_path),
            "-o",
            str(output_path),
        ],
    )
    assert result.exit_code == 0


def test_compute_transfer_blank_config():
    runner = CliRunner()
    for option in ("-c ", "--config-path "):
        cmd = "compute-tf " + option
        result = runner.invoke(cli, cmd)
        assert result.exit_code == 2
        assert "Error" in result.output


def test_compute_transfer_blank_output():
    runner = CliRunner()
    for option in ("-o ", "--output-path "):
        cmd = "compute-tf " + option
        result = runner.invoke(cli, cmd)
        assert result.exit_code == 2
        assert "Error" in result.output


def test_compute_transfer_output_file(tmp_path, example_plate):
    recon_settings = settings.ReconstructionSettings(
        input_channel_names=["BF"],
        reconstruction_dimension=3,
        phase=settings.PhaseSettings(),
    )
    config_path = tmp_path / "test.yml"
    utils.model_to_yaml(recon_settings, config_path)

    plate_path, _ = example_plate
    runner = CliRunner()
    for option in ("-o", "--output-dirpath"):
        for output_folder in ["test1.zarr", "test2/test.zarr"]:
            output_path = tmp_path.joinpath(output_folder)
            result = runner.invoke(
                cli,
                [
                    "compute-tf",
                    "-i",
                    str(plate_path / "A" / "1" / "0"),
                    "-c",
                    str(config_path),
                    str(option),
                    str(output_path),
                ],
            )
            assert result.exit_code == 0
            assert str(output_path) in result.output
            assert output_path.exists()


def test_stokes_matrix_write(birefringence_phase_recon_settings_function):
    recon_settings, dataset = birefringence_phase_recon_settings_function
    czyx_data = _make_czyx_data()
    tf_ds = birefringence.compute_transfer_function(
        czyx_data,
        recon_settings.birefringence,
        recon_settings.input_channel_names,
    )
    _write_birefringence_tf(dataset, tf_ds)
    assert dataset["intensity_to_stokes_matrix"]


def test_absorption_and_phase_write(
    birefringence_phase_recon_settings_function,
):
    recon_settings, dataset = birefringence_phase_recon_settings_function
    czyx_data = _make_czyx_data()
    tf_ds = phase.compute_transfer_function(czyx_data, 3, recon_settings.phase)
    _write_phase_tf(dataset, tf_ds, (3, 4, 5), 3)
    assert dataset["real_potential_transfer_function"]
    assert dataset["imaginary_potential_transfer_function"]
    assert dataset["imaginary_potential_transfer_function"].shape == (
        1,
        1,
        3,
        4,
        5,
    )
    assert "absorption_transfer_function" not in dataset
    assert "phase_transfer_function" not in dataset


def test_phase_3dim_write(birefringence_phase_recon_settings_function):
    recon_settings, dataset = birefringence_phase_recon_settings_function
    recon_settings.reconstruction_dimension = 2
    czyx_data = _make_czyx_data()
    tf_ds = phase.compute_transfer_function(czyx_data, 2, recon_settings.phase)
    _write_phase_tf(dataset, tf_ds, (3, 4, 5), 2)
    assert dataset["singular_system_U"]
    assert dataset["singular_system_U"].shape == (1, 2, 2, 4, 5)
    assert dataset["singular_system_S"]
    assert dataset["singular_system_Vh"]
    assert "real_potential_transfer_function" not in dataset
    assert "imaginary_potential_transfer_function" not in dataset


def test_fluorescence_write(fluorescence_recon_settings_function):
    recon_settings, dataset = fluorescence_recon_settings_function
    czyx_data = _make_czyx_data(n_channels=1)
    tf_ds = fluorescence.compute_transfer_function(
        czyx_data, 3, recon_settings.fluorescence
    )
    _write_fluorescence_tf(dataset, tf_ds, (3, 4, 5), 3)
    assert dataset["optical_transfer_function"]
    assert dataset["optical_transfer_function"].shape == (1, 1, 3, 4, 5)
    assert "real_potential_transfer_function" not in dataset
    assert "imaginary_potential_transfer_function" not in dataset


def test_fluorescence_2d_write(fluorescence_recon_settings_function):
    """Test 2D fluorescence transfer function generation"""
    recon_settings, dataset = fluorescence_recon_settings_function
    recon_settings.reconstruction_dimension = 2
    czyx_data = _make_czyx_data(n_channels=1)
    tf_ds = fluorescence.compute_transfer_function(
        czyx_data, 2, recon_settings.fluorescence
    )
    _write_fluorescence_tf(dataset, tf_ds, (3, 4, 5), 2)
    # Should generate singular system components, not optical transfer function
    assert dataset["singular_system_U"]
    assert dataset["singular_system_S"]
    assert dataset["singular_system_Vh"]
    assert dataset["singular_system_U"].shape == (1, 1, 1, 4, 5)
    assert dataset["singular_system_S"].shape == (1, 1, 1, 4, 5)
    assert dataset["singular_system_Vh"].shape == (1, 1, 3, 4, 5)
    assert "optical_transfer_function" not in dataset
