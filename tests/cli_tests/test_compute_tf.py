import numpy as np
import pytest
from typer.testing import CliRunner

from waveorder.api import birefringence, fluorescence, phase
from waveorder.cli import settings
from waveorder.cli.apply_inverse_transfer_function import _load_transfer_function
from waveorder.cli.compute_transfer_function import (
    _write_birefringence_tf,
    _write_fluorescence_tf,
    _write_phase_tf,
)
from waveorder.cli.main import app
from waveorder.cli.utils import cast_float16
from waveorder.io import utils

ZYX_SHAPE = (3, 4, 5)


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
        app,
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
        result = runner.invoke(app, cmd)
        assert result.exit_code == 2
        assert "Error" in result.output


def test_compute_transfer_blank_output():
    runner = CliRunner()
    for option in ("-o ", "--output-path "):
        cmd = "compute-tf " + option
        result = runner.invoke(app, cmd)
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
                app,
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


def test_stokes_matrix_write(birefringence_phase_recon_settings_function, make_czyx):
    recon_settings, dataset = birefringence_phase_recon_settings_function
    czyx_data = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=4)
    tf_ds = birefringence.compute_transfer_function(
        czyx_data,
        recon_settings.birefringence,
        recon_settings.input_channel_names,
    )
    _write_birefringence_tf(dataset, tf_ds)
    assert dataset["intensity_to_stokes_matrix"]


def test_absorption_and_phase_write(birefringence_phase_recon_settings_function, make_czyx):
    recon_settings, dataset = birefringence_phase_recon_settings_function
    czyx_data = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=4)
    tf_ds = phase.compute_transfer_function(czyx_data, 3, recon_settings.phase)
    _write_phase_tf(dataset, tf_ds, 3)
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


def test_phase_3dim_write(birefringence_phase_recon_settings_function, make_czyx):
    recon_settings, dataset = birefringence_phase_recon_settings_function
    recon_settings.reconstruction_dimension = 2
    czyx_data = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=4)
    tf_ds = phase.compute_transfer_function(czyx_data, 2, recon_settings.phase)
    _write_phase_tf(dataset, tf_ds, 2)
    assert dataset["singular_system_U"]
    assert dataset["singular_system_U"].shape == (1, 2, 2, 4, 5)
    assert dataset["singular_system_S"]
    assert dataset["singular_system_Vh"]
    assert "real_potential_transfer_function" not in dataset
    assert "imaginary_potential_transfer_function" not in dataset


def test_fluorescence_write(fluorescence_recon_settings_function, make_czyx):
    recon_settings, dataset = fluorescence_recon_settings_function
    czyx_data = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = fluorescence.compute_transfer_function(czyx_data, 3, recon_settings.fluorescence)
    _write_fluorescence_tf(dataset, tf_ds, 3)
    assert dataset["optical_transfer_function"]
    assert dataset["optical_transfer_function"].shape == (1, 1, 3, 4, 5)
    assert "real_potential_transfer_function" not in dataset
    assert "imaginary_potential_transfer_function" not in dataset


def test_fluorescence_2d_write(fluorescence_recon_settings_function, make_czyx):
    """Test 2D fluorescence transfer function generation"""
    recon_settings, dataset = fluorescence_recon_settings_function
    recon_settings.reconstruction_dimension = 2
    czyx_data = make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1)
    tf_ds = fluorescence.compute_transfer_function(czyx_data, 2, recon_settings.fluorescence)
    _write_fluorescence_tf(dataset, tf_ds, 2)
    # Should generate singular system components, not optical transfer function
    assert dataset["singular_system_U"]
    assert dataset["singular_system_S"]
    assert dataset["singular_system_Vh"]
    assert dataset["singular_system_U"].shape == (1, 1, 1, 4, 5)
    assert dataset["singular_system_S"].shape == (1, 1, 1, 4, 5)
    assert dataset["singular_system_Vh"].shape == (1, 1, 3, 4, 5)
    assert "optical_transfer_function" not in dataset


@pytest.mark.parametrize("recon_dim", [2, 3])
def test_float16_transfer_function_loads_back_as_written(
    birefringence_phase_recon_settings_function, make_czyx, recon_dim
):
    """A float16 store loads back at its original dtype and shape, within float16 precision.

    2D covers a real array (S) beside complex ones (U, Vh); 3D covers complex only.
    """
    recon_settings, dataset = birefringence_phase_recon_settings_function
    tf_ds = phase.compute_transfer_function(
        make_czyx(zyx_shape=ZYX_SHAPE, n_channels=1), recon_dim, recon_settings.phase
    )
    _write_phase_tf(dataset, tf_ds, recon_dim, float16=True)

    loaded = _load_transfer_function(dataset, False, True, False, recon_dim)

    assert loaded.keys() == set(tf_ds.data_vars)
    for name, tensor in loaded.items():
        assert dataset[name].dtype == np.float16
        expected = tf_ds[name].values
        assert tensor.numpy().dtype == expected.dtype
        np.testing.assert_allclose(tensor.numpy(), expected, rtol=1e-3, atol=1e-7)


def test_cast_float16_rejects_values_beyond_its_range():
    np.testing.assert_array_equal(cast_float16(np.array([-65504.0, 1.5]), "x"), [-65504.0, 1.5])
    with pytest.raises(ValueError, match="65504"):
        cast_float16(np.array([1.0, np.nan, 7e4]), "x")
