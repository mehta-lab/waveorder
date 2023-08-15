import os

from click.testing import CliRunner

from recOrder.cli import settings
from recOrder.cli.compute_transfer_function import (
    generate_and_save_birefringence_transfer_function,
    generate_and_save_fluorescence_transfer_function,
    generate_and_save_phase_transfer_function,
)
from recOrder.cli.main import cli
from recOrder.io import utils


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
    settings, dataset = birefringence_phase_recon_settings_function
    generate_and_save_birefringence_transfer_function(settings, dataset)
    assert dataset["intensity_to_stokes_matrix"]


def test_absorption_and_phase_write(
    birefringence_phase_recon_settings_function,
):
    settings, dataset = birefringence_phase_recon_settings_function
    generate_and_save_phase_transfer_function(settings, dataset, (3, 4, 5))
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
    settings, dataset = birefringence_phase_recon_settings_function
    settings.reconstruction_dimension = 2
    generate_and_save_phase_transfer_function(settings, dataset, (3, 4, 5))
    assert dataset["absorption_transfer_function"]
    assert dataset["phase_transfer_function"]
    assert dataset["phase_transfer_function"].shape == (1, 1, 3, 4, 5)
    assert "real_potential_transfer_function" not in dataset
    assert "imaginary_potential_transfer_function" not in dataset


def test_fluorescence_write(fluorescence_recon_settings_function):
    settings, dataset = fluorescence_recon_settings_function
    generate_and_save_fluorescence_transfer_function(
        settings, dataset, (3, 4, 5)
    )
    assert dataset["optical_transfer_function"]
    assert dataset["optical_transfer_function"].shape == (1, 1, 3, 4, 5)
    assert "real_potential_transfer_function" not in dataset
    assert "imaginary_potential_transfer_function" not in dataset
