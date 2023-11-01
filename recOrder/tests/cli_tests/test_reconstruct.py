import os

import numpy as np
from click.testing import CliRunner
from iohub.ngff import open_ome_zarr
from iohub.ngff_meta import TransformationMeta

from recOrder.cli import settings
from recOrder.cli.main import cli
from recOrder.io import utils
from recOrder.cli.apply_inverse_transfer_function import (
    apply_inverse_transfer_function_cli,
)
from unittest.mock import patch
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory


input_scale = [1, 2, 3, 4, 5]
# Setup options
birefringence_settings = settings.BirefringenceSettings(
    transfer_function=settings.BirefringenceTransferFunctionSettings()
)

# birefringence_option, time_indices, phase_option, dimension_option, time_length_target
all_options = [
    (birefringence_settings, [0, 3, 4], None, 2, 5),
    (birefringence_settings, 0, settings.PhaseSettings(), 2, 5),
    (birefringence_settings, [0, 1], None, 3, 5),
    (birefringence_settings, "all", settings.PhaseSettings(), 3, 5),
]


@pytest.fixture(scope="session")
def tmp_input_path_zarr():
    tmp_path = TemporaryDirectory()
    yield Path(tmp_path.name) / "input.zarr", Path(tmp_path.name) / "test.yml"
    tmp_path.cleanup()


def test_reconstruct(tmp_input_path_zarr):
    input_path, tmp_config_yml = tmp_input_path_zarr
    # Generate input "dataset"
    channel_names = [f"State{x}" for x in range(4)]
    dataset = open_ome_zarr(
        input_path,
        layout="hcs",
        mode="w",
        channel_names=channel_names,
    )

    position = dataset.create_position("0", "0", "0")
    position.create_zeros(
        "0",
        (5, 4, 4, 5, 6),
        dtype=np.uint16,
        transform=[TransformationMeta(type="scale", scale=input_scale)],
    )

    for i, (
        birefringence_option,
        time_indices,
        phase_option,
        dimension_option,
        time_length_target,
    ) in enumerate(all_options):
        if (birefringence_option is None) and (phase_option is None):
            continue

        # Generate recon settings
        recon_settings = settings.ReconstructionSettings(
            input_channel_names=channel_names,
            time_indices=time_indices,
            reconstruction_dimension=dimension_option,
            birefringence=birefringence_option,
            phase=phase_option,
        )
        config_path = tmp_config_yml.with_name(f"{i}.yml")
        utils.model_to_yaml(recon_settings, config_path)

        # Run CLI
        runner = CliRunner()
        tf_path = input_path.with_name(f"tf_{i}.zarr")
        runner.invoke(
            cli,
            [
                "compute-tf",
                "-i",
                str(input_path / "0" / "0" / "0"),
                "-c",
                str(config_path),
                "-o",
                str(tf_path),
            ],
            catch_exceptions=False,
        )
        assert tf_path.exists()


def test_append_channel_reconstruction(tmp_input_path_zarr):
    input_path, tmp_config_yml = tmp_input_path_zarr
    output_path = input_path.with_name(f"output.zarr")

    # Generate input "dataset"
    channel_names = [f"State{x}" for x in range(4)] + ["GFP"]
    dataset = open_ome_zarr(
        input_path,
        layout="hcs",
        mode="w",
        channel_names=channel_names,
    )
    position = dataset.create_position("0", "0", "0")
    position.create_zeros(
        "0",
        (5, 5, 4, 5, 6),
        dtype=np.uint16,
        transform=[TransformationMeta(type="scale", scale=input_scale)],
    )

    # Generate recon settings
    biref_settings = settings.ReconstructionSettings(
        input_channel_names=[f"State{x}" for x in range(4)],
        time_indices="all",
        reconstruction_dimension=3,
        birefringence=settings.BirefringenceSettings(),
        phase=None,
        fluorescence=None,
    )
    fluor_settings = settings.ReconstructionSettings(
        input_channel_names=["GFP"],
        time_indices="all",
        reconstruction_dimension=3,
        birefringence=None,
        phase=None,
        fluorescence=settings.FluorescenceSettings(),
    )
    biref_config_path = tmp_config_yml.with_name(f"biref.yml")
    fluor_config_path = tmp_config_yml.with_name(f"fluor.yml")

    utils.model_to_yaml(biref_settings, biref_config_path)
    utils.model_to_yaml(fluor_settings, fluor_config_path)

    # Apply birefringence reconstruction
    runner = CliRunner()
    runner.invoke(
        cli,
        [
            "reconstruct",
            "-i",
            str(input_path / "0" / "0" / "0"),
            "-c",
            str(biref_config_path),
            "-o",
            str(output_path),
        ],
        catch_exceptions=False,
    )
    assert output_path.exists()
    with open_ome_zarr(output_path) as dataset:
        assert dataset["0/0/0"]["0"].shape[1] == 4

    # Append fluoresncence reconstruction
    runner.invoke(
        cli,
        [
            "reconstruct",
            "-i",
            str(input_path / "0" / "0" / "0"),
            "-c",
            str(fluor_config_path),
            "-o",
            str(output_path),
        ],
        catch_exceptions=False,
    )
    assert output_path.exists()
    with open_ome_zarr(output_path) as dataset:
        assert dataset["0/0/0"]["0"].shape[1] == 5
        assert dataset.channel_names[-1] == "GFP_Density3D"
        assert dataset.channel_names[-2] == "Pol"


def test_cli_apply_inv_tf_mock(tmp_input_path_zarr):
    tmp_input_zarr, tmp_config_yml = tmp_input_path_zarr
    tmp_config_yml = tmp_config_yml.with_name("0.yml").resolve()
    tf_path = tmp_input_zarr.with_name("tf_0.zarr").resolve()
    input_path = (tmp_input_zarr / "0" / "0" / "0").resolve()
    result_path = tmp_input_zarr.with_name("result.zarr").resolve()

    assert tmp_config_yml.exists()
    assert tf_path.exists()
    assert input_path.exists()
    assert not result_path.exists()

    runner = CliRunner()
    with patch(
        "recOrder.cli.apply_inverse_transfer_function.apply_inverse_transfer_function_cli"
    ) as mock:
        cmd = [
            "apply-inv-tf",
            "-i",
            str(input_path),
            "-t",
            str(tf_path),
            "-c",
            str(tmp_config_yml),
            "-o",
            str(result_path),
            "-j",
            str(1),
        ]
        result_inv = runner.invoke(
            cli,
            cmd,
            catch_exceptions=False,
        )
        mock.assert_called_with(
            [input_path],
            Path(tf_path),
            Path(tmp_config_yml),
            Path(result_path),
            1,
        )
        assert result_inv.exit_code == 0


def test_cli_apply_inv_tf_output(tmp_input_path_zarr, capsys):
    tmp_input_zarr, tmp_config_yml = tmp_input_path_zarr
    input_path = tmp_input_zarr / "0" / "0" / "0"

    for i, (
        birefringence_option,
        time_indices,
        phase_option,
        dimension_option,
        time_length_target,
    ) in enumerate(all_options):
        if (birefringence_option is None) and (phase_option is None):
            continue

        result_path = tmp_input_zarr.with_name(f"result{i}.zarr").resolve()

        tf_path = tmp_input_zarr.with_name(f"tf_{i}.zarr")
        tmp_config_yml = tmp_config_yml.with_name(f"{i}.yml")

        # # Check output
        apply_inverse_transfer_function_cli(
            [input_path], tf_path, tmp_config_yml, result_path, 1
        )

        result_dataset = open_ome_zarr(str(result_path / "0" / "0" / "0"))
        assert result_dataset["0"].shape[0] == time_length_target
        assert result_dataset["0"].shape[3:] == (5, 6)

        assert result_path.exists()
        captured = capsys.readouterr()
        assert "Reconstructing" in captured.out

        # Check scale transformations pass through
        assert input_scale == result_dataset.scale
