from pathlib import Path

import click
import numpy as np
from iohub import open_ome_zarr
from waveorder.models import (
    inplane_oriented_thick_pol3d,
    isotropic_fluorescent_thick_3d,
    isotropic_thin_3d,
    phase_thick_3d,
)

from recOrder.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
)
from recOrder.cli.printing import echo_headline, echo_settings
from recOrder.cli.settings import ReconstructionSettings
from recOrder.io import utils


def generate_and_save_birefringence_transfer_function(settings, dataset):
    """Generates and saves the birefringence transfer function to the dataset, based on the settings.

    Parameters
    ----------
    settings: ReconstructionSettings
    dataset: NGFF Node
        The dataset that will be updated.
    """
    echo_headline("Generating birefringence transfer function with settings:")
    echo_settings(settings.birefringence.transfer_function)

    # Calculate transfer functions
    intensity_to_stokes_matrix = (
        inplane_oriented_thick_pol3d.calculate_transfer_function(
            scheme=str(len(settings.input_channel_names)) + "-State",
            **settings.birefringence.transfer_function.dict(),
        )
    )
    # Save
    dataset[
        "intensity_to_stokes_matrix"
    ] = intensity_to_stokes_matrix.cpu().numpy()[None, None, None, ...]


def generate_and_save_phase_transfer_function(settings, dataset, zyx_shape):
    """Generates and saves the phase transfer function to the dataset, based on the settings.

    Parameters
    ----------
    settings: ReconstructionSettings
    dataset: NGFF Node
        The dataset that will be updated.
    zyx_shape : tuple
        A tuple of integers specifying the input data's shape in (Z, Y, X) order
    """
    echo_headline("Generating phase transfer function with settings:")
    echo_settings(settings.phase.transfer_function)

    if settings.reconstruction_dimension == 2:
        # Convert zyx_shape and z_pixel_size into yx_shape and z_position_list
        settings_dict = settings.phase.transfer_function.dict()
        settings_dict["yx_shape"] = [zyx_shape[1], zyx_shape[2]]
        settings_dict["z_position_list"] = list(
            -(np.arange(zyx_shape[0]) - zyx_shape[0] // 2)
            * settings_dict["z_pixel_size"]
        )

        # Remove unused parameters
        settings_dict.pop("z_pixel_size")
        settings_dict.pop("z_padding")

        # Calculate transfer functions
        (
            absorption_transfer_function,
            phase_transfer_function,
        ) = isotropic_thin_3d.calculate_transfer_function(
            **settings_dict,
        )

        # Save
        dataset[
            "absorption_transfer_function"
        ] = absorption_transfer_function.cpu().numpy()[None, None, ...]
        dataset[
            "phase_transfer_function"
        ] = phase_transfer_function.cpu().numpy()[None, None, ...]

    elif settings.reconstruction_dimension == 3:
        # Calculate transfer functions
        (
            real_potential_transfer_function,
            imaginary_potential_transfer_function,
        ) = phase_thick_3d.calculate_transfer_function(
            zyx_shape=zyx_shape,
            **settings.phase.transfer_function.dict(),
        )
        # Save
        dataset[
            "real_potential_transfer_function"
        ] = real_potential_transfer_function.cpu().numpy()[None, None, ...]
        dataset[
            "imaginary_potential_transfer_function"
        ] = imaginary_potential_transfer_function.cpu().numpy()[
            None, None, ...
        ]


def generate_and_save_fluorescence_transfer_function(
    settings, dataset, zyx_shape
):
    """Generates and saves the fluorescence transfer function to the dataset, based on the settings.

    Parameters
    ----------
    settings: ReconstructionSettings
    dataset: NGFF Node
        The dataset that will be updated.
    zyx_shape : tuple
        A tuple of integers specifying the input data's shape in (Z, Y, X) order
    """
    echo_headline("Generating fluorescence transfer function with settings:")
    echo_settings(settings.fluorescence.transfer_function)

    if settings.reconstruction_dimension == 2:
        raise NotImplementedError
    elif settings.reconstruction_dimension == 3:
        # Calculate transfer functions
        optical_transfer_function = (
            isotropic_fluorescent_thick_3d.calculate_transfer_function(
                zyx_shape=zyx_shape,
                **settings.fluorescence.transfer_function.dict(),
            )
        )
        # Save
        dataset[
            "optical_transfer_function"
        ] = optical_transfer_function.cpu().numpy()[None, None, ...]


def compute_transfer_function_cli(
    input_position_dirpath: Path, config_filepath: Path, output_dirpath: Path
) -> None:
    """CLI command to compute the transfer function given a configuration file path
    and a desired output path.
    """

    # Load config file
    settings = utils.yaml_to_model(config_filepath, ReconstructionSettings)

    echo_headline(
        f"Generating transfer functions and storing in {output_dirpath}\n"
    )

    # Read shape from input dataset
    input_dataset = open_ome_zarr(
        input_position_dirpath, layout="fov", mode="r"
    )
    zyx_shape = input_dataset.data.shape[
        2:
    ]  # only loads a single position "0"

    # Check input channel names
    if not set(settings.input_channel_names).issubset(
        input_dataset.channel_names
    ):
        raise ValueError(
            f"Each of the input_channel_names = {settings.input_channel_names} in {config_filepath} must appear in the dataset {input_position_dirpaths[0]} which currently contains channel_names = {input_dataset.channel_names}."
        )

    # Prepare output dataset
    output_dataset = open_ome_zarr(
        output_dirpath, layout="fov", mode="w", channel_names=["None"]
    )

    # Pass settings to appropriate calculate_transfer_function and save
    if settings.birefringence is not None:
        generate_and_save_birefringence_transfer_function(
            settings, output_dataset
        )
    if settings.phase is not None:
        generate_and_save_phase_transfer_function(
            settings, output_dataset, zyx_shape
        )
    if settings.fluorescence is not None:
        generate_and_save_fluorescence_transfer_function(
            settings, output_dataset, zyx_shape
        )

    # Write settings to metadata
    output_dataset.zattrs["settings"] = settings.dict()

    echo_headline(f"Closing {output_dirpath}\n")
    output_dataset.close()

    echo_headline(
        f"Recreate this transfer function with:\n$ recorder compute-tf {input_position_dirpaths} -c {config_filepath} -o {output_dirpath}"
    )


@click.command()
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
def compute_tf(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
) -> None:
    """
    Compute a transfer function using a dataset and configuration file.

    Calculates the transfer function based on the shape of the first position
    in the list `input-position-dirpaths`.

    See /examples for example configuration files.

    >> recorder compute-tf -i ./input.zarr/0/0/0 -c ./examples/birefringence.yml -o ./transfer_function.zarr
    """
    compute_transfer_function_cli(
        input_position_dirpaths[0], config_filepath, output_dirpath
    )
