from pathlib import Path

import click
import numpy as np
from iohub.ngff import Position, open_ome_zarr

from waveorder import focus
from waveorder.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
)
from waveorder.cli.printing import echo_headline, echo_settings
from waveorder.cli.settings import ReconstructionSettings
from waveorder.io import utils
from waveorder.models import (
    inplane_oriented_thick_pol3d,
    isotropic_fluorescent_thick_3d,
    isotropic_thin_3d,
    phase_thick_3d,
)


def _position_list_from_shape_scale_offset(
    shape: int, scale: float, offset: float
) -> list:
    """
    Generates a list of positions based on the given array shape, pixel size (scale), and offset.

    Examples
    --------
    >>> _position_list_from_shape_scale_offset(5, 1.0, 0.0)
    [2.0, 1.0, 0.0, -1.0, -2.0]
    >>> _position_list_from_shape_scale_offset(4, 0.5, 1.0)
    [1.5, 1.0, 0.5, 0.0]
    """
    return list((-np.arange(shape) + (shape // 2) + offset) * scale)


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
    dataset["intensity_to_stokes_matrix"] = (
        intensity_to_stokes_matrix.cpu().numpy()[None, None, None, ...]
    )


def generate_and_save_phase_transfer_function(
    settings: ReconstructionSettings, dataset: Position, zyx_shape: tuple
):
    """Generates and saves the phase transfer function to the dataset, based on the settings.

    Parameters
    ----------
    settings: ReconstructionSettings
    dataset: Position
        The dataset that will be updated.
    zyx_shape : tuple
        A tuple of integers specifying the input data's shape in (Z, Y, X) order
    """
    echo_headline("Generating phase transfer function with settings:")
    echo_settings(settings.phase.transfer_function)

    settings_dict = settings.phase.transfer_function.dict()
    if settings.reconstruction_dimension == 2:
        # Convert zyx_shape and z_pixel_size into yx_shape and z_position_list
        settings_dict["yx_shape"] = [zyx_shape[1], zyx_shape[2]]
        settings_dict["z_position_list"] = (
            _position_list_from_shape_scale_offset(
                shape=zyx_shape[0],
                scale=settings_dict["z_pixel_size"],
                offset=settings_dict["z_focus_offset"],
            )
        )

        # Remove unused parameters
        settings_dict.pop("z_pixel_size")
        settings_dict.pop("z_padding")
        settings_dict.pop("z_focus_offset")

        # Calculate transfer functions
        (
            absorption_transfer_function,
            phase_transfer_function,
        ) = isotropic_thin_3d.calculate_transfer_function(
            **settings_dict,
        )

        # Calculate singular system
        U, S, Vh = isotropic_thin_3d.calculate_singular_system(
            absorption_transfer_function,
            phase_transfer_function,
        )

        # Save
        dataset.create_image(
            "singular_system_U",
            U.cpu().numpy()[None],
        )
        dataset.create_image(
            "singular_system_S",
            S.cpu().numpy()[None, None],
        )
        dataset.create_image(
            "singular_system_Vh",
            Vh.cpu().numpy()[None],
        )

    elif settings.reconstruction_dimension == 3:
        settings_dict.pop("z_focus_offset")  # not used in 3D

        # Calculate transfer functions
        (
            real_potential_transfer_function,
            imaginary_potential_transfer_function,
        ) = phase_thick_3d.calculate_transfer_function(
            zyx_shape=zyx_shape,
            **settings_dict,
        )
        # Save
        dataset.create_image(
            "real_potential_transfer_function",
            real_potential_transfer_function.cpu().numpy()[None, None, ...],
            chunks=(1, 1, 1, zyx_shape[1], zyx_shape[2]),
        )
        dataset.create_image(
            "imaginary_potential_transfer_function",
            imaginary_potential_transfer_function.cpu().numpy()[
                None, None, ...
            ],
            chunks=(1, 1, 1, zyx_shape[1], zyx_shape[2]),
        )


def generate_and_save_fluorescence_transfer_function(
    settings: ReconstructionSettings, dataset: Position, zyx_shape: tuple
):
    """Generates and saves the fluorescence transfer function to the dataset, based on the settings.

    Parameters
    ----------
    settings: ReconstructionSettings
    dataset: Position
        The dataset that will be updated.
    zyx_shape : tuple
        A tuple of integers specifying the input data's shape in (Z, Y, X) order
    """
    echo_headline("Generating fluorescence transfer function with settings:")
    echo_settings(settings.fluorescence.transfer_function)
    # Remove unused parameters
    settings_dict = settings.fluorescence.transfer_function.dict()
    settings_dict.pop("z_focus_offset")

    if settings.reconstruction_dimension == 2:
        raise NotImplementedError
    elif settings.reconstruction_dimension == 3:
        # Calculate transfer functions
        optical_transfer_function = (
            isotropic_fluorescent_thick_3d.calculate_transfer_function(
                zyx_shape=zyx_shape,
                **settings_dict,
            )
        )
        # Save
        dataset.create_image(
            "optical_transfer_function",
            optical_transfer_function.cpu().numpy()[None, None, ...],
            chunks=(1, 1, 1, zyx_shape[1], zyx_shape[2]),
        )


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

    # Find in-focus slices for 2D reconstruction in "auto" mode
    if (
        settings.phase is not None
        and settings.reconstruction_dimension == 2
        and settings.phase.transfer_function.z_focus_offset == "auto"
    ):

        c_idx = input_dataset.get_channel_index(
            settings.input_channel_names[0]
        )
        zyx_array = input_dataset["0"][0, c_idx]

        in_focus_index = focus.focus_from_transverse_band(
            zyx_array,
            NA_det=settings.phase.transfer_function.numerical_aperture_detection,
            lambda_ill=settings.phase.transfer_function.wavelength_illumination,
            pixel_size=settings.phase.transfer_function.yx_pixel_size,
            mode="min",
            polynomial_fit_order=4,
        )

        z_focus_offset = in_focus_index - (zyx_shape[0] // 2)
        settings.phase.transfer_function.z_focus_offset = z_focus_offset
        print("Found z_focus_offset:", z_focus_offset)

    # Prepare output dataset
    num_channels = (
        2 if settings.reconstruction_dimension == 2 else 1
    )  # space for SVD
    output_dataset = open_ome_zarr(
        output_dirpath,
        layout="fov",
        mode="w",
        channel_names=num_channels * ["None"],
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
        f"Recreate this transfer function with:\n$ waveorder compute-tf {input_position_dirpaths} -c {config_filepath} -o {output_dirpath}"
    )


@click.command("compute-tf")
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
def _compute_transfer_function_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
) -> None:
    """
    Compute a transfer function using a dataset and configuration file.

    Calculates the transfer function based on the shape of the first position
    in the list `input-position-dirpaths`.

    See /examples for example configuration files.

    >> waveorder compute-tf -i ./input.zarr/0/0/0 -c ./examples/birefringence.yml -o ./transfer_function.zarr
    """
    compute_transfer_function_cli(
        input_position_dirpaths[0], config_filepath, output_dirpath
    )
