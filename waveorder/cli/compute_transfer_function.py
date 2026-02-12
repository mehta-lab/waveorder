from pathlib import Path

import click
import numpy as np
from iohub.ngff import Position, open_ome_zarr

from waveorder import focus
from waveorder.api import (
    birefringence,
    birefringence_and_phase,
    fluorescence,
    phase,
)
from waveorder.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
)
from waveorder.cli.printing import echo_headline, echo_settings
from waveorder.cli.settings import ReconstructionSettings
from waveorder.io import utils


def _write_birefringence_tf(dataset: Position, tf_ds):
    """Write birefringence TF arrays to zarr."""
    dataset["intensity_to_stokes_matrix"] = tf_ds[
        "intensity_to_stokes_matrix"
    ].values[None, None, None, ...]


def _write_phase_tf(dataset: Position, tf_ds, zyx_shape, recon_dim):
    """Write phase TF arrays to zarr."""
    if recon_dim == 2:
        dataset.create_image(
            "singular_system_U",
            tf_ds["singular_system_U"].values[None],
        )
        dataset.create_image(
            "singular_system_S",
            tf_ds["singular_system_S"].values[None, None],
        )
        dataset.create_image(
            "singular_system_Vh",
            tf_ds["singular_system_Vh"].values[None],
        )
    elif recon_dim == 3:
        chunks = (1, 1, 1, zyx_shape[1], zyx_shape[2])
        dataset.create_image(
            "real_potential_transfer_function",
            tf_ds["real_potential_transfer_function"].values[None, None, ...],
            chunks=chunks,
        )
        dataset.create_image(
            "imaginary_potential_transfer_function",
            tf_ds["imaginary_potential_transfer_function"].values[
                None, None, ...
            ],
            chunks=chunks,
        )


def _write_fluorescence_tf(dataset: Position, tf_ds, zyx_shape, recon_dim):
    """Write fluorescence TF arrays to zarr."""
    yx_shape = zyx_shape[1:]
    if recon_dim == 2:
        dataset.create_image(
            "singular_system_U",
            tf_ds["singular_system_U"].values[None, ...],
            chunks=(1, 1, 1, yx_shape[0], yx_shape[1]),
        )
        dataset.create_image(
            "singular_system_S",
            tf_ds["singular_system_S"].values[None, None, ...],
            chunks=(1, 1, 1, yx_shape[0], yx_shape[1]),
        )
        dataset.create_image(
            "singular_system_Vh",
            tf_ds["singular_system_Vh"].values[None, ...],
            chunks=(1, 1, zyx_shape[0], yx_shape[0], yx_shape[1]),
        )
    elif recon_dim == 3:
        dataset.create_image(
            "optical_transfer_function",
            tf_ds["optical_transfer_function"].values[None, None, ...],
            chunks=(1, 1, 1, zyx_shape[1], zyx_shape[2]),
        )


def _write_vector_birefringence_tf(dataset: Position, tf_ds, zyx_shape):
    """Write vector birefringence TF arrays to zarr."""
    chunks = (1, 1, 1, zyx_shape[1], zyx_shape[2])

    # Add dummy channels needed by iohub for additional images
    for i in range(3):
        dataset.append_channel(f"ch{i}")

    dataset.create_image(
        "vector_transfer_function",
        tf_ds["vector_transfer_function"].values,
        chunks=chunks,
    )
    dataset.create_image(
        "vector_singular_system_U",
        tf_ds["vector_singular_system_U"].values,
        chunks=chunks,
    )
    dataset.create_image(
        "vector_singular_system_S",
        tf_ds["vector_singular_system_S"].values[None],
        chunks=chunks,
    )
    dataset.create_image(
        "vector_singular_system_Vh",
        tf_ds["vector_singular_system_Vh"].values,
        chunks=chunks,
    )


def compute_transfer_function_cli(
    input_position_dirpath: Path,
    config_filepath: Path,
    output_dirpath: Path,
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
    input_version = input_dataset.version
    zyx_shape = input_dataset.data.shape[2:]

    # Check input channel names
    if not set(settings.input_channel_names).issubset(
        input_dataset.channel_names
    ):
        raise ValueError(
            f"Each of the input_channel_names = {settings.input_channel_names} in {config_filepath} must appear in the dataset {input_position_dirpath} which currently contains channel_names = {input_dataset.channel_names}."
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

    # Get input data as CZYX xarray for the API
    czyx_data = input_dataset.to_xarray().isel(t=0)

    recon_dim = settings.reconstruction_dimension

    # Prepare output dataset
    num_channels = 2 if recon_dim == 2 else 1
    output_dataset = open_ome_zarr(
        output_dirpath,
        layout="fov",
        mode="w",
        channel_names=num_channels * ["None"],
        version=input_version,
    )

    # Compute and save transfer functions
    if settings.birefringence is not None and settings.phase is not None:
        echo_headline(
            "Generating birefringence and phase transfer functions with settings:"
        )
        echo_settings(settings.birefringence.transfer_function)
        echo_settings(settings.phase.transfer_function)

        tf_ds = birefringence_and_phase.compute_transfer_function(
            czyx_data,
            settings.birefringence,
            settings.phase,
            settings.input_channel_names,
            recon_dim,
        )

        _write_birefringence_tf(output_dataset, tf_ds)

        # Write phase TFs (only for 3D; 2D phase uses vector singular system)
        if recon_dim == 3:
            _write_phase_tf(output_dataset, tf_ds, zyx_shape, recon_dim)

        echo_headline(
            f"Downsampling transfer function in X and Y by "
            f"{int(np.ceil(np.sqrt(np.array(zyx_shape).prod() / 1e7)))}x"
        )
        _write_vector_birefringence_tf(output_dataset, tf_ds, zyx_shape)

    else:
        if settings.birefringence is not None:
            echo_headline(
                "Generating birefringence transfer function with settings:"
            )
            echo_settings(settings.birefringence.transfer_function)

            tf_ds = birefringence.compute_transfer_function(
                czyx_data,
                settings.birefringence,
                settings.input_channel_names,
            )
            _write_birefringence_tf(output_dataset, tf_ds)

        if settings.phase is not None:
            echo_headline("Generating phase transfer function with settings:")
            echo_settings(settings.phase.transfer_function)

            tf_ds = phase.compute_transfer_function(
                czyx_data, recon_dim, settings.phase
            )
            _write_phase_tf(output_dataset, tf_ds, zyx_shape, recon_dim)

        if settings.fluorescence is not None:
            echo_headline(
                "Generating fluorescence transfer function with settings:"
            )
            echo_settings(settings.fluorescence.transfer_function)

            tf_ds = fluorescence.compute_transfer_function(
                czyx_data, recon_dim, settings.fluorescence
            )
            _write_fluorescence_tf(output_dataset, tf_ds, zyx_shape, recon_dim)

    # Write settings to metadata
    output_dataset.zattrs["settings"] = settings.model_dump()

    echo_headline(f"Closing {output_dirpath}\n")
    output_dataset.close()

    echo_headline(
        f"Recreate this transfer function with:\n$ waveorder compute-tf {input_position_dirpath} -c {config_filepath} -o {output_dirpath}"
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
