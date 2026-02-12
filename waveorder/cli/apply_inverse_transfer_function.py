import itertools
import warnings
from functools import partial
from pathlib import Path
from typing import Literal

import click
import numpy as np
import torch
import torch.multiprocessing as mp
import xarray as xr
from iohub import open_ome_zarr
from iohub.ngff import Position

from waveorder.api import (
    birefringence,
    birefringence_and_phase,
    fluorescence,
    phase,
)
from waveorder.api._utils import _named_dataarray
from waveorder.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
    processes_option,
    transfer_function_dirpath,
)
from waveorder.cli.printing import echo_headline, echo_settings
from waveorder.cli.settings import ReconstructionSettings
from waveorder.cli.utils import (
    apply_inverse_to_zyx_and_save,
    create_empty_hcs_zarr,
    generate_valid_position_key,
    is_single_position_store,
)
from waveorder.io import utils


def _check_background_consistency(
    background_shape, data_shape, input_channel_names
):
    data_cyx_shape = (len(input_channel_names),) + data_shape[3:]
    if background_shape != data_cyx_shape:
        raise ValueError(
            f"Background shape {background_shape} does not match data shape {data_cyx_shape}"
        )


def _load_transfer_function_dataset(
    transfer_function_dataset: Position,
    recon_biref: bool,
    recon_phase: bool,
    recon_fluo: bool,
    recon_dim: Literal[2, 3],
) -> xr.Dataset:
    """Load transfer function arrays from a zarr store into an xr.Dataset.

    Returns an xr.Dataset with the same variable names as produced by
    compute_transfer_function, so it can be passed directly to
    apply_inverse functions.
    """

    def _load(key, idx):
        return _named_dataarray(
            np.array(transfer_function_dataset[key][idx]), key
        )

    variables = {}

    if recon_biref:
        variables["intensity_to_stokes_matrix"] = _load(
            "intensity_to_stokes_matrix", (0, 0, 0)
        )

    if recon_phase and not recon_biref:
        if recon_dim == 2:
            variables["singular_system_U"] = _load("singular_system_U", (0,))
            variables["singular_system_S"] = _load("singular_system_S", (0, 0))
            variables["singular_system_Vh"] = _load("singular_system_Vh", (0,))
        elif recon_dim == 3:
            variables["real_potential_transfer_function"] = _load(
                "real_potential_transfer_function", (0, 0)
            )
            variables["imaginary_potential_transfer_function"] = _load(
                "imaginary_potential_transfer_function", (0, 0)
            )

    if recon_biref and recon_phase:
        if recon_dim == 2:
            variables["vector_singular_system_U"] = _load(
                "vector_singular_system_U", (0,)
            )
            variables["vector_singular_system_S"] = _load(
                "vector_singular_system_S", (0, 0)
            )
            variables["vector_singular_system_Vh"] = _load(
                "vector_singular_system_Vh", (0,)
            )
        elif recon_dim == 3:
            variables["real_potential_transfer_function"] = _load(
                "real_potential_transfer_function", (0, 0)
            )
            variables["imaginary_potential_transfer_function"] = _load(
                "imaginary_potential_transfer_function", (0, 0)
            )
            variables["vector_singular_system_U"] = _load(
                "vector_singular_system_U", ()
            )
            variables["vector_singular_system_S"] = _load(
                "vector_singular_system_S", (0,)
            )
            variables["vector_singular_system_Vh"] = _load(
                "vector_singular_system_Vh", ()
            )

    if recon_fluo:
        if recon_dim == 2:
            variables["singular_system_U"] = _load("singular_system_U", (0,))
            variables["singular_system_S"] = _load("singular_system_S", (0, 0))
            variables["singular_system_Vh"] = _load("singular_system_Vh", (0,))
        elif recon_dim == 3:
            variables["optical_transfer_function"] = _load(
                "optical_transfer_function", (0, 0)
            )

    return xr.Dataset(variables)


def get_reconstruction_output_metadata(position_path: Path, config_path: Path):
    # Get non-OME-Zarr plate-level metadata if it's available
    plate_metadata = {}
    input_version = "0.4"
    try:
        input_plate = open_ome_zarr(
            position_path.parent.parent.parent, mode="r"
        )
        input_version = input_plate.version
        plate_metadata = dict(input_plate.zattrs)
        # In v0.5 (zarr v3), OME metadata is nested inside an "ome" key
        if "ome" in plate_metadata:
            plate_metadata.pop("ome")
        else:
            plate_metadata.pop("plate")
    except (RuntimeError, FileNotFoundError):
        warnings.warn(
            "Position is not part of a plate...no plate metadata will be copied."
        )

    # Load the first position to infer dataset information
    input_dataset = open_ome_zarr(str(position_path), mode="r")
    T, _, Z, Y, X = input_dataset.data.shape

    settings = utils.yaml_to_model(config_path, ReconstructionSettings)

    channel_names = settings.output_channel_names
    output_z_shape = 1 if settings.output_z_is_singleton else Z

    return {
        "shape": (T, len(channel_names), output_z_shape, Y, X),
        "chunks": (1, 1, 1, Y, X),
        "scale": input_dataset.scale,
        "channel_names": channel_names,
        "dtype": np.float32,
        "plate_metadata": plate_metadata,
        "version": input_version,
    }


def apply_inverse_transfer_function_single_position(
    input_position_dirpath: Path,
    transfer_function_dirpath: Path,
    config_filepath: Path,
    output_position_dirpath: Path,
    num_processes,
    output_channel_names: list[str],
) -> None:

    echo_headline("\nStarting reconstruction...")

    # Load datasets
    transfer_function_dataset = open_ome_zarr(transfer_function_dirpath)
    input_dataset = open_ome_zarr(input_position_dirpath)
    output_dataset = open_ome_zarr(output_position_dirpath, mode="r+")

    # Get input data as xarray
    input_xa = input_dataset.to_xarray()

    # Load config file
    settings = utils.yaml_to_model(config_filepath, ReconstructionSettings)

    # Check input channel names
    if not set(settings.input_channel_names).issubset(
        input_dataset.channel_names
    ):
        raise ValueError(
            f"Each of the input_channel_names = {settings.input_channel_names} in {config_filepath} must appear in the dataset {input_position_dirpath} which currently contains channel_names = {input_dataset.channel_names}."
        )

    # Find time indices
    if settings.time_indices == "all":
        time_indices = range(input_dataset.data.shape[0])
    elif isinstance(settings.time_indices, list):
        time_indices = settings.time_indices
    elif isinstance(settings.time_indices, int):
        time_indices = [settings.time_indices]

    # Check for invalid times
    time_ubound = input_dataset.data.shape[0] - 1
    if np.max(time_indices) > time_ubound:
        raise ValueError(
            f"time_indices = {time_indices} includes a time index beyond the maximum index of the dataset = {time_ubound}"
        )

    # Simplify important settings names
    recon_biref = settings.birefringence is not None
    recon_phase = settings.phase is not None
    recon_fluo = settings.fluorescence is not None
    recon_dim = settings.reconstruction_dimension

    # Load transfer function as xr.Dataset
    tf_dataset = _load_transfer_function_dataset(
        transfer_function_dataset,
        recon_biref,
        recon_phase,
        recon_fluo,
        recon_dim,
    )

    # Close transfer function dataset early (no longer needed)
    transfer_function_dataset.close()

    # Resolve background data for birefringence
    cyx_no_sample_data = None
    if settings.birefringence is not None:
        background_path = settings.birefringence.apply_inverse.background_path
        if background_path != "":
            cyx_no_sample_data = utils.load_background(background_path)
            _check_background_consistency(
                cyx_no_sample_data.shape,
                input_dataset.data.shape,
                settings.input_channel_names,
            )

    # Prepare the apply_inverse_model_function and its arguments

    # [biref only]
    if recon_biref and (not recon_phase):
        echo_headline("Reconstructing birefringence with settings:")
        echo_settings(settings.birefringence)

        apply_inverse_model_function = (
            birefringence.apply_inverse_transfer_function
        )
        apply_inverse_args = {
            "transfer_function": tf_dataset,
            "recon_dim": recon_dim,
            "settings": settings.birefringence,
            "cyx_no_sample_data": cyx_no_sample_data,
        }

    # [phase only]
    if recon_phase and (not recon_biref):
        echo_headline("Reconstructing phase with settings:")
        echo_settings(settings.phase.apply_inverse)

        apply_inverse_model_function = phase.apply_inverse_transfer_function
        apply_inverse_args = {
            "transfer_function": tf_dataset,
            "recon_dim": recon_dim,
            "settings": settings.phase,
        }

    # [biref and phase]
    if recon_biref and recon_phase:
        echo_headline("Reconstructing birefringence and phase with settings:")
        echo_settings(settings.birefringence.apply_inverse)
        echo_settings(settings.phase.apply_inverse)

        apply_inverse_model_function = (
            birefringence_and_phase.apply_inverse_transfer_function
        )
        apply_inverse_args = {
            "transfer_function": tf_dataset,
            "recon_dim": recon_dim,
            "settings_biref": settings.birefringence,
            "settings_phase": settings.phase,
            "cyx_no_sample_data": cyx_no_sample_data,
        }

    # [fluo]
    if recon_fluo:
        echo_headline("Reconstructing fluorescence with settings:")
        echo_settings(settings.fluorescence.apply_inverse)

        apply_inverse_model_function = (
            fluorescence.apply_inverse_transfer_function
        )
        apply_inverse_args = {
            "transfer_function": tf_dataset,
            "recon_dim": recon_dim,
            "settings": settings.fluorescence,
            "fluor_channel_name": settings.input_channel_names[0],
        }

    # Make the partial function for apply inverse
    partial_apply_inverse_to_zyx_and_save = partial(
        apply_inverse_to_zyx_and_save,
        apply_inverse_model_function,
        input_xa,
        output_position_dirpath,
        settings.input_channel_names,
        **apply_inverse_args,
    )

    # Multiprocessing logic
    if num_processes > 1:
        # Loop through T, processing and writing as we go
        click.echo(
            f"\nStarting multiprocess pool with {num_processes} processes"
        )
        with mp.Pool(num_processes) as p:
            p.starmap(
                partial_apply_inverse_to_zyx_and_save,
                itertools.product(time_indices),
            )
    else:
        for t_idx in time_indices:
            partial_apply_inverse_to_zyx_and_save(t_idx)

    # Save metadata at position level
    output_dataset.zattrs["settings"] = settings.model_dump()

    echo_headline(f"Closing {output_position_dirpath}\n")

    output_dataset.close()
    input_dataset.close()

    echo_headline(
        f"Recreate this reconstruction with:\n$ waveorder apply-inv-tf {input_position_dirpath} {transfer_function_dirpath} -c {config_filepath} -o {output_position_dirpath}"
    )


def apply_inverse_transfer_function_cli(
    input_position_dirpaths: list[Path],
    transfer_function_dirpath: Path,
    config_filepath: Path,
    output_dirpath: Path,
    num_processes,
) -> None:
    # Prepare output store
    output_metadata = get_reconstruction_output_metadata(
        input_position_dirpaths[0], config_filepath
    )

    # Generate position keys - use valid HCS keys for single-position stores
    position_keys = []
    for i, input_path in enumerate(input_position_dirpaths):
        if is_single_position_store(input_path):
            position_key = generate_valid_position_key(i)
        else:
            # Use original HCS plate structure
            position_key = input_path.parts[-3:]
        position_keys.append(position_key)

    create_empty_hcs_zarr(
        store_path=output_dirpath,
        position_keys=position_keys,
        **output_metadata,
    )

    # Initialize torch threads
    if num_processes > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # Loop through positions
    for i, input_position_dirpath in enumerate(input_position_dirpaths):
        # Use the same position key generation logic
        if is_single_position_store(input_position_dirpath):
            position_key = generate_valid_position_key(i)
        else:
            position_key = input_position_dirpath.parts[-3:]

        output_position_path = output_dirpath / Path(*position_key)

        apply_inverse_transfer_function_single_position(
            input_position_dirpath,
            transfer_function_dirpath,
            config_filepath,
            output_position_path,
            num_processes,
            output_metadata["channel_names"],
        )


@click.command("apply-inv-tf")
@input_position_dirpaths()
@transfer_function_dirpath()
@config_filepath()
@output_dirpath()
@processes_option(default=1)
def _apply_inverse_transfer_function_cli(
    input_position_dirpaths: list[Path],
    transfer_function_dirpath: Path,
    config_filepath: Path,
    output_dirpath: Path,
    num_processes,
) -> None:
    """
    Apply an inverse transfer function to a dataset using a configuration file.

    Applies a transfer function to all positions in the list `input-position-dirpaths`,
    so all positions must have the same TCZYX shape.

    Appends channels to ./output.zarr, so multiple reconstructions can fill a single store.

    See /examples for example configuration files.

    >> waveorder apply-inv-tf -i ./input.zarr/*/*/* -t ./transfer-function.zarr -c /examples/birefringence.yml -o ./output.zarr
    """
    apply_inverse_transfer_function_cli(
        input_position_dirpaths,
        transfer_function_dirpath,
        config_filepath,
        output_dirpath,
        num_processes,
    )
