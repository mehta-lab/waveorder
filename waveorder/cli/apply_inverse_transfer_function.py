import math
import warnings
from pathlib import Path
from typing import Literal

import click

from waveorder._pixel_size import YXPixelSize
from waveorder.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
    processes_option,
    resume_option,
    transfer_function_dirpath,
    write_config_scale_to_output,
)


def _check_background_consistency(background_shape, data_shape, input_channel_names):
    data_cyx_shape = (len(input_channel_names),) + data_shape[3:]
    if background_shape != data_cyx_shape:
        raise ValueError(f"Background shape {background_shape} does not match data shape {data_cyx_shape}")


def _load_transfer_function(
    transfer_function_dataset,
    recon_biref: bool,
    recon_phase: bool,
    recon_fluo: bool,
    recon_dim: Literal[2, 3],
) -> dict:
    """Load transfer function arrays from a zarr store as torch tensors.

    Returns a dict keyed by the variable names produced by
    compute_transfer_function. The apply_inverse API functions accept it in
    place of compute_transfer_function's xr.Dataset and use the tensors
    without copying them.
    """

    # Deferred imports for fast CLI help
    import numpy as np
    import torch

    def _load(key, idx):
        return torch.from_numpy(np.array(transfer_function_dataset[key][idx]))

    variables = {}

    if recon_biref:
        variables["intensity_to_stokes_matrix"] = _load("intensity_to_stokes_matrix", (0, 0, 0))

    if recon_phase and not recon_biref:
        if recon_dim == 2:
            variables["singular_system_U"] = _load("singular_system_U", (0,))
            variables["singular_system_S"] = _load("singular_system_S", (0, 0))
            variables["singular_system_Vh"] = _load("singular_system_Vh", (0,))
        elif recon_dim == 3:
            variables["real_potential_transfer_function"] = _load("real_potential_transfer_function", (0, 0))
            variables["imaginary_potential_transfer_function"] = _load("imaginary_potential_transfer_function", (0, 0))

    if recon_biref and recon_phase:
        if recon_dim == 2:
            variables["vector_singular_system_U"] = _load("vector_singular_system_U", (0,))
            variables["vector_singular_system_S"] = _load("vector_singular_system_S", (0, 0))
            variables["vector_singular_system_Vh"] = _load("vector_singular_system_Vh", (0,))
        elif recon_dim == 3:
            variables["real_potential_transfer_function"] = _load("real_potential_transfer_function", (0, 0))
            variables["imaginary_potential_transfer_function"] = _load("imaginary_potential_transfer_function", (0, 0))
            variables["vector_singular_system_U"] = _load("vector_singular_system_U", ())
            variables["vector_singular_system_S"] = _load("vector_singular_system_S", (0,))
            variables["vector_singular_system_Vh"] = _load("vector_singular_system_Vh", ())

    if recon_fluo:
        if recon_dim == 2:
            variables["singular_system_U"] = _load("singular_system_U", (0,))
            variables["singular_system_S"] = _load("singular_system_S", (0, 0))
            variables["singular_system_Vh"] = _load("singular_system_Vh", (0,))
        elif recon_dim == 3:
            variables["optical_transfer_function"] = _load("optical_transfer_function", (0, 0))

    return variables


class PixelSizeMismatchWarning(UserWarning):
    """Input zarr pixel sizes disagree with the reconstruction config."""


# The CLI suppresses UserWarning at startup to silence torch/CUDA noise;
# carve out this subclass so the mismatch warning is still shown to the user.
warnings.filterwarnings("always", category=PixelSizeMismatchWarning)


def _get_config_pixel_sizes(settings):
    """Return (z_pixel_size, yx_pixel_size) from config, or None for birefringence-only.

    Parameters
    ----------
    settings : ReconstructionSettings
        Top-level reconstruction settings.

    Returns
    -------
    tuple[float, float] or None
        (z_pixel_size, yx_pixel_size) if available, None otherwise.
    """
    for attr in ("phase", "fluorescence"):
        sub = getattr(settings, attr, None)
        if sub is not None:
            tf = sub.transfer_function
            return tf.z_pixel_size, tf.yx_pixel_size
    return None


def _warn_pixel_size_mismatch(input_scale, config_pixel_sizes):
    """Warn if input zarr scale and config pixel sizes differ by more than 5%.

    Parameters
    ----------
    input_scale : tuple[float, ...]
        TCZYX scale from the input dataset.
    config_pixel_sizes : tuple[float, YXPixelSize]
        (z_pixel_size, yx_pixel_size) from the reconstruction config. The
        lateral entry is a :class:`YXPixelSize` with ``.y`` and ``.x``
        spacings.
    """
    rel_tol = 0.05
    z_pixel_size, yx_pixel_size = config_pixel_sizes
    yx_pixel_size = YXPixelSize.from_value(yx_pixel_size)
    _, _, z_scale, y_scale, x_scale = input_scale

    mismatches = []
    if not math.isclose(z_scale, z_pixel_size, rel_tol=rel_tol):
        mismatches.append(f"  z: input={z_scale}, config={z_pixel_size}")
    if not math.isclose(y_scale, yx_pixel_size.y, rel_tol=rel_tol):
        mismatches.append(f"  y: input={y_scale}, config={yx_pixel_size.y}")
    if not math.isclose(x_scale, yx_pixel_size.x, rel_tol=rel_tol):
        mismatches.append(f"  x: input={x_scale}, config={yx_pixel_size.x}")

    if mismatches:
        detail = "\n".join(mismatches)
        warnings.warn(
            f"Input pixel sizes do not match reconstruction config "
            f"(>{rel_tol:.0%} relative difference):\n{detail}\n"
            f"The input zarr's pixel sizes will be used in the output. "
            f"Use --write-config-scale-to-output to use the config's pixel sizes instead.",
            PixelSizeMismatchWarning,
            stacklevel=2,
        )


def get_reconstruction_output_metadata(
    position_path: Path,
    config_path: Path,
    write_config_scale_to_output: bool = False,
):
    # Deferred imports for fast CLI help
    import numpy as np
    from iohub import open_ome_zarr

    from waveorder.cli.settings import ReconstructionSettings
    from waveorder.io import utils

    # Get non-OME-Zarr plate-level metadata if it's available
    plate_metadata = {}
    input_version = "0.4"
    try:
        input_plate = open_ome_zarr(position_path.parent.parent.parent, mode="r")
        input_version = input_plate.version
        plate_metadata = dict(input_plate.zattrs)
        # In v0.5 (zarr v3), OME metadata is nested inside an "ome" key
        if "ome" in plate_metadata:
            plate_metadata.pop("ome")
        else:
            plate_metadata.pop("plate")
    except (RuntimeError, FileNotFoundError):
        warnings.warn("Position is not part of a plate...no plate metadata will be copied.")

    # Load the first position to infer dataset information
    input_dataset = open_ome_zarr(str(position_path), mode="r")
    T, _, Z, Y, X = input_dataset.data.shape

    settings = utils.yaml_to_model(config_path, ReconstructionSettings)

    channel_names = settings.output_channel_names
    output_z_shape = 1 if settings.output_z_is_singleton else Z

    scale = input_dataset.scale
    config_pixel_sizes = _get_config_pixel_sizes(settings)
    if config_pixel_sizes is not None:
        if write_config_scale_to_output:
            z_pixel_size, yx_pixel_size = config_pixel_sizes
            yx_pixel_size = YXPixelSize.from_value(yx_pixel_size)
            scale = (scale[0], scale[1], z_pixel_size, yx_pixel_size.y, yx_pixel_size.x)
        else:
            _warn_pixel_size_mismatch(scale, config_pixel_sizes)

    return {
        "shape": (T, len(channel_names), output_z_shape, Y, X),
        "scale": scale,
        "channel_names": channel_names,
        "dtype": np.float32,
        "plate_metadata": plate_metadata,
        "version": input_version,
    }


def get_reconstruction_provenance(settings) -> dict:
    """Provenance to record on each output position, as ``create_empty_plate(extra_metadata=...)``.

    One top-level key per reconstruction (``settings.provenance_key``, e.g.
    ``waveorder-Phase3D``), so a reconstruction appending channels to a plate
    overwrites only its own entry. Recorded once, at plate creation, like the
    ``biahub-<step>`` keys of the other processing steps.
    """
    return {settings.provenance_key: settings.model_dump()}


def apply_inverse_transfer_function_single_position(
    input_position_dirpath: Path,
    transfer_function_dirpath: Path,
    config_filepath: Path,
    output_position_dirpath: Path,
    num_processes,
    output_channel_names: list[str],
    verbose: bool = True,
    resume: bool = False,
) -> None:
    """Reconstruct one position, one timepoint per task, through iohub.

    Timepoints and channels are addressed by index: channels are resolved by
    name, in config order, against the input and output stores, so several
    configs can write different channel groups into one plate. With
    ``resume=True``, timepoints a previous interrupted run already finished
    are skipped, unless the settings or the transfer function changed since.

    Writes no metadata: the output plate's creator records the settings (see
    `get_reconstruction_provenance`).
    """

    # Deferred imports for fast CLI help
    import numpy as np
    from iohub import open_ome_zarr
    from iohub.ngff.utils import process_single_position

    from waveorder.api import (
        birefringence,
        birefringence_and_phase,
        fluorescence,
        phase,
    )
    from waveorder.cli.printing import echo_headline, echo_settings
    from waveorder.cli.settings import ReconstructionSettings
    from waveorder.cli.utils import (
        apply_inverse_czyx,
        reconstruction_fingerprint,
        resolve_time_indices,
    )
    from waveorder.io import utils

    if verbose:
        echo_headline("\nStarting reconstruction...")

    # Load datasets
    transfer_function_dataset = open_ome_zarr(transfer_function_dirpath)
    input_dataset = open_ome_zarr(input_position_dirpath)

    # Input coordinates, for labeling each volume iohub hands the model (the
    # data itself is read by iohub, once per timepoint)
    input_xa = input_dataset.to_xarray()

    # Load config file
    settings = utils.yaml_to_model(config_filepath, ReconstructionSettings)

    # Check input channel names
    if not set(settings.input_channel_names).issubset(input_dataset.channel_names):
        raise ValueError(
            f"Each of the input_channel_names = {settings.input_channel_names} in {config_filepath} must appear in the dataset {input_position_dirpath} which currently contains channel_names = {input_dataset.channel_names}."
        )

    # Find time indices
    time_indices = resolve_time_indices(settings.time_indices, input_dataset.data.shape[0])

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

    # Load the transfer function once, as torch tensors, and pass it with
    # every timepoint's task. iohub's process pool pickles each task's
    # arguments, but importing torch registers pickling reductions (from
    # torch.multiprocessing) that move a CPU tensor into shared memory and send
    # a handle to it instead of its data: every worker maps this one copy, and
    # nothing large goes through the worker pipes. This relies on them being
    # torch tensors -- numpy arrays (or an xr.Dataset of them) pickle by value,
    # which re-sent the whole transfer function (~2-3 GB in 3D) for every
    # timepoint.
    transfer_function = _load_transfer_function(
        transfer_function_dataset,
        recon_biref,
        recon_phase,
        recon_fluo,
        recon_dim,
    )

    # Fingerprint what determines the output, so resuming after a settings
    # change or a recomputed transfer function recomputes instead of reusing
    # stale timepoints. Read before closing the transfer function dataset.
    resume_token = reconstruction_fingerprint(
        settings,
        dict(transfer_function_dataset.zattrs.get("settings", {})),
        transfer_function,
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

        apply_inverse_model_function = birefringence.apply_inverse_transfer_function
        apply_inverse_args = {
            "transfer_function": transfer_function,
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
            "transfer_function": transfer_function,
            "recon_dim": recon_dim,
            "settings": settings.phase,
        }

    # [biref and phase]
    if recon_biref and recon_phase:
        echo_headline("Reconstructing birefringence and phase with settings:")
        echo_settings(settings.birefringence.apply_inverse)
        echo_settings(settings.phase.apply_inverse)

        apply_inverse_model_function = birefringence_and_phase.apply_inverse_transfer_function
        apply_inverse_args = {
            "transfer_function": transfer_function,
            "recon_dim": recon_dim,
            "settings_biref": settings.birefringence,
            "settings_phase": settings.phase,
            "cyx_no_sample_data": cyx_no_sample_data,
        }

    # [fluo]
    if recon_fluo:
        echo_headline("Reconstructing fluorescence with settings:")
        echo_settings(settings.fluorescence.apply_inverse)

        apply_inverse_model_function = fluorescence.apply_inverse_transfer_function
        apply_inverse_args = {
            "transfer_function": transfer_function,
            "recon_dim": recon_dim,
            "settings": settings.fluorescence,
            "fluor_channel_name": settings.input_channel_names[0],
        }

    # Label each CZYX volume with the input's own coordinates, taken once here
    czyx_coords = {
        "c": ("c", settings.input_channel_names),
        **{dim: (dim, input_xa.coords[dim].values, dict(input_xa.coords[dim].attrs)) for dim in ("z", "y", "x")},
    }

    # Channels by name, in config order, as one group, so the model receives
    # all input channels in one CZYX and writes the channels it owns -- not
    # necessarily the plate's first ones
    input_channel_indices = [[input_dataset.channel_names.index(name) for name in settings.input_channel_names]]
    with open_ome_zarr(output_position_dirpath, mode="r") as output_dataset:
        output_channel_indices = [[output_dataset.channel_names.index(name) for name in output_channel_names]]
        if resume and output_dataset.version == "0.4":
            # iohub also warns, but the CLI silences UserWarnings
            click.echo(
                "resume was requested, but progress can only be tracked in an OME-Zarr v0.5 output; "
                "recomputing every timepoint."
            )
    input_dataset.close()

    # The output plate has the input's full T, so input and output timepoints
    # share indices, whatever the input's time coordinates
    process_single_position(
        apply_inverse_czyx,
        input_position_path=input_position_dirpath,
        output_position_path=output_position_dirpath,
        input_channel_indices=input_channel_indices,
        output_channel_indices=output_channel_indices,
        input_time_indices=time_indices,
        output_time_indices=time_indices,
        num_workers=num_processes,
        resume=resume,
        resume_token=resume_token,
        model_function=apply_inverse_model_function,
        czyx_coords=czyx_coords,
        **apply_inverse_args,
    )

    if verbose:
        echo_headline(f"Closing {output_position_dirpath}\n")


def apply_inverse_transfer_function_cli(
    input_position_dirpaths: list[Path],
    transfer_function_dirpath: Path,
    config_filepath: Path,
    output_dirpath: Path,
    num_processes,
    write_config_scale_to_output: bool = False,
    resume: bool = False,
) -> None:
    # Deferred imports for fast CLI help
    import torch
    from iohub import open_ome_zarr
    from iohub.ngff.utils import create_empty_plate

    from waveorder.cli.settings import ReconstructionSettings
    from waveorder.cli.utils import (
        generate_valid_position_key,
        is_single_position_store,
    )
    from waveorder.io import utils

    # Prepare output store
    output_metadata = get_reconstruction_output_metadata(
        input_position_dirpaths[0], config_filepath, write_config_scale_to_output
    )

    # `plate_metadata` is not a `create_empty_plate` parameter; write it to
    # the plate's zattrs after creation. `output_metadata["version"]` is read
    # from the input plate by `get_reconstruction_output_metadata`, so the
    # output preserves the input's OME-Zarr version.
    plate_metadata = output_metadata.pop("plate_metadata", {})

    # Generate position keys - use valid HCS keys for single-position stores
    position_keys = []
    for i, input_path in enumerate(input_position_dirpaths):
        if is_single_position_store(input_path):
            position_key = generate_valid_position_key(i)
        else:
            # Use original HCS plate structure
            position_key = input_path.parts[-3:]
        position_keys.append(position_key)

    create_empty_plate(
        store_path=output_dirpath,
        position_keys=position_keys,
        **output_metadata,
        extra_metadata=get_reconstruction_provenance(utils.yaml_to_model(config_filepath, ReconstructionSettings)),
    )

    if plate_metadata:
        with open_ome_zarr(str(output_dirpath), mode="r+") as output_plate:
            output_plate.zattrs.update(plate_metadata)

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
            resume=resume,
        )


@click.command("apply-inv-tf", no_args_is_help=True)
@input_position_dirpaths()
@transfer_function_dirpath()
@config_filepath()
@output_dirpath()
@processes_option(default=1)
@write_config_scale_to_output()
@resume_option()
def _apply_inverse_transfer_function_cli(
    input_position_dirpaths: list[Path],
    transfer_function_dirpath: Path,
    config_filepath: Path,
    output_dirpath: Path,
    num_processes,
    write_config_scale_to_output: bool,
    resume: bool,
) -> None:
    """Apply an inverse transfer function to a dataset.

    Applies a transfer function to all positions in the list
    `input-position-dirpaths`, so all positions must have the same TCZYX shape.

    \b
    Example:
      \033[92mwo apply-inv-tf -i ./input.zarr/*/*/* -t ./tf.zarr -c ./config.yml -o ./output.zarr\033[0m
    """
    apply_inverse_transfer_function_cli(
        input_position_dirpaths,
        transfer_function_dirpath,
        config_filepath,
        output_dirpath,
        num_processes,
        write_config_scale_to_output,
        resume,
    )
