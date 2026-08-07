import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import click

from waveorder._pixel_size import YXPixelSize
from waveorder.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
    processes_option,
    transfer_function_dirpath,
    write_config_scale_to_output,
)


def _check_background_consistency(background_shape, data_shape, input_channel_names):
    data_cyx_shape = (len(input_channel_names),) + data_shape[3:]
    if background_shape != data_cyx_shape:
        raise ValueError(f"Background shape {background_shape} does not match data shape {data_cyx_shape}")


def _load_transfer_function_dataset(
    transfer_function_dataset,
    recon_biref: bool,
    recon_phase: bool,
    recon_fluo: bool,
    recon_dim: Literal[2, 3],
):
    """Load transfer function arrays from a zarr store into an xr.Dataset.

    Returns an xr.Dataset with the same variable names as produced by
    compute_transfer_function, so it can be passed directly to
    apply_inverse functions.
    """

    # Deferred imports for fast CLI help
    import numpy as np
    import xarray as xr

    from waveorder.api._utils import _named_dataarray

    def _load(key, idx):
        return _named_dataarray(np.array(transfer_function_dataset[key][idx]), key)

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

    return xr.Dataset(variables)


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
        with open_ome_zarr(position_path.parent.parent.parent, mode="r") as input_plate:
            input_version = input_plate.version
            plate_metadata = dict(input_plate.zattrs)
        # In v0.5 (zarr v3), OME metadata is nested inside an "ome" key
        if "ome" in plate_metadata:
            plate_metadata.pop("ome")
        else:
            plate_metadata.pop("plate")
    except (RuntimeError, FileNotFoundError):
        warnings.warn("Position is not part of a plate...no plate metadata will be copied.")

    with open_ome_zarr(str(position_path), mode="r") as input_dataset:
        T, _, Z, Y, X = input_dataset.data.shape
        scale = input_dataset.scale

    settings = utils.yaml_to_model(config_path, ReconstructionSettings)

    channel_names = settings.output_channel_names
    output_z_shape = 1 if settings.output_z_is_singleton else Z

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


def apply_inverse_transfer_function_single_position(
    input_position_dirpath: Path,
    transfer_function_dirpath: Path,
    config_filepath: Path,
    output_position_dirpath: Path,
    num_processes,
    output_channel_names: list[str],
    verbose: bool = True,
) -> None:

    # Deferred imports for fast CLI help
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial

    import numpy as np
    import torch.multiprocessing as mp
    from iohub import open_ome_zarr

    from waveorder.api import (
        birefringence,
        birefringence_and_phase,
        fluorescence,
        phase,
    )
    from waveorder.cli.printing import echo_headline, echo_settings
    from waveorder.cli.settings import ReconstructionSettings
    from waveorder.cli.utils import (
        apply_inverse_to_zyx_and_save,
        resolve_time_indices,
    )
    from waveorder.io import utils

    if verbose:
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

        apply_inverse_model_function = birefringence.apply_inverse_transfer_function
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

        apply_inverse_model_function = birefringence_and_phase.apply_inverse_transfer_function
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

        apply_inverse_model_function = fluorescence.apply_inverse_transfer_function
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
        verbose=verbose,
        **apply_inverse_args,
    )

    # Multiprocessing logic
    if num_processes > 1:
        if verbose:
            click.echo(f"\nStarting multiprocess pool with {num_processes} processes")
        # NOTE: spawn (not fork) — tensorstore runs internal C++ threads
        # that are not fork-safe, so a forked worker can deadlock or
        # segfault before our code runs. See google/tensorstore#61.
        # NOTE: ProcessPoolExecutor (not mp.Pool) so silent worker death
        # (e.g. cgroup OOM-kill) surfaces as BrokenProcessPool instead
        # of hanging indefinitely on pool.starmap.
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_processes, mp_context=context) as p:
            futures = [p.submit(partial_apply_inverse_to_zyx_and_save, t_idx) for t_idx in time_indices]
            for fut in as_completed(futures):
                fut.result()
    else:
        for t_idx in time_indices:
            partial_apply_inverse_to_zyx_and_save(t_idx)

    # Save metadata at position level, keyed by output channel names
    waveorder_meta = dict(output_dataset.zattrs.get("waveorder", {}))
    channel_key = ",".join(output_channel_names)
    waveorder_meta[channel_key] = settings.model_dump()
    output_dataset.zattrs["waveorder"] = waveorder_meta

    if verbose:
        echo_headline(f"Closing {output_position_dirpath}\n")

    output_dataset.close()
    input_dataset.close()


def _check_uniform_zyx_shapes(input_position_dirpaths: list[Path]) -> None:
    """Raise if the positions do not all share one ZYX shape.

    A transfer function is only valid for the ZYX shape it was computed from,
    so a single transfer function cannot cover positions of mixed shapes.
    """
    # Deferred import for fast CLI help
    from waveorder.cli.utils import read_zyx_shapes

    positions_by_zyx_shape: dict[tuple[int, ...], list[Path]] = {}
    for zyx_shape, position_dirpath in zip(
        read_zyx_shapes(input_position_dirpaths), input_position_dirpaths, strict=True
    ):
        positions_by_zyx_shape.setdefault(zyx_shape, []).append(position_dirpath)

    if len(positions_by_zyx_shape) == 1:
        return

    max_listed = 3
    detail_lines = []
    for zyx_shape, position_dirpaths in positions_by_zyx_shape.items():
        listed = ", ".join(str(dirpath) for dirpath in position_dirpaths[:max_listed])
        if len(position_dirpaths) > max_listed:
            listed += f", and {len(position_dirpaths) - max_listed} more"
        detail_lines.append(f"  ZYX {zyx_shape}: {listed}")
    detail = "\n".join(detail_lines)

    raise click.ClickException(
        "apply-inv-tf applies one transfer function to every position, so all "
        "positions must have the same ZYX shape, but the input positions have "
        f"{len(positions_by_zyx_shape)} different shapes:\n{detail}\n"
        "Use `waveorder reconstruct`, which computes a transfer function for "
        "each distinct shape, or run apply-inv-tf once per shape."
    )


def _resolve_transfer_function_dirpaths(
    transfer_function_dirpaths: str | Path | Mapping[tuple[int, ...], Path],
    input_position_dirpaths: list[Path],
) -> list[Path]:
    """Return one transfer function dirpath per input position.

    A single dirpath is reused for every position, which is only valid when the
    positions share a ZYX shape. A mapping selects the transfer function by
    each position's ZYX shape.
    """
    # Deferred import for fast CLI help
    from waveorder.cli.utils import read_zyx_shapes

    if isinstance(transfer_function_dirpaths, (str, Path)):
        _check_uniform_zyx_shapes(input_position_dirpaths)
        return [Path(transfer_function_dirpaths)] * len(input_position_dirpaths)

    zyx_shapes = read_zyx_shapes(input_position_dirpaths)
    missing_shapes = set(zyx_shapes) - transfer_function_dirpaths.keys()
    if missing_shapes:
        raise ValueError(
            "No transfer function was provided for ZYX shape(s): "
            + ", ".join(map(str, sorted(missing_shapes)))
        )
    return [Path(transfer_function_dirpaths[zyx_shape]) for zyx_shape in zyx_shapes]


def apply_inverse_transfer_function_cli(
    input_position_dirpaths: list[Path],
    transfer_function_dirpath: str | Path | Mapping[tuple[int, ...], Path],
    config_filepath: Path,
    output_dirpath: Path,
    num_processes,
    write_config_scale_to_output: bool = False,
) -> None:
    """Reconstruct every position in ``input_position_dirpaths``.

    ``transfer_function_dirpath`` is either a single transfer function, which
    requires every position to share one ZYX shape, or a mapping from ZYX
    shapes to transfer functions.
    """
    # Deferred imports for fast CLI help
    import torch
    from iohub import open_ome_zarr
    from iohub.ngff.utils import create_empty_plate

    from waveorder.cli.utils import (
        generate_valid_position_key,
        is_single_position_store,
    )

    per_position_transfer_function_dirpaths = _resolve_transfer_function_dirpaths(
        transfer_function_dirpath, input_position_dirpaths
    )

    # Positions may differ in shape and scale, so each gets its own metadata.
    output_metadatas = [
        get_reconstruction_output_metadata(position_dirpath, config_filepath, write_config_scale_to_output)
        for position_dirpath in input_position_dirpaths
    ]

    # `plate_metadata` is not a `create_empty_plate` parameter; write it to
    # the plate's zattrs after creation. `output_metadata["version"]` is read
    # from the input plate by `get_reconstruction_output_metadata`, so the
    # output preserves the input's OME-Zarr version.
    plate_metadata = output_metadatas[0].pop("plate_metadata", {})
    for output_metadata in output_metadatas[1:]:
        output_metadata.pop("plate_metadata", None)

    # Generate position keys - use valid HCS keys for single-position stores
    position_keys = []
    for i, input_path in enumerate(input_position_dirpaths):
        if is_single_position_store(input_path):
            position_key = generate_valid_position_key(i)
        else:
            # Use original HCS plate structure
            position_key = input_path.parts[-3:]
        position_keys.append(position_key)

    # `create_empty_plate` allocates one shape and scale per call.
    for position_key, output_metadata in zip(position_keys, output_metadatas, strict=True):
        create_empty_plate(
            store_path=output_dirpath,
            position_keys=[position_key],
            **output_metadata,
        )

    if plate_metadata:
        with open_ome_zarr(str(output_dirpath), mode="r+") as output_plate:
            output_plate.zattrs.update(plate_metadata)

    # Initialize torch threads
    if num_processes > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # Loop through positions
    for input_position_dirpath, tf_dirpath, position_key, output_metadata in zip(
        input_position_dirpaths,
        per_position_transfer_function_dirpaths,
        position_keys,
        output_metadatas,
        strict=True,
    ):
        apply_inverse_transfer_function_single_position(
            input_position_dirpath,
            tf_dirpath,
            config_filepath,
            output_dirpath / Path(*position_key),
            num_processes,
            output_metadata["channel_names"],
        )


@click.command("apply-inv-tf", no_args_is_help=True)
@input_position_dirpaths()
@transfer_function_dirpath()
@config_filepath()
@output_dirpath()
@processes_option(default=1)
@write_config_scale_to_output()
def _apply_inverse_transfer_function_cli(
    input_position_dirpaths: list[Path],
    transfer_function_dirpath: Path,
    config_filepath: Path,
    output_dirpath: Path,
    num_processes,
    write_config_scale_to_output: bool,
) -> None:
    """Apply an inverse transfer function to a dataset.

    Applies a transfer function to all positions in the list
    `input-position-dirpaths`. A transfer function is only valid for the ZYX
    shape it was computed from, so this command errors out unless all
    positions have the same ZYX shape. To reconstruct positions of mixed
    shapes, use `waveorder reconstruct`, which computes a transfer function
    for each distinct shape.

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
    )
