"""CLI: ``wo tile-stitch`` — single-process tiled reconstruction.

Reads an OME-Zarr input volume + a TileStitchSettings YAML config,
reconstructs each input tile with the modality selected by
``settings.recon.kind``, blends overlap, and writes the assembled
output volume back as OME-Zarr.

This is the single-process reference implementation. For distributed
execution across many workers (SLURM, multi-GPU), use the biahub
package, which orchestrates the same primitives over a dask cluster.
"""

from __future__ import annotations

from pathlib import Path

import click

from waveorder.cli.parsing import (
    config_filepath,
    input_position_dirpaths,
    output_dirpath,
)


def tile_stitch_cli(
    input_position_dirpath: Path,
    config_filepath_arg: Path,
    output_dirpath_arg: Path,
    *,
    device: str | None = None,
) -> None:
    """Single-process tile-stitch reconstruction driven by a YAML config.

    All run-time selectors (channel(s), timepoint(s), 2D vs 3D) live in
    the YAML under ``recon``. ``--device`` is the only CLI override.
    """
    import numpy as np
    from iohub.ngff import open_ome_zarr

    from waveorder.api.tile_stitch import (
        TileStitchSettings,
        prepare_transfer_function,
    )
    from waveorder.cli.printing import echo_headline, echo_settings
    from waveorder.io import utils
    from waveorder.tile_stitch._engine import tile_stitch_reconstruction

    settings = utils.yaml_to_model(config_filepath_arg, TileStitchSettings)
    effective_device = device if device is not None else settings.recon.device

    echo_headline("Tile-stitching with settings:")
    echo_settings(settings)

    input_dataset = open_ome_zarr(input_position_dirpath, layout="fov", mode="r")
    input_version = input_dataset.version

    requested_channels = list(settings.recon.input_channel_names)
    available = list(input_dataset.channel_names)
    missing = [c for c in requested_channels if c not in available]
    if missing:
        raise click.UsageError(
            f"recon.input_channel_names={requested_channels} not all present in dataset channels {available}: missing {missing}"
        )

    requested_t = settings.recon.time_indices
    if requested_t == "all":
        time_idxs = list(range(input_dataset.data.shape[0]))
    elif isinstance(requested_t, int):
        time_idxs = [requested_t]
    else:
        time_idxs = list(requested_t)
    if len(time_idxs) != 1:
        raise click.UsageError(
            f"wo tile-stitch processes one timepoint at a time; recon.time_indices selected {len(time_idxs)}"
        )
    timepoint = time_idxs[0]

    czyx_data = input_dataset.to_xarray().isel(t=timepoint).sel(c=requested_channels)

    transfer_function = prepare_transfer_function(settings, device=effective_device)

    output = tile_stitch_reconstruction(
        czyx_data,
        settings,
        transfer_function=transfer_function,
        device=effective_device,
    )

    echo_headline(f"Writing output to {output_dirpath_arg}\n")
    out_arr = np.asarray(output.values, dtype=np.float32)
    out_5d = out_arr[None, ...]  # add t axis: (T, C, Z, Y, X) for FOV layout
    out_channel_names = [f"{c}_recon" for c in requested_channels]

    output_dataset = open_ome_zarr(
        output_dirpath_arg,
        layout="fov",
        mode="w",
        channel_names=out_channel_names,
        version=input_version,
    )
    output_dataset.create_image("0", out_5d)
    output_dataset.zattrs["tile_stitch_settings"] = settings.model_dump()
    output_dataset.close()

    echo_headline(
        f"Recreate this reconstruction with:\n"
        f"$ wo tile-stitch -i {input_position_dirpath} -c {config_filepath_arg} -o {output_dirpath_arg}"
    )


@click.command("tile-stitch", no_args_is_help=True)
@input_position_dirpaths()
@config_filepath()
@output_dirpath()
@click.option(
    "--device",
    type=str,
    default=None,
    help='Override ``recon.device`` from the YAML ("cpu", "cuda", "mps", ...).',
)
def _tile_stitch_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    device: str | None,
) -> None:
    """Single-process tiled reconstruction.

    Reconstructs an input OME-Zarr volume by partitioning into overlapping
    input tiles, applying the configured reconstruction per tile, then
    blending contributions over each non-overlapping output tile.

    Channel selection, timepoint, and 2D vs 3D reconstruction all live in
    the YAML config under ``recon`` (the same schema used by ``wo rec``).

    For distributed (multi-node, multi-GPU) execution, use biahub's
    ``biahub tile-stitch`` command — it drives the same waveorder
    primitives over a dask cluster.

    \b
    Example:
      \033[92mwo tile-stitch -i ./input.zarr/0/0/0 -c ./tile_stitch.yml -o ./output.zarr\033[0m
    """
    tile_stitch_cli(
        input_position_dirpaths[0],
        config_filepath,
        output_dirpath,
        device=device,
    )
