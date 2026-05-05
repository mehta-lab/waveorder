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
    timepoint: int = 0,
    channel: str | None = None,
    device: str | None = None,
) -> None:
    """Single-process tile-stitch reconstruction driven by a YAML config."""
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

    echo_headline(f"Tile-stitching with settings:")
    echo_settings(settings)

    input_dataset = open_ome_zarr(input_position_dirpath, layout="fov", mode="r")
    input_version = input_dataset.version

    if channel is None:
        if len(input_dataset.channel_names) != 1:
            raise click.UsageError(
                f"Input has multiple channels {input_dataset.channel_names}; "
                "specify --channel."
            )
        channel = input_dataset.channel_names[0]
    elif channel not in input_dataset.channel_names:
        raise click.UsageError(
            f"--channel {channel!r} not in dataset channels {input_dataset.channel_names}"
        )

    czyx_data = (
        input_dataset.to_xarray()
        .isel(t=timepoint)
        .sel(c=[channel])
    )

    transfer_function = prepare_transfer_function(settings, device=device)

    output = tile_stitch_reconstruction(
        czyx_data,
        settings,
        transfer_function=transfer_function,
        device=device,
    )

    echo_headline(f"Writing output to {output_dirpath_arg}\n")
    out_arr = np.asarray(output.values, dtype=np.float32)
    out_5d = out_arr[None, ...]  # add t axis: (T, C, Z, Y, X) for FOV layout

    output_dataset = open_ome_zarr(
        output_dirpath_arg,
        layout="fov",
        mode="w",
        channel_names=[f"{channel}_recon"],
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
    "--timepoint",
    type=int,
    default=0,
    show_default=True,
    help="Timepoint index to reconstruct.",
)
@click.option(
    "--channel",
    type=str,
    default=None,
    help="Channel name to reconstruct (auto-selected when input has one channel).",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help='Compute device ("cpu", "cuda", or None for default).',
)
def _tile_stitch_cli(
    input_position_dirpaths: list[Path],
    config_filepath: Path,
    output_dirpath: Path,
    timepoint: int,
    channel: str | None,
    device: str | None,
) -> None:
    """Single-process tiled reconstruction.

    Reconstructs an input OME-Zarr volume by partitioning into overlapping
    input tiles, applying the configured reconstruction per tile, then
    blending contributions over each non-overlapping output tile.

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
        timepoint=timepoint,
        channel=channel,
        device=device,
    )
