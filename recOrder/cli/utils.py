from pathlib import Path
from typing import Tuple

import click
import numpy as np
import torch
from iohub.ngff import Position, open_ome_zarr
from iohub.ngff_meta import TransformationMeta
from numpy.typing import DTypeLike


def create_empty_hcs_zarr(
    store_path: Path,
    position_keys: list[Tuple[str]],
    shape: Tuple[int],
    chunks: Tuple[int],
    scale: Tuple[float],
    channel_names: list[str],
    dtype: DTypeLike,
) -> None:
    """Create an empty zarr plate, appending position if they do not exist"""

    # Create plate
    output_plate = open_ome_zarr(
        str(store_path), layout="hcs", mode="a", channel_names=channel_names
    )

    # Create positions
    for position_key in position_keys:
        if "/".join(position_key) not in output_plate.zgroup:
            position = output_plate.create_position(*position_key)

            _ = position.create_zeros(
                name="0",
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                transform=[TransformationMeta(type="scale", scale=scale)],
            )


def apply_inverse_to_zyx_and_save(
    func,
    position: Position,
    output_path: Path,
    channel_indices,
    t_idx: int = 0,
    **kwargs,
) -> None:
    """Load a zyx array from a Position object, apply a transformation and save the result to file"""
    click.echo(f"Reconstructing t={t_idx}")

    # Load data
    czyx_uint16_numpy = position.data.oindex[t_idx, channel_indices]
    # convert to np.int32 (torch doesn't accept np.uint16), then convert to tensor float32
    czyx_data = torch.tensor(
        np.int32(czyx_uint16_numpy), dtype=torch.float32
    )

    # Apply transformation
    reconstruction_czyx = func(czyx_data, **kwargs)

    # Write to file
    # for c, recon_zyx in enumerate(reconstruction_zyx):
    with open_ome_zarr(output_path, mode="r+") as output_dataset:
        output_dataset[0][t_idx] = reconstruction_czyx
    click.echo(f"Finished Writing.. t={t_idx}")
