from pathlib import Path

import numpy as np
from iohub.ngff import Position, open_ome_zarr

from recOrder.cli.utils import create_empty_hcs_zarr


def test_create_empty_hcs_zarr():
    store_path = Path("test_store.zarr")
    position_keys: list[tuple[str]] = [
        ("A", "0", "3"),
        ("B", "10", "4"),
    ]
    shape = (1, 2, 1, 1024, 1024)
    chunks = (1, 1, 1, 256, 256)
    scale = (1, 1, 1, 0.5, 0.5)
    channel_names = ["Channel1", "Channel2"]
    dtype = np.uint16

    create_empty_hcs_zarr(
        store_path, position_keys, shape, chunks, scale, channel_names, dtype
    )

    # Verify existence of positions and channels
    with open_ome_zarr(store_path, mode="r") as plate:
        for position_key in position_keys:
            position = plate["/".join(position_key)]
            assert isinstance(position, Position)
            assert position[0].shape == shape

    # Repeat creation should not fail
    more_channel_names = ["Channel3"]
    create_empty_hcs_zarr(
        store_path,
        position_keys,
        shape,
        chunks,
        scale,
        more_channel_names,
        dtype,
    )

    # Verify existence of appended channel names
    channel_names += more_channel_names
    for position_key in position_keys:
        position_path = store_path
        for element in position_key:
            position_path /= element
        with open_ome_zarr(position_path, mode="r") as position:
            assert position.channel_names == channel_names
