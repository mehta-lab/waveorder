from pathlib import Path

import numpy as np
from iohub import open_ome_zarr

from recOrder.cli.utils import create_empty_hcs_zarr


def test_create_empty_hcs_zarr():
    store_path = Path("test_store.zarr")
    position_keys = [
        ("A", "0", "3"),
        ("B", "10", "4"),
    ]
    shape = (1, 1, 1, 1024, 1024)
    chunks = (1, 1, 1, 256, 256)
    scale = (1, 1, 1, 0.5, 0.5)
    channel_names = ["Channel1", "Channel2"]
    dtype = np.uint16

    create_empty_hcs_zarr(
        store_path, position_keys, shape, chunks, scale, channel_names, dtype
    )

    # Verify existence of positions and channels
    with open_ome_zarr(store_path, mode="r") as z:
        for position_key in position_keys:
            position_path = "/".join(position_key)
            assert position_path in z.zgroup
            assert "Channel1" in z.channel_names
            assert "Channel2" in z.channel_names

    # Repeat creation should not fail
    create_empty_hcs_zarr(
        store_path, position_keys, shape, chunks, scale, channel_names, dtype
    )
