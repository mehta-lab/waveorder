from pathlib import Path

import numpy as np
from iohub.ngff import Position, open_ome_zarr

from waveorder.cli.utils import (
    create_empty_hcs_zarr,
    generate_valid_position_key,
    is_single_position_store,
)


def test_create_empty_hcs_zarr(tmp_path):
    store_path = tmp_path / Path("test_store.zarr")
    position_keys: list[tuple[str]] = [
        ("A", "0", "3"),
        ("B", "10", "4"),
    ]
    shape = (1, 2, 1, 1024, 1024)
    chunks = (1, 1, 1, 256, 256)
    scale = (1, 1, 1, 0.5, 0.5)
    channel_names = ["Channel1", "Channel2"]
    dtype = np.uint16
    plate_metadata = {"test": 2}

    create_empty_hcs_zarr(
        store_path,
        position_keys,
        shape,
        chunks,
        scale,
        channel_names,
        dtype,
        plate_metadata,
    )

    # Verify existence of positions and channels
    with open_ome_zarr(store_path, mode="r") as plate:
        assert plate.zattrs["test"] == 2
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


def test_generate_valid_position_key():
    """Test that position key generation produces valid alphanumeric keys."""
    # Test first few positions
    assert generate_valid_position_key(0) == ("A", "1", "0")
    assert generate_valid_position_key(1) == ("A", "2", "0")
    assert generate_valid_position_key(9) == ("A", "10", "0")
    assert generate_valid_position_key(10) == ("B", "1", "0")

    # Test all keys are alphanumeric
    for i in range(20):
        key = generate_valid_position_key(i)
        for part in key:
            assert part.isalnum(), f"Part '{part}' is not alphanumeric"


def test_single_position_detection(tmp_path):
    """Test single position store detection."""
    # Create an HCS plate structure first
    hcs_plate = tmp_path / "plate.zarr"
    with open_ome_zarr(
        str(hcs_plate), layout="hcs", mode="a", channel_names=["test"]
    ) as plate:
        position = plate.create_position("A", "1", "0")
        position.create_zeros(
            name="0",
            shape=(1, 1, 10, 512, 512),
            chunks=(1, 1, 1, 256, 256),
            dtype=np.uint16,
        )

    hcs_position = hcs_plate / "A" / "1" / "0"
    assert not is_single_position_store(
        hcs_position
    ), "Should detect as HCS plate position"

    # Create a simple single-position zarr store path (simulate what happens in real usage)
    # In real usage, the path would be something like /path/to/single-pos.zarr/0/
    # and when we try to open 3 levels up, it would fail
    fake_single_pos = tmp_path / "does-not-exist" / "single.zarr" / "0"
    fake_single_pos.mkdir(parents=True)

    # This should detect as single position because 3 levels up doesn't exist as plate
    assert is_single_position_store(
        fake_single_pos
    ), "Should detect as single position"
