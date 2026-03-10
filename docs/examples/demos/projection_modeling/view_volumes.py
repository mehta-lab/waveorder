"""Launch neuroglancer to view projection modeling zarr volumes.

Usage:
    /hpc/mydata/shalin.mehta/envs/neuroglancer_iohub/bin/python view_volumes.py

VS Code SSH will auto-forward the port. Click the printed URL to open
in your local browser.
"""

from pathlib import Path

import neuroglancer
import numpy as np
import zarr

ZARR_PATH = Path(__file__).parent / "data" / "projection_modeling.zarr"
VOXEL_SIZE_UM = 0.05


def main():
    store = zarr.open(str(ZARR_PATH), mode="r")
    voxel_size = store.attrs.get("voxel_size_um", VOXEL_SIZE_UM)

    dimensions = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"],
        units=["um", "um", "um"],
        scales=[voxel_size] * 3,
    )

    neuroglancer.set_server_bind_address("127.0.0.1")
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        for group_name in store.group_keys():
            group = store[group_name]
            for arr_name in group.array_keys():
                data = np.array(group[arr_name])
                layer_name = f"{group_name}/{arr_name}"
                lo, hi = float(data.min()), float(data.max())
                if lo == hi:
                    hi = lo + 1.0
                s.layers[layer_name] = neuroglancer.ImageLayer(
                    source=neuroglancer.LocalVolume(data, dimensions=dimensions),
                    shader="#uicontrol invlerp normalized\nvoid main() { emitGrayscale(normalized()); }",
                    shader_controls={"normalized": neuroglancer.InvlerpParameters(range=[lo, hi])},
                )

    print(f"Neuroglancer viewer: {viewer}")
    try:
        input("Press Enter to quit...")
    except EOFError:
        import time

        print("No interactive stdin; keeping server alive. Ctrl-C to quit.")
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()
