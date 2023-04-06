from iohub import read_micromanager, open_ome_zarr
from recOrder.compute.reconstructions import (
    initialize_reconstructor,
    reconstruct_phase3D,
)
from recOrder.compute.phantoms import bf_3D_from_phantom
from datetime import datetime
import numpy as np
import napari

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

## Load a dataset

# Option 1: use data simulated from a phantom and run this script as is.
data = bf_3D_from_phantom()  # (Z, Y, X)

# Option 2: load from file
# reader = read_micromanager('/path/to/ome-tiffs/or/zarr/store/')
# position, time, channel = 0, 0, 0
# data = reader.get_array(position)[time, channel, ...]  # read 3D volume

## Set up a reconstructor.
Z, Y, X = data.shape
reconstructor_args = {
    "image_dim": (Y, X),
    "mag": 20,  # magnification
    "pixel_size_um": 6.5,  # pixel size in um
    "n_slices": Z,  # number of slices in z-stack
    "z_step_um": 2,  # z-step size in um
    "wavelength_nm": 532,
    "NA_obj": 0.4,  # numerical aperture of objective
    "NA_illu": 0.2,  # numerical aperture of condenser
    "n_obj_media": 1.0,  # refractive index of objective immersion media
    "pad_z": 5,  # slices to pad for phase reconstruction boundary artifacts
    "mode": "3D",  # phase reconstruction mode, "2D" or "3D"
    "use_gpu": False,
    "gpu_id": 0,
}
reconstructor = initialize_reconstructor(
    pipeline="PhaseFromBF", **reconstructor_args
)

# Setup a single writer and viewer
reg_powers = 10.0 ** np.arange(-3, 3)

dataset = open_ome_zarr(
    "./output/reconstructions_" + timestamp + ".zarr",
    layout="fov",
    mode="w-",
    channel_names=[f"{reg:.1e}" for reg in reg_powers],
)

dataset.create_zeros(
    "0", shape=(1, len(reg_powers), Z, Y, X), dtype=np.float32
)

v = napari.Viewer()
v.add_image(data)

# Loop through regularizations
for i, reg in enumerate(reg_powers):
    print(f"Reconstructing with 3D phase with reg = {reg}")

    phase3D = reconstruct_phase3D(
        data, reconstructor, method="Tikhonov", reg_re=reg
    )

    # Save each regularization into a "position" of the output zarr
    dataset["0"][0, i, ...] = phase3D

    # Add the reconstructions to the viewer
    v.add_image(phase3D, name=f"reg = {reg}")

napari.run()
