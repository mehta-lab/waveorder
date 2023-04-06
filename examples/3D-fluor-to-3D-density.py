from iohub import read_micromanager, open_ome_zarr
from recOrder.compute.reconstructions import (
    initialize_reconstructor,
    reconstruct_density_from_fluorescence,
)
from recOrder.compute.phantoms import fluorescence_from_phantom
from datetime import datetime
import numpy as np
import napari

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

## Load a dataset

# Option 1: use data simulated from a phantom and run this script as is.
data = fluorescence_from_phantom()  # (Z, Y, X)

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
    "n_obj_media": 1.0,  # refractive index of objective immersion media
    "pad_z": 5,  # slices to pad for phase reconstruction boundary artifacts
    "mode": "3D",  # phase reconstruction mode, "2D" or "3D"
    "use_gpu": False,
    "gpu_id": 0,
}
reconstructor = initialize_reconstructor(
    pipeline="fluorescence", **reconstructor_args
)

density = reconstruct_density_from_fluorescence(data, reconstructor, reg=1e-2)
print(f"Shape of 3D density data: {np.shape(density)}")

## Save to zarr
with open_ome_zarr(
    "./output/reconstructions_" + timestamp + ".zarr",
    layout="fov",
    mode="w-",
    channel_names=["Density"],
) as dataset:
    # Write to position "0", with length-one time and channel dimensions
    dataset["0"] = density[np.newaxis, np.newaxis, ...]

# These lines open the reconstructed images
# Alternatively, drag and drop the zarr store into napari and use the recOrder-napari reader.
v = napari.Viewer()
v.add_image(data)
v.add_image(density)
napari.run()
