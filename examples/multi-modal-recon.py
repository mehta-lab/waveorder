from iohub import read_micromanager, open_ome_zarr
from recOrder.compute.reconstructions import (
    initialize_reconstructor,
    reconstruct_phase3D,
    reconstruct_density_from_fluorescence,
)
from recOrder.compute.phantoms import (
    bf_3D_from_phantom,
    fluorescence_from_phantom,
)
from datetime import datetime
import numpy as np
import napari

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

## Load a dataset

## Option 1: use data simulated from a phantom and run this script as is.
bf_data = bf_3D_from_phantom()  # (Z, Y, X)
fluor_data = fluorescence_from_phantom()  # (Z, Y, X)

## Option 2: load from file. Uncomment all lines with a single #.
# reader = read_micromanager('/path/to/ome-tiffs/or/zarr/store/')
# position, time = 0, 0

## Set these by hand or use `reader.channel_names.index('BF')` to read from metadata
# bf_index = 0
# fluor_index = 1

## Read
# bf_data = reader.get_array(position)[time, bf_index, ...]
# fluor_data = reader.get_array(position)[time, fluor_index, ...]

## Set up the reconstructors.

# Set arguments that are shared by both modalities
reconstructor_args = {
    "image_dim": bf_data.shape[1:],  # (Y, X)
    "mag": 20,  # magnification
    "pixel_size_um": 6.5,  # pixel size in um
    "z_step_um": 2,  # z-step size in um
    "NA_obj": 0.4,  # numerical aperture of objective
    "NA_illu": 0.2,  # numerical aperture of condenser
    "n_obj_media": 1.0,  # refractive index of objective immersion media
    "pad_z": 5,  # slices to pad for phase reconstruction boundary artifacts
    "mode": "3D",  # phase reconstruction mode, "2D" or "3D"
    "use_gpu": False,
    "gpu_id": 0,
}

# Initialize reconstructors with parameters that are not shared
bf_reconstructor = initialize_reconstructor(
    pipeline="PhaseFromBF",
    wavelength_nm=532,
    n_slices=bf_data.shape[0],
    **reconstructor_args
)

fluor_reconstructor = initialize_reconstructor(
    pipeline="fluorescence",
    wavelength_nm=450,
    n_slices=fluor_data.shape[0],
    **reconstructor_args
)

# Reconstruct
phase = reconstruct_phase3D(
    bf_data, bf_reconstructor, method="Tikhonov", reg_re=1e-2
)

density = reconstruct_density_from_fluorescence(
    fluor_data, fluor_reconstructor, reg=1e-2
)

## Save to zarr
with open_ome_zarr(
    "./output/reconstructions_" + timestamp + ".zarr",
    layout="fov",
    mode="w-",
    channel_names=["Phase"],
) as dataset:
    # Write to position "0", with length-one time dimension
    dataset["0"] = phase[np.newaxis, np.newaxis, ...]
    dataset.append_channel("Density")
    dataset["0"][0, -1] = density

# These lines open the reconstructed images
# Alternatively, drag and drop the zarr store into napari and use the recOrder-napari reader.
v = napari.Viewer()
v.add_image(bf_data)
v.add_image(fluor_data)
v.add_image(phase)
v.add_image(density)
v.dims.current_step = (15, 256, 256)

napari.run()
