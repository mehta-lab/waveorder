from waveorder.io.reader import WaveorderReader
from waveorder.io.writer import WaveorderWriter
from recOrder.compute.qlipp_compute import (
    initialize_reconstructor,
    reconstruct_phase3D,
    bf_3D_from_phantom,
    fluorescence_from_phantom,
    reconstruct_density_from_fluorescence,
)
from datetime import datetime
import numpy as np
import napari

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

## Load a dataset

## Option 1: use data simulated from a phantom and run this script as is.
bf_data = bf_3D_from_phantom()  # (Z, Y, X)
fluor_data = fluorescence_from_phantom()  # (Z, Y, Z)

## Option 2: load from file. Uncomment all lines with a single #.
# reader = WaveorderReader('/path/to/ome-tiffs/or/zarr/store/')
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

# Save to zarr
writer = WaveorderWriter("./output-multi-modal")
writer.create_zarr_root("phase-density" + timestamp)
writer.init_array(
    position=0,
    data_shape=(1, 2) + fluor_data.shape,
    chunk_size=(1, 1, 1) + fluor_data.shape[1:],
    chan_names=["Phase", "Density"],
)
writer.write(phase, p=0, t=0, c=0, z=slice(0, phase.shape[0]))
writer.write(density, p=0, t=0, c=1, z=slice(0, density.shape[0]))

# These lines open the reconstructed images
# Alternatively, drag and drop the zarr store into napari and use the recOrder-napari reader.
v = napari.Viewer()
v.add_image(bf_data)
v.add_image(fluor_data)
v.add_image(phase)
v.add_image(density)

napari.run()
