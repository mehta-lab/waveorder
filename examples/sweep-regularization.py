from waveorder.io.reader import WaveorderReader
from waveorder.io.writer import WaveorderWriter
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
# reader = WaveorderReader('/path/to/ome-tiffs/or/zarr/store/')
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
writer = WaveorderWriter("./output")
writer.create_zarr_root("reconstructions_" + timestamp)
v = napari.Viewer()
v.add_image(data)

# Loop through regularizations
reg_powers = np.arange(-3, 3)
for i, reg_power in enumerate(reg_powers):
    reg = 10.0**reg_power
    print(f"Reconstructing with 3D phase with reg = {reg:.1e}")

    phase3D = reconstruct_phase3D(
        data, reconstructor, method="Tikhonov", reg_re=reg
    )

    # Save each regularization into a "position" of the output zarr
    writer.init_array(
        position=i,
        data_shape=(1, 1, Z, Y, X),
        chunk_size=(1, 1, 1, Y, X),
        chan_names=["Phase"],
    )
    writer.write(phase3D, p=i, t=0, c=0, z=slice(0, Z))

    # Add the reconstructions to the viewer
    v.add_image(phase3D, name=f"reg = {reg:.1e}")

napari.run()
