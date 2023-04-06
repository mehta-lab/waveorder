from iohub import read_micromanager, open_ome_zarr
from recOrder.io.utils import load_bg
from recOrder.compute.reconstructions import (
    initialize_reconstructor,
    reconstruct_qlipp_stokes,
    reconstruct_qlipp_birefringence,
    reconstruct_phase3D,
)
from recOrder.compute.phantoms import pol_3D_from_phantom
from datetime import datetime
import numpy as np
import napari

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

## Load a dataset

# Option 1: use random data and run this script as is.
data, bg_data = pol_3D_from_phantom()  # (C, Z, Y, X) and (C, Y, X)
C, Z, Y, X = data.shape

# Option 2: load from file
# reader = read_micromanager('/path/to/ome-tiffs/or/zarr/store/')
# position, time = 0, 0
# data = reader.get_array(position)[time, ...]
# C, Z, Y, X = data.shape
# bg_data = load_bg("/path/to/recorder/BG", height=Y, width=X)

## Set up a reconstructor.
reconstructor_args = {
    "image_dim": (Y, X),
    "mag": 20,  # magnification
    "pixel_size_um": 6.5,  # pixel size in um
    "n_slices": Z,  # number of slices in z-stack
    "z_step_um": 2,  # z-step size in um
    "wavelength_nm": 532,
    "swing": 0.1,
    "calibration_scheme": "5-State",  # "4-State" or "5-State"
    "NA_obj": 0.4,  # numerical aperture of objective
    "NA_illu": 0.2,  # numerical aperture of condenser
    "n_obj_media": 1.0,  # refractive index of objective immersion media
    "pad_z": 5,  # slices to pad for phase reconstruction boundary artifacts
    "bg_correction": "local_fit",  # BG correction method: "None", "local_fit", "global"
    "mode": "3D",  # phase reconstruction mode, "2D" or "3D"
    "use_gpu": False,
    "gpu_id": 0,
}
reconstructor = initialize_reconstructor(
    pipeline="QLIPP", **reconstructor_args
)

# Reconstruct background Stokes
bg_stokes = reconstruct_qlipp_stokes(bg_data, reconstructor)
print(f"Shape of BG Stokes: {np.shape(bg_stokes)}")

# Reconstruct data Stokes w/ background correction
stokes = reconstruct_qlipp_stokes(data, reconstructor, bg_stokes)
print(f"Shape of background corrected data Stokes: {np.shape(stokes)}")

# Reconstruct Birefringence from Stokes
# Shape of the output birefringence will be (C, Z, Y, X) where
# Channel order = Retardance [nm], Orientation [rad], Brightfield (S0), Degree of Polarization
birefringence = reconstruct_qlipp_birefringence(stokes, reconstructor)
birefringence[0] = (
    birefringence[0] / (2 * np.pi) * reconstructor_args["wavelength_nm"]
)
print(f"Shape of birefringence data: {np.shape(birefringence)}")

# Reconstruct Phase3D from S0
S0 = birefringence[2]

phase3D = reconstruct_phase3D(
    S0, reconstructor, method="Tikhonov", reg_re=1e-2
)
print(f"Shape of 3D phase data: {np.shape(phase3D)}")

## Save to zarr
with open_ome_zarr(
    "./output/reconstructions_" + timestamp + ".zarr",
    layout="fov",
    mode="w-",
    channel_names=[
        "Retardance",
        "Orientation",
        "BF - computed",
        "DoP",
        "Phase",
    ],
) as dataset:
    img = dataset.create_zeros("0", (1, 5, Z, Y, X), dtype=birefringence.dtype)
    img[0, 0:4] = birefringence
    img[0, 4] = phase3D

# These lines open the reconstructed images
# Alternatively, drag and drop the zarr store into napari and use the recOrder-napari reader.
v = napari.Viewer()
v.add_image(data)
v.add_image(phase3D)
v.add_image(birefringence, contrast_limits=(0, 25))
v.dims.current_step = (0, 5, 256, 256)
napari.run()
