from waveorder.io.reader import WaveorderReader
from waveorder.io.writer import WaveorderWriter
from recOrder.io.utils import load_bg
from recOrder.compute.qlipp_compute import (
    initialize_reconstructor,
    reconstruct_qlipp_stokes,
    reconstruct_qlipp_birefringence,
    reconstruct_phase3D,
)
from datetime import datetime
import numpy as np
import napari

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

## Load a dataset

# Option 1: use random data and run this script as is.
data = np.random.random((4, 16, 256, 256))  # (C, Z, Y, X)
C, Z, Y, X = data.shape
bg_data = np.random.random((4, 256, 256))  # (C, Y, X)

# Option 2: use test data or previously acquired data.
# For test data, use this link to download from Zenodo,
# https://zenodo.org/record/6983916/files/recOrder_test_data.zip?download=1
# unzip the folder, and modify test_data_root to point to the unzipped folder.
# Uncomment the next 8 lines then run this script.
# test_data_root = "/Users/talon.chandler/Downloads/recOrder_test_data/"
# dataset = "2022_08_04_recOrder_pytest_20x_04NA/"
# reader = WaveorderReader(
#    test_data_root + dataset + "2T_3P_16Z_128Y_256X_Kazansky_1"
# )
# data = reader.get_array(0)[0, ...]
# C, Z, Y, X = data.shape
# bg_data = load_bg(test_data_root + dataset + "BG/", height=Y, width=X)

## Set up a reconstructor.
reconstructor_args = {
    "image_dim": (Y, X),
    "mag": 20,  # magnification
    "pixel_size_um": 6.5,  # pixel size in um
    "n_slices": Z,  # number of slices in z-stack
    "z_step_um": 2,  # z-step size in um
    "wavelength_nm": 532,
    "swing": 0.1,
    "calibration_scheme": "4-State",  # "4-State" or "5-State"
    "NA_obj": 0.4,  # numerical aperture of objective
    "NA_illu": 0.2,  # numerical aperture of condenser
    "n_obj_media": 1.0,  # refractive index of objective immersion media
    "pad_z": 0,  # slices to pad for phase reconstruction boundary artifacts
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
S0 = birefringence[3]

phase3D = reconstruct_phase3D(
    S0, reconstructor, method="Tikhonov", reg_re=1e-2
)
print(f"Shape of 3D phase data: {np.shape(phase3D)}")

## Save to zarr
# Save birefringence
writer = WaveorderWriter("./output")
writer.create_zarr_root("birefringence_" + timestamp)
writer.init_array(
    position=0,
    data_shape=(1, 4, Z, Y, X),
    chunk_size=(1, 1, 1, Y, X),
    chan_names=["Retardance", "Orientation", "BF", "Pol"],
)
writer.write(birefringence, p=0, t=0, c=slice(0, 4), z=slice(0, Z))

## Save phase
writer.create_zarr_root("phase_" + timestamp)
writer.init_array(
    position=0,
    data_shape=(1, 1, Z, Y, X),
    chunk_size=(1, 1, 1, Y, X),
    chan_names=["Phase"],
)
writer.write(phase3D, p=0, t=0, c=0, z=slice(0, Z))

# These lines opens the reconstructed images
# Alternatively, drag and drop the zarr store into napari and use the recOrder-napari reader.
v = napari.Viewer()
v.add_image(birefringence)
v.add_image(phase3D)
napari.run()
