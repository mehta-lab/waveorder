# 3D partially coherent optical diffraction tomography (ODT) simulation
# J. M. Soto, J. A. Rodrigo, and T. Alieva, "Label-free quantitative
# 3D tomographic imaging for partially coherent light microscopy," Opt. Express
# 25, 15699-15712 (2017)

import napari
import numpy as np
from waveorder import util
from waveorder.models import phase3D_3D


viewer = napari.Viewer()

# 3D OTF parameters
# all lengths must use consistent units e.g. um
args = {
    "zyx_shape": (100, 256, 256),
    "yx_pixel_size": 6.5 / 63,
    "z_pixel_size": 0.25,
    "z_padding": 0,
    "wavelength_illumination": 0.532,
    "index_of_refraction_media": 1.3,
    "numerical_aperture_illumination": 0.9,
    "numerical_aperture_detection": 1.2,
}

# Calculate and display OTF
(
    real_potential_transfer_function,
    imag_potential_transfer_function,
) = phase3D_3D.calculate_transfer_function(**args)

zyx_scale = np.array(
    [args["z_pixel_size"], args["yx_pixel_size"], args["yx_pixel_size"]]
)
phase3D_3D.visualize_transfer_function(
    viewer,
    real_potential_transfer_function,
    imag_potential_transfer_function,
    zyx_scale,
)
input("Showing OTFs. Press <enter> to continue...")
viewer.layers.select_all()
viewer.layers.remove_selected()

# Create a phantom
index_of_refraction_sample = 1.50
sphere, _, _ = util.generate_sphere_target(
    args["zyx_shape"],
    args["yx_pixel_size"],
    args["z_pixel_size"],
    radius=5,
    blur_size=2 * args["yx_pixel_size"],
)
zyx_phase = (
    sphere
    * (index_of_refraction_sample - args["index_of_refraction_media"])
    * args["z_pixel_size"]
    / args["wavelength_illumination"]
)  # phase in radians

# Perform simulation, reconstruction, and display both
zyx_data = phase3D_3D.apply_transfer_function(
    zyx_phase, real_potential_transfer_function, args["z_padding"]
)
zyx_recon = phase3D_3D.apply_inverse_transfer_function(
    zyx_data,
    real_potential_transfer_function,
    imag_potential_transfer_function,
    args["z_padding"],
    args["z_pixel_size"],
    args["wavelength_illumination"],
)

viewer.add_image(zyx_phase.numpy(), name="Phantom", scale=zyx_scale)
viewer.add_image(zyx_data.numpy(), name="Data", scale=zyx_scale)
viewer.add_image(zyx_recon.numpy(), name="Reconstruction", scale=zyx_scale)
input("Showing object, data, and recon. Press <enter> to quit...")
