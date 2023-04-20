# 3D partially coherent optical diffraction tomography (ODT) simulation
# J. M. Soto, J. A. Rodrigo, and T. Alieva, "Label-free quantitative
# 3D tomographic imaging for partially coherent light microscopy," Opt. Express
# 25, 15699-15712 (2017)

import napari
import numpy as np
from waveorder import util
from waveorder.models import phase2D_3D


viewer = napari.Viewer()

# 3D OTF parameters
# all lengths must use consistent units e.g. um
z_pixel_size = 0.25
z_shape = 100
args = {
    "yx_shape": (256, 256),
    "yx_pixel_size": 6.5 / 63,
    "z_position_list": (np.arange(z_shape) - z_shape // 2) * z_pixel_size,
    "wavelength_illumination": 0.532,
    "index_of_refraction_media": 1.3,
    "numerical_aperture_illumination": 0.9,
    "numerical_aperture_detection": 1.2,
}

# Calculate and display OTF
(
    absorption_transfer_function,
    phase_2D_to_3D_transfer_function,
) = phase2D_3D.calculate_transfer_function(**args)

zyx_scale = np.array(
    [z_pixel_size, args["yx_pixel_size"], args["yx_pixel_size"]]
)
phase2D_3D.visualize_transfer_function(
    viewer,
    absorption_transfer_function,
    phase_2D_to_3D_transfer_function,
)
input("Showing OTFs. Press <enter> to continue...")
viewer.layers.select_all()
viewer.layers.remove_selected()

# Create a phantom
index_of_refraction_sample = 1.50
sphere, _, _ = util.generate_sphere_target(
    (z_shape,) + args["yx_shape"],
    args["yx_pixel_size"],
    z_pixel_size,
    radius=5,
    blur_size=2 * args["yx_pixel_size"],
)
zyx_phase = (
    sphere
    * (index_of_refraction_sample - args["index_of_refraction_media"])
    * z_pixel_size
    / args["wavelength_illumination"]
)  # phase in radians

# Perform simulation, reconstruction, and display both
zyx_data = phase2D_3D.apply_transfer_function(
    zyx_phase, phase_2D_to_3D_transfer_function
)
zyx_recon = phase2D_3D.apply_inverse_transfer_function(
    zyx_data, absorption_transfer_function, phase_2D_to_3D_transfer_function
)

viewer.add_image(zyx_phase.cpu().numpy(), name="Phantom", scale=zyx_scale)
viewer.add_image(zyx_data.cpu().numpy(), name="Data", scale=zyx_scale)
viewer.add_image(
    zyx_recon.cpu().numpy(), name="Reconstruction", scale=zyx_scale[1:]
)
input("Showing object, data, and recon. Press <enter> to quit...")
