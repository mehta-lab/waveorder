# 3D partially coherent optical diffraction tomography (ODT) simulation
# J. M. Soto, J. A. Rodrigo, and T. Alieva, "Label-free quantitative
# 3D tomographic imaging for partially coherent light microscopy," Opt. Express
# 25, 15699-15712 (2017)

import napari
import numpy as np
from waveorder import util
from waveorder.models import isotropic_thin_3d


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
    absorption_2D_to_3D_transfer_function,
    phase_2D_to_3D_transfer_function,
) = isotropic_thin_3d.calculate_transfer_function(**args)

zyx_scale = np.array(
    [z_pixel_size, args["yx_pixel_size"], args["yx_pixel_size"]]
)
isotropic_thin_3d.visualize_transfer_function(
    viewer,
    absorption_2D_to_3D_transfer_function,
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
yx_phase = (
    sphere[z_shape // 2]
    * (index_of_refraction_sample - args["index_of_refraction_media"])
    * z_pixel_size
    / args["wavelength_illumination"]
)  # phase in radians

yx_absorption = 0.99 * sphere[z_shape // 2]

# Perform simulation, reconstruction, and display both
zyx_data = isotropic_thin_3d.apply_transfer_function(
    yx_absorption,
    yx_phase,
    absorption_2D_to_3D_transfer_function,
    phase_2D_to_3D_transfer_function,
)

(
    yx_absorption_recon,
    yx_phase_recon,
) = isotropic_thin_3d.apply_inverse_transfer_function(
    zyx_data,
    absorption_2D_to_3D_transfer_function,
    phase_2D_to_3D_transfer_function,
)

arrays = [
    (yx_absorption, "Phantom - absorption"),
    (yx_phase, "Phantom - phase"),
    (zyx_data, "Data"),
    (yx_absorption_recon, "Reconstruction - absorption"),
    (yx_phase_recon, "Reconstruction - phase"),
]

for array in arrays:
    viewer.add_image(array[0].cpu().numpy(), name=array[1])
input("Showing object, data, and recon. Press <enter> to quit...")
