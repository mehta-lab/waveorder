# 3D partially coherent optical diffraction tomography (ODT) simulation
# J. M. Soto, J. A. Rodrigo, and T. Alieva, "Label-free quantitative
# 3D tomographic imaging for partially coherent light microscopy," Opt. Express
# 25, 15699-15712 (2017)

import napari
import numpy as np
from waveorder import util
from waveorder.models import isotropic_thin_3d

# Parameters
# all lengths must use consistent units e.g. um
simulation_arguments = {
    "yx_shape": (256, 256),
    "yx_pixel_size": 6.5 / 63,
    "wavelength_illumination": 0.532,
    "index_of_refraction_media": 1.3,
}
phantom_arguments = {"index_of_refraction_sample": 1.33, "sphere_radius": 5}
z_shape = 100
z_pixel_size = 0.25
transfer_function_arguments = {
    "z_position_list": (np.arange(z_shape) - z_shape // 2) * z_pixel_size,
    "numerical_aperture_illumination": 0.9,
    "numerical_aperture_detection": 1.2,
}

# Create a phantom
yx_absorption, yx_phase = isotropic_thin_3d.generate_test_phantom(
    **simulation_arguments, **phantom_arguments
)

# Calculate transfer function
(
    absorption_2d_to_3d_transfer_function,
    phase_2d_to_3d_transfer_function,
) = isotropic_thin_3d.calculate_transfer_function(
    **simulation_arguments, **transfer_function_arguments
)

# Display transfer function
viewer = napari.Viewer()
zyx_scale = np.array(
    [
        z_pixel_size,
        simulation_arguments["yx_pixel_size"],
        simulation_arguments["yx_pixel_size"],
    ]
)
isotropic_thin_3d.visualize_transfer_function(
    viewer,
    absorption_2d_to_3d_transfer_function,
    phase_2d_to_3d_transfer_function,
)
input("Showing OTFs. Press <enter> to continue...")
viewer.layers.select_all()
viewer.layers.remove_selected()

# Simulate
zyx_data = isotropic_thin_3d.apply_transfer_function(
    yx_absorption,
    yx_phase,
    absorption_2d_to_3d_transfer_function,
    phase_2d_to_3d_transfer_function,
)

# Reconstruct
(
    yx_absorption_recon,
    yx_phase_recon,
) = isotropic_thin_3d.apply_inverse_transfer_function(
    zyx_data,
    absorption_2d_to_3d_transfer_function,
    phase_2d_to_3d_transfer_function,
)

# Display
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
