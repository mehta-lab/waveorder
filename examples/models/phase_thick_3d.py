# 3D partially coherent optical diffraction tomography (ODT) simulation
# J. M. Soto, J. A. Rodrigo, and T. Alieva, "Label-free quantitative
# 3D tomographic imaging for partially coherent light microscopy," Opt. Express
# 25, 15699-15712 (2017)

import napari
import numpy as np
from waveorder import util
from waveorder.models import phase_thick_3d

# Parameters
# all lengths must use consistent units e.g. um
simulation_arguments = {
    "zyx_shape": (100, 256, 256),
    "yx_pixel_size": 6.5 / 63,
    "z_pixel_size": 0.25,
    "wavelength_illumination": 0.532,
    "index_of_refraction_media": 1.3,
}
phantom_arguments = {"index_of_refraction_sample": 1.50, "sphere_radius": 5}
transfer_function_arguments = {
    "z_padding": 0,
    "numerical_aperture_illumination": 0.9,
    "numerical_aperture_detection": 1.2,
}

# Create a phantom
zyx_phase = phase_thick_3d.generate_test_phantom(
    **simulation_arguments, **phantom_arguments
)

# Calculate transfer function
(
    real_potential_transfer_function,
    imag_potential_transfer_function,
) = phase_thick_3d.calculate_transfer_function(
    **simulation_arguments, **transfer_function_arguments
)

# Display transfer function
viewer = napari.Viewer()
zyx_scale = np.array(
    [
        simulation_arguments["z_pixel_size"],
        simulation_arguments["yx_pixel_size"],
        simulation_arguments["yx_pixel_size"],
    ]
)
phase_thick_3d.visualize_transfer_function(
    viewer,
    real_potential_transfer_function,
    imag_potential_transfer_function,
    zyx_scale,
)
input("Showing OTFs. Press <enter> to continue...")
viewer.layers.select_all()
viewer.layers.remove_selected()

# Simulate
zyx_data = phase_thick_3d.apply_transfer_function(
    zyx_phase,
    real_potential_transfer_function,
    transfer_function_arguments["z_padding"],
)

# Reconstruct
zyx_recon = phase_thick_3d.apply_inverse_transfer_function(
    zyx_data,
    real_potential_transfer_function,
    imag_potential_transfer_function,
    transfer_function_arguments["z_padding"],
    simulation_arguments["z_pixel_size"],
    simulation_arguments["wavelength_illumination"],
)

# Display
viewer.add_image(zyx_phase.numpy(), name="Phantom", scale=zyx_scale)
viewer.add_image(zyx_data.numpy(), name="Data", scale=zyx_scale)
viewer.add_image(zyx_recon.numpy(), name="Reconstruction", scale=zyx_scale)
input("Showing object, data, and recon. Press <enter> to quit...")
