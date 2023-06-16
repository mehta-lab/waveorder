import napari
import numpy as np

from waveorder.models import isotropic_fluorescent_thick_3d

# Parameters
# all lengths must use consistent units e.g. um
simulation_arguments = {
    "zyx_shape": (100, 256, 256),
    "yx_pixel_size": 6.5 / 63,
    "z_pixel_size": 0.25,
}
phantom_arguments = {"sphere_radius": 5}
transfer_function_arguments = {
    "wavelength_illumination": 0.532,
    "z_padding": 0,
    "index_of_refraction_media": 1.3,
    "numerical_aperture_detection": 1.2,
}

# Create a phantom
zyx_fluorescence_density = (
    isotropic_fluorescent_thick_3d.generate_test_phantom(
        **simulation_arguments, **phantom_arguments
    )
)

# Calculate transfer function
optical_transfer_function = (
    isotropic_fluorescent_thick_3d.calculate_transfer_function(
        **simulation_arguments, **transfer_function_arguments
    )
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
isotropic_fluorescent_thick_3d.visualize_transfer_function(
    viewer,
    optical_transfer_function,
    zyx_scale,
)
input("Showing OTFs. Press <enter> to continue...")
viewer.layers.select_all()
viewer.layers.remove_selected()

# Simulate
zyx_data = isotropic_fluorescent_thick_3d.apply_transfer_function(
    zyx_fluorescence_density,
    optical_transfer_function,
    transfer_function_arguments["z_padding"],
)

# Reconstruct
zyx_recon = isotropic_fluorescent_thick_3d.apply_inverse_transfer_function(
    zyx_data,
    optical_transfer_function,
    transfer_function_arguments["z_padding"],
)

# Display
viewer.add_image(
    zyx_fluorescence_density.numpy(), name="Phantom", scale=zyx_scale
)
viewer.add_image(zyx_data.numpy(), name="Data", scale=zyx_scale)
viewer.add_image(zyx_recon.numpy(), name="Reconstruction", scale=zyx_scale)
input("Showing object, data, and recon. Press <enter> to quit...")
