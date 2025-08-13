# Fluorescent thin object 3D simulation and reconstruction

import napari
import numpy as np

from waveorder.models import isotropic_fluorescent_thin_3d

# Parameters
# all lengths must use consistent units e.g. um
simulation_arguments = {
    "yx_shape": (256, 256),
    "yx_pixel_size": 6.5 / 63,
}
phantom_arguments = {"sphere_radius": 5}
z_shape = 100
z_pixel_size = 0.25
zyx_scale = np.array(
    [
        z_pixel_size,
        simulation_arguments["yx_pixel_size"],
        simulation_arguments["yx_pixel_size"],
    ]
)
transfer_function_arguments = {
    "z_position_list": (np.arange(z_shape) - z_shape // 2) * z_pixel_size,
    "wavelength_emission": 0.532,
    "index_of_refraction_media": 1.3,
    "numerical_aperture_detection": 1.2,
}

# Create a fluorescent phantom (2D thin object)
yx_fluorescence_density = isotropic_fluorescent_thin_3d.generate_test_phantom(
    **simulation_arguments, **phantom_arguments
)

# Calculate transfer function
fluorescent_2d_to_3d_transfer_function = (
    isotropic_fluorescent_thin_3d.calculate_transfer_function(
        **simulation_arguments, **transfer_function_arguments
    )
)

# Calculate singular system
singular_system = isotropic_fluorescent_thin_3d.calculate_singular_system(
    fluorescent_2d_to_3d_transfer_function
)

# Display transfer function
viewer = napari.Viewer()
isotropic_fluorescent_thin_3d.visualize_transfer_function(
    viewer,
    fluorescent_2d_to_3d_transfer_function,
    zyx_scale,
)
input("Showing fluorescent OTF. Press <enter> to continue...")
viewer.layers.select_all()
viewer.layers.remove_selected()

# Simulate fluorescent imaging
zyx_data = isotropic_fluorescent_thin_3d.apply_transfer_function(
    yx_fluorescence_density,
    fluorescent_2d_to_3d_transfer_function,
)

# Reconstruct fluorescence density
yx_fluorescence_recon = (
    isotropic_fluorescent_thin_3d.apply_inverse_transfer_function(
        zyx_data,
        singular_system,
        regularization_strength=1e-2,
    )
)

# Display results
arrays = [
    (yx_fluorescence_density, "Phantom - fluorescence density"),
    (zyx_data, "Data - defocus stack"),
    (yx_fluorescence_recon, "Reconstruction - fluorescence density"),
]

for array in arrays:
    scale = zyx_scale[1:] if array[0].ndim == 2 else zyx_scale
    viewer.add_image(array[0].cpu().numpy(), name=array[1], scale=scale)

viewer.grid.enabled = True
viewer.dims.current_step = (z_shape // 2, 0, 0)

input("Showing object, data, and reconstruction. Press <enter> to quit...")
