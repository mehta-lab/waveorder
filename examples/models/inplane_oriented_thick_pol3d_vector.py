import torch
import napari

from waveorder.models import (
    inplane_oriented_thick_pol3d,
    inplane_oriented_thick_pol3d_vector,
)

# Parameters
# all lengths must use consistent units e.g. um
oversample_factor = 2
zyx_shape = (50, 128, 128) # (100, 256, 256)
swing = 0.1
scheme = "5-State"
yx_pixel_size = 0.325 / oversample_factor  # 0.325
z_pixel_size = 2.0 / oversample_factor  # 2.0
wavelength_illumination = 0.532
z_padding = 0
index_of_refraction_media = 1.0
numerical_aperture_illumination = 0.4
numerical_aperture_detection = 0.55

# Create a phantom
fzyx_object = inplane_oriented_thick_pol3d_vector.generate_test_phantom(
    zyx_shape
)

# Calculate transfer function
singular_system, sfZYX_transfer_function, intensity_to_stokes_matrix = (
    inplane_oriented_thick_pol3d_vector.calculate_transfer_function(
        swing,
        scheme,
        zyx_shape,
        yx_pixel_size,
        z_pixel_size,
        wavelength_illumination,
        z_padding,
        index_of_refraction_media,
        numerical_aperture_illumination,
        numerical_aperture_detection,
    )
)

# Display transfer function
viewer = napari.Viewer()
inplane_oriented_thick_pol3d_vector.visualize_transfer_function(
    viewer,
    sfZYX_transfer_function,
    zyx_scale=(z_pixel_size, yx_pixel_size, yx_pixel_size),
)

input("Showing transfer functions. Press <enter> to continue...")
viewer.layers.select_all()
viewer.layers.remove_selected()

# Simulate
szyx_data = inplane_oriented_thick_pol3d_vector.apply_transfer_function(
    fzyx_object,
    sfZYX_transfer_function,
    intensity_to_stokes_matrix,
)

# Reconstruct
fzyx_object_recon = (
    inplane_oriented_thick_pol3d_vector.apply_inverse_transfer_function(
        szyx_data,
        singular_system,
        intensity_to_stokes_matrix,
        regularization_strength=1e-1,
    )
)

# Display
arrays = [
    (fzyx_object_recon, "Object - recon"),
    (szyx_data, "Data"),
    (fzyx_object, "Object"),
]

for array in arrays:
    viewer.add_image(torch.real(array[0]).cpu().numpy(), name=array[1])

# viewer.grid.enabled = True
# viewer.grid.shape = (2, 5)
input("Showing object, data, and recon. Press <enter> to quit...")
