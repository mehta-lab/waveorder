import torch
import napari

from waveorder.models import (
    inplane_oriented_thick_pol3d_vector,
)

# Parameters
# all lengths must use consistent units e.g. um
zyx_shape = (101, 256, 256)
swing = 0.1
scheme = "5-State"
yx_pixel_size = 6.5 / 63
z_pixel_size = 0.15
wavelength_illumination = 0.532
z_padding = 0
index_of_refraction_media = 1.3
numerical_aperture_illumination = 0.5
numerical_aperture_detection = 1.2
fourier_oversample_factor = 1

# Create a phantom
fzyx_object = inplane_oriented_thick_pol3d_vector.generate_test_phantom(
    zyx_shape
)

# Calculate transfer function
sfZYX_transfer_function, intensity_to_stokes_matrix = (
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
        fourier_oversample_factor=fourier_oversample_factor,
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

singular_system = (
    inplane_oriented_thick_pol3d_vector.calculate_singular_system(
        sfZYX_transfer_function
    )
)

# Simulate
szyx_data = inplane_oriented_thick_pol3d_vector.apply_transfer_function(
    fzyx_object,
    sfZYX_transfer_function,
    intensity_to_stokes_matrix,
)

# from waveorder.visuals.napari_visuals import add_transfer_function_to_viewer

# add_transfer_function_to_viewer(
#     viewer,
#     singular_system[1],
#     zyx_scale=(z_pixel_size, yx_pixel_size, yx_pixel_size),
#     layer_name="Singular Values",
# )
# import pdb; pdb.set_trace()


# Display
arrays = [
    (szyx_data, "Data"),
    (fzyx_object, "Object"),
]

for array in arrays:
    viewer.add_image(torch.real(array[0]).cpu().numpy(), name=array[1])


# Reconstruct
for reg_strength in [0.005, 0.008, 0.01, 0.05, 0.1]:
    fzyx_object_recon = (
        inplane_oriented_thick_pol3d_vector.apply_inverse_transfer_function(
            szyx_data,
            singular_system,
            intensity_to_stokes_matrix,
            regularization_strength=reg_strength,
        )
    )
    viewer.add_image(
        torch.real(fzyx_object_recon).cpu().numpy(),
        name=f"Object - recon, reg_strength={reg_strength}",
    )

viewer.grid.enabled = True
viewer.grid.shape = (2, 5)
import pdb

pdb.set_trace()
input("Showing object, data, and recon. Press <enter> to quit...")
