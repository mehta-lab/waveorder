"""
phase thick 3d with sector illumination
========================================

# 3D phase reconstruction with oblique sector illumination
# This example demonstrates multi-channel phase reconstruction where each channel
# corresponds to a different illumination sector angle.
"""

import napari
import numpy as np
import torch

from waveorder.models import phase_thick_3d

# Parameters
# all lengths must use consistent units e.g. um
simulation_arguments = {
    "zyx_shape": (100, 256, 256),
    "yx_pixel_size": 6.5 / 63,
    "z_pixel_size": 0.25,
    "index_of_refraction_media": 1.3,
}
phantom_arguments = {"index_of_refraction_sample": 1.50, "sphere_radius": 5}
transfer_function_arguments = {
    "z_padding": 0,
    "wavelength_illumination": 0.532,
    "numerical_aperture_illumination": 0.9,
    "numerical_aperture_detection": 1.2,
}

# Define 9 sector illumination angles
# 8 sectors at 45-degree intervals + 1 full aperture
sector_angle = 45
illumination_sector_angles = [
    (i * sector_angle, (i + 1) * sector_angle) for i in range(8)
] + [(0, 360)]

print(f"Using {len(illumination_sector_angles)} illumination sectors")

# Create a phantom
zyx_phase = phase_thick_3d.generate_test_phantom(
    **simulation_arguments, **phantom_arguments
)

# Calculate multi-channel transfer function (one for each sector)
(
    real_potential_transfer_function,
    imag_potential_transfer_function,
) = phase_thick_3d.calculate_transfer_function(
    **simulation_arguments,
    **transfer_function_arguments,
    illumination_sector_angles=illumination_sector_angles,
)

print(
    f"Transfer function shape: {real_potential_transfer_function.shape}"
)  # Should be (C, Z, Y, X)

# Display complete multi-channel transfer function
viewer = napari.Viewer()
zyx_scale = np.array(
    [
        simulation_arguments["z_pixel_size"],
        simulation_arguments["yx_pixel_size"],
        simulation_arguments["yx_pixel_size"],
    ]
)

# Add full CZYX transfer function (imaginary part) as single 4D layer
# Match the visualization style from add_transfer_function_to_viewer
czyx_shape = imag_potential_transfer_function.shape
voxel_scale = np.array(
    [
        czyx_shape[1] * zyx_scale[0],  # Z extent
        czyx_shape[2] * zyx_scale[1],  # Y extent
        czyx_shape[3] * zyx_scale[2],  # X extent
    ]
)
lim = 0.5 * torch.max(torch.abs(imag_potential_transfer_function)).item()

viewer.add_image(
    torch.fft.ifftshift(
        torch.imag(imag_potential_transfer_function), dim=(-3, -2, -1)
    )
    .cpu()
    .numpy(),
    name="Imag pot. TF (CZYX)",
    colormap="bwr",
    contrast_limits=(-lim, lim),
    scale=(1,) + tuple(1 / voxel_scale),  # No scaling on C dimension
)

# Set up XZ view with C and Y as sliders
viewer.dims.order = [0, 2, 1, 3]  # (C, Y, Z, X) for XZ display
viewer.dims.current_step = (
    0,
    czyx_shape[1] // 2,
    czyx_shape[2] // 2,
    czyx_shape[3] // 2,
)

input(
    "Showing CZYX OTF in XZ view (use C and Y sliders). Press <enter> to continue..."
)
viewer.layers.select_all()
viewer.layers.remove_selected()

# Simulate multi-channel data (one channel per sector)
# In practice, these would come from your microscope as separate acquisitions
zyx_data_multi_channel = []
for c in range(len(illumination_sector_angles)):
    zyx_data_channel = phase_thick_3d.apply_transfer_function(
        zyx_phase,
        real_potential_transfer_function[c],
        transfer_function_arguments["z_padding"],
        brightness=1e3,
    )
    zyx_data_multi_channel.append(zyx_data_channel)

# Stack into (C, Z, Y, X) tensor
zyx_data_multi_channel = torch.stack(zyx_data_multi_channel, dim=0)
print(f"Multi-channel data shape: {zyx_data_multi_channel.shape}")

# Reconstruct phase from all channels combined
zyx_recon = phase_thick_3d.apply_inverse_transfer_function(
    zyx_data_multi_channel,
    real_potential_transfer_function,
    imag_potential_transfer_function,
    transfer_function_arguments["z_padding"],
)

# Display
viewer.add_image(zyx_phase.numpy(), name="Phantom", scale=zyx_scale)
viewer.add_image(
    zyx_data_multi_channel.numpy(),
    name="Data (CZYX)",
    scale=zyx_scale,
)
viewer.add_image(zyx_recon.numpy(), name="Reconstruction", scale=zyx_scale)

# Show comparison with single channel (full aperture) for reference
print("\nComparing with single-channel (full aperture) reconstruction...")
(
    real_tf_single,
    imag_tf_single,
) = phase_thick_3d.calculate_transfer_function(
    **simulation_arguments,
    **transfer_function_arguments,
    illumination_sector_angles=None,  # Full aperture
)
zyx_data_single = phase_thick_3d.apply_transfer_function(
    zyx_phase,
    real_tf_single[0],  # Single channel
    transfer_function_arguments["z_padding"],
    brightness=1e3,
)
zyx_recon_single = phase_thick_3d.apply_inverse_transfer_function(
    zyx_data_single[None, ...],  # Add channel dimension
    real_tf_single,
    imag_tf_single,
    transfer_function_arguments["z_padding"],
)
viewer.add_image(
    zyx_recon_single.numpy(),
    name="Reconstruction (single channel)",
    scale=zyx_scale,
)

print(
    f"\nReconstruction error (multi-channel): {torch.mean(torch.abs(zyx_recon - zyx_phase)).item():.6f}"
)
print(
    f"Reconstruction error (single channel): {torch.mean(torch.abs(zyx_recon_single - zyx_phase)).item():.6f}"
)

input("\nShowing phantom, data, and reconstructions. Press <enter> to quit...")
