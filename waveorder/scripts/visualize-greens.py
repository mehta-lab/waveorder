import napari
from napari.experimental import link_layers
import numpy as np
import torch
import os
from waveorder import util, optics
from scipy.ndimage import gaussian_filter

# Parameters
# all lengths must use consistent units e.g. um
output_dirpath = "./greens_plots"
grid_size = 100
blur_width = 2
zyx_shape = 3 * (grid_size,)
yx_pixel_size = 6.5 / 63
z_pixel_size = 0.15
wavelength_illumination = 0.532
index_of_refraction_media = 1.3

# Calculate coordinate grids
zyx_pixel_size = (z_pixel_size, yx_pixel_size, yx_pixel_size)
y_frequencies, x_frequencies = util.generate_frequencies(
    zyx_shape[1:], yx_pixel_size
)
radial_frequencies = torch.sqrt(x_frequencies**2 + y_frequencies**2)
z_position_list = torch.fft.ifftshift(
    (torch.arange(zyx_shape[0]) - zyx_shape[0] // 2) * z_pixel_size
)
z_frequencies = torch.fft.fftfreq(zyx_shape[0], d=z_pixel_size)

freq_shape = z_position_list.shape + x_frequencies.shape
z_broadcast = torch.broadcast_to(z_frequencies[:, None, None], freq_shape)
y_broadcast = torch.broadcast_to(y_frequencies[None, :, :], freq_shape)
x_broadcast = torch.broadcast_to(x_frequencies[None, :, :], freq_shape)
nu_rr = torch.sqrt(z_broadcast**2 + y_broadcast**2 + x_broadcast**2)

freq_voxel_size = [1 / (d * n) for d, n in zip(zyx_pixel_size, zyx_shape)]

# Calculate Greens tensor spectrum
G_3D = optics.generate_greens_tensor_spectrum(
    zyx_shape=zyx_shape,
    zyx_pixel_size=zyx_pixel_size,
    wavelength=wavelength_illumination / index_of_refraction_media,
)

# Mask to zero outside of a spherical shell
wavelength = wavelength_illumination / index_of_refraction_media
nu_max = (33 / 32) / (wavelength)
nu_min = (31 / 32) / (wavelength)
mask = torch.logical_and(nu_rr < nu_max, nu_rr > nu_min)
G_3D *= mask

# Split into positve and negative imaginary parts
G3D_imag = torch.imag(torch.fft.fftshift(G_3D, dim=(-3, -2, -1)))
G_pos = G3D_imag * (G3D_imag > 0)
G_neg = G3D_imag * (G3D_imag < 0)

# Blur to reduce aliasing
sigma = (
    0,
    0,
) + 3 * (blur_width,)
G_pos = gaussian_filter(np.array(G_pos), sigma=sigma)
G_neg = gaussian_filter(np.array(G_neg), sigma=sigma)

# Add to napari
viewer = napari.Viewer()

settings = [
    (slice(grid_size // 2, None), "botton", True),
    (slice(None, grid_size // 2), "top", False),
]
for my_slice, name, visible in settings:
    G_pos_copy = np.array(G_pos)
    G_neg_copy = np.array(G_neg)

    G_pos_copy[:, :, my_slice] = 0
    G_neg_copy[:, :, my_slice] = 0

    viewer.add_image(
        -G_neg_copy,
        colormap="I Purple",
        scale=freq_voxel_size,
        contrast_limits=(0, 0.1),
        blending="minimum",
        rendering="mip",
        name=name + "-negative",
        visible=visible,
    )
    viewer.add_image(
        G_pos_copy,
        colormap="greens",
        scale=freq_voxel_size,
        contrast_limits=(0, 0.1),
        blending="minimum",
        rendering="mip",
        name=name + "-positive",
        visible=visible,
    )
    link_layers(viewer.layers[-2:])

viewer.theme = "light"
viewer.dims.ndisplay = 3
viewer.camera.set_view_direction(
    view_direction=[1, 1, 1], up_direction=[1, 0, 0]
)
viewer.camera.zoom = 100

input("Press <enter> to save screenshots...")

# Save screenshots
os.makedirs(output_dirpath, exist_ok=True)


def screenshots_to_folder(folder):
    out_folder = os.path.join(output_dirpath, folder)
    os.makedirs(out_folder, exist_ok=True)
    for i in range(3):
        for j in range(3):
            viewer.dims.current_step = (i, j, 0, 0, 0)
            viewer.screenshot(
                os.path.join(out_folder, f"{i}_{j}.png"), scale=2
            )


screenshots_to_folder("bottoms")
viewer.layers[0].visible = True
viewer.layers[-1].visible = False
screenshots_to_folder("tops")
