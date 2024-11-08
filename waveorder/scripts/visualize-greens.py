# %%
import numpy as np
import torch
import os
from waveorder import util, optics
from waveorder.visuals.matplotlib_visuals import plot_5d_ortho
from waveorder.visuals.utils import complex_tensor_to_rgb
from scipy.ndimage import gaussian_filter

output_folder = "2024-11-07"
os.makedirs(output_folder, exist_ok=True)

# Parameters
# all lengths must use consistent units e.g. um
N = 350
blur_width = 2
zyx_shape = 3 * (N,)
swing = 0.1
scheme = "5-State"
yx_pixel_size = 6.5 / 63
z_pixel_size = 0.15
zyx_pixel_size = (z_pixel_size, yx_pixel_size, yx_pixel_size)
wavelength_illumination = 0.532
z_padding = 0
index_of_refraction_media = 1.3
numerical_aperture_detection = 1.2
numerical_aperture_illumination = 0.01

input_jones = torch.tensor([0.0 - 1.0j, 1.0 + 0j])  # circular

# Calculate frequencies
y_frequencies, x_frequencies = util.generate_frequencies(zyx_shape[1:], yx_pixel_size)
radial_frequencies = torch.sqrt(x_frequencies**2 + y_frequencies**2)

z_total = zyx_shape[0] + 2 * z_padding
z_position_list = torch.fft.ifftshift(
    (torch.arange(z_total) - z_total // 2) * z_pixel_size
)
z_frequencies = torch.fft.fftfreq(z_total, d=z_pixel_size)


G_3D = optics.generate_greens_tensor_spectrum(
    zyx_shape=(z_total, zyx_shape[1], zyx_shape[2]),
    zyx_pixel_size=(z_pixel_size, yx_pixel_size, yx_pixel_size),
    wavelength=wavelength_illumination / index_of_refraction_media,
)

freq_shape = z_position_list.shape + x_frequencies.shape

z_broadcast = torch.broadcast_to(z_frequencies[:, None, None], freq_shape)
y_broadcast = torch.broadcast_to(y_frequencies[None, :, :], freq_shape)
x_broadcast = torch.broadcast_to(x_frequencies[None, :, :], freq_shape)

nu_rr = torch.sqrt(z_broadcast**2 + y_broadcast**2 + x_broadcast**2)
wavelength = wavelength_illumination / index_of_refraction_media
nu_max = (33 / 32) / (wavelength)
nu_min = (31 / 32) / (wavelength)

mask = torch.logical_and(nu_rr < nu_max, nu_rr > nu_min)

G_3D *= mask

# Make plots
voxel_size = [1 / (d * n) for d, n in zip(zyx_pixel_size, zyx_shape)]

from waveorder.visuals.napari_visuals import add_transfer_function_to_viewer
import napari

v = napari.Viewer()
# add_transfer_function_to_viewer(
#     v, G_3D, zyx_pixel_size, layer_name="G", complex_rgb=True
# )


G3D_imag = torch.imag(torch.fft.fftshift(G_3D, dim=(-3, -2, -1)))
# G3D_imag[:, :, N // 2 :] = 0  # bottoms
G3D_imag[:, :, : N // 2] = 0 # tops
G_pos = G3D_imag * (G3D_imag > 0)
G_neg = G3D_imag * (G3D_imag < 0)

sigma = (
    0,
    0,
) + 3 * (blur_width,)
G_pos = gaussian_filter(np.array(G_pos), sigma=sigma)
G_neg = gaussian_filter(np.array(G_neg), sigma=sigma)

v.add_image(
    -G_neg,
    colormap="I Purple",
    scale=voxel_size,
    contrast_limits=(0, 0.1),
    blending="minimum",
    rendering="mip",
)
v.add_image(
    G_pos,
    colormap="greens",
    scale=voxel_size,
    contrast_limits=(0, 0.1),
    blending="minimum",
    rendering="mip",
)


v.theme = "light"
v.dims.ndisplay = 3
v.camera.set_view_direction(view_direction=[1, 1, 1], up_direction=[1, 0, 0])
v.camera.zoom = 100

# Create a new folder named
folder = os.path.join(output_folder, "tops") #"bottoms")
os.makedirs(folder, exist_ok=True)
for i in range(3):
    for j in range(3):
        v.dims.current_step = (i, j, 0, 0, 0)
        v.screenshot(os.path.join(folder, f"{i}_{j}.png"), scale=2)

import pdb

pdb.set_trace()
