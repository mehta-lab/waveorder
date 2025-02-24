import os

import napari
import numpy as np
import torch
from napari.experimental import link_layers
from scipy.ndimage import gaussian_filter
from skimage import measure

from waveorder import optics, util

# Parameters
# all lengths must use consistent units e.g. um
output_dirpath = "./greens_plots"
os.makedirs(output_dirpath, exist_ok=True)
grid_size = 100  # 300 for publication
blur_width = grid_size // 35  # blurring to smooth sharp corners
zyx_shape = 3 * (grid_size,)
yx_pixel_size = 6.5 / 63
z_pixel_size = 6.5 / 63
wavelength_illumination = 0.532
index_of_refraction_media = 1.3
threshold = 0.5  # for marching cubes


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

viewer.theme = "light"
viewer.dims.ndisplay = 3
viewer.camera.zoom = 100

for i in range(3):
    for j in range(3):
        name = f"{i}_{j}"

        volume = G_pos[i, j]
        verts, faces, normals, _ = measure.marching_cubes(
            volume, level=threshold * np.max(volume)
        )
        viewer.add_surface(
            (verts, faces),
            name=name + "-positive-surface",
            colormap="greens",
            scale=freq_voxel_size,
            shading="smooth",
        )

        volume = -G_neg[i, j]
        if i != j:
            verts, faces, normals, _ = measure.marching_cubes(
                volume, level=threshold * np.max(volume)
            )
            viewer.add_surface(
                (verts, faces),
                opacity=1.0,
                name=name + "-negative-surface",
                colormap="I Purple",
                scale=freq_voxel_size,
                blending="translucent",
                shading="smooth",
            )
        else:
            viewer.add_surface(
                (
                    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                    np.array([[0, 1, 2]]),
                ),
                opacity=1.0,
                name=name + "-dummy-surface",
                colormap="gray",
                scale=freq_voxel_size,
                blending="translucent",
                shading="smooth",
            )
        link_layers(viewer.layers[-2:])

        print(f"Screenshotting {i}_{j}")
        viewer.camera.set_view_direction(
            view_direction=[-1, -1, -1], up_direction=[0, 0, 1]
        )
        viewer.screenshot(
            os.path.join(output_dirpath, f"{i}_{j}.png"), scale=2
        )
        viewer.layers[-1].visible = False
        viewer.layers[-2].visible = False

# Show in complete grid
for layer in viewer.layers:
    layer.visible = True
viewer.grid.enabled = True
viewer.grid.stride = 2
viewer.grid.shape = (-1, 3)
viewer.theme = "dark"
napari.run()
