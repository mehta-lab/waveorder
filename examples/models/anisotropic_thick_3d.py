import torch
import napari
import numpy as np
from waveorder import optics, util
from waveorder.models import phase_thick_3d

# Parameters
# all lengths must use consistent units e.g. um
margin = 50
simulation_arguments = {
    "zyx_shape": (129, 256, 256),
    "yx_pixel_size": 6.5 / 65,
    "z_pixel_size": 0.1,
    "index_of_refraction_media": 1.25,
}
# phantom_arguments = {"index_of_refraction_sample": 1.50, "sphere_radius": 5}
transfer_function_arguments = {
    "z_padding": 0,
    "wavelength_illumination": 0.5,
    "numerical_aperture_illumination": 0.75,  # 75,
    "numerical_aperture_detection": 1.0,
}
input_jones = torch.tensor([0.0 + 1.0j, 1.0 + 0j])

# # Create a phantom
# zyx_phase = phase_thick_3d.generate_test_phantom(
#     **simulation_arguments, **phantom_arguments
# )

# Convert
zyx_shape = simulation_arguments["zyx_shape"]
yx_pixel_size = simulation_arguments["yx_pixel_size"]
z_pixel_size = simulation_arguments["z_pixel_size"]
index_of_refraction_media = simulation_arguments["index_of_refraction_media"]
z_padding = transfer_function_arguments["z_padding"]
wavelength_illumination = transfer_function_arguments[
    "wavelength_illumination"
]
numerical_aperture_illumination = transfer_function_arguments[
    "numerical_aperture_illumination"
]
numerical_aperture_detection = transfer_function_arguments[
    "numerical_aperture_detection"
]

# Precalculations
z_total = zyx_shape[0] + 2 * z_padding
z_position_list = torch.fft.ifftshift(
    (torch.arange(z_total) - z_total // 2) * z_pixel_size
)

# Calculate frequencies
y_frequencies, x_frequencies = util.generate_frequencies(
    zyx_shape[1:], yx_pixel_size
)
radial_frequencies = np.sqrt(x_frequencies**2 + y_frequencies**2)

# 2D pupils
ill_pupil = optics.generate_pupil(
    radial_frequencies,
    numerical_aperture_illumination,
    wavelength_illumination,
)
det_pupil = optics.generate_pupil(
    radial_frequencies,
    numerical_aperture_detection,
    wavelength_illumination,
)
pupil = optics.generate_pupil(
    radial_frequencies,
    index_of_refraction_media,  # largest possible NA
    wavelength_illumination,
)

# Defocus pupils
defocus_pupil = optics.generate_propagation_kernel(
    radial_frequencies,
    pupil,
    wavelength_illumination / index_of_refraction_media,
    z_position_list,
)

greens_functions_z = optics.generate_greens_function_z(
    radial_frequencies,
    pupil,
    wavelength_illumination / index_of_refraction_media,
    z_position_list,
)

# Calculate vector defocus pupils
S = optics.generate_vector_source_defocus_pupil(
    x_frequencies,
    y_frequencies,
    z_position_list,
    defocus_pupil,
    input_jones,
    ill_pupil,
    wavelength_illumination / index_of_refraction_media,
)

# Simplified scalar pupil
sP = optics.generate_propagation_kernel(
    radial_frequencies,
    det_pupil,
    wavelength_illumination / index_of_refraction_media,
    z_position_list,
)

P = optics.generate_vector_detection_defocus_pupil(
    x_frequencies,
    y_frequencies,
    z_position_list,
    defocus_pupil,
    det_pupil,
    wavelength_illumination / index_of_refraction_media,
)

G = optics.generate_defocus_greens_tensor(
    x_frequencies,
    y_frequencies,
    greens_functions_z,
    pupil,
    lambda_in=wavelength_illumination / index_of_refraction_media,
)

# window = torch.fft.ifftshift(
#    torch.hann_window(z_position_list.shape[0], periodic=False)
# )

# ###### LATEST

# # abs() and *(1j) are hacks to correct for tricky phase shifts
# P_3D = torch.abs(torch.fft.ifft(P, dim=-3)).type(torch.complex64)
# G_3D = torch.abs(torch.fft.ifft(G, dim=-3)) * (-1j)
# S_3D = torch.fft.ifft(S, dim=-3)

# # Normalize
# P_3D /= torch.amax(torch.abs(P_3D))
# G_3D /= torch.amax(torch.abs(G_3D))
# S_3D /= torch.amax(torch.abs(S_3D))

# # Main part
# PG_3D = torch.einsum("ijzyx,jpzyx->ipzyx", P_3D, G_3D)
# PS_3D = torch.einsum("jlzyx,lzyx,kzyx->jlzyx", P_3D, S_3D, torch.conj(S_3D))

# # PG_3D /= torch.amax(torch.abs(PG_3D))
# # PS_3D /= torch.amax(torch.abs(PS_3D))

# pg = torch.fft.fftn(PG_3D, dim=(-3, -2, -1))
# ps = torch.fft.fftn(PS_3D, dim=(-3, -2, -1))

# H1 = torch.fft.ifftn(
#     torch.einsum("ipzyx,jkzyx->ijpkzyx", pg, torch.conj(ps)),
#     dim=(-3, -2, -1),
# )

# H2 = torch.fft.ifftn(
#     torch.einsum("ikzyx,jpzyx->ijpkzyx", ps, torch.conj(pg)),
#     dim=(-3, -2, -1),
# )

# MAY 12 Simplified
P_3D = torch.abs(torch.fft.ifft(sP, dim=-3)).type(torch.complex64)
G_3D = torch.abs(torch.fft.ifft(G, dim=-3)) * (-1j)
S_3D = torch.fft.ifft(S, dim=-3)

# Normalize
P_3D /= torch.amax(torch.abs(P_3D))
G_3D /= torch.amax(torch.abs(G_3D))
S_3D /= torch.amax(torch.abs(S_3D))

# Main part
PG_3D = torch.einsum("zyx,ipzyx->ipzyx", P_3D, G_3D)
PS_3D = torch.einsum("zyx,jzyx,kzyx->jkzyx", P_3D, S_3D, torch.conj(S_3D))

PG_3D /= torch.amax(torch.abs(PG_3D))
PS_3D /= torch.amax(torch.abs(PS_3D))

pg = torch.fft.fftn(PG_3D, dim=(-3, -2, -1))
ps = torch.fft.fftn(PS_3D, dim=(-3, -2, -1))

H1 = torch.fft.ifftn(
    torch.einsum("ipzyx,jkzyx->ijpkzyx", pg, torch.conj(ps)),
    dim=(-3, -2, -1),
)

H2 = torch.fft.ifftn(
    torch.einsum("ikzyx,jpzyx->ijpkzyx", ps, torch.conj(pg)),
    dim=(-3, -2, -1),
)

H_re = H1[1:, 1:] + H2[1:, 1:]
# H_im = 1j * (H1 - H2)

s = util.pauli()
Y = util.gellmann()

H_re_stokes = torch.einsum("sik,ikpjzyx,lpj->slzyx", s, H_re, Y)

print("H_re_stokes: (RE, IM, ABS)")
torch.set_printoptions(precision=1)
print(torch.log10(torch.sum(torch.real(H_re_stokes) ** 2, dim=(-3, -2, -1))))
print(torch.log10(torch.sum(torch.imag(H_re_stokes) ** 2, dim=(-3, -2, -1))))
print(torch.log10(torch.sum(torch.abs(H_re_stokes) ** 2, dim=(-3, -2, -1))))

# Display transfer function
v = napari.Viewer()


def view_transfer_function(
    transfer_function,
):
    shift_dims = (-3, -2, -1)
    lim = 1e-3
    zyx_scale = np.array(
        [
            zyx_shape[0] * z_pixel_size,
            zyx_shape[1] * yx_pixel_size,
            zyx_shape[2] * yx_pixel_size,
        ]
    )

    v.add_image(
        torch.fft.ifftshift(torch.real(transfer_function), dim=shift_dims)
        .cpu()
        .numpy(),
        colormap="bwr",
        contrast_limits=(-lim, lim),
        scale=1 / zyx_scale,
    )
    if transfer_function.dtype == torch.complex64:
        v.add_image(
            torch.fft.ifftshift(torch.imag(transfer_function), dim=shift_dims)
            .cpu()
            .numpy(),
            colormap="bwr",
            contrast_limits=(-lim, lim),
            scale=1 / zyx_scale,
        )

    # v.dims.order = (2, 1, 0)


# view_transfer_function(H_re_stokes)
# view_transfer_function(G_3D)
# view_transfer_function(H_re)
# view_transfer_function(P_3D)

# PLOT transfer function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_data(data, y_slices, filename):
    fig, axs = plt.subplots(4, 9, figsize=(20, 10))  # Adjust the size as needed

    for i in range(data.shape[0]):  # Stokes parameter
        for j in range(data.shape[1]):  # Object parameter
            for k, y in enumerate(y_slices):  # Y slices
                z = data[i, j, :, y, :]
                hue = np.angle(z) / (2 * np.pi) + 0.5  # Normalize and shift to make red at 0
                sat = np.abs(z) / np.amax(np.abs(z))
                hsv = np.stack((hue, sat, np.ones_like(sat)), axis=-1)
                rgb = mcolors.hsv_to_rgb(hsv)
                
                ax = axs[i, j]
                ax.imshow(rgb, aspect='auto')
                ax.set_title('')  # Remove titles
                ax.set_xticks([])  # Remove x-axis ticks
                ax.set_yticks([])  # Remove y-axis ticks
                ax.spines['top'].set_visible(False)  # Hide top spine
                ax.spines['right'].set_visible(False)  # Hide right spine
                ax.spines['bottom'].set_visible(False)  # Hide bottom spine
                ax.spines['left'].set_visible(False)  # Hide left spine
                ax.set_xlabel('')  # Remove x-axis labels

    plt.tight_layout()
    plt.savefig(filename, format='pdf')

# Adjust y_slices according to your index base (check if your array index starts at 0)
y_center = 128 # Assuming the middle index for Y dimension
y_slices = [y_center - 10, y_center, y_center + 10]
plot_data(torch.fft.ifftshift(H_re_stokes, dim=(-3, -2, -1)).numpy(), y_slices, './output.pdf')

# Simulate
yx_star, yx_theta, _ = util.generate_star_target(
    yx_shape=zyx_shape[1:],
    blur_px=1,
    margin=margin,
)
c00 = yx_star
c2_2 = -torch.sin(2 * yx_theta) * yx_star
c22 = torch.cos(2 * yx_theta) * yx_star

# Put in in a center slices of a 3D object
center_slice_object = torch.stack((c00, c2_2, c22), dim=0)
object = torch.zeros((3,) + zyx_shape)
object[:, zyx_shape[0] // 2, ...] = center_slice_object

# Simulate
object_spectrum = torch.fft.fftn(object, dim=(-3, -2, -1))
data_spectrum = torch.einsum(
    "slzyx,lzyx->szyx", H_re_stokes[:, (0, 4, 8), ...], object_spectrum
)
data = torch.fft.ifftn(data_spectrum, dim=(-3, -2, -1))

v.add_image(object.numpy())
v.add_image(torch.real(data).numpy())
v.add_image(torch.imag(data).numpy())

import pdb

pdb.set_trace()


zyx_data = phase_thick_3d.apply_transfer_function(
    zyx_phase,
    real_potential_transfer_function,
    transfer_function_arguments["z_padding"],
    brightness=1e3,
)

# Reconstruct
zyx_recon = phase_thick_3d.apply_inverse_transfer_function(
    zyx_data,
    real_potential_transfer_function,
    imag_potential_transfer_function,
    transfer_function_arguments["z_padding"],
)

# Display
viewer.add_image(zyx_phase.numpy(), name="Phantom", scale=zyx_scale)
viewer.add_image(zyx_data.numpy(), name="Data", scale=zyx_scale)
viewer.add_image(zyx_recon.numpy(), name="Reconstruction", scale=zyx_scale)
input("Showing object, data, and recon. Press <enter> to quit...")

# %%
