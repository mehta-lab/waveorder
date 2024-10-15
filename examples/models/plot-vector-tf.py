import torch
import os
import numpy as np
from waveorder import util, optics
from waveorder.visuals.matplotlib_visuals import plot_5d_ortho

output_folder = "2024-10-14"
os.makedirs(output_folder, exist_ok=True)

# Parameters
# all lengths must use consistent units e.g. um
zyx_shape = (101, 128, 128)  # (101, 256, 256)
swing = 0.1
scheme = "5-State"
yx_pixel_size = 6.5 / 63
z_pixel_size = 0.15
wavelength_illumination = 0.532
z_padding = 0
index_of_refraction_media = 1.3
numerical_aperture_detection = 1.0

for i, numerical_aperture_illumination in enumerate([0.01, 0.75]):
    file_suffix = str(i)

    input_jones = torch.tensor([-1j, 1.0 + 0j]) / torch.sqrt(
        torch.tensor([2])
    )  # circular

    # input_jones = torch.tensor([0, 1.0 + 0j])

    # Calculate frequencies
    y_frequencies, x_frequencies = util.generate_frequencies(
        zyx_shape[1:], yx_pixel_size
    )
    radial_frequencies = torch.sqrt(x_frequencies**2 + y_frequencies**2)

    z_total = zyx_shape[0] + 2 * z_padding
    z_position_list = torch.fft.ifftshift(
        (torch.arange(z_total) - z_total // 2) * z_pixel_size
    )
    z_frequencies = torch.fft.fftfreq(z_total, d=z_pixel_size)

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
        axially_even=True,
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
    P = optics.generate_propagation_kernel(
        radial_frequencies,
        det_pupil,
        wavelength_illumination / index_of_refraction_media,
        z_position_list,
    )


    P_3D = torch.abs(torch.fft.ifft(P, dim=-3)).type(torch.complex64)
    S_3D = torch.fft.ifft(S, dim=-3)

    ## CANDIDATE FOR REMOVAL
    # cleanup some ringing
    freq_shape = z_position_list.shape + x_frequencies.shape

    z_broadcast = torch.broadcast_to(z_frequencies[:, None, None], freq_shape)
    y_broadcast = torch.broadcast_to(y_frequencies[None, :, :], freq_shape)
    x_broadcast = torch.broadcast_to(x_frequencies[None, :, :], freq_shape)

    nu_rr = torch.sqrt(z_broadcast**2 + y_broadcast**2 + x_broadcast**2)
    wavelength = wavelength_illumination / index_of_refraction_media
    nu_max = (33 / 32) / (wavelength)
    nu_min = (31 / 32) / (wavelength)

    mask = torch.logical_and(nu_rr < nu_max, nu_rr > nu_min)

    P_3D *= mask
    S_3D *= mask

    ## <end> CANDIDATE FOR REMOVAL <end>

    # TODO REFACTOR: Direct green's tensor
    Z = z_total
    Y = zyx_shape[1]
    X = zyx_shape[2]

    z_step = torch.fft.ifftshift(
        (torch.arange(z_total) - z_total // 2) * z_pixel_size
    )
    y_step = torch.fft.ifftshift((torch.arange(Y) - Y // 2) * yx_pixel_size)
    x_step = torch.fft.ifftshift((torch.arange(X) - X // 2) * yx_pixel_size)

    zz = torch.broadcast_to(z_step[:, None, None], (Z, Y, X))
    yy = torch.broadcast_to(y_step[None, :, None], (Z, Y, X))
    xx = torch.broadcast_to(x_step[None, None, :], (Z, Y, X))

    rr = torch.sqrt(xx**2 + yy**2 + zz**2)
    rhat = torch.stack([zz, yy, xx], dim=0) / rr

    scalar_g = torch.exp(1j * 2 * torch.pi * rr / wavelength) / (
        4 * torch.pi * rr
    )

    eye = torch.zeros((3, 3, Z, Y, X))
    eye[0, 0] = 1
    eye[1, 1] = 1
    eye[2, 2] = 1

    Q = eye - torch.einsum("izyx,jzyx->ijzyx", rhat, rhat)
    g_3d = Q * scalar_g
    g_3d = torch.nan_to_num(g_3d)

    G_3D = torch.fft.fftn(g_3d, dim=(-3, -2, -1))
    G_3D = torch.imag(G_3D) * 1j
    G_3D /= torch.amax(torch.abs(G_3D))

    # Main transfer function calculation
    PG_3D = torch.einsum("zyx,ipzyx->ipzyx", P_3D, G_3D)
    PS_3D = torch.einsum("zyx,jzyx,kzyx->jkzyx", P_3D, S_3D, torch.conj(S_3D))

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
    
    # Not necessary yet
    illum_intensity = torch.sum(S_3D * torch.conj(S_3D), dim=0)
    direct_intensity = torch.real(
        torch.sum(illum_intensity * P_3D) * torch.sum(P_3D)
    )

    H_re = H1[1:, 1:] + H2[1:, 1:]  # drop data-side z components
    # H_im = 1j * (H1[1:, 1:] - H2[1:,1:]) # ignore absorptive terms

    # H_re /= torch.amax(torch.abs(H_re))

    s_labels = [0, 1, 2, 3]
    f_labels = [0, 4, 8]
    s = util.pauli()[s_labels]  # select s0, s1, and s2 (drop s3)
    Y = util.gellmann()[f_labels]
    # select phase f00 and transverse linear isotropic terms 2-2, and f22
    sfZYX_transfer_function = torch.einsum(
        "sik,ikpjzyx,lpj->slzyx", s, H_re, Y
    )

    sfzyx_point_response_function = torch.fft.fftn(
        sfZYX_transfer_function, dim=(-3, -2, -1)
    )

    # # Use this to visualize Green's tensor
    # from waveorder.visuals.napari_visuals import (
    #     add_transfer_function_to_viewer,
    # )
    # import napari

    # v = napari.Viewer()
    # add_transfer_function_to_viewer(
    #     v,
    #     sfZYX_transfer_function,
    #     zyx_scale=(z_pixel_size, yx_pixel_size, yx_pixel_size),
    #     complex_rgb=True,
    # )
    # import pdb
    # pdb.set_trace()

    plot_5d_ortho(
        g_3d,
        filename=os.path.join(output_folder, f"greens_{file_suffix}.pdf"),
        zyx_scale=(z_pixel_size, yx_pixel_size, yx_pixel_size),
        z_slice=0,
        row_labels=["Z", "Y", "X"],
        column_labels=["Z", "Y", "X"],
        rose_path=None,
        inches_per_column=1,
        saturate_clim_fraction=1.0,
        trim_edges=50,
        fourier_space=False,
    )

    plot_5d_ortho(
        G_3D,
        filename=os.path.join(output_folder, f"G_{file_suffix}.pdf"),
        zyx_scale=(z_pixel_size, yx_pixel_size, yx_pixel_size),
        z_slice=0,
        row_labels=["Z", "Y", "X"],
        column_labels=["Z", "Y", "X"],
        rose_path=None,
        inches_per_column=1,
        saturate_clim_fraction=0.7,
        trim_edges=0,
    )

    plot_5d_ortho(
        S_3D[1:][None],
        filename=os.path.join(output_folder, f"source_{file_suffix}.pdf"),
        zyx_scale=(z_pixel_size, yx_pixel_size, yx_pixel_size),
        z_slice=-35,
        row_labels=[""],
        column_labels=["Y", "X"],
        rose_path=None,
        inches_per_column=1,
        saturate_clim_fraction=0.75,
        trim_edges=0,
        fourier_space=True,
    )

    plot_5d_ortho(
        sfZYX_transfer_function,
        filename=os.path.join(output_folder, f"tf_{file_suffix}.pdf"),
        zyx_scale=(z_pixel_size, yx_pixel_size, yx_pixel_size),
        z_slice=-5,
        row_labels=s_labels,
        column_labels=f_labels,
        rose_path=None,
        inches_per_column=1,
        saturate_clim_fraction=0.2,
        trim_edges=40,
    )

    plot_5d_ortho(
        sfzyx_point_response_function,
        filename=os.path.join(output_folder, f"psf_{file_suffix}.pdf"),
        zyx_scale=(z_pixel_size, yx_pixel_size, yx_pixel_size),
        z_slice=-10,
        row_labels=s_labels,
        column_labels=f_labels,
        rose_path=None,
        inches_per_column=1,
        saturate_clim_fraction=0.2,
        trim_edges=60,
    )
