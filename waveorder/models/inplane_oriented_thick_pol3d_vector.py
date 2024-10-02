import torch
import numpy as np

from torch import Tensor
from torch.nn.functional import avg_pool3d
from waveorder import optics, sampling, stokes, util
from waveorder.visuals.napari_visuals import add_transfer_function_to_viewer


def generate_test_phantom(zyx_shape):

    # Simulate
    yx_star, yx_theta, _ = util.generate_star_target(
        yx_shape=zyx_shape[1:],
        blur_px=1,
        margin=50,
    )
    c00 = yx_star
    c2_2 = -torch.sin(2 * yx_theta) * yx_star
    c22 = -torch.cos(2 * yx_theta) * yx_star

    # Put in a center slices of a 3D object
    center_slice_object = torch.stack((c00, c2_2, c22), dim=0)
    object = torch.zeros((3,) + zyx_shape)
    object[:, zyx_shape[0] // 2, ...] = center_slice_object
    return object


def calculate_transfer_function(
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
    invert_phase_contrast=False,
    fourier_oversample_factor=1,
):
    transverse_nyquist = sampling.transverse_nyquist(
        wavelength_illumination,
        numerical_aperture_illumination,
        numerical_aperture_detection,
    )
    axial_nyquist = sampling.axial_nyquist(
        wavelength_illumination,
        numerical_aperture_detection,
        index_of_refraction_media,
    )

    yx_factor = int(np.ceil(yx_pixel_size / transverse_nyquist))
    z_factor = int(np.ceil(z_pixel_size / axial_nyquist))

    sfZYX_transfer_function, intensity_to_stokes_matrix = (
        _calculate_wrap_unsafe_transfer_function(
            swing,
            scheme,
            (
                zyx_shape[0] * z_factor * fourier_oversample_factor,
                zyx_shape[1] * yx_factor * fourier_oversample_factor,
                zyx_shape[2] * yx_factor * fourier_oversample_factor,
            ),
            yx_pixel_size / yx_factor,
            z_pixel_size / z_factor,
            wavelength_illumination,
            z_padding,
            index_of_refraction_media,
            numerical_aperture_illumination,
            numerical_aperture_detection,
            invert_phase_contrast=invert_phase_contrast,
        )
    )

    # avg_pool3d does not support complex numbers
    pooled_sfZYX_transfer_function_real = avg_pool3d(
        sfZYX_transfer_function.real, (fourier_oversample_factor,) * 3
    )
    pooled_sfZYX_transfer_function_imag = avg_pool3d(
        sfZYX_transfer_function.imag, (fourier_oversample_factor,) * 3
    )
    pooled_sfZYX_transfer_function = (
        pooled_sfZYX_transfer_function_real
        + 1j * pooled_sfZYX_transfer_function_imag
    )

    # Crop to original size
    sfzyx_out_shape = (
        3,
        3,
        zyx_shape[0] + 2 * z_padding,
    ) + zyx_shape[1:]
    return (
        sampling.nd_fourier_central_cuboid(
            pooled_sfZYX_transfer_function, sfzyx_out_shape
        ),
        intensity_to_stokes_matrix,
    )


def _calculate_wrap_unsafe_transfer_function(
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
    invert_phase_contrast=False,
):
    intensity_to_stokes_matrix = stokes.calculate_intensity_to_stokes_matrix(
        swing, scheme=scheme
    )

    input_jones = torch.tensor([0.0 + 1.0j, 1.0 + 0j])  # circular

    # Calculate frequencies
    y_frequencies, x_frequencies = util.generate_frequencies(
        zyx_shape[1:], yx_pixel_size
    )
    radial_frequencies = torch.sqrt(x_frequencies**2 + y_frequencies**2)

    z_total = zyx_shape[0] + 2 * z_padding
    z_position_list = torch.fft.ifftshift(
        (torch.arange(z_total) - z_total // 2) * z_pixel_size
    )
    if invert_phase_contrast:
        z_position_list = torch.flip(z_position_list, dims=(0,))

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
    P = optics.generate_propagation_kernel(
        radial_frequencies,
        det_pupil,
        wavelength_illumination / index_of_refraction_media,
        z_position_list,
    )

    G = optics.generate_defocus_greens_tensor(
        x_frequencies,
        y_frequencies,
        greens_functions_z,
        pupil,
        lambda_in=wavelength_illumination / index_of_refraction_media,
    )

    P_3D = torch.abs(torch.fft.ifft(P, dim=-3)).type(torch.complex64)
    G_3D = torch.abs(torch.fft.ifft(G, dim=-3)) * (-1j)
    S_3D = torch.fft.ifft(S, dim=-3)


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

    H_re = H1[1:, 1:] + H2[1:, 1:]  # drop data-side z components
    # H_im = 1j * (H1 - H2) # ignore absorptive terms

    H_re /= torch.amax(torch.abs(H_re))

    s = util.pauli()[[0, 1, 2]]  # select s0, s1, and s2 (drop s3)
    Y = util.gellmann()[[0, 4, 8]]
    # select phase f00 and transverse linear isotropic terms 2-2, and f22

    sfZYX_transfer_function = torch.einsum(
        "sik,ikpjzyx,lpj->slzyx", s, H_re, Y
    )

    return (
        sfZYX_transfer_function,
        intensity_to_stokes_matrix,
    )


def calculate_singular_system(sfZYX_transfer_function):
    # Compute regularized inverse filter
    print("Computing SVD")
    ZYXsf_transfer_function = sfZYX_transfer_function.permute(2, 3, 4, 0, 1)
    U, S, Vh = torch.linalg.svd(ZYXsf_transfer_function, full_matrices=False)
    S /= torch.max(S)
    singular_system = (U, S, Vh)
    return singular_system


def visualize_transfer_function(viewer, sfZYX_transfer_function, zyx_scale):
    add_transfer_function_to_viewer(
        viewer,
        sfZYX_transfer_function,
        zyx_scale=zyx_scale,
        layer_name="Transfer Function",
        complex_rgb=True,
        clim_factor=0.5,
    )


def apply_transfer_function(
    fzyx_object,
    sfZYX_transfer_function,
    intensity_to_stokes_matrix,  # TODO use this to simulate intensities
):
    fZYX_object = torch.fft.fftn(fzyx_object, dim=(1, 2, 3))
    sZYX_data = torch.einsum(
        "fzyx,sfzyx->szyx", fZYX_object, sfZYX_transfer_function
    )
    szyx_data = torch.fft.ifftn(sZYX_data, dim=(1, 2, 3))

    return (50 * szyx_data) + 0.1 * torch.randn(szyx_data.shape)


def apply_inverse_transfer_function(
    szyx_data: Tensor,
    singular_system: tuple[Tensor],
    intensity_to_stokes_matrix: Tensor,
    regularization_strength: float = 1e-3,
):
    sZYX_data = torch.fft.fftn(szyx_data, dim=(1, 2, 3))

    # Key computation
    print("Computing inverse filter")
    U, S, Vh = singular_system
    S_reg = S / (S**2 + regularization_strength**2)
    ZYXsf_inverse_filter = torch.einsum(
        "zyxij,zyxj,zyxjk->zyxki", U, S_reg, Vh
    )

    # Apply inverse filter
    fZYX_reconstructed = torch.einsum(
        "szyx,zyxsf->fzyx", sZYX_data, ZYXsf_inverse_filter
    )

    return torch.real(torch.fft.ifftn(fZYX_reconstructed, dim=(1, 2, 3)))
