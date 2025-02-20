from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import avg_pool3d

from waveorder import optics, sampling, stokes, util
from waveorder.filter import apply_filter_bank
from waveorder.visuals.napari_visuals import add_transfer_function_to_viewer


def generate_test_phantom(zyx_shape: tuple[int, int, int]) -> torch.Tensor:
    # Simulate
    yx_star, yx_theta, _ = util.generate_star_target(
        yx_shape=zyx_shape[1:],
        blur_px=1,
        margin=50,
    )
    c00 = yx_star
    c2_2 = -torch.sin(2 * yx_theta) * yx_star  # torch.zeros_like(c00)
    c22 = -torch.cos(2 * yx_theta) * yx_star  # torch.zeros_like(c00)  #

    # Put in a center slices of a 3D object
    center_slice_object = torch.stack((c00, c2_2, c22), dim=0)
    object = torch.zeros((3,) + zyx_shape)
    object[:, zyx_shape[0] // 2, ...] = center_slice_object
    return object


def calculate_transfer_function(
    swing: float,
    scheme: str,
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_illumination: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_illumination: float,
    numerical_aperture_detection: float,
    invert_phase_contrast: bool = False,
    fourier_oversample_factor: int = 1,
) -> tuple[
    torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]:
    if z_padding != 0:
        raise NotImplementedError("Padding not implemented for this model")

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

    print("YX factor:", yx_factor)
    print("Z factor:", z_factor)

    tf_calculation_shape = (
        zyx_shape[0] * z_factor * fourier_oversample_factor,
        int(np.ceil(zyx_shape[1] * yx_factor * fourier_oversample_factor)),
        int(np.ceil(zyx_shape[2] * yx_factor * fourier_oversample_factor)),
    )

    (
        sfZYX_transfer_function,
        intensity_to_stokes_matrix,
    ) = _calculate_wrap_unsafe_transfer_function(
        swing,
        scheme,
        tf_calculation_shape,
        yx_pixel_size / yx_factor,
        z_pixel_size / z_factor,
        wavelength_illumination,
        z_padding,
        index_of_refraction_media,
        numerical_aperture_illumination,
        numerical_aperture_detection,
        invert_phase_contrast=invert_phase_contrast,
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
        pooled_sfZYX_transfer_function.shape[0],
        pooled_sfZYX_transfer_function.shape[1],
        zyx_shape[0] + 2 * z_padding,
    ) + zyx_shape[1:]

    cropped = sampling.nd_fourier_central_cuboid(
        pooled_sfZYX_transfer_function, sfzyx_out_shape
    )

    # Compute singular system on cropped and downsampled
    singular_system = calculate_singular_system(cropped)

    return (
        cropped,
        intensity_to_stokes_matrix,
        singular_system,
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
    print("Computing transfer function")
    intensity_to_stokes_matrix = stokes.calculate_intensity_to_stokes_matrix(
        swing, scheme=scheme
    )

    input_jones = torch.tensor([0.0 - 1.0j, 1.0 + 0j])  # circular
    # input_jones = torch.tensor([0 + 0j, 1 + 0j]) # linear

    # Calculate frequencies
    y_frequencies, x_frequencies = util.generate_frequencies(
        zyx_shape[1:], yx_pixel_size
    )
    radial_frequencies = torch.sqrt(x_frequencies**2 + y_frequencies**2)

    z_total = zyx_shape[0] + 2 * z_padding
    z_position_list = torch.fft.ifftshift(
        (torch.arange(z_total) - z_total // 2) * z_pixel_size
    )
    if (
        not invert_phase_contrast
    ):  # opposite sign of direct phase reconstruction
        z_position_list = torch.flip(z_position_list, dims=(0,))
    z_frequencies = torch.fft.fftfreq(z_total, d=z_pixel_size)

    # 2D pupils
    print("\tCalculating pupils...")
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

    print("\tCalculating greens tensor spectrum...")
    G_3D = optics.generate_greens_tensor_spectrum(
        zyx_shape=(z_total, zyx_shape[1], zyx_shape[2]),
        zyx_pixel_size=(z_pixel_size, yx_pixel_size, yx_pixel_size),
        wavelength=wavelength_illumination / index_of_refraction_media,
    )

    # Main part
    PG_3D = torch.einsum("zyx,ipzyx->ipzyx", P_3D, G_3D)
    PS_3D = torch.einsum("zyx,jzyx,kzyx->jkzyx", P_3D, S_3D, torch.conj(S_3D))

    del P_3D, G_3D, S_3D

    print("\tComputing pg and ps...")
    pg = torch.fft.fftn(PG_3D, dim=(-3, -2, -1))
    ps = torch.fft.fftn(PS_3D, dim=(-3, -2, -1))

    del PG_3D, PS_3D

    print("\tComputing H1 and H2...")
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

    del H1, H2

    H_re /= torch.amax(torch.abs(H_re))

    s = util.pauli()[[0, 1, 2, 3]]  # select s0, s1, and s2
    Y = util.gellmann()[[0, 4, 8]]
    # select phase f00 and transverse linear isotropic terms 2-2, and f22

    print("\tComputing final transfer function...")
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
    singular_system = (
        U.permute(3, 4, 0, 1, 2),
        S.permute(3, 0, 1, 2),
        Vh.permute(3, 4, 0, 1, 2),
    )
    return singular_system


def visualize_transfer_function(
    viewer: "napari.Viewer",
    sfZYX_transfer_function: torch.Tensor,
    zyx_scale: tuple[float, float, float],
) -> None:
    add_transfer_function_to_viewer(
        viewer,
        sfZYX_transfer_function,
        zyx_scale=zyx_scale,
        layer_name="Transfer Function",
        complex_rgb=True,
        clim_factor=0.5,
    )


def apply_transfer_function(
    fzyx_object: torch.Tensor,
    sfZYX_transfer_function: torch.Tensor,
    intensity_to_stokes_matrix: torch.Tensor,  # TODO use this to simulate intensities
) -> torch.Tensor:
    fZYX_object = torch.fft.fftn(fzyx_object, dim=(1, 2, 3))
    sZYX_data = torch.einsum(
        "fzyx,sfzyx->szyx", fZYX_object, sfZYX_transfer_function
    )
    szyx_data = torch.fft.ifftn(sZYX_data, dim=(1, 2, 3))

    return 50 * szyx_data  # + 0.1 * torch.randn(szyx_data.shape)


def apply_inverse_transfer_function(
    szyx_data: Tensor,
    singular_system: tuple[Tensor],
    intensity_to_stokes_matrix: Tensor,
    reconstruction_algorithm: Literal["Tikhonov", "TV"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    TV_rho_strength: float = 1e-3,
    TV_iterations: int = 10,
):
    # Key computation
    print("Computing inverse filter")
    U, S, Vh = singular_system
    S_reg = S / (S**2 + regularization_strength)
    sfzyx_inverse_filter = torch.einsum(
        "sjzyx,jzyx,jfzyx->sfzyx", U, S_reg, Vh
    )

    fzyx_recon = apply_filter_bank(sfzyx_inverse_filter, szyx_data)

    return fzyx_recon
