import warnings
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from waveorder import backprojector, optics, rlgc, sampling, util
from waveorder._pixel_size import YXPixelSize
from waveorder.backprojector import BackProjectorType
from waveorder.reconstruct import tikhonov_regularized_inverse_filter
from waveorder.visuals.napari_visuals import add_transfer_function_to_viewer


def generate_test_phantom(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    sphere_radius: float,
) -> Tensor:
    sphere, _, _ = util.generate_sphere_target(zyx_shape, yx_pixel_size, z_pixel_size, sphere_radius)

    return sphere


def calculate_transfer_function(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_emission: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
    confocal_pinhole_diameter: float | None = None,
) -> Tensor:
    """Calculate the optical transfer function for fluorescence imaging.

    Supports both widefield and confocal microscopy modes. When
    confocal_pinhole_diameter is None, computes widefield OTF. When specified,
    computes confocal OTF by multiplying excitation and detection PSFs, where
    the detection PSF is downweighted by the pinhole aperture function.

    Parameters
    ----------
    zyx_shape : tuple[int, int, int]
        Shape of the 3D volume
    yx_pixel_size : float
        Pixel size in YX plane
    z_pixel_size : float
        Pixel size in Z dimension
    wavelength_emission : float
        Emission wavelength
    z_padding : int
        Padding for axial dimension
    index_of_refraction_media : float
        Refractive index of imaging medium
    numerical_aperture_detection : float
        Numerical aperture of detection objective
    confocal_pinhole_diameter : float | None, optional
        Diameter of confocal pinhole in image space (demagnified). If None,
        computes widefield OTF. If specified, computes confocal OTF.

    Returns
    -------
    Tensor
        3D optical transfer function
    """
    na_det_val = float(torch.as_tensor(numerical_aperture_detection).detach())
    transverse_nyquist = sampling.transverse_nyquist(
        wavelength_emission,
        na_det_val,  # ill = det for fluorescence
        na_det_val,
    )
    axial_nyquist = sampling.axial_nyquist(
        wavelength_emission,
        na_det_val,
        index_of_refraction_media,
    )

    # For confocal, double the Nyquist range (half the sampling requirement)
    if confocal_pinhole_diameter is not None:
        transverse_nyquist = transverse_nyquist / 2
        axial_nyquist = axial_nyquist / 2

    yx_pixel_size = YXPixelSize.from_value(yx_pixel_size)
    y_factor = int(np.ceil(yx_pixel_size.y / transverse_nyquist))
    x_factor = int(np.ceil(yx_pixel_size.x / transverse_nyquist))
    z_factor = int(np.ceil(z_pixel_size / axial_nyquist))

    optical_transfer_function = _calculate_wrap_unsafe_transfer_function(
        (
            zyx_shape[0] * z_factor,
            zyx_shape[1] * y_factor,
            zyx_shape[2] * x_factor,
        ),
        YXPixelSize(y=yx_pixel_size.y / y_factor, x=yx_pixel_size.x / x_factor),
        z_pixel_size / z_factor,
        wavelength_emission,
        z_padding,
        index_of_refraction_media,
        numerical_aperture_detection,
        confocal_pinhole_diameter,
    )
    zyx_out_shape = (zyx_shape[0] + 2 * z_padding,) + zyx_shape[1:]
    optical_transfer_function = sampling.nd_fourier_central_cuboid(optical_transfer_function, zyx_out_shape)
    return _enforce_nonnegative_psf(optical_transfer_function)


def _enforce_nonnegative_psf(optical_transfer_function: Tensor) -> Tensor:
    """Return an OTF whose real-space incoherent PSF is nonnegative.

    The intensity PSF is built as ``|field|**2`` and so is nonnegative, but
    cropping the OTF to the working resolution (``nd_fourier_central_cuboid``,
    an ideal Fourier-domain low-pass) makes the PSF ring below zero. A physical
    fluorescence PSF cannot be negative, and Richardson-Lucy's convergence
    guarantee requires a nonnegative forward operator, so we clip the (small,
    sub-percent) negative lobes and rebuild the normalized OTF.

    Parameters
    ----------
    optical_transfer_function : Tensor
        3D OTF, shape ``(Z, Y, X)``.

    Returns
    -------
    Tensor
        OTF whose inverse transform is nonnegative, normalized to unit peak.
    """
    psf = torch.real(torch.fft.ifftn(optical_transfer_function, dim=(-3, -2, -1)))
    psf = torch.clamp(psf, min=0)
    otf = torch.fft.fftn(psf, dim=(-3, -2, -1))
    return otf / torch.clamp(torch.max(torch.abs(otf)), min=1e-12)


def _calculate_pinhole_aperture_otf(
    radial_frequencies: Tensor,
    pinhole_diameter: float,
) -> Tensor:
    """Calculate the pinhole aperture OTF for confocal microscopy.

    The pinhole acts as a spatial filter in the image plane. A smaller pinhole
    (approaching a point) gives a broader OTF (approaching flat/ones).
    A larger pinhole gives a narrower OTF (approaching a delta function).

    Parameters
    ----------
    radial_frequencies : Tensor
        Radial spatial frequencies (units of 1/length)
    pinhole_diameter : float
        Diameter (not radius) of the confocal pinhole (units of length, matching
        radial_frequencies)

    Returns
    -------
    Tensor
        Pinhole aperture OTF (jinc^2 function)
    """
    argument = pinhole_diameter * radial_frequencies
    j1_values = torch.special.bessel_j1(np.pi * argument)
    jinc = torch.where(argument > 1e-10, j1_values / (2 * argument), 0.5)
    return jinc**2


def _calculate_wrap_unsafe_transfer_function(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_emission: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
    confocal_pinhole_diameter: float | None = None,
) -> Tensor:
    radial_frequencies = util.generate_radial_frequencies(zyx_shape[1:], yx_pixel_size)

    z_total = zyx_shape[0] + 2 * z_padding
    z_position_list = torch.fft.ifftshift((torch.arange(z_total) - z_total // 2) * z_pixel_size)

    det_pupil = optics.generate_pupil(
        radial_frequencies,
        numerical_aperture_detection,
        wavelength_emission,
    )

    propagation_kernel = optics.generate_propagation_kernel(
        radial_frequencies,
        det_pupil,
        wavelength_emission / index_of_refraction_media,
        z_position_list,
    )

    point_spread_function = torch.abs(torch.fft.ifft2(propagation_kernel, dim=(1, 2))) ** 2
    optical_transfer_function = torch.fft.fftn(point_spread_function, dim=(0, 1, 2))

    # Confocal: multiply excitation PSF with detection PSF (downweighted by pinhole)
    if confocal_pinhole_diameter is not None:
        pinhole_otf_2d = _calculate_pinhole_aperture_otf(radial_frequencies, confocal_pinhole_diameter)
        # Detection OTF is downweighted by pinhole
        otf_detection = optical_transfer_function * pinhole_otf_2d[None, :, :]

        # Convert to PSFs
        psf_excitation = torch.abs(torch.fft.ifftn(optical_transfer_function, dim=(0, 1, 2)))
        psf_detection = torch.abs(torch.fft.ifftn(otf_detection, dim=(0, 1, 2)))

        # Confocal PSF = excitation PSF * detection PSF (in real space)
        psf_confocal = psf_excitation * psf_detection

        # Convert back to OTF
        optical_transfer_function = torch.fft.fftn(psf_confocal, dim=(0, 1, 2))

    optical_transfer_function = optical_transfer_function / torch.clamp(
        torch.max(torch.abs(optical_transfer_function)), min=1e-12
    )

    return optical_transfer_function


def visualize_transfer_function(
    viewer,
    optical_transfer_function: Tensor,
    zyx_scale: tuple[float, float, float],
) -> None:
    add_transfer_function_to_viewer(
        viewer,
        torch.real(optical_transfer_function),
        zyx_scale,
        clim_factor=0.05,
    )


def apply_transfer_function(
    zyx_object: Tensor,
    optical_transfer_function: Tensor,
    z_padding: int,
    background: int = 10,
) -> Tensor:
    """Simulate imaging by applying a transfer function

    Parameters
    ----------
    zyx_object : torch.Tensor
    optical_transfer_function : torch.Tensor
    z_padding : int
    background : int, optional
        constant background counts added to each voxel, by default 10

    Returns
    -------
    Simulated data : torch.Tensor

    """
    if zyx_object.shape[0] + 2 * z_padding != optical_transfer_function.shape[0]:
        raise ValueError("Please check padding: ZYX_obj.shape[0] + 2 * Z_pad != H_re.shape[0]")
    if z_padding > 0:
        optical_transfer_function = optical_transfer_function[z_padding:-z_padding]

    # Very simple simulation, consider adding noise and bkg knobs
    zyx_obj_hat = torch.fft.fftn(zyx_object)
    zyx_data = zyx_obj_hat * optical_transfer_function
    data = torch.real(torch.fft.ifftn(zyx_data))

    data += background  # Add a direct background
    return data


def apply_inverse_transfer_function(
    zyx_data: Tensor,
    optical_transfer_function: Tensor,
    z_padding: int,
    reconstruction_algorithm: Literal["Tikhonov", "TV", "RL", "RLGC"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    TV_rho_strength: float = 1e-3,
    TV_iterations: int = 10,
    rl_iterations: int = 25,
    rl_background: float = 0.0,
    rl_stopping_tolerance: float | None = None,
    rl_back_projector: BackProjectorType = "matched",
    rl_bp_alpha: float | None = None,
    rl_bp_beta: float | None = None,
    rl_bp_order: int = 8,
    rl_bp_resolution_mode: Literal["fwhm", "fwhm_over_sqrt2"] = "fwhm",
    back_projector_otf: Tensor | None = None,
    apodization_rolloff: float = 0.0,
) -> Tensor:
    """Reconstructs fluorescence density from defocus data.

    Parameters
    ----------
    zyx_data : Tensor
        Raw data of shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
    optical_transfer_function : Tensor
        3D optical transfer function (shared, not batched)
    z_padding : int
        Padding for axial dimension. Use zero for defocus stacks that
        extend ~3 PSF widths beyond the sample. Pad by ~3 PSF widths otherwise.
    reconstruction_algorithm : {"Tikhonov", "TV", "RL", "RLGC"}, optional
        By default "Tikhonov". "TV" is not implemented. "RL" is
        Richardson-Lucy deconvolution and "RLGC" is its Gradient-Consensus
        variant, which resists overfitting noise (see :mod:`waveorder.rlgc`).
    regularization_strength : float, optional
        Regularization parameter (Tikhonov), by default 1e-3
    TV_rho_strength : float, optional
        TV-specific regularization parameter, by default 1e-3
    TV_iterations : int, optional
        TV-specific number of iterations, by default 10
    rl_iterations : int, optional
        Maximum RL / RLGC iterations, by default 25
    rl_background : float, optional
        Constant background (dark counts / offset) folded into the RL / RLGC
        Poisson forward model, by default 0.0
    rl_stopping_tolerance : float, optional
        If set, RL / RLGC stop early once the relative change of the estimate
        falls below this value, by default None (run all iterations)
    rl_back_projector : str, optional
        Back projector for RL, by default "matched" (the matched transpose,
        i.e. classic Richardson-Lucy). The unmatched alternatives "gaussian",
        "butterworth", "wiener" and "wiener_butterworth" flatten the spectral
        product and so converge in far fewer iterations; see
        :mod:`waveorder.backprojector`. Unmatched choices are RL-only, and one
        iteration is a good rule of thumb for them.
    rl_bp_alpha : float, optional
        Wiener regularization for the "wiener"/"wiener_butterworth" back
        projectors, by default None (use the matched cutoff gain)
    rl_bp_beta : float, optional
        Cutoff gain for the "butterworth"/"wiener_butterworth" back projectors,
        by default None (use the matched cutoff gain)
    rl_bp_order : int, optional
        Butterworth order for the "butterworth"/"wiener_butterworth" back
        projectors, by default 8
    rl_bp_resolution_mode : str, optional
        How the back projector sets its cutoff frequency, by default "fwhm".
        Use "fwhm_over_sqrt2" for iSIM.
    back_projector_otf : Tensor, optional
        Prebuilt back projector, skipping the rl_bp_* construction. Building it
        costs a few seconds on a large OTF, so callers reconstructing many tiles
        should build it once with :func:`waveorder.backprojector.calculate_back_projector`
        and pass it here. By default None (build it on every call).
    apodization_rolloff : float, optional
        Raised-cosine roll-off fraction applied to the Tikhonov inverse
        filter at the transverse Nyquist edge (Y and X only). Suppresses
        Nyquist-rate checkerboard artifacts in the reconstruction when
        the optical band limit exceeds the sampling Nyquist frequency.
        Tikhonov only. By default 0.0 (no apodization, previous behavior). Must be
        between 0 and 1; if you see checkerboarding artifacts, start
        with 0.25.

    Returns
    -------
    Tensor
        Fluorescence density, shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
    """
    batched = zyx_data.ndim == 4
    if not batched:
        zyx_data = zyx_data.unsqueeze(0)

    # Handle padding: (B, Z, Y, X) -> (B, Z+2*pad, Y, X)
    zyx_padded = util.pad_zyx_along_z(zyx_data, z_padding)

    # Reconstruct
    if reconstruction_algorithm == "Tikhonov":
        inverse_filter = tikhonov_regularized_inverse_filter(optical_transfer_function, regularization_strength)

        if apodization_rolloff > 0:
            window = sampling.raised_cosine_window(
                inverse_filter.shape[-2], apodization_rolloff, device=inverse_filter.device
            )[:, None] * sampling.raised_cosine_window(
                inverse_filter.shape[-1], apodization_rolloff, device=inverse_filter.device
            )
            inverse_filter = inverse_filter * window

        # Batched FFT multiply: inverse_filter (Z,Y,X) broadcasts over B
        zyx_fft = torch.fft.fftn(zyx_padded, dim=(-3, -2, -1))
        f_real = torch.real(torch.fft.ifftn(zyx_fft * inverse_filter, dim=(-3, -2, -1)))

    elif reconstruction_algorithm == "TV":
        raise NotImplementedError

    elif reconstruction_algorithm in ("RL", "RLGC"):
        # The OTF (shared, shape (Z,Y,X)) broadcasts over the batch axis.
        otf = optical_transfer_function

        # RLGC reads the sign of transpose(forward(.)) to ask whether the two
        # photon halves agree. Only a true adjoint makes that question
        # meaningful: an unmatched back projector's negative lobes flip the
        # sign on their own, freezing good voxels.
        if reconstruction_algorithm == "RLGC" and rl_back_projector != "matched":
            raise NotImplementedError(
                f"rl_back_projector={rl_back_projector!r} is only supported with "
                f"reconstruction_algorithm='RL'; RLGC requires the matched "
                f"back projector for its gradient-consensus test."
            )

        # An unmatched back projector abandons Richardson-Lucy's fixed point at
        # the maximum-likelihood solution, so past a few iterations the estimate
        # degrades instead of settling. Guo et al. recommend a single iteration,
        # or up to five at low filter orders.
        if rl_back_projector != "matched" and rl_iterations > 5:
            warnings.warn(
                f"rl_iterations={rl_iterations} with rl_back_projector={rl_back_projector!r}: "
                f"unmatched back projectors reach a resolution-limited result in 1-5 iterations "
                f"and introduce artifacts beyond that. Consider rl_iterations=1.",
                UserWarning,
                stacklevel=2,
            )

        # Depends only on the OTF and the rl_bp_* knobs, all fixed for a run, so a
        # caller that reconstructs many tiles should build it once and pass it in;
        # otherwise it is rebuilt on every call.
        if back_projector_otf is None:
            back_projector_otf = backprojector.calculate_back_projector(
                otf,
                rl_back_projector,
                alpha=rl_bp_alpha,
                beta=rl_bp_beta,
                order=rl_bp_order,
                resolution_mode=rl_bp_resolution_mode,
            )

        def forward(x: Tensor) -> Tensor:
            return torch.real(torch.fft.ifftn(torch.fft.fftn(x, dim=(-3, -2, -1)) * otf, dim=(-3, -2, -1)))

        def transpose(y: Tensor) -> Tensor:
            return torch.real(
                torch.fft.ifftn(torch.fft.fftn(y, dim=(-3, -2, -1)) * back_projector_otf, dim=(-3, -2, -1))
            )

        f_real = rlgc.richardson_lucy(
            torch.clamp(zyx_padded, min=0.0),
            forward,
            transpose,
            num_iterations=rl_iterations,
            method=reconstruction_algorithm,
            background=rl_background,
            stopping_tolerance=rl_stopping_tolerance,
        )

    else:
        raise NotImplementedError(f"Unknown reconstruction_algorithm: {reconstruction_algorithm}")

    # Unpad
    if z_padding != 0:
        f_real = f_real[:, z_padding:-z_padding]

    if not batched:
        f_real = f_real.squeeze(0)

    return f_real


def reconstruct(
    zyx_data: Tensor,
    yx_pixel_size: float,
    z_pixel_size: float,
    wavelength_emission: float,
    z_padding: int,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
    confocal_pinhole_diameter: float | None = None,
    reconstruction_algorithm: Literal["Tikhonov", "TV", "RL", "RLGC"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    TV_rho_strength: float = 1e-3,
    TV_iterations: int = 10,
    rl_iterations: int = 25,
    rl_background: float = 0.0,
    rl_stopping_tolerance: float | None = None,
    rl_back_projector: BackProjectorType = "matched",
    rl_bp_alpha: float | None = None,
    rl_bp_beta: float | None = None,
    rl_bp_order: int = 8,
    rl_bp_resolution_mode: Literal["fwhm", "fwhm_over_sqrt2"] = "fwhm",
    apodization_rolloff: float = 0.0,
) -> Tensor:
    """Reconstruct 3D fluorescence density from a defocus stack.

    Parameters
    ----------
    zyx_data : Tensor
        Raw data of shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``
    yx_pixel_size : float
        Pixel size in the transverse (Y, X) dimensions
    z_pixel_size : float
        Pixel size in the axial (Z) dimension
    wavelength_emission : float
        Emission wavelength
    z_padding : int
        Padding for axial dimension
    index_of_refraction_media : float
        Refractive index of the surrounding medium
    numerical_aperture_detection : float
        Detection numerical aperture
    confocal_pinhole_diameter : float | None, optional
        Confocal pinhole diameter, by default None (widefield)
    reconstruction_algorithm : {"Tikhonov", "TV", "RL", "RLGC"}, optional
        By default "Tikhonov". "RL"/"RLGC" are Richardson-Lucy and its
        Gradient-Consensus variant.
    regularization_strength : float, optional
        Regularization parameter (Tikhonov), by default 1e-3
    TV_rho_strength : float, optional
        TV-specific regularization parameter, by default 1e-3
    TV_iterations : int, optional
        TV-specific number of iterations, by default 10
    rl_iterations : int, optional
        Maximum RL / RLGC iterations, by default 25
    rl_background : float, optional
        Constant background folded into the RL / RLGC forward model, by default 0.0
    rl_stopping_tolerance : float, optional
        Relative-change early-stop threshold for RL / RLGC, by default None
    rl_back_projector : str, optional
        Back projector for RL, by default "matched" (the matched transpose,
        i.e. classic Richardson-Lucy). The unmatched alternatives "gaussian",
        "butterworth", "wiener" and "wiener_butterworth" flatten the spectral
        product and so converge in far fewer iterations; see
        :mod:`waveorder.backprojector`. Unmatched choices are RL-only, and one
        iteration is a good rule of thumb for them.
    rl_bp_alpha : float, optional
        Wiener regularization for the "wiener"/"wiener_butterworth" back
        projectors, by default None (use the matched cutoff gain)
    rl_bp_beta : float, optional
        Cutoff gain for the "butterworth"/"wiener_butterworth" back projectors,
        by default None (use the matched cutoff gain)
    rl_bp_order : int, optional
        Butterworth order for the "butterworth"/"wiener_butterworth" back
        projectors, by default 8
    rl_bp_resolution_mode : str, optional
        How the back projector sets its cutoff frequency, by default "fwhm".
        Use "fwhm_over_sqrt2" for iSIM.
    apodization_rolloff : float, optional
        Raised-cosine roll-off fraction applied to the Tikhonov inverse
        filter at the transverse Nyquist edge, by default 0.0
        (no apodization). Suppresses Nyquist-rate checkerboard
        artifacts. Tikhonov only. Must be between 0 and 1; if you see
        checkerboarding artifacts, start with 0.25. See
        ``apply_inverse_transfer_function``.

    Returns
    -------
    Tensor
        Fluorescence density, shape ``(Z, Y, X)`` or ``(B, Z, Y, X)``

    Notes
    -----
    This recomputes the transfer function on every call, so it does not take a
    prebuilt back projector. Callers reconstructing many tiles with RL should
    call :func:`calculate_transfer_function` and
    :func:`apply_inverse_transfer_function` directly, building the back projector
    once with :func:`waveorder.backprojector.calculate_back_projector` and
    passing it as ``back_projector_otf``.
    """
    # Use last 3 dims as zyx_shape for TF computation
    zyx_shape = zyx_data.shape[-3:]
    optical_transfer_function = calculate_transfer_function(
        zyx_shape,
        yx_pixel_size,
        z_pixel_size,
        wavelength_emission,
        z_padding,
        index_of_refraction_media,
        numerical_aperture_detection,
        confocal_pinhole_diameter=confocal_pinhole_diameter,
    )
    return apply_inverse_transfer_function(
        zyx_data,
        optical_transfer_function,
        z_padding,
        reconstruction_algorithm=reconstruction_algorithm,
        regularization_strength=regularization_strength,
        TV_rho_strength=TV_rho_strength,
        TV_iterations=TV_iterations,
        rl_iterations=rl_iterations,
        rl_background=rl_background,
        rl_stopping_tolerance=rl_stopping_tolerance,
        rl_back_projector=rl_back_projector,
        rl_bp_alpha=rl_bp_alpha,
        rl_bp_beta=rl_bp_beta,
        rl_bp_order=rl_bp_order,
        rl_bp_resolution_mode=rl_bp_resolution_mode,
        apodization_rolloff=apodization_rolloff,
    )
