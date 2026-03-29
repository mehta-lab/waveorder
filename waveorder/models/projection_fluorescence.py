"""Projection tomography model with fluorescence OTF blur.

Forward model: 3D OTF convolution + Siddon line-integral projection.
By the Fourier-slice theorem, projecting an OTF-blurred volume at
angle theta is equivalent to applying the central slice of the 3D OTF
at that angle as a 2D transfer function on the projection.

Inverse algorithms:
    - Tikhonov: per-angle OTF-slice inverse filter + ramp + backproject.
    - CG: iterative solver with full 3D OTF in the forward/adjoint model.

This module follows the standard waveorder model pattern.
"""

from typing import Literal

import torch
from torch import Tensor

from waveorder.models import isotropic_fluorescent_thick_3d
from waveorder.models.projection_no_blur import generate_test_phantom  # noqa: F401 — re-export
from waveorder.projection import (
    SiddonOperator,
    cg_tikhonov,
    extract_otf_slices,
)
from waveorder.reconstruct import tikhonov_regularized_inverse_filter_2d


def calculate_transfer_function(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    angles: list[float],
    wavelength_emission: float,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
    z_padding: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[SiddonOperator, list[Tensor], Tensor]:
    """Build Siddon operator and extract fluorescence OTF slices.

    Parameters
    ----------
    zyx_shape : tuple[int, int, int]
    yx_pixel_size : float
    z_pixel_size : float
    angles : list of float
        Tilt angles in degrees.
    wavelength_emission : float
    index_of_refraction_media : float
    numerical_aperture_detection : float
    z_padding : int
    device : str or torch.device

    Returns
    -------
    siddon_op : SiddonOperator
    otf_slices : list of Tensor
        Per-angle 2D OTF slices, each (ny, n_lateral_i).
    otf_3d : Tensor
        Full 3D fluorescence OTF on device.
    """
    device = torch.device(device)

    siddon_op = SiddonOperator(zyx_shape, angles, yx_pixel_size, device)

    otf_3d = isotropic_fluorescent_thick_3d.calculate_transfer_function(
        zyx_shape=zyx_shape,
        yx_pixel_size=yx_pixel_size,
        z_pixel_size=z_pixel_size,
        wavelength_emission=wavelength_emission,
        z_padding=z_padding,
        index_of_refraction_media=index_of_refraction_media,
        numerical_aperture_detection=numerical_aperture_detection,
    ).to(device)

    otf_slices = extract_otf_slices(otf_3d, angles, siddon_op.n_laterals)

    return siddon_op, otf_slices, otf_3d


def apply_transfer_function(
    zyx_object: Tensor,
    siddon_op: SiddonOperator,
    otf_3d: Tensor,
) -> list[Tensor]:
    """Forward: OTF blur (3D FFT convolution) then Siddon projection.

    Parameters
    ----------
    zyx_object : Tensor, shape (nz, ny, nx)
    siddon_op : SiddonOperator
    otf_3d : Tensor, shape (nz, ny, nx)
        3D fluorescence OTF.

    Returns
    -------
    projections : list of Tensor, each (ny, n_lateral_i)
    """
    blurred = torch.fft.ifftn(torch.fft.fftn(zyx_object) * otf_3d).real
    return siddon_op.project_all(blurred)


def apply_inverse_transfer_function(
    projections: list[Tensor],
    siddon_op: SiddonOperator,
    otf_slices: list[Tensor],
    reconstruction_algorithm: Literal["Tikhonov", "CG"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    n_iter: int = 50,
    ramp_filter: bool = True,
    otf_3d: Tensor | None = None,
) -> Tensor:
    """Reconstruct from projections using fluorescence OTF model.

    Parameters
    ----------
    projections : list of Tensor
    siddon_op : SiddonOperator
    otf_slices : list of Tensor
        Per-angle 2D OTF slices for Tikhonov.
    reconstruction_algorithm : {"Tikhonov", "CG"}
        "Tikhonov": per-angle OTF-slice Wiener filter + ramp + backproject.
        "CG": iterative solver with full 3D OTF blur in forward/adjoint.
    regularization_strength : float
    n_iter : int
    ramp_filter : bool
    otf_3d : Tensor or None
        Required for CG algorithm (full 3D OTF for blur).

    Returns
    -------
    Tensor
        Reconstructed volume, shape zyx_shape.
    """
    device = siddon_op.device

    if reconstruction_algorithm == "Tikhonov":
        vol = torch.zeros(siddon_op.zyx_shape, dtype=torch.float32, device=device)
        for i, (proj, otf_slice) in enumerate(zip(projections, otf_slices)):
            inv_filter = tikhonov_regularized_inverse_filter_2d(otf_slice, regularization_strength)
            if ramp_filter:
                ny, n_lat = proj.shape
                freqs = torch.fft.fftfreq(n_lat, device=device)
                ramp = freqs.abs()
                ramp[0] = 0.5 * freqs[1].abs()
                ramp = ramp / ramp.max()
                inv_filter = inv_filter * ramp.unsqueeze(0)

            proj_ft = torch.fft.fft(proj, dim=1)
            filtered = torch.fft.ifft(proj_ft * inv_filter, dim=1).real
            vol += siddon_op.backproject(filtered, i)
        return vol

    elif reconstruction_algorithm == "CG":
        if otf_3d is None:
            raise ValueError("otf_3d is required for CG reconstruction.")

        otf_conj = torch.conj(otf_3d)

        def forward_blur(vol):
            return torch.fft.ifftn(torch.fft.fftn(vol) * otf_3d).real

        def adjoint_blur(vol):
            return torch.fft.ifftn(torch.fft.fftn(vol) * otf_conj).real

        def forward(vol):
            return siddon_op.project_all(forward_blur(vol))

        def adjoint(projs):
            bp = siddon_op.backproject_all(projs, ramp_filter=ramp_filter)
            return adjoint_blur(bp)

        return cg_tikhonov(
            forward,
            adjoint,
            projections,
            siddon_op.zyx_shape,
            regularization_strength,
            n_iter,
            device,
        )
    else:
        raise ValueError(f"Unknown reconstruction_algorithm: {reconstruction_algorithm!r}")


def reconstruct(
    zyx_object: Tensor,
    yx_pixel_size: float,
    z_pixel_size: float,
    angles: list[float],
    wavelength_emission: float,
    index_of_refraction_media: float,
    numerical_aperture_detection: float,
    z_padding: int = 0,
    reconstruction_algorithm: Literal["Tikhonov", "CG"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    n_iter: int = 50,
    ramp_filter: bool = True,
    device: str | torch.device = "cpu",
) -> Tensor:
    """One-liner: compute OTF, forward project with blur, and reconstruct.

    Parameters
    ----------
    zyx_object : Tensor
    yx_pixel_size : float
    z_pixel_size : float
    angles : list of float
    wavelength_emission : float
    index_of_refraction_media : float
    numerical_aperture_detection : float
    z_padding : int
    reconstruction_algorithm : {"Tikhonov", "CG"}
    regularization_strength : float
    n_iter : int
    ramp_filter : bool
    device : str or torch.device

    Returns
    -------
    Tensor
    """
    siddon_op, otf_slices, otf_3d = calculate_transfer_function(
        zyx_object.shape,
        yx_pixel_size,
        z_pixel_size,
        angles,
        wavelength_emission,
        index_of_refraction_media,
        numerical_aperture_detection,
        z_padding,
        device,
    )
    zyx_object = zyx_object.to(device)
    projections = apply_transfer_function(zyx_object, siddon_op, otf_3d)
    return apply_inverse_transfer_function(
        projections,
        siddon_op,
        otf_slices,
        reconstruction_algorithm=reconstruction_algorithm,
        regularization_strength=regularization_strength,
        n_iter=n_iter,
        ramp_filter=ramp_filter,
        otf_3d=otf_3d,
    )
