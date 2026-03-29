"""Projection tomography model with phase transfer function.

Forward model: 3D phase TF convolution + Siddon line-integral projection.
The phase TF comes from the weak object (first Born) approximation via
``phase_thick_3d.calculate_transfer_function``.

Inverse algorithms:
    - Tikhonov: per-angle TF-slice inverse filter + ramp + backproject.
    - CG: iterative solver with full 3D TF in the forward/adjoint model.

This module follows the standard waveorder model pattern.
"""

from typing import Literal

import torch
from torch import Tensor

from waveorder.models import phase_thick_3d
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
    wavelength_illumination: float,
    index_of_refraction_media: float,
    numerical_aperture_illumination: float,
    numerical_aperture_detection: float,
    z_padding: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[SiddonOperator, list[Tensor], Tensor]:
    """Build Siddon operator and extract phase TF slices.

    Parameters
    ----------
    zyx_shape : tuple[int, int, int]
    yx_pixel_size : float
    z_pixel_size : float
    angles : list of float
        Tilt angles in degrees.
    wavelength_illumination : float
    index_of_refraction_media : float
    numerical_aperture_illumination : float
    numerical_aperture_detection : float
    z_padding : int
    device : str or torch.device

    Returns
    -------
    siddon_op : SiddonOperator
    otf_slices : list of Tensor
        Per-angle 2D TF slices, each (ny, n_lateral_i).
    real_tf : Tensor
        Full 3D real-potential transfer function on device.
    """
    device = torch.device(device)

    siddon_op = SiddonOperator(zyx_shape, angles, yx_pixel_size, device)

    real_tf, _imag_tf = phase_thick_3d.calculate_transfer_function(
        zyx_shape=zyx_shape,
        yx_pixel_size=yx_pixel_size,
        z_pixel_size=z_pixel_size,
        wavelength_illumination=wavelength_illumination,
        z_padding=z_padding,
        index_of_refraction_media=index_of_refraction_media,
        numerical_aperture_illumination=numerical_aperture_illumination,
        numerical_aperture_detection=numerical_aperture_detection,
    )
    real_tf = real_tf.to(device)

    otf_slices = extract_otf_slices(real_tf, angles, siddon_op.n_laterals)

    return siddon_op, otf_slices, real_tf


def apply_transfer_function(
    zyx_object: Tensor,
    siddon_op: SiddonOperator,
    real_tf: Tensor,
) -> list[Tensor]:
    """Forward: phase TF convolution then Siddon projection.

    Parameters
    ----------
    zyx_object : Tensor, shape (nz, ny, nx)
        Phase object (cycles per voxel).
    siddon_op : SiddonOperator
    real_tf : Tensor, shape (nz, ny, nx)
        3D real-potential transfer function.

    Returns
    -------
    projections : list of Tensor, each (ny, n_lateral_i)
    """
    blurred = torch.fft.ifftn(torch.fft.fftn(zyx_object) * real_tf).real
    return siddon_op.project_all(blurred)


def apply_inverse_transfer_function(
    projections: list[Tensor],
    siddon_op: SiddonOperator,
    otf_slices: list[Tensor],
    reconstruction_algorithm: Literal["Tikhonov", "CG"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    n_iter: int = 50,
    ramp_filter: bool = True,
    real_tf: Tensor | None = None,
) -> Tensor:
    """Reconstruct from projections using phase TF model.

    Parameters
    ----------
    projections : list of Tensor
    siddon_op : SiddonOperator
    otf_slices : list of Tensor
        Per-angle 2D TF slices for Tikhonov.
    reconstruction_algorithm : {"Tikhonov", "CG"}
    regularization_strength : float
    n_iter : int
    ramp_filter : bool
    real_tf : Tensor or None
        Required for CG algorithm (full 3D TF for convolution).

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
        if real_tf is None:
            raise ValueError("real_tf is required for CG reconstruction.")

        tf_conj = torch.conj(real_tf)

        def forward_blur(vol):
            return torch.fft.ifftn(torch.fft.fftn(vol) * real_tf).real

        def adjoint_blur(vol):
            return torch.fft.ifftn(torch.fft.fftn(vol) * tf_conj).real

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
    wavelength_illumination: float,
    index_of_refraction_media: float,
    numerical_aperture_illumination: float,
    numerical_aperture_detection: float,
    z_padding: int = 0,
    reconstruction_algorithm: Literal["Tikhonov", "CG"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    n_iter: int = 50,
    ramp_filter: bool = True,
    device: str | torch.device = "cpu",
) -> Tensor:
    """One-liner: compute phase TF, forward project with blur, and reconstruct.

    Parameters
    ----------
    zyx_object : Tensor
    yx_pixel_size : float
    z_pixel_size : float
    angles : list of float
    wavelength_illumination : float
    index_of_refraction_media : float
    numerical_aperture_illumination : float
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
    siddon_op, otf_slices, real_tf = calculate_transfer_function(
        zyx_object.shape,
        yx_pixel_size,
        z_pixel_size,
        angles,
        wavelength_illumination,
        index_of_refraction_media,
        numerical_aperture_illumination,
        numerical_aperture_detection,
        z_padding,
        device,
    )
    zyx_object = zyx_object.to(device)
    projections = apply_transfer_function(zyx_object, siddon_op, real_tf)
    return apply_inverse_transfer_function(
        projections,
        siddon_op,
        otf_slices,
        reconstruction_algorithm=reconstruction_algorithm,
        regularization_strength=regularization_strength,
        n_iter=n_iter,
        ramp_filter=ramp_filter,
        real_tf=real_tf,
    )
