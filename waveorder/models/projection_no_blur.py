"""Projection tomography model without optical blur (geometric Siddon only).

Forward model: line-integral projection via Siddon ray-tracing.
Transfer function: identity (OTF = 1 at every angle).
Inverse: ramp-filtered backprojection (Tikhonov) or CG-Tikhonov.

This module follows the standard waveorder model pattern:
    generate_test_phantom -> calculate_transfer_function ->
    apply_transfer_function -> apply_inverse_transfer_function -> reconstruct
"""

from typing import Literal

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor

from waveorder.projection import (
    SiddonOperator,
    cg_tikhonov,
)
from waveorder.reconstruct import tikhonov_regularized_inverse_filter_2d


# ---------------------------------------------------------------------------
# Phantom generators
# ---------------------------------------------------------------------------
def _generate_isolated_bead(zyx_shape, voxel_size, sphere_radius):
    """Single sphere at volume center. Returns binary volume (0/1)."""
    nz, ny, nx = zyx_shape
    z = (np.arange(nz) - nz / 2) * voxel_size
    y = (np.arange(ny) - ny / 2) * voxel_size
    x = (np.arange(nx) - nx / 2) * voxel_size
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dist = np.sqrt(zz**2 + yy**2 + xx**2)
    return (dist <= sphere_radius).astype(np.float32)


def _generate_line_pattern(
    zyx_shape,
    voxel_size,
    line_radius=0.25,
    bracket_half_extent=2.0,
    bracket_x_offset=2.0,
    bracket_cap_length=0.6,
    torus_major_radius=0.8,
):
    """[o] pattern from bracket-shaped cylinders and a torus ring."""
    nz, ny, nx = zyx_shape
    z = (np.arange(nz) - nz / 2) * voxel_size
    y = (np.arange(ny) - ny / 2) * voxel_size
    x = (np.arange(nx) - nx / 2) * voxel_size
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    volume = np.zeros(zyx_shape, dtype=np.float32)

    def _add_cylinder_y(vol, x_center, y_min, y_max):
        y_clamped = np.clip(yy, y_min, y_max)
        dist = np.sqrt((xx - x_center) ** 2 + zz**2 + (yy - y_clamped) ** 2)
        vol[dist <= line_radius] = 1.0

    def _add_cylinder_x(vol, y_center, x_min, x_max):
        x_clamped = np.clip(xx, x_min, x_max)
        dist = np.sqrt((yy - y_center) ** 2 + zz**2 + (xx - x_clamped) ** 2)
        vol[dist <= line_radius] = 1.0

    # '[' bracket
    x_left = -bracket_x_offset
    _add_cylinder_y(volume, x_left, -bracket_half_extent, bracket_half_extent)
    _add_cylinder_x(volume, +bracket_half_extent, x_left, x_left + bracket_cap_length)
    _add_cylinder_x(volume, -bracket_half_extent, x_left, x_left + bracket_cap_length)

    # ']' bracket
    x_right = +bracket_x_offset
    _add_cylinder_y(volume, x_right, -bracket_half_extent, bracket_half_extent)
    _add_cylinder_x(volume, +bracket_half_extent, x_right - bracket_cap_length, x_right)
    _add_cylinder_x(volume, -bracket_half_extent, x_right - bracket_cap_length, x_right)

    # 'o' torus
    rho = np.sqrt(xx**2 + yy**2)
    dist_torus = np.sqrt((rho - torus_major_radius) ** 2 + zz**2)
    volume[dist_torus <= line_radius] = 1.0

    blur_sigma = 1.0
    volume = gaussian_filter(volume, sigma=blur_sigma)
    if volume.max() > 0:
        volume /= volume.max()
    return volume


def _generate_shepp_logan_3d(zyx_shape):
    """3D Shepp-Logan phantom normalized to [0, 1]."""
    ellipsoids = [
        (1.0, 0.0, 0.0, 0.0, 0.69, 0.92, 0.81, 0),
        (-0.8, 0.0, -0.0184, 0.0, 0.6624, 0.8740, 0.78, 0),
        (-0.2, 0.22, 0.0, 0.0, 0.11, 0.31, 0.22, -18),
        (-0.2, -0.22, 0.0, 0.0, 0.16, 0.41, 0.28, 18),
        (0.1, 0.0, 0.35, 0.0, 0.21, 0.25, 0.41, 0),
        (0.1, 0.0, 0.1, 0.0, 0.046, 0.046, 0.05, 0),
        (0.1, 0.0, -0.1, 0.0, 0.046, 0.046, 0.05, 0),
        (0.1, -0.08, -0.605, 0.0, 0.046, 0.023, 0.05, 0),
        (0.1, 0.0, -0.605, 0.0, 0.023, 0.023, 0.02, 0),
        (0.1, 0.06, -0.605, 0.0, 0.046, 0.023, 0.02, 0),
    ]
    nz, ny, nx = zyx_shape
    phantom = np.zeros(zyx_shape, dtype=np.float32)
    z = np.linspace(-1, 1, nz)
    y = np.linspace(-1, 1, ny)
    x = np.linspace(-1, 1, nx)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    for density, cx, cy, cz, sa, sb, sc, phi_deg in ellipsoids:
        phi = np.radians(phi_deg)
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        xp, yp, zp = xx - cx, yy - cy, zz - cz
        xr = xp * cos_phi + yp * sin_phi
        yr = -xp * sin_phi + yp * cos_phi
        inside = (xr / sa) ** 2 + (yr / sb) ** 2 + (zp / sc) ** 2 <= 1.0
        phantom[inside] += density

    phantom = np.clip(phantom, 0, None)
    if phantom.max() > 0:
        phantom /= phantom.max()
    return phantom


# ---------------------------------------------------------------------------
# Standard model API
# ---------------------------------------------------------------------------
def generate_test_phantom(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    phantom_type: str = "point",
    sphere_radius: float = 0.25,
) -> Tensor:
    """Generate a 3D fluorescence-density phantom.

    Parameters
    ----------
    zyx_shape : tuple[int, int, int]
    yx_pixel_size : float
        Transverse voxel spacing (used as isotropic voxel_size).
    z_pixel_size : float
        Axial voxel spacing (unused; kept for API consistency).
    phantom_type : str
        One of "point", "lines", "shepplogan".
    sphere_radius : float
        Radius for the "point" phantom.

    Returns
    -------
    Tensor
        Fluorescence density in [0, 1], shape zyx_shape.
    """
    if phantom_type == "point":
        vol = _generate_isolated_bead(zyx_shape, yx_pixel_size, sphere_radius)
    elif phantom_type == "lines":
        vol = _generate_line_pattern(zyx_shape, yx_pixel_size)
    elif phantom_type == "shepplogan":
        vol = _generate_shepp_logan_3d(zyx_shape)
    else:
        raise ValueError(f"Unknown phantom_type: {phantom_type!r}")
    return torch.tensor(vol, dtype=torch.float32)


def calculate_transfer_function(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    angles: list[float],
    device: str | torch.device = "cpu",
) -> tuple[SiddonOperator, list[Tensor]]:
    """Build Siddon operator and trivial (identity) OTF slices.

    Parameters
    ----------
    zyx_shape : tuple[int, int, int]
    yx_pixel_size : float
        Isotropic voxel spacing.
    z_pixel_size : float
        Unused; kept for API consistency.
    angles : list of float
        Tilt angles in degrees.
    device : str or torch.device

    Returns
    -------
    siddon_op : SiddonOperator
    otf_slices : list of Tensor
        List of all-ones tensors, one per angle, shape (ny, n_lateral_i).
    """
    device = torch.device(device)
    siddon_op = SiddonOperator(zyx_shape, angles, yx_pixel_size, device)
    ny = zyx_shape[1]
    otf_slices = [torch.ones(ny, n_lat, dtype=torch.complex64, device=device) for n_lat in siddon_op.n_laterals]
    return siddon_op, otf_slices


def apply_transfer_function(
    zyx_object: Tensor,
    siddon_op: SiddonOperator,
) -> list[Tensor]:
    """Forward projection via Siddon ray-tracing (no blur).

    Parameters
    ----------
    zyx_object : Tensor, shape (nz, ny, nx)
    siddon_op : SiddonOperator

    Returns
    -------
    projections : list of Tensor, each (ny, n_lateral_i)
    """
    return siddon_op.project_all(zyx_object)


def apply_inverse_transfer_function(
    projections: list[Tensor],
    siddon_op: SiddonOperator,
    otf_slices: list[Tensor],
    reconstruction_algorithm: Literal["Tikhonov", "CG"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    n_iter: int = 50,
    ramp_filter: bool = True,
) -> Tensor:
    """Reconstruct a 3D volume from projection measurements.

    Parameters
    ----------
    projections : list of Tensor
        Measured projections, one per angle.
    siddon_op : SiddonOperator
    otf_slices : list of Tensor
        Per-angle 2D OTF slices (ones for no-blur model).
    reconstruction_algorithm : {"Tikhonov", "CG"}
        "Tikhonov": per-angle inverse filter + ramp + backproject (single-shot).
        "CG": iterative conjugate gradient on the normal equation.
    regularization_strength : float
    n_iter : int
        CG iterations (ignored for Tikhonov).
    ramp_filter : bool
        Apply ramp filter (Tikhonov: in the filter; CG: as preconditioner).

    Returns
    -------
    Tensor
        Reconstructed volume, shape zyx_shape.
    """
    device = siddon_op.device

    if reconstruction_algorithm == "Tikhonov":
        vol = torch.zeros(siddon_op.zyx_shape, dtype=torch.float32, device=device)
        for i, (proj, otf_slice) in enumerate(zip(projections, otf_slices)):
            # Inverse filter: W = ramp * conj(H) / (|H|^2 + lambda)
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

        def forward(vol):
            return siddon_op.project_all(vol)

        def adjoint(projs):
            return siddon_op.backproject_all(projs, ramp_filter=ramp_filter)

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
    reconstruction_algorithm: Literal["Tikhonov", "CG"] = "Tikhonov",
    regularization_strength: float = 1e-3,
    n_iter: int = 50,
    ramp_filter: bool = True,
    device: str | torch.device = "cpu",
) -> Tensor:
    """One-liner: calculate transfer function, forward project, and reconstruct.

    Parameters
    ----------
    zyx_object : Tensor
        Ground-truth volume to project as measurements.
    yx_pixel_size : float
    z_pixel_size : float
    angles : list of float
    reconstruction_algorithm : {"Tikhonov", "CG"}
    regularization_strength : float
    n_iter : int
    ramp_filter : bool
    device : str or torch.device

    Returns
    -------
    Tensor
        Reconstructed volume.
    """
    siddon_op, otf_slices = calculate_transfer_function(
        zyx_object.shape,
        yx_pixel_size,
        z_pixel_size,
        angles,
        device,
    )
    zyx_object = zyx_object.to(device)
    projections = apply_transfer_function(zyx_object, siddon_op)
    return apply_inverse_transfer_function(
        projections,
        siddon_op,
        otf_slices,
        reconstruction_algorithm=reconstruction_algorithm,
        regularization_strength=regularization_strength,
        n_iter=n_iter,
        ramp_filter=ramp_filter,
    )
