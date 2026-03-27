"""Phantom generation for benchmarks, testing, and simulation.

Provides explicit, reproducible phantom generation with full metadata.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class Phantom:
    """A simulated sample with ground truth.

    Attributes
    ----------
    phase : Tensor
        Refractive index difference from background, shape ``(..., Y, X)``.
        Units: dimensionless (dn). Positive = higher RI than medium.
    absorption : Tensor
        Absorption coefficient, shape ``(..., Y, X)``.
        Units: arbitrary (non-negative).
    fluorescence : Tensor
        Fluorophore concentration, shape ``(..., Y, X)``.
        Units: arbitrary (non-negative).
    pixel_sizes : tuple[float, float, float]
        (z, y, x) pixel sizes in um.
    metadata : dict
        All parameters used to generate this phantom, sufficient to
        reproduce it exactly.
    """

    phase: Tensor
    absorption: Tensor
    fluorescence: Tensor
    pixel_sizes: tuple[float, float, float]
    metadata: dict


def single_bead(
    shape: tuple[int, int, int] = (64, 128, 128),
    pixel_sizes: tuple[float, float, float] = (0.25, 0.1, 0.1),
    bead_radius_um: float = 2.5,
    refractive_index_diff: float = 0.05,
    absorption_coefficient: float = 0.0,
    fluorescence_intensity: float = 1.0,
    center: tuple[float, float, float] | None = None,
    blur_size_um: float = 0.1,
) -> Phantom:
    """Generate a single bead phantom.

    Creates a sphere with specified refractive index difference,
    absorption, and fluorescence intensity, blurred by a Gaussian
    to avoid sharp edges.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape (Z, Y, X) in pixels.
    pixel_sizes : tuple[float, float, float]
        (z, y, x) pixel sizes in um.
    bead_radius_um : float
        Bead radius in um.
    refractive_index_diff : float
        RI difference from background (dn). Can be positive or negative.
    absorption_coefficient : float
        Peak absorption coefficient inside the bead.
    fluorescence_intensity : float
        Peak fluorophore concentration inside the bead.
    center : tuple[float, float, float] or None
        Bead center in um, relative to volume center. None = (0, 0, 0).
    blur_size_um : float
        Gaussian blur standard deviation in um.

    Returns
    -------
    Phantom
        Phase, absorption, and fluorescence ground truth with metadata.

    Examples
    --------
    >>> phantom = single_bead(shape=(32, 64, 64), bead_radius_um=2.0)
    >>> phantom.phase.shape
    torch.Size([32, 64, 64])
    >>> phantom.metadata["bead_radius_um"]
    2.0
    """
    if center is None:
        center = (0.0, 0.0, 0.0)

    metadata = {
        "type": "single_bead",
        "shape": list(shape),
        "pixel_sizes": list(pixel_sizes),
        "bead_radius_um": bead_radius_um,
        "refractive_index_diff": refractive_index_diff,
        "absorption_coefficient": absorption_coefficient,
        "fluorescence_intensity": fluorescence_intensity,
        "center": list(center),
        "blur_size_um": blur_size_um,
    }

    mask = _sphere_mask(shape, pixel_sizes, bead_radius_um, center)
    blurred = _gaussian_blur(mask, pixel_sizes, blur_size_um)

    phase = blurred * refractive_index_diff
    absorption = blurred * absorption_coefficient
    fluorescence = blurred * fluorescence_intensity

    return Phantom(
        phase=phase,
        absorption=absorption,
        fluorescence=fluorescence,
        pixel_sizes=pixel_sizes,
        metadata=metadata,
    )


def random_beads(
    shape: tuple[int, int, int] = (64, 128, 128),
    pixel_sizes: tuple[float, float, float] = (0.25, 0.1, 0.1),
    n_beads: int = 10,
    bead_radius_um: float = 1.0,
    refractive_index_diff: float = 0.03,
    absorption_coefficient: float = 0.0,
    fluorescence_intensity: float = 1.0,
    blur_size_um: float = 0.1,
    seed: int = 42,
) -> Phantom:
    """Generate randomly placed beads.

    Creates multiple spheres at random positions, with the same radius
    and optical properties. Beads are placed with no overlap
    (minimum center-to-center distance of 2 * bead_radius_um).

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape (Z, Y, X) in pixels.
    pixel_sizes : tuple[float, float, float]
        (z, y, x) pixel sizes in um.
    n_beads : int
        Number of beads.
    bead_radius_um : float
        Bead radius in um.
    refractive_index_diff : float
        RI difference from background (dn).
    absorption_coefficient : float
        Peak absorption coefficient inside each bead.
    fluorescence_intensity : float
        Peak fluorophore concentration inside each bead.
    blur_size_um : float
        Gaussian blur standard deviation in um.
        This blur softens the edges of the object e.g. from motion.
        Optical blur comes later during imaging.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Phantom
        Phase, absorption, and fluorescence ground truth with metadata.

    Examples
    --------
    >>> phantom = random_beads(shape=(32, 64, 64), n_beads=5, seed=0)
    >>> phantom.phase.shape
    torch.Size([32, 64, 64])
    >>> phantom.metadata["n_beads"]
    5
    """
    metadata = {
        "type": "random_beads",
        "shape": list(shape),
        "pixel_sizes": list(pixel_sizes),
        "n_beads": n_beads,
        "bead_radius_um": bead_radius_um,
        "refractive_index_diff": refractive_index_diff,
        "absorption_coefficient": absorption_coefficient,
        "fluorescence_intensity": fluorescence_intensity,
        "blur_size_um": blur_size_um,
        "seed": seed,
    }

    rng = np.random.default_rng(seed)

    # Volume extent in um
    z_extent = shape[0] * pixel_sizes[0]
    y_extent = shape[1] * pixel_sizes[1]
    x_extent = shape[2] * pixel_sizes[2]

    # Place beads with rejection sampling (no overlap)
    margin = bead_radius_um
    min_dist = 2 * bead_radius_um
    bounds_z = (-z_extent / 2 + margin, z_extent / 2 - margin)
    bounds_y = (-y_extent / 2 + margin, y_extent / 2 - margin)
    bounds_x = (-x_extent / 2 + margin, x_extent / 2 - margin)

    if bounds_z[0] >= bounds_z[1] or bounds_y[0] >= bounds_y[1] or bounds_x[0] >= bounds_x[1]:
        raise ValueError(
            f"Could only place 0/{n_beads} non-overlapping "
            f"beads. Try fewer beads, a smaller radius, or a larger volume."
        )

    centers = []
    max_attempts = n_beads * 1000
    attempts = 0
    while len(centers) < n_beads and attempts < max_attempts:
        candidate = np.array(
            [
                rng.uniform(*bounds_z),
                rng.uniform(*bounds_y),
                rng.uniform(*bounds_x),
            ]
        )
        if all(np.linalg.norm(candidate - c) >= min_dist for c in centers):
            centers.append(candidate)
        attempts += 1

    if len(centers) < n_beads:
        raise ValueError(
            f"Could only place {len(centers)}/{n_beads} non-overlapping "
            f"beads. Try fewer beads, a smaller radius, or a larger volume."
        )

    # Accumulate all beads into one mask
    combined_mask = torch.zeros(shape, dtype=torch.float32)
    for c in centers:
        center = (float(c[0]), float(c[1]), float(c[2]))
        mask = _sphere_mask(shape, pixel_sizes, bead_radius_um, center)
        combined_mask = torch.maximum(combined_mask, mask)

    blurred = _gaussian_blur(combined_mask, pixel_sizes, blur_size_um)

    phase = blurred * refractive_index_diff
    absorption = blurred * absorption_coefficient
    fluorescence = blurred * fluorescence_intensity

    return Phantom(
        phase=phase,
        absorption=absorption,
        fluorescence=fluorescence,
        pixel_sizes=pixel_sizes,
        metadata=metadata,
    )


def _sphere_mask(
    shape: tuple[int, int, int],
    pixel_sizes: tuple[float, float, float],
    radius_um: float,
    center_um: tuple[float, float, float],
) -> Tensor:
    """Create a binary sphere mask.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape (Z, Y, X).
    pixel_sizes : tuple[float, float, float]
        (z, y, x) pixel sizes in um.
    radius_um : float
        Sphere radius in um.
    center_um : tuple[float, float, float]
        Sphere center in um, relative to volume center.

    Returns
    -------
    Tensor
        Binary mask, shape (Z, Y, X), float32.
    """
    Z, Y, X = shape
    z = (torch.arange(Z) - Z // 2) * pixel_sizes[0] - center_um[0]
    y = (torch.arange(Y) - Y // 2) * pixel_sizes[1] - center_um[1]
    x = (torch.arange(X) - X // 2) * pixel_sizes[2] - center_um[2]

    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    r_sq = xx**2 + yy**2 + zz**2

    mask = torch.zeros(shape, dtype=torch.float32)
    mask[r_sq < radius_um**2] = 1.0
    return mask


def _gaussian_blur(
    volume: Tensor,
    pixel_sizes: tuple[float, float, float],
    sigma_um: float,
) -> Tensor:
    """Apply Gaussian blur in Fourier domain.

    Parameters
    ----------
    volume : Tensor
        Input volume, shape (Z, Y, X).
    pixel_sizes : tuple[float, float, float]
        (z, y, x) pixel sizes in um.
    sigma_um : float
        Gaussian standard deviation in um.

    Returns
    -------
    Tensor
        Blurred volume, non-negative, peak-normalized to [0, 1].
    """
    if sigma_um <= 0:
        peak = volume.max()
        if peak > 0:
            return volume / peak
        return volume

    Z, Y, X = volume.shape
    z = (torch.arange(Z) - Z // 2) * pixel_sizes[0]
    y = (torch.arange(Y) - Y // 2) * pixel_sizes[1]
    x = (torch.arange(X) - X // 2) * pixel_sizes[2]

    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    r_sq = xx**2 + yy**2 + zz**2

    gaussian = torch.exp(-r_sq / (2 * sigma_um**2))

    blurred = torch.real(torch.fft.ifftn(torch.fft.fftn(volume) * torch.fft.fftn(torch.fft.ifftshift(gaussian))))
    blurred = torch.clamp(blurred, min=0)

    peak = blurred.max()
    if peak > 0:
        blurred = blurred / peak

    return blurred
