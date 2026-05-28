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
    fluorescence_background: float = 0.0,
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
    fluorescence_background : float
        Constant baseline added to the fluorescence channel everywhere
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
        "fluorescence_background": fluorescence_background,
        "center": list(center),
        "blur_size_um": blur_size_um,
    }

    mask = _sphere_mask(shape, pixel_sizes, bead_radius_um, center)
    blurred = _gaussian_blur(mask, pixel_sizes, blur_size_um)

    phase = blurred * refractive_index_diff
    absorption = blurred * absorption_coefficient
    fluorescence = blurred * fluorescence_intensity + fluorescence_background

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
    fluorescence_background: float = 0.0,
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
    fluorescence_background : float
        Constant baseline added to the fluorescence channel everywhere
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
        "fluorescence_background": fluorescence_background,
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
    fluorescence = blurred * fluorescence_intensity + fluorescence_background

    return Phantom(
        phase=phase,
        absorption=absorption,
        fluorescence=fluorescence,
        pixel_sizes=pixel_sizes,
        metadata=metadata,
    )


def grid_beads(
    shape: tuple[int, int, int] = (32, 256, 256),
    pixel_sizes: tuple[float, float, float] = (0.25, 0.1, 0.1),
    grid_shape: tuple[int, int] = (5, 5),
    bead_radius_um: float = 0.5,
    refractive_index_diff: float = 0.03,
    absorption_coefficient: float = 0.0,
    fluorescence_intensity: float = 1.0,
    fluorescence_background: float = 0.0,
    z_plane_um: float = 0.0,
    margin_um: float | None = None,
    blur_size_um: float = 0.05,
) -> Phantom:
    """Generate a regular 2D grid of beads in a single Z plane.

    All beads share radius and optical properties; centers are placed on
    a uniformly spaced ``grid_shape`` lattice in the YX plane at axial
    position ``z_plane_um``. The resulting phantom is "thin" — all
    contrast lies on the requested plane.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape ``(Z, Y, X)`` in pixels.
    pixel_sizes : tuple[float, float, float]
        ``(z, y, x)`` pixel sizes in microns.
    grid_shape : tuple[int, int]
        Number of beads along ``(Y, X)``.
    bead_radius_um : float
        Bead radius in microns. Default 0.5 um (1 um diameter).
    refractive_index_diff : float
        RI difference from background (dn).
    absorption_coefficient : float
        Peak absorption inside each bead.
    fluorescence_intensity : float
        Peak fluorophore concentration inside each bead.
    fluorescence_background : float
        Constant baseline added to the fluorescence channel.
    z_plane_um : float
        Axial position of the bead plane, relative to volume center.
    margin_um : float, optional
        Margin between beads and the FOV boundary (in microns). If
        ``None``, half a tile is used so beads sit at tile centers.
    blur_size_um : float
        Gaussian blur standard deviation in microns to soften edges.

    Returns
    -------
    Phantom
        Phase, absorption, and fluorescence ground truth with metadata
        including the bead-center grid (``centers_um``).

    Examples
    --------
    >>> phantom = grid_beads(grid_shape=(3, 3))
    >>> len(phantom.metadata["centers_um"]) == 9
    True
    """
    metadata = {
        "type": "grid_beads",
        "shape": list(shape),
        "pixel_sizes": list(pixel_sizes),
        "grid_shape": list(grid_shape),
        "bead_radius_um": bead_radius_um,
        "refractive_index_diff": refractive_index_diff,
        "absorption_coefficient": absorption_coefficient,
        "fluorescence_intensity": fluorescence_intensity,
        "fluorescence_background": fluorescence_background,
        "z_plane_um": z_plane_um,
        "blur_size_um": blur_size_um,
    }

    y_extent = shape[1] * pixel_sizes[1]
    x_extent = shape[2] * pixel_sizes[2]

    ny, nx = grid_shape
    if margin_um is None:
        margin_y = y_extent / (2 * ny)
        margin_x = x_extent / (2 * nx)
    else:
        margin_y = margin_x = float(margin_um)

    if ny == 1:
        ys = np.array([0.0])
    else:
        ys = np.linspace(-y_extent / 2 + margin_y, y_extent / 2 - margin_y, ny)
    if nx == 1:
        xs = np.array([0.0])
    else:
        xs = np.linspace(-x_extent / 2 + margin_x, x_extent / 2 - margin_x, nx)

    centers = []
    for y in ys:
        for x in xs:
            centers.append((float(z_plane_um), float(y), float(x)))
    metadata["centers_um"] = [list(c) for c in centers]

    combined_mask = torch.zeros(shape, dtype=torch.float32)
    for c in centers:
        combined_mask = torch.maximum(combined_mask, _sphere_mask(shape, pixel_sizes, bead_radius_um, c))

    blurred = _gaussian_blur(combined_mask, pixel_sizes, blur_size_um)

    phase = blurred * refractive_index_diff
    absorption = blurred * absorption_coefficient
    fluorescence = blurred * fluorescence_intensity + fluorescence_background

    return Phantom(
        phase=phase,
        absorption=absorption,
        fluorescence=fluorescence,
        pixel_sizes=pixel_sizes,
        metadata=metadata,
    )


def grid_beads_gaussian(
    shape: tuple[int, int, int] = (32, 256, 256),
    pixel_sizes: tuple[float, float, float] = (0.25, 0.1, 0.1),
    grid_shape: tuple[int, int] = (5, 5),
    sigma_um: tuple[float, float, float] = (0.15, 0.1, 0.1),
    fluorescence_intensity: float = 1.0,
    fluorescence_background: float = 0.0,
    refractive_index_diff: float = 0.0,
    z_plane_um: float = 0.0,
    margin_um: float | None = None,
) -> Phantom:
    """Pixel-aligned 2D grid of 3D Gaussian beads.

    Replaces the sphere-mask + blur composition used by ``grid_beads``
    with a direct anisotropic Gaussian intensity profile centered on the
    nearest *integer pixel* of each grid position. Both choices remove
    sub-pixel sampling artefacts (octagonal beads, peak-intensity drift
    across the grid) that surface when bead radii are close to one voxel.

    Each bead contributes ``exp(-(z² + y² + x²) / (2 σ²))`` (with
    independent σ per axis) and beads add together where they overlap.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape ``(Z, Y, X)`` in pixels.
    pixel_sizes : tuple[float, float, float]
        ``(z, y, x)`` pixel sizes in microns.
    grid_shape : tuple[int, int]
        Number of beads along ``(Y, X)``.
    sigma_um : tuple[float, float, float]
        Gaussian σ along ``(z, y, x)`` in microns. Default ``(0.15, 0.1,
        0.1)`` matches a sub-diffraction bead at 100 nm pixels.
    fluorescence_intensity : float
        Peak fluorophore concentration at each bead center.
    fluorescence_background : float
        Constant baseline added to the fluorescence channel.
    refractive_index_diff : float
        RI difference from background (dn). Phase channel uses the same
        Gaussian profile multiplied by ``refractive_index_diff``.
    z_plane_um : float
        Axial position of the bead plane, relative to volume center.
    margin_um : float, optional
        Margin between beads and the FOV boundary in microns. If None,
        defaults to half a tile cell so beads sit at tile centers for
        the matching ``grid_shape`` tile partition.

    Returns
    -------
    Phantom
        Phase, absorption, and fluorescence ground truth, plus metadata
        with the snapped pixel centers (``centers_pix`` and ``centers_um``).
    """
    Z, Y, X = shape
    pz, py, px = pixel_sizes
    sz, sy, sx = sigma_um
    ny, nx = grid_shape

    y_extent = Y * py
    x_extent = X * px
    if margin_um is None:
        margin_y = y_extent / (2 * ny)
        margin_x = x_extent / (2 * nx)
    else:
        margin_y = margin_x = float(margin_um)

    if ny == 1:
        ys_um = np.array([0.0])
    else:
        ys_um = np.linspace(-y_extent / 2 + margin_y, y_extent / 2 - margin_y, ny)
    if nx == 1:
        xs_um = np.array([0.0])
    else:
        xs_um = np.linspace(-x_extent / 2 + margin_x, x_extent / 2 - margin_x, nx)
    z_pix = int(np.round(z_plane_um / pz)) + Z // 2

    ys_pix = (np.round(ys_um / py) + Y // 2).astype(int)
    xs_pix = (np.round(xs_um / px) + X // 2).astype(int)
    centers_pix = [(int(z_pix), int(yp), int(xp)) for yp in ys_pix for xp in xs_pix]
    centers_um = [
        (
            (cz - Z // 2) * pz,
            (cy - Y // 2) * py,
            (cx - X // 2) * px,
        )
        for cz, cy, cx in centers_pix
    ]

    metadata = {
        "type": "grid_beads_gaussian",
        "shape": list(shape),
        "pixel_sizes": list(pixel_sizes),
        "grid_shape": list(grid_shape),
        "sigma_um": list(sigma_um),
        "fluorescence_intensity": fluorescence_intensity,
        "fluorescence_background": fluorescence_background,
        "refractive_index_diff": refractive_index_diff,
        "z_plane_um": z_plane_um,
        "centers_pix": [list(c) for c in centers_pix],
        "centers_um": [list(c) for c in centers_um],
    }

    z = (torch.arange(Z) - Z // 2) * pz
    y = (torch.arange(Y) - Y // 2) * py
    x = (torch.arange(X) - X // 2) * px
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    accum = torch.zeros(shape, dtype=torch.float32)
    for cz_um, cy_um, cx_um in centers_um:
        bead = torch.exp(
            -(((zz - cz_um) ** 2) / (2 * sz**2) + ((yy - cy_um) ** 2) / (2 * sy**2) + ((xx - cx_um) ** 2) / (2 * sx**2))
        )
        accum = accum + bead

    fluorescence = accum * fluorescence_intensity + fluorescence_background
    phase = accum * refractive_index_diff
    absorption = torch.zeros_like(accum)

    return Phantom(
        phase=phase,
        absorption=absorption,
        fluorescence=fluorescence,
        pixel_sizes=pixel_sizes,
        metadata=metadata,
    )


def ring_beads_gaussian(
    shape: tuple[int, int, int] = (32, 256, 256),
    pixel_sizes: tuple[float, float, float] = (0.25, 0.1, 0.1),
    n_rings: int = 8,
    beads_per_unit: int = 6,
    include_center: bool = True,
    r_max_frac: float = 0.94,
    sigma_um: tuple[float, float, float] = (0.15, 0.1, 0.1),
    fluorescence_intensity: float = 1.0,
    fluorescence_background: float = 0.0,
    refractive_index_diff: float = 0.0,
    z_plane_um: float = 0.0,
) -> Phantom:
    """Beads arranged on concentric rings in a single Z plane.

    Reproduces the RDM Fig. 5d-right layout: ``n_rings`` concentric rings
    of beads, with ring ``k`` carrying ``beads_per_unit * k`` beads so the
    angular density stays roughly constant. Ring radii are evenly spaced
    out to ``r_max_frac`` of the inscribed-circle radius
    ``min(Y * py, X * px) / 2``. Beads are 3D Gaussians with anisotropic
    ``sigma_um``, snapped to integer pixels in the same style as
    :func:`grid_beads_gaussian`.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Volume shape ``(Z, Y, X)`` in pixels.
    pixel_sizes : tuple[float, float, float]
        ``(z, y, x)`` pixel sizes in microns.
    n_rings : int
        Number of rings (excluding the center bead).
    beads_per_unit : int
        Beads on ring ``k`` is ``beads_per_unit * k``. Constant angular
        density: outer rings get more beads than inner rings.
    include_center : bool
        Place a single bead at the optical axis.
    r_max_frac : float
        Outermost ring radius as a fraction of the inscribed-circle
        radius. Stay below 1.0 so beads remain inside the FOV.
    sigma_um : tuple[float, float, float]
        Gaussian sigma along ``(z, y, x)`` in microns.
    fluorescence_intensity : float
        Peak fluorophore concentration at each bead center.
    fluorescence_background : float
        Constant baseline added to the fluorescence channel.
    refractive_index_diff : float
        RI difference from background (dn) applied to the phase channel.
    z_plane_um : float
        Axial position of the bead plane, relative to volume center.

    Returns
    -------
    Phantom
        Phase, absorption (zero), and fluorescence ground truth, plus
        metadata with ``centers_pix``, ``centers_um``, and
        ``beads_per_ring``.

    Examples
    --------
    >>> phantom = ring_beads_gaussian(shape=(8, 64, 64), n_rings=2, beads_per_unit=4)
    >>> phantom.fluorescence.shape
    torch.Size([8, 64, 64])
    >>> 1 + 4 + 8 == len(phantom.metadata["centers_pix"])
    True
    """
    Z, Y, X = shape
    pz, py, px = pixel_sizes
    sz, sy, sx = sigma_um

    inscribed_radius_um = min(Y * py, X * px) / 2
    radii_um = np.linspace(0.0, inscribed_radius_um * r_max_frac, n_rings + 1)[1:]
    beads_per_ring = [int(beads_per_unit * (k + 1)) for k in range(n_rings)]

    centers_um: list[tuple[float, float, float]] = []
    if include_center:
        centers_um.append((float(z_plane_um), 0.0, 0.0))

    for r_um, n_beads in zip(radii_um, beads_per_ring):
        # Start ring k at angle pi/(2*n_beads) so beads aren't on the axes
        offset = np.pi / (2 * n_beads)
        angles = offset + np.linspace(0.0, 2 * np.pi, n_beads, endpoint=False)
        for theta in angles:
            centers_um.append(
                (
                    float(z_plane_um),
                    float(r_um * np.sin(theta)),
                    float(r_um * np.cos(theta)),
                )
            )

    centers_pix: list[tuple[int, int, int]] = []
    for cz_um, cy_um, cx_um in centers_um:
        centers_pix.append(
            (
                int(np.round(cz_um / pz)) + Z // 2,
                int(np.round(cy_um / py)) + Y // 2,
                int(np.round(cx_um / px)) + X // 2,
            )
        )

    metadata = {
        "type": "ring_beads_gaussian",
        "shape": list(shape),
        "pixel_sizes": list(pixel_sizes),
        "n_rings": n_rings,
        "beads_per_unit": beads_per_unit,
        "include_center": include_center,
        "r_max_frac": r_max_frac,
        "sigma_um": list(sigma_um),
        "fluorescence_intensity": fluorescence_intensity,
        "fluorescence_background": fluorescence_background,
        "refractive_index_diff": refractive_index_diff,
        "z_plane_um": z_plane_um,
        "beads_per_ring": beads_per_ring,
        "centers_pix": [list(c) for c in centers_pix],
        "centers_um": [list(c) for c in centers_um],
    }

    z_axis = (torch.arange(Z) - Z // 2) * pz
    y_axis = (torch.arange(Y) - Y // 2) * py
    x_axis = (torch.arange(X) - X // 2) * px
    zz, yy, xx = torch.meshgrid(z_axis, y_axis, x_axis, indexing="ij")

    accum = torch.zeros(shape, dtype=torch.float32)
    # Center each bead at its snapped pixel position so beads with the
    # same r_um are placed identically up to rotation.
    for cz_pix, cy_pix, cx_pix in centers_pix:
        cz_um = (cz_pix - Z // 2) * pz
        cy_um = (cy_pix - Y // 2) * py
        cx_um = (cx_pix - X // 2) * px
        bead = torch.exp(
            -(((zz - cz_um) ** 2) / (2 * sz**2) + ((yy - cy_um) ** 2) / (2 * sy**2) + ((xx - cx_um) ** 2) / (2 * sx**2))
        )
        accum = accum + bead

    fluorescence = accum * fluorescence_intensity + fluorescence_background
    phase = accum * refractive_index_diff
    absorption = torch.zeros_like(accum)

    return Phantom(
        phase=phase,
        absorption=absorption,
        fluorescence=fluorescence,
        pixel_sizes=pixel_sizes,
        metadata=metadata,
    )


def ring_beads_2d_gaussian(
    shape: tuple[int, int, int] = (32, 192, 192),
    pixel_sizes: tuple[float, float, float] = (0.25, 0.1, 0.1),
    n_rings: int = 6,
    beads_per_unit: int = 6,
    include_center: bool = True,
    r_max_frac: float = 0.94,
    sigma_um: tuple[float, float] = (0.1, 0.1),
    fluorescence_intensity: float = 1.0,
    fluorescence_background: float = 0.0,
    refractive_index_diff: float = 0.0,
) -> Phantom:
    """Truly 2D phantom of beads on concentric rings.

    Returns a :class:`Phantom` whose ``fluorescence`` and ``phase``
    arrays are 2D ``(Y, X)``. Use with ``simulate_*_2d_to_3d`` forward
    models that take a 2D source and emit a 3D defocus stack.

    Parameters
    ----------
    shape : tuple[int, int, int]
        ``(Z, Y, X)`` shape. Only ``Y, X`` define the 2D phantom array;
        ``Z`` is recorded in metadata so the simulator knows how many
        defocus planes to render.
    pixel_sizes : tuple[float, float, float]
        ``(z, y, x)`` pixel sizes in microns.
    n_rings : int
        Number of rings (excluding the center bead).
    beads_per_unit : int
        Beads on ring ``k`` = ``beads_per_unit * k``.
    include_center : bool
        Place one bead at the optical axis.
    r_max_frac : float
        Outermost ring radius as a fraction of the inscribed-circle
        radius ``min(Y * py, X * px) / 2``.
    sigma_um : tuple[float, float]
        Gaussian sigma along ``(y, x)`` in microns.
    fluorescence_intensity : float
        Peak fluorophore concentration at each bead center.
    fluorescence_background : float
        Constant baseline added to the fluorescence channel.
    refractive_index_diff : float
        RI difference from background applied to the phase channel.

    Returns
    -------
    Phantom
        ``fluorescence``, ``phase``, and ``absorption`` are 2D tensors
        of shape ``(Y, X)``. ``pixel_sizes`` retains the 3-tuple form so
        downstream simulators can read the z-spacing.

    Examples
    --------
    >>> phantom = ring_beads_2d_gaussian(shape=(8, 64, 64), n_rings=2, beads_per_unit=4)
    >>> phantom.fluorescence.shape
    torch.Size([64, 64])
    >>> 1 + 4 + 8 == len(phantom.metadata["centers_pix"])
    True
    """
    Z, Y, X = shape
    sy, sx = sigma_um
    pz, py, px = (float(v) for v in pixel_sizes)

    inscribed_radius_um = min(Y * py, X * px) / 2
    radii_um = np.linspace(0.0, inscribed_radius_um * r_max_frac, n_rings + 1)[1:]
    beads_per_ring = [int(beads_per_unit * (k + 1)) for k in range(n_rings)]

    centers_um: list[tuple[float, float]] = []
    if include_center:
        centers_um.append((0.0, 0.0))
    for r_um, n_beads in zip(radii_um, beads_per_ring):
        offset = np.pi / (2 * n_beads)
        angles = offset + np.linspace(0.0, 2 * np.pi, n_beads, endpoint=False)
        for theta in angles:
            centers_um.append((float(r_um * np.sin(theta)), float(r_um * np.cos(theta))))

    centers_pix: list[tuple[int, int]] = []
    for cy_um, cx_um in centers_um:
        centers_pix.append(
            (
                int(np.round(cy_um / py)) + Y // 2,
                int(np.round(cx_um / px)) + X // 2,
            )
        )

    metadata = {
        "type": "ring_beads_2d_gaussian",
        "shape": list(shape),
        "pixel_sizes": list(pixel_sizes),
        "Z_stack": Z,
        "n_rings": n_rings,
        "beads_per_unit": beads_per_unit,
        "include_center": include_center,
        "r_max_frac": r_max_frac,
        "sigma_um": list(sigma_um),
        "fluorescence_intensity": fluorescence_intensity,
        "fluorescence_background": fluorescence_background,
        "refractive_index_diff": refractive_index_diff,
        "beads_per_ring": beads_per_ring,
        "centers_pix": [list(c) for c in centers_pix],
        "centers_um": [list(c) for c in centers_um],
    }

    y_axis = (torch.arange(Y) - Y // 2) * py
    x_axis = (torch.arange(X) - X // 2) * px
    yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")

    accum = torch.zeros((Y, X), dtype=torch.float32)
    for cy_pix, cx_pix in centers_pix:
        cy_um = (cy_pix - Y // 2) * py
        cx_um = (cx_pix - X // 2) * px
        bead = torch.exp(-(((yy - cy_um) ** 2) / (2 * sy**2) + ((xx - cx_um) ** 2) / (2 * sx**2)))
        accum = accum + bead

    fluorescence = accum * fluorescence_intensity + fluorescence_background
    phase = accum * refractive_index_diff
    absorption = torch.zeros_like(accum)

    return Phantom(
        phase=phase,
        absorption=absorption,
        fluorescence=fluorescence,
        pixel_sizes=(pz, py, px),
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
