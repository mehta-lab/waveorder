"""Siddon ray-tracing via sparse matrices, ramp filter, CG-Tikhonov solver,
and OTF slice extraction for projection tomography.

Forward and adjoint operators for line-integral projection through a 3D
volume at arbitrary tilt angles (rotation around Y in the ZX plane).
All heavy computation uses PyTorch sparse matmul on CPU or GPU.
"""

import math

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Ray geometry (numpy, runs once during operator construction)
# ---------------------------------------------------------------------------
def _ray_trace(volume_shape, angle_deg, voxel_size):
    """Compute ray geometry for Siddon projection at a given angle.

    Returns a list of (il, iz, ix, lengths) tuples -- one per detector
    column that intersects the volume. Rays propagate in the ZX plane;
    all Y rows share the same trace.

    Parameters
    ----------
    volume_shape : tuple of int
        (nz, ny, nx).
    angle_deg : float
        Tilt angle in degrees from the Z-axis.
    voxel_size : float
        Isotropic voxel spacing in um.

    Returns
    -------
    n_lateral : int
        Number of detector columns.
    traces : list of (il, iz, ix, lengths)
        Per-column voxel indices and segment lengths.
    """
    nz, _ny, nx = volume_shape

    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    lateral_extent = nz * abs(sin_t) + nx * abs(cos_t)
    n_lateral = int(np.ceil(lateral_extent))

    cz, cx = nz / 2.0, nx / 2.0
    z_planes = np.arange(1, nz, dtype=np.float64)
    x_planes = np.arange(1, nx, dtype=np.float64)

    traces = []
    for il in range(n_lateral):
        lat_offset = il - n_lateral / 2.0
        origin_z = cz + lat_offset * (-sin_t)
        origin_x = cx + lat_offset * cos_t

        t_min = -1e10
        t_max = 1e10

        if abs(cos_t) > 1e-10:
            t_z0 = -origin_z / cos_t
            t_zn = (nz - origin_z) / cos_t
            t_min = max(t_min, min(t_z0, t_zn))
            t_max = min(t_max, max(t_z0, t_zn))
        elif origin_z < 0 or origin_z >= nz:
            continue

        if abs(sin_t) > 1e-10:
            t_x0 = -origin_x / sin_t
            t_xn = (nx - origin_x) / sin_t
            t_min = max(t_min, min(t_x0, t_xn))
            t_max = min(t_max, max(t_x0, t_xn))
        elif origin_x < 0 or origin_x >= nx:
            continue

        if t_min >= t_max:
            continue

        t_vals = [t_min]
        if abs(cos_t) > 1e-10:
            t_z = (z_planes - origin_z) / cos_t
            t_z = t_z[(t_z > t_min) & (t_z < t_max)]
            t_vals.append(t_z)
        if abs(sin_t) > 1e-10:
            t_x = (x_planes - origin_x) / sin_t
            t_x = t_x[(t_x > t_min) & (t_x < t_max)]
            t_vals.append(t_x)
        t_vals.append(np.array([t_max]))
        all_t = np.unique(np.concatenate([np.atleast_1d(v) for v in t_vals]))

        if len(all_t) < 2:
            continue

        t_mid = (all_t[:-1] + all_t[1:]) / 2.0
        iz = np.floor(origin_z + t_mid * cos_t).astype(int)
        ix = np.floor(origin_x + t_mid * sin_t).astype(int)

        valid = (iz >= 0) & (iz < nz) & (ix >= 0) & (ix < nx)
        iz = iz[valid]
        ix = ix[valid]
        if len(iz) == 0:
            continue

        lengths = np.diff(all_t)[valid] * voxel_size
        traces.append((il, iz, ix, lengths))

    return n_lateral, traces


def _build_sparse_matrix(volume_shape, angle_deg, voxel_size, device):
    """Build a sparse projection matrix for one angle.

    The matrix A has shape (n_lateral, nz * nx). Each row corresponds
    to a detector column; nonzero entries are the intersection lengths
    of that ray with the voxels it traverses.

    Parameters
    ----------
    volume_shape : tuple of int
        (nz, ny, nx).
    angle_deg : float
        Tilt angle in degrees.
    voxel_size : float
        Isotropic voxel spacing in um.
    device : torch.device
        Target device for the sparse tensor.

    Returns
    -------
    A : torch.sparse_coo_tensor, shape (n_lateral, nz * nx)
    n_lateral : int
    """
    nz, _ny, nx = volume_shape
    n_lateral, traces = _ray_trace(volume_shape, angle_deg, voxel_size)

    if len(traces) == 0:
        indices = torch.zeros((2, 0), dtype=torch.long)
        values = torch.zeros(0, dtype=torch.float32)
    else:
        row_indices = []
        col_indices = []
        vals = []
        for il, iz, ix, lengths in traces:
            n_seg = len(iz)
            row_indices.append(np.full(n_seg, il, dtype=np.int64))
            col_indices.append(iz * nx + ix)
            vals.append(lengths.astype(np.float32))

        rows = np.concatenate(row_indices)
        cols = np.concatenate(col_indices)
        values = np.concatenate(vals)

        indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
        values = torch.tensor(values, dtype=torch.float32)

    A = torch.sparse_coo_tensor(indices, values, size=(n_lateral, nz * nx)).coalesce().to(device)

    return A, n_lateral


# ---------------------------------------------------------------------------
# SiddonOperator: precomputed sparse matrices for all angles
# ---------------------------------------------------------------------------
class SiddonOperator:
    """Sparse-matrix Siddon projector for a fixed set of tilt angles.

    Precomputes one sparse matrix per angle at construction time.
    Forward projection and backprojection reduce to
    ``torch.sparse.mm``, which runs on CPU or GPU.

    Parameters
    ----------
    zyx_shape : tuple of int
        Volume shape (nz, ny, nx).
    angles : list of float
        Tilt angles in degrees.
    voxel_size : float
        Isotropic voxel spacing in um.
    device : torch.device
        CPU or CUDA device.
    """

    def __init__(self, zyx_shape, angles, voxel_size, device):
        self.zyx_shape = zyx_shape
        self.nz, self.ny, self.nx = zyx_shape
        self.angles = list(angles)
        self.voxel_size = voxel_size
        self.device = device

        self.matrices = []
        self.matrices_T = []
        self.n_laterals = []

        for angle in self.angles:
            A, n_lat = _build_sparse_matrix(zyx_shape, angle, voxel_size, device)
            self.matrices.append(A)
            self.matrices_T.append(A.t())
            self.n_laterals.append(n_lat)

    def project(self, volume, angle_idx):
        """Forward: 3D volume -> 2D projection at one angle.

        Parameters
        ----------
        volume : torch.Tensor, shape (nz, ny, nx)
        angle_idx : int

        Returns
        -------
        projection : torch.Tensor, shape (ny, n_lateral)
        """
        A = self.matrices[angle_idx]
        vol_zx = volume.permute(0, 2, 1).reshape(self.nz * self.nx, self.ny)
        proj = torch.sparse.mm(A, vol_zx)
        return proj.T

    def backproject(self, projection, angle_idx):
        """Adjoint: 2D projection -> 3D volume at one angle.

        Parameters
        ----------
        projection : torch.Tensor, shape (ny, n_lateral)
        angle_idx : int

        Returns
        -------
        volume : torch.Tensor, shape (nz, ny, nx)
        """
        AT = self.matrices_T[angle_idx]
        proj_T = projection.T
        vol_zx = torch.sparse.mm(AT, proj_T)
        return vol_zx.reshape(self.nz, self.nx, self.ny).permute(0, 2, 1)

    def project_all(self, volume):
        """Forward: project volume at all angles.

        Returns
        -------
        projections : list of torch.Tensor, each (ny, n_lateral_i)
        """
        return [self.project(volume, i) for i in range(len(self.angles))]

    def backproject_all(self, projections, ramp_filter=False):
        """Adjoint: sum backprojections from all angles.

        Parameters
        ----------
        projections : list of torch.Tensor, each (ny, n_lateral_i)
        ramp_filter : bool
            Apply Ram-Lak ramp filter before backprojection.

        Returns
        -------
        volume : torch.Tensor, shape (nz, ny, nx)
        """
        vol = torch.zeros(self.zyx_shape, dtype=torch.float32, device=self.device)
        for i, proj in enumerate(projections):
            if ramp_filter:
                proj = ramp_filter_sinogram(proj)
            vol += self.backproject(proj, i)
        return vol


# ---------------------------------------------------------------------------
# Ramp filter (torch)
# ---------------------------------------------------------------------------
def ramp_filter_sinogram(projection):
    """Apply Ram-Lak (ramp) filter along the lateral detector direction.

    The filter compensates for the 1/|omega| blur introduced by
    backprojection, preconditioning the normal operator so that all
    spatial frequencies converge at a similar rate in CG.

    Parameters
    ----------
    projection : torch.Tensor, shape (ny, n_lateral)

    Returns
    -------
    filtered : torch.Tensor, shape (ny, n_lateral)
    """
    ny, nx = projection.shape
    freqs = torch.fft.fftfreq(nx, device=projection.device)
    ramp = freqs.abs()
    ramp[0] = 0.5 * freqs[1].abs()
    ramp = ramp / ramp.max()

    ft = torch.fft.fft(projection, dim=1)
    ft = ft * ramp.unsqueeze(0)
    return torch.fft.ifft(ft, dim=1).real


# ---------------------------------------------------------------------------
# CG-Tikhonov solver (torch)
# ---------------------------------------------------------------------------
def cg_tikhonov(forward_op, adjoint_op, measurements, zyx_shape, reg_strength, n_iter, device):
    """Conjugate gradient solver for the Tikhonov normal equation.

    Solves (H* H + lambda I) x = H* y where H is the forward operator,
    H* is the adjoint, y is the measurement data, and lambda is the
    Tikhonov regularization strength.

    Parameters
    ----------
    forward_op : callable
        H: Tensor (nz, ny, nx) -> list of Tensor projections.
    adjoint_op : callable
        H*: list of Tensor projections -> Tensor (nz, ny, nx).
    measurements : list of torch.Tensor
        Observed projections y.
    zyx_shape : tuple of int
        Shape of the unknown volume.
    reg_strength : float
        Tikhonov regularization lambda.
    n_iter : int
        Maximum CG iterations.
    device : torch.device

    Returns
    -------
    x : torch.Tensor
        Reconstructed 3D volume.
    """

    def normal_op(vol):
        return adjoint_op(forward_op(vol)) + reg_strength * vol

    b = adjoint_op(measurements)
    x = torch.zeros(zyx_shape, dtype=torch.float32, device=device)
    r = b - normal_op(x)
    p = r.clone()
    rs_old = torch.sum(r * r).item()

    for _i in range(n_iter):
        Ap = normal_op(p)
        pAp = torch.sum(p * Ap).item()
        if pAp < 1e-30:
            break

        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = torch.sum(r * r).item()
        if rs_new < 1e-30:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


# ---------------------------------------------------------------------------
# OTF slice extraction (Fourier-slice theorem)
# ---------------------------------------------------------------------------
def extract_otf_slices(
    otf_3d: Tensor,
    angles: list[float],
    n_laterals: list[int],
) -> list[Tensor]:
    """Extract 2D central slices from a 3D OTF at each projection angle.

    By the Fourier-slice theorem, projecting a volume at angle theta
    samples a central plane of the 3D Fourier transform. This function
    extracts those planes from a pre-computed 3D OTF via bilinear
    interpolation.

    For rotation around Y in the ZX plane, the slice at angle theta
    maps detector lateral frequency k_l to 3D frequencies:
        kz = -sin(theta) * k_l
        kx = cos(theta) * k_l
        ky = ky  (unchanged)

    Parameters
    ----------
    otf_3d : Tensor, shape (nz, ny, nx)
        3D optical transfer function in FFT order.
    angles : list of float
        Tilt angles in degrees.
    n_laterals : list of int
        Number of lateral detector columns per angle.

    Returns
    -------
    slices : list of Tensor, each shape (ny, n_lateral_i)
        Complex 2D OTF slices in FFT order, one per angle.
    """
    nz, ny, nx = otf_3d.shape
    device = otf_3d.device

    # Work in centered (fftshift) coordinates for interpolation
    otf_shifted = torch.fft.fftshift(otf_3d)

    # Split real/imag for grid_sample (which requires real tensors)
    otf_re = otf_shifted.real.unsqueeze(0).unsqueeze(0)  # (1, 1, nz, ny, nx)
    otf_im = otf_shifted.imag.unsqueeze(0).unsqueeze(0)

    slices = []
    for angle_deg, n_lat in zip(angles, n_laterals):
        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Lateral detector frequencies in cycles/pixel (FFT order)
        k_lat = torch.fft.fftfreq(n_lat, device=device)
        # Y frequencies (FFT order)
        ky = torch.fft.fftfreq(ny, device=device)

        # Map to 3D frequency coordinates (in cycles/voxel)
        kz_freq = -sin_t * k_lat  # (n_lat,)
        kx_freq = cos_t * k_lat  # (n_lat,)

        # Convert to shifted-grid indices:
        #   frequency f -> shifted index = f * N + N/2
        kz_idx = kz_freq * nz + nz / 2.0
        kx_idx = kx_freq * nx + nx / 2.0
        ky_idx = ky * ny + ny / 2.0

        # Normalize to [-1, 1] for grid_sample (align_corners=True)
        kz_norm = kz_idx / (nz - 1) * 2 - 1  # (n_lat,)
        kx_norm = kx_idx / (nx - 1) * 2 - 1  # (n_lat,)
        ky_norm = ky_idx / (ny - 1) * 2 - 1  # (ny,)

        # Build 3D sampling grid: shape (1, 1, ny, n_lat, 3)
        # grid_sample 5D expects grid[..., 0]=W(kx), [..1]=H(ky), [..2]=D(kz)
        ky_g = ky_norm[:, None].expand(ny, n_lat)  # (ny, n_lat)
        kz_g = kz_norm[None, :].expand(ny, n_lat)
        kx_g = kx_norm[None, :].expand(ny, n_lat)

        grid = torch.stack([kx_g, ky_g, kz_g], dim=-1)  # (ny, n_lat, 3)
        grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, ny, n_lat, 3)

        slice_re = torch.nn.functional.grid_sample(
            otf_re, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )[0, 0, 0]  # (ny, n_lat)

        slice_im = torch.nn.functional.grid_sample(
            otf_im, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )[0, 0, 0]

        otf_slice = torch.complex(slice_re, slice_im)
        slices.append(otf_slice)

    return slices


# ---------------------------------------------------------------------------
# Convenience wrappers (numpy interface, backward compatibility)
# ---------------------------------------------------------------------------
def siddon_project(volume_np, angle_deg, voxel_size, mode="sum"):
    """Forward: 3D numpy volume -> 2D numpy projection at a tilt angle.

    For repeated projections at multiple angles, use SiddonOperator
    directly to avoid rebuilding the sparse matrix each time.

    Parameters
    ----------
    volume_np : np.ndarray, shape (Z, Y, X)
    angle_deg : float
    voxel_size : float
    mode : str
        "sum" for line-integral projection, "mean" divides by path length.

    Returns
    -------
    projection : np.ndarray, shape (Y, N_lateral)
    """
    device = torch.device("cpu")
    op = SiddonOperator(volume_np.shape, [angle_deg], voxel_size, device)
    vol = torch.tensor(volume_np, dtype=torch.float32, device=device)
    proj = op.project(vol, 0)
    result = proj.numpy()

    if mode == "mean":
        _n_lateral, traces = _ray_trace(volume_np.shape, angle_deg, voxel_size)
        for il, _iz, _ix, lengths in traces:
            path_len = lengths.sum()
            if path_len > 0 and il < result.shape[1]:
                result[:, il] /= path_len

    return result


def siddon_backproject(projection_np, angle_deg, zyx_shape, voxel_size):
    """Adjoint: 2D numpy projection -> 3D numpy volume.

    Parameters
    ----------
    projection_np : np.ndarray, shape (Y, N_lateral)
    angle_deg : float
    zyx_shape : tuple of int
    voxel_size : float

    Returns
    -------
    volume : np.ndarray, shape (Z, Y, X)
    """
    device = torch.device("cpu")
    op = SiddonOperator(zyx_shape, [angle_deg], voxel_size, device)
    proj = torch.tensor(projection_np, dtype=torch.float32, device=device)
    vol = op.backproject(proj, 0)
    return vol.numpy()
