"""Siddon ray-tracing via sparse matrices, ramp filter, and CG-Tikhonov solver.

Forward and adjoint operators for line-integral projection through a 3D
volume at arbitrary tilt angles (rotation around Y in the ZX plane).
All heavy computation uses PyTorch sparse matmul on CPU or GPU.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Ray geometry (numpy, runs once during operator construction)
# ---------------------------------------------------------------------------
def _ray_trace(volume_shape, angle_deg, voxel_size):
    """Compute ray geometry for Siddon projection at a given angle.

    Returns a list of (il, iz, ix, lengths) tuples — one per detector
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
    of that ray with the voxels it traverses. Since all Y rows share
    the same (Z, X) ray geometry, the 2D projection of a volume slice
    vol[:, y, :] is simply A @ vol[:, y, :].ravel().

    Forward projection (sum mode):
        proj = (A @ vol_zx).T           # vol_zx: (nz*nx, ny), proj: (ny, n_lat)

    Adjoint (backprojection):
        vol_zx += (A.T @ proj.T)        # proj.T: (n_lat, ny), vol_zx: (nz*nx, ny)

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
        Sparse projection matrix.
    n_lateral : int
        Number of detector columns.
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

    A = torch.sparse_coo_tensor(
        indices, values, size=(n_lateral, nz * nx)
    ).coalesce().to(device)

    return A, n_lateral


# ---------------------------------------------------------------------------
# SiddonOperator: precomputed sparse matrices for all angles
# ---------------------------------------------------------------------------
class SiddonOperator:
    """Sparse-matrix Siddon projector for a fixed set of tilt angles.

    Precomputes one sparse matrix per angle at construction time.
    Forward projection and backprojection then reduce to
    ``torch.sparse.mm``, which runs on CPU or GPU without Python loops
    over voxels.

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

        # Precompute sparse matrices and their transposes
        self.matrices = []      # A_i: (n_lateral_i, nz*nx)
        self.matrices_T = []    # A_i^T: (nz*nx, n_lateral_i)
        self.n_laterals = []

        for angle in self.angles:
            A, n_lat = _build_sparse_matrix(zyx_shape, angle, voxel_size, device)
            self.matrices.append(A)
            self.matrices_T.append(A.t())
            self.n_laterals.append(n_lat)

    def project(self, volume, angle_idx):
        """Forward: 3D volume -> 2D projection at one angle.

        Computes proj = (A @ vol_zx) where vol_zx is the volume
        reshaped to (nz*nx, ny). Result has shape (ny, n_lateral).

        Parameters
        ----------
        volume : torch.Tensor, shape (nz, ny, nx)
        angle_idx : int
            Index into self.angles.

        Returns
        -------
        projection : torch.Tensor, shape (ny, n_lateral)
        """
        A = self.matrices[angle_idx]
        # Permute (nz, ny, nx) → (nz, nx, ny) so that flattening the first
        # two dims gives (nz*nx, ny) with column index = iz*nx + ix,
        # matching the sparse matrix layout.
        vol_zx = volume.permute(0, 2, 1).reshape(self.nz * self.nx, self.ny)
        proj = torch.sparse.mm(A, vol_zx)  # (n_lateral, ny)
        return proj.T  # (ny, n_lateral)

    def backproject(self, projection, angle_idx):
        """Adjoint: 2D projection -> 3D volume at one angle.

        Computes vol_zx = A^T @ proj^T, then reshapes to (nz, ny, nx).
        This is the exact transpose of ``project``.

        Parameters
        ----------
        projection : torch.Tensor, shape (ny, n_lateral)
        angle_idx : int
            Index into self.angles.

        Returns
        -------
        volume : torch.Tensor, shape (nz, ny, nx)
        """
        AT = self.matrices_T[angle_idx]
        proj_T = projection.T  # (n_lateral, ny)
        vol_zx = torch.sparse.mm(AT, proj_T)  # (nz*nx, ny)
        # Reshape to (nz, nx, ny) then permute back to (nz, ny, nx)
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

    In standard parallel-beam CT, backprojection blurs the reconstruction
    by a factor proportional to 1/|omega|.  The ramp filter |omega|
    compensates for this blur.  When used inside the adjoint of an
    iterative solver, it preconditions the normal operator H^T H so that
    all spatial frequencies converge at a similar rate, reducing Gibbs
    ringing near sharp edges.

    The filter is applied per Y-row along the detector (lateral) direction
    via FFT.  DC is floored at half the first nonzero frequency to avoid
    zeroing the mean.

    Parameters
    ----------
    projection : torch.Tensor, shape (ny, n_lateral)
        2D projection image.

    Returns
    -------
    filtered : torch.Tensor, shape (ny, n_lateral)
        Ramp-filtered projection.
    """
    ny, nx = projection.shape
    freqs = torch.fft.fftfreq(nx, device=projection.device)
    ramp = freqs.abs()
    # Floor DC at half the first nonzero frequency to preserve the mean
    ramp[0] = 0.5 * freqs[1].abs()
    # Normalize so the peak weight is 1
    ramp = ramp / ramp.max()

    ft = torch.fft.fft(projection, dim=1)
    ft = ft * ramp.unsqueeze(0)
    return torch.fft.ifft(ft, dim=1).real


# ---------------------------------------------------------------------------
# CG-Tikhonov solver (torch)
# ---------------------------------------------------------------------------
def cg_tikhonov(forward_op, adjoint_op, measurements, zyx_shape, reg_strength, n_iter, device):
    """Conjugate gradient solver for the Tikhonov normal equation.

    Solves::

        (H* H + lambda I) x = H* y

    where H is the forward operator (projection), H* is the adjoint
    (backprojection, optionally ramp-filtered), y is the measurement
    data, and lambda is the Tikhonov regularization strength.

    **Why CG?**  The normal operator A = H*H + lambda I is symmetric
    positive-definite, so CG converges to the unique minimizer without
    storing any matrix.  Each iteration requires one forward and one
    adjoint evaluation.

    **Preconditioning via ramp-filtered adjoint.**  When H* is plain
    backprojection, H*H has a frequency response that falls as ~1/|omega|.
    Low frequencies dominate the gradient, so CG resolves them first;
    high-frequency edges converge slowly and ring in the interim.
    Replacing H* with ramp-filtered backprojection (|omega| weighting)
    flattens the spectrum of H*H, making all frequencies converge at
    a similar rate.  This is equivalent to left-preconditioning with
    the ramp filter.  The regularization term lambda I then acts
    uniformly across frequencies rather than predominantly on the
    poorly-conditioned high frequencies.

    Parameters
    ----------
    forward_op : callable
        H: torch.Tensor (nz, ny, nx) -> list of torch.Tensor projections.
    adjoint_op : callable
        H*: list of torch.Tensor projections -> torch.Tensor (nz, ny, nx).
        May include ramp filtering for preconditioning.
    measurements : list of torch.Tensor
        Observed projections y.
    zyx_shape : tuple of int
        Shape of the unknown volume x.
    reg_strength : float
        Tikhonov regularization lambda.
    n_iter : int
        Maximum number of CG iterations.
    device : torch.device
        CPU or CUDA device.

    Returns
    -------
    x : torch.Tensor
        Reconstructed 3D volume.
    """

    # The normal operator: A(vol) = H* H(vol) + lambda * vol
    # This is the left-hand side of the normal equation.
    def normal_op(vol):
        return adjoint_op(forward_op(vol)) + reg_strength * vol

    # Right-hand side: b = H* y  (backproject the measurements)
    b = adjoint_op(measurements)

    # Initialize x = 0.  The initial residual r = b - A(0) = b.
    x = torch.zeros(zyx_shape, dtype=torch.float32, device=device)
    r = b - normal_op(x)

    # CG search direction starts aligned with the residual.
    p = r.clone()

    # Squared norm of the residual: <r, r>.
    # CG tracks this to compute step sizes without extra inner products.
    rs_old = torch.sum(r * r).item()

    for i in range(n_iter):
        # Apply the normal operator to the search direction.
        Ap = normal_op(p)

        # Curvature along the search direction: <p, Ap>.
        # If near zero, the search direction is in the null space — stop.
        pAp = torch.sum(p * Ap).item()
        if pAp < 1e-30:
            break

        # Step size: move along p to minimize the quadratic.
        # alpha = <r, r> / <p, Ap>
        alpha = rs_old / pAp

        # Update the solution and residual.
        x = x + alpha * p
        r = r - alpha * Ap

        # New residual norm.  If near zero, we have converged.
        rs_new = torch.sum(r * r).item()
        if rs_new < 1e-30:
            break

        # Conjugate direction update: beta ensures <p_new, A p_old> = 0.
        # This orthogonality (in the A-inner product) is what makes CG
        # converge in at most n iterations for an n-dimensional problem.
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


# ---------------------------------------------------------------------------
# Convenience wrappers (numpy interface, for backward compatibility)
# ---------------------------------------------------------------------------
def siddon_project(volume_np, angle_deg, voxel_size, mode="sum"):
    """Forward: 3D numpy volume -> 2D numpy projection at a tilt angle.

    Thin wrapper that builds a one-angle SiddonOperator and projects.
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
        # Divide by path length per detector column
        _n_lateral, traces = _ray_trace(volume_np.shape, angle_deg, voxel_size)
        for il, _iz, _ix, lengths in traces:
            path_len = lengths.sum()
            if path_len > 0 and il < result.shape[1]:
                result[:, il] /= path_len

    return result


def siddon_backproject(projection_np, angle_deg, zyx_shape, voxel_size):
    """Adjoint: 2D numpy projection -> 3D numpy volume.

    Thin wrapper around SiddonOperator for backward compatibility.

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
