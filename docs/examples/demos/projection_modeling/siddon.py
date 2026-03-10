"""Siddon ray-tracing projection/backprojection and CG-Tikhonov solver.

Forward and adjoint operators for line-integral projection through a 3D
volume at an arbitrary tilt angle (rotation around Y in the ZX plane).
"""

import numpy as np


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


def siddon_project(volume, angle_deg, voxel_size, mode="sum"):
    """Forward: 3D volume -> 2D projection at a tilt angle.

    Rays propagate in the ZX plane, tilted by angle_deg from the Z-axis
    (rotation around Y). Since rays have no Y component, each lateral
    detector position shares the same (Z, X) voxel trace across all
    Y rows.

    Parameters
    ----------
    volume : np.ndarray, shape (Z, Y, X)
        3D volume to project.
    angle_deg : float
        Tilt angle in degrees from the Z-axis.
    voxel_size : float
        Isotropic voxel spacing in um.
    mode : str
        "sum" for line-integral projection, "max" for MIP.

    Returns
    -------
    projection : np.ndarray, shape (Y, N_lateral)
        2D projection image.
    """
    nz, ny, nx = volume.shape

    if abs(angle_deg) < 1e-6:
        if mode == "sum":
            return volume.sum(axis=0) * voxel_size
        return volume.max(axis=0)

    n_lateral, traces = _ray_trace(volume.shape, angle_deg, voxel_size)
    projection = np.zeros((ny, n_lateral), dtype=np.float32)

    for il, iz, ix, lengths in traces:
        slices = volume[iz, :, ix]  # (n_seg, ny)
        if mode == "sum":
            projection[:, il] = (slices * lengths[:, np.newaxis]).sum(axis=0)
        elif mode == "max":
            projection[:, il] = slices.max(axis=0)

    return projection


def siddon_backproject(projection, angle_deg, zyx_shape, voxel_size):
    """Adjoint: 2D projection -> 3D volume (transpose of siddon_project).

    For each ray (same geometry as siddon_project), distributes the
    detector pixel value back into traversed voxels, weighted by
    intersection length.

    Parameters
    ----------
    projection : np.ndarray, shape (Y, N_lateral)
        2D projection image.
    angle_deg : float
        Tilt angle in degrees from the Z-axis.
    zyx_shape : tuple of int
        (nz, ny, nx) output volume shape.
    voxel_size : float
        Isotropic voxel spacing in um.

    Returns
    -------
    volume : np.ndarray, shape (Z, Y, X)
        Backprojected 3D volume.
    """
    nz, ny, nx = zyx_shape
    volume = np.zeros(zyx_shape, dtype=np.float32)

    if abs(angle_deg) < 1e-6:
        # Adjoint of sum-along-Z: replicate projection into every Z slice
        volume[:] = projection[np.newaxis, :, :nx] * voxel_size
        return volume

    _n_lateral, traces = _ray_trace(zyx_shape, angle_deg, voxel_size)

    for il, iz, ix, lengths in traces:
        # projection[:, il] has shape (ny,); scatter into volume
        col = projection[:, il]  # (ny,)
        # Accumulate weighted values along the ray
        for k in range(len(iz)):
            volume[iz[k], :, ix[k]] += col * lengths[k]

    return volume


def cg_tikhonov(forward_op, adjoint_op, measurements, zyx_shape, reg_strength, n_iter):
    """Conjugate gradient solver for (H^T H + lambda I) x = H^T y.

    Parameters
    ----------
    forward_op : callable
        Maps 3D volume (np.ndarray) -> list of 2D projections.
    adjoint_op : callable
        Maps list of 2D projections -> 3D volume (np.ndarray).
    measurements : list of np.ndarray
        Observed projections.
    zyx_shape : tuple of int
        Shape of the unknown volume.
    reg_strength : float
        Tikhonov regularization lambda.
    n_iter : int
        Number of CG iterations.

    Returns
    -------
    x : np.ndarray
        Reconstructed 3D volume.
    """

    def normal_op(vol):
        return adjoint_op(forward_op(vol)) + reg_strength * vol

    b = adjoint_op(measurements)
    x = np.zeros(zyx_shape, dtype=np.float32)
    r = b - normal_op(x)
    p = r.copy()
    rs_old = np.sum(r * r)

    for i in range(n_iter):
        Ap = normal_op(p)
        pAp = np.sum(p * Ap)
        if pAp < 1e-30:
            break
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.sum(r * r)
        if rs_new < 1e-30:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x
