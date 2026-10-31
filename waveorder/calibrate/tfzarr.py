"""The ``tf.zarr`` transfer-function store (design document section 3).

Layout::

    tf.zarr/
      parameters              [N_tiles, P]        always
      parameters_covariance   [N_tiles, P, P]     zeros when the score has no curvature
      transfer_function       [Z, Y, X] or [N_tiles, Z, Y, X], one tile per chunk
      attrs:
        parameter_names       [P]
        tile_grid             {size_yx, overlap_yx, centers_yx}
        provenance            {kind, sources, config_hash, ...}

``parameters`` rows are per tile; a single row is broadcast across tiles by
consumers. ``transfer_function`` is optional for tiled stores, where it is
regenerated per tile from ``parameters`` at apply time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr


@dataclass
class TransferFunctionStore:
    """In-memory view of a ``tf.zarr``."""

    parameters: np.ndarray  # [N_tiles, P]
    parameter_names: list[str]
    tile_grid: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    parameters_covariance: np.ndarray | None = None  # [N_tiles, P, P]
    transfer_function: np.ndarray | None = None  # [Z, Y, X] or [N_tiles, Z, Y, X]

    @property
    def n_tiles(self) -> int:
        return self.parameters.shape[0]


def write_tf_zarr(
    path: Path | str,
    parameters: np.ndarray,
    parameter_names: list[str],
    tile_grid: dict[str, Any],
    provenance: dict[str, Any] | None = None,
    parameters_covariance: np.ndarray | None = None,
    transfer_function: np.ndarray | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    """Write a ``tf.zarr`` store.

    Parameters
    ----------
    path : Path or str
        Output store path.
    parameters : ndarray
        ``[N_tiles, P]`` estimated (or nominal) parameter values.
    parameter_names : list[str]
        ``P`` names, e.g. ``["defocus", "astigmatism_0", ...]``.
    tile_grid : dict
        ``{"size_yx": [y, x], "overlap_yx": [y, x], "centers_yx": [[y_um, x_um], ...]}``;
        one center when shift-invariant.
    provenance : dict, optional
        ``{"kind": "nominal" | "beads" | "refined" | "simulated", "sources": [...],
        "config_hash": ...}`` plus free-form entries.
    parameters_covariance : ndarray, optional
        ``[N_tiles, P, P]``; written as zeros when None.
    transfer_function : ndarray, optional
        ``[Z, Y, X]`` (shift-invariant) or ``[N_tiles, Z, Y, X]`` (tiled,
        chunked one tile per chunk).
    extra_arrays : dict, optional
        Additional named arrays, e.g. ``score_trace`` (per gradient
        iteration) and ``score_grid`` (one row per evaluated point:
        parameter coordinates followed by the score).
    """
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.ndim != 2:
        raise ValueError(f"parameters must be [N_tiles, P], got shape {parameters.shape}")
    n_tiles, n_params = parameters.shape
    if len(parameter_names) != n_params:
        raise ValueError(f"{len(parameter_names)} parameter_names for {n_params} parameters")

    if parameters_covariance is None:
        parameters_covariance = np.zeros((n_tiles, n_params, n_params), dtype=np.float64)
    parameters_covariance = np.asarray(parameters_covariance, dtype=np.float64)
    if parameters_covariance.shape != (n_tiles, n_params, n_params):
        raise ValueError(
            f"parameters_covariance must be [N_tiles, P, P] = {(n_tiles, n_params, n_params)}, "
            f"got {parameters_covariance.shape}"
        )

    group = zarr.open_group(str(path), mode="w")
    group.create_array("parameters", shape=parameters.shape, dtype="float64")[:] = parameters
    group.create_array("parameters_covariance", shape=parameters_covariance.shape, dtype="float64")[:] = (
        parameters_covariance
    )

    if transfer_function is not None:
        transfer_function = np.asarray(transfer_function)
        if transfer_function.ndim == 4:
            chunks = (1, *transfer_function.shape[1:])
        else:
            chunks = transfer_function.shape
        tf_array = group.create_array(
            "transfer_function",
            shape=transfer_function.shape,
            dtype=str(transfer_function.dtype),
            chunks=chunks,
        )
        tf_array[:] = transfer_function

    for name, array in (extra_arrays or {}).items():
        array = np.asarray(array)
        group.create_array(name, shape=array.shape, dtype=str(array.dtype))[:] = array

    group.attrs["parameter_names"] = list(parameter_names)
    group.attrs["tile_grid"] = tile_grid
    group.attrs["provenance"] = provenance or {}


def read_tf_zarr(path: Path | str) -> TransferFunctionStore:
    """Read a ``tf.zarr`` store written by :func:`write_tf_zarr`.

    Parameters
    ----------
    path : Path or str
        Store path.

    Returns
    -------
    TransferFunctionStore
    """
    group = zarr.open_group(str(path), mode="r")
    transfer_function = None
    if "transfer_function" in group:
        transfer_function = group["transfer_function"][:]
    return TransferFunctionStore(
        parameters=group["parameters"][:],
        parameters_covariance=group["parameters_covariance"][:],
        parameter_names=list(group.attrs["parameter_names"]),
        tile_grid=dict(group.attrs["tile_grid"]),
        provenance=dict(group.attrs.get("provenance", {})),
        transfer_function=transfer_function,
    )
