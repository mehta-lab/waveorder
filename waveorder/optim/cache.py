"""Transfer function cache for fast multi-tile reconstruction.

Precomputes singular systems (U, S, Vh) on a parameter grid and saves
them to a zarr store on disk. Multiple reconstruction workers can then
read from the same cache, skipping the expensive TF computation and
paying only the cheap inverse filter step.

Examples
--------
Build the cache once::

    cache = TransferFunctionCache.build(
        compute_tf_fn=my_compute_tf,
        cache_specs={"z_focus_offset": CacheSpec(-2, 6, 1)},
        cache_dir="./tf_cache.zarr",
    )

Then look up from any worker::

    cache = TransferFunctionCache("./tf_cache.zarr")
    tf = cache.lookup(z_focus_offset=2.3)
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable

import numpy as np
import xarray as xr
import zarr

from waveorder.optim._types import CacheSpec


class TransferFunctionCache:
    """Disk-backed cache of transfer functions on a parameter grid.

    The zarr store layout is::

        cache.zarr/
          .zattrs          # param_names, grids, tf_keys
          grid_000_000/    # one group per grid point
            U, S, Vh       # singular system arrays
          grid_000_001/
            ...

    Parameters
    ----------
    cache_dir : str or Path
        Path to an existing cache zarr store.
    interpolation : {"nearest", "linear"}
        Lookup strategy.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        interpolation: str = "nearest",
    ):
        self._cache_dir = Path(cache_dir)
        self._interpolation = interpolation
        self._store = zarr.open(str(self._cache_dir), mode="r")

        meta = dict(self._store.attrs)
        self._param_names: list[str] = meta["param_names"]
        self._grids: dict[str, np.ndarray] = {name: np.array(vals) for name, vals in meta["grids"].items()}
        self._tf_keys: list[str] = meta["tf_keys"]

    @property
    def param_names(self) -> list[str]:
        """Parameter names in the cache."""
        return list(self._param_names)

    @property
    def grids(self) -> dict[str, np.ndarray]:
        """Grid values for each parameter."""
        return dict(self._grids)

    @property
    def cache_size(self) -> int:
        """Number of cached grid points."""
        size = 1
        for g in self._grids.values():
            size *= len(g)
        return size

    @staticmethod
    def build(
        compute_tf_fn: Callable[..., xr.Dataset],
        cache_specs: dict[str, CacheSpec],
        cache_dir: str | Path,
        fixed_kwargs: dict | None = None,
    ) -> "TransferFunctionCache":
        """Precompute transfer functions on a parameter grid.

        Parameters
        ----------
        compute_tf_fn : callable
            Function that takes ``(**params)`` and returns an
            ``xr.Dataset`` with singular system arrays.
        cache_specs : dict[str, CacheSpec]
            Grid specification for each parameter.
        cache_dir : str or Path
            Where to write the zarr cache.
        fixed_kwargs : dict, optional
            Extra keyword arguments passed to ``compute_tf_fn``.

        Returns
        -------
        TransferFunctionCache
            The opened cache, ready for lookups.
        """
        cache_dir = Path(cache_dir)
        fixed_kwargs = fixed_kwargs or {}

        param_names = list(cache_specs.keys())
        grids = {}
        for name, spec in cache_specs.items():
            grids[name] = np.arange(
                spec.start,
                spec.stop + spec.step / 2,
                spec.step,
            ).tolist()

        store = zarr.open(str(cache_dir), mode="w")

        grid_arrays = [grids[n] for n in param_names]
        all_combos = list(itertools.product(*grid_arrays))
        tf_keys = None

        for combo in all_combos:
            kwargs = dict(fixed_kwargs)
            for i, name in enumerate(param_names):
                kwargs[name] = combo[i]

            tf_dataset = compute_tf_fn(**kwargs)

            if tf_keys is None:
                tf_keys = list(tf_dataset.data_vars.keys())

            idx_str = _combo_to_key(combo, grid_arrays, param_names)
            group = store.require_group(idx_str)
            for key in tf_keys:
                arr = tf_dataset[key].values
                group.create_array(key, data=arr, overwrite=True)

        store.attrs["param_names"] = param_names
        store.attrs["grids"] = grids
        store.attrs["tf_keys"] = tf_keys

        return TransferFunctionCache(cache_dir)

    def lookup(self, **params) -> xr.Dataset:
        """Look up a transfer function for the given parameter values.

        Parameters
        ----------
        **params : float
            Parameter values. Names must match those used during build.

        Returns
        -------
        xr.Dataset
            Transfer function dataset in the same format as
            ``compute_transfer_function`` output.
        """
        if self._interpolation == "nearest":
            return self._lookup_nearest(params)
        elif self._interpolation == "linear":
            return self._lookup_linear(params)
        else:
            raise ValueError(f"Unknown interpolation: {self._interpolation}")

    def _find_nearest_index(self, name: str, value: float) -> int:
        grid = self._grids[name]
        return int(np.argmin(np.abs(grid - value)))

    def _load_grid_point(self, indices: dict[str, int]) -> xr.Dataset:
        """Load a single cached TF from disk."""
        grid_arrays = [self._grids[n].tolist() for n in self._param_names]
        combo = tuple(grid_arrays[i][indices[n]] for i, n in enumerate(self._param_names))
        key = _combo_to_key(combo, grid_arrays, self._param_names)
        group = self._store[key]

        from waveorder.api._utils import _named_dataarray

        variables = {}
        for tf_key in self._tf_keys:
            arr = np.array(group[tf_key])
            variables[tf_key] = _named_dataarray(arr, tf_key)
        return xr.Dataset(variables)

    def _lookup_nearest(self, params: dict) -> xr.Dataset:
        indices = {
            name: self._find_nearest_index(
                name,
                params.get(name, 0.0),
            )
            for name in self._param_names
        }
        return self._load_grid_point(indices)

    def _lookup_linear(self, params: dict) -> xr.Dataset:
        """Multilinear interpolation of singular system arrays."""
        brackets = []
        for name in self._param_names:
            val = float(params.get(name, 0.0))
            grid = self._grids[name]
            val = max(grid[0], min(grid[-1], val))
            idx = np.searchsorted(grid, val, side="right") - 1
            idx = max(0, min(len(grid) - 2, idx))
            lo, hi = grid[idx], grid[idx + 1]
            w = 0.0 if hi == lo else (val - lo) / (hi - lo)
            brackets.append((idx, w))

        result_arrays: dict[str, np.ndarray] | None = None

        for corner in itertools.product(
            *[(0, 1)] * len(self._param_names),
        ):
            weight = 1.0
            indices = {}
            for dim, bit in enumerate(corner):
                lo_idx, w = brackets[dim]
                indices[self._param_names[dim]] = lo_idx + bit
                weight *= w if bit == 1 else (1 - w)
            if weight == 0:
                continue

            ds = self._load_grid_point(indices)
            if result_arrays is None:
                result_arrays = {k: ds[k].values * weight for k in self._tf_keys}
            else:
                for k in self._tf_keys:
                    result_arrays[k] += ds[k].values * weight

        from waveorder.api._utils import _named_dataarray

        return xr.Dataset({k: _named_dataarray(v, k) for k, v in result_arrays.items()})


def _combo_to_key(combo, grid_arrays, param_names):
    """Convert a parameter combo to a zarr group key."""
    parts = []
    for i, name in enumerate(param_names):
        idx = grid_arrays[i].index(combo[i])
        parts.append(f"{idx:03d}")
    return "grid_" + "_".join(parts)
