"""Tile-stitching settings — public surface for configuring tiled reconstruction.

``TileStitchSettings`` is the single user-facing config object. The
``recon`` field is a discriminated union over per-modality reconstruction
settings (``phase.Settings``, ``fluorescence.Settings``,
``birefringence.Settings``) keyed by the ``kind`` literal each carries.

The blend factory is itself declarative — ``BlendSettings.kind`` selects
one of the built-in reductions, ``.build()`` materializes the
``Blend`` dataclass.

This module is pure-python: no dask, no cupy, no torch beyond what the
recon settings already need. Distributed orchestration happens in the
biahub package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Union

import numpy as np
import xarray as xr
from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from waveorder.api._settings import MyBaseModel
from waveorder.api.birefringence import Settings as BirefringenceSettings
from waveorder.api.fluorescence import Settings as FluorescenceSettings
from waveorder.api.phase import Settings as PhaseSettings
from waveorder.tile_stitch._engine import (
    TileStitchPlan,
    blend_output_tile,
    build_plan,
    reconstruct_tile,
    tile_stitch_reconstruction,
)
from waveorder.tile_stitch.blend import (
    Blend,
    gaussian_mean,
    max_blend,
    min_blend,
    uniform_mean,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch

# Discriminated union of the per-modality recon settings. The ``kind``
# literal on each ``Settings`` class is the discriminator key.
ReconSettings = Annotated[
    Union[PhaseSettings, FluorescenceSettings, BirefringenceSettings],
    Field(discriminator="kind"),
]


class BlendSettings(MyBaseModel):
    """Declarative form of :class:`Blend`. ``.build()`` materializes the dataclass."""

    kind: Literal["uniform_mean", "gaussian_mean", "max", "min"] = Field(
        default="uniform_mean",
        description="reduction kernel used to combine overlapping tile contributions",
    )
    sigma_fraction: float | None = Field(
        default=None,
        description="Gaussian σ as a fraction of each tile axis (kind='gaussian_mean' only); "
        "default 1/8 applied at build time when None",
    )
    accumulator_dtype: Literal["float32", "float64"] = Field(
        default="float32",
        description="dtype used for the running sum during tree reduction; "
        "fp32 matches input precision, fp64 reduces accumulation error at extreme N_contributors",
    )

    @model_validator(mode="after")
    def _sigma_only_when_gaussian(self) -> BlendSettings:
        if self.kind != "gaussian_mean" and self.sigma_fraction is not None:
            raise ValueError(
                "sigma_fraction is only valid when kind='gaussian_mean'; "
                f"got kind={self.kind!r}"
            )
        return self

    def build(self) -> Blend:
        if self.kind == "uniform_mean":
            return uniform_mean()
        if self.kind == "gaussian_mean":
            return gaussian_mean(self.sigma_fraction if self.sigma_fraction is not None else 1.0 / 8.0)
        if self.kind == "max":
            return max_blend()
        if self.kind == "min":
            return min_blend()
        raise ValueError(f"unknown blend kind: {self.kind!r}")


class TileSettings(MyBaseModel):
    """Spatial tiling geometry: per-dim tile size + overlap.

    Output zarr chunks always shadow ``tile_size`` (one output tile == one
    chunk). Not a user knob — making it adjustable creates a footgun where
    chunks misalign with tile boundaries and producer-vs-consumer write
    contention can corrupt output.
    """

    tile_size: dict[str, PositiveInt] = Field(
        description="per-dimension input tile size in pixels (e.g. {'y': 512, 'x': 512})",
    )
    overlap: dict[str, NonNegativeInt] = Field(
        default_factory=dict,
        description="per-dimension tile overlap in pixels; dims absent default to 0",
    )

    @model_validator(mode="after")
    def _overlap_dims_subset_of_tile(self) -> TileSettings:
        extra = set(self.overlap) - set(self.tile_size)
        if extra:
            raise ValueError(
                f"overlap has dims {sorted(extra)} not present in tile_size {sorted(self.tile_size)}"
            )
        for d, ov in self.overlap.items():
            if ov >= self.tile_size[d]:
                raise ValueError(
                    f"overlap[{d!r}]={ov} must be strictly less than tile_size[{d!r}]={self.tile_size[d]}"
                )
        return self


# --- Worker-cache contract --------------------------------------------------
#
# Distributed pipelines (biahub) compute the transfer function once per
# worker process and cache it across dask tasks. ``prepare_transfer_function``
# is the cache contract: stable across dask client.run calls within the same
# Python process, keyed by the settings + recon_dim + device tuple.
#
# Cache lives at module scope so it persists across function calls in the
# worker process. dask-jobqueue keeps worker processes alive across tasks,
# so a 5s TF build amortizes across hundreds of subsequent reconstructions.

_TF_CACHE: dict[tuple, xr.Dataset] = {}


def prepare_transfer_function(
    settings: TileStitchSettings,
    *,
    recon_dim: Literal[2, 3] = 3,
    device: "str | torch.device | None" = None,
) -> xr.Dataset:
    """Compute (or return cached) transfer function for the modality in
    ``settings.recon`` at the spatial shape implied by ``settings.tile.tile_size``.

    **Worker-cache contract.** Distributed pipelines call this on each
    worker process once per task; the second and subsequent calls with
    identical ``(settings, recon_dim, device)`` return the cached
    ``xr.Dataset`` without recomputing. The cache is a plain module-level
    dict — there is no eviction. Workers should call
    :func:`prepare_transfer_function` exactly once at the start of each
    task.

    The cache key is the JSON serialization of ``settings`` plus
    ``recon_dim`` plus ``str(device)``. Mutating the returned ``xr.Dataset``
    will mutate the cache; treat it as read-only.

    Parameters
    ----------
    settings : TileStitchSettings
        Full tile-stitch config — the modality is selected by
        ``settings.recon.kind``; the spatial shape is taken from
        ``settings.tile.tile_size``.
    recon_dim : {2, 3}, default 3
    device : str, torch.device, or None
        Forwarded to the modality's ``compute_transfer_function``.

    Returns
    -------
    xr.Dataset
        The transfer function dataset for the selected modality.
    """
    key = (
        settings.model_dump_json(),
        recon_dim,
        str(device) if device is not None else None,
    )
    cached = _TF_CACHE.get(key)
    if cached is not None:
        return cached

    tile_size = dict(settings.tile.tile_size)
    spatial_dims = ("z", "y", "x") if "z" in tile_size else ("y", "x")
    sample_shape = (1,) + tuple(tile_size.get(d, 1) for d in spatial_dims)
    sample = xr.DataArray(
        np.empty(sample_shape, dtype=np.float32),
        dims=("c",) + spatial_dims,
    )

    recon_kind = settings.recon.kind
    if recon_kind == "phase":
        from waveorder.api import phase

        tf = phase.compute_transfer_function(
            sample, recon_dim=recon_dim, settings=settings.recon, device=device
        )
    elif recon_kind == "fluorescence":
        from waveorder.api import fluorescence

        tf = fluorescence.compute_transfer_function(
            sample, recon_dim=recon_dim, settings=settings.recon, device=device
        )
    elif recon_kind == "birefringence":
        from waveorder.api import birefringence

        tf = birefringence.compute_transfer_function(sample, settings.recon)
    else:  # pragma: no cover - exhaustive over the discriminated union
        raise ValueError(f"unknown recon kind: {recon_kind!r}")

    _TF_CACHE[key] = tf
    return tf


def clear_transfer_function_cache() -> None:
    """Clear the process-local TF cache.

    Useful for tests; rarely needed in production. Distributed pipelines
    let the cache persist for the lifetime of the worker process.
    """
    _TF_CACHE.clear()


# --- Settings ---------------------------------------------------------------


class TileStitchSettings(MyBaseModel):
    """User-facing config: tiling geometry + blend + per-modality recon settings.

    ``schema_version`` is bumped on a backward-incompatible change to this
    schema. v0.5 only emits / reads ``"1"``; downstream tools that read
    serialized configs should validate the version on load and refuse to
    proceed silently on mismatch.
    """

    schema_version: Literal["1"] = Field(
        default="1",
        description="schema version of this settings document (bumped on breaking changes)",
    )
    tile: TileSettings = Field(
        description="spatial tiling geometry (tile_size, overlap, output_chunk)",
    )
    blend: BlendSettings = Field(
        default_factory=BlendSettings,
        description="overlap-blend reduction kernel selection",
    )
    recon: ReconSettings = Field(
        description="per-modality reconstruction settings; discriminated by 'kind'",
    )
