"""Tile-stitching settings — public surface for configuring tiled reconstruction.

``TileStitchSettings`` is the single user-facing config object. The
``recon`` field embeds the full ``ReconstructionSettings`` schema used
by ``wo rec`` — same shape, same fields. The modality is selected by
populating exactly one of ``recon.phase`` / ``recon.fluorescence`` /
``recon.birefringence`` (or ``birefringence`` + ``phase`` together);
``recon.reconstruction_dimension`` picks 2D vs 3D and
``recon.input_channel_names`` selects which channel(s) to process.

The blend factory is itself declarative — ``BlendSettings.kind`` selects
one of the built-in reductions, ``.build()`` materializes the
``Blend`` dataclass.

This module is pure-python: no dask, no cupy, no torch beyond what the
recon settings already need. Distributed orchestration happens in the
biahub package.
"""

from typing import TYPE_CHECKING, Literal, Self

import numpy as np
import xarray as xr
from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from waveorder.api._settings import MyBaseModel
from waveorder.cli.settings import ReconstructionSettings
from waveorder.tile_stitch._engine import (  # noqa: F401  (public re-exports)
    TileStitchPlan,
    blend_output_tile,
    build_plan,
    reconstruct_tile,
    tile_stitch_reconstruction,
)
from waveorder.tile_stitch.blend import (  # noqa: F401  (public re-exports)
    Blend,
    clipped_mean,
    gaussian_mean,
    max_blend,
    min_blend,
    uniform_mean,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch


def select_recon_modality(recon: ReconstructionSettings):
    """Return ``(name, settings)`` for the populated modality block.

    ``ReconstructionSettings`` enforces that exactly one of ``phase`` or
    ``fluorescence`` is set, optionally alongside ``birefringence``.
    Tile-stitch dispatches on whichever non-None block is present, with
    ``birefringence + phase`` resolving to the birefringence block since
    that's the joint-recovery pathway.
    """
    if recon.fluorescence is not None:
        return "fluorescence", recon.fluorescence
    if recon.birefringence is not None:
        return "birefringence", recon.birefringence
    if recon.phase is not None:
        return "phase", recon.phase
    raise ValueError("ReconstructionSettings must populate one of phase / fluorescence / birefringence")


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
    def _sigma_only_when_gaussian(self) -> Self:
        if self.kind != "gaussian_mean" and self.sigma_fraction is not None:
            raise ValueError(f"sigma_fraction is only valid when kind='gaussian_mean'; got kind={self.kind!r}")
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
    def _overlap_dims_subset_of_tile(self) -> Self:
        extra = set(self.overlap) - set(self.tile_size)
        if extra:
            raise ValueError(f"overlap has dims {sorted(extra)} not present in tile_size {sorted(self.tile_size)}")
        for d, ov in self.overlap.items():
            if ov >= self.tile_size[d]:
                raise ValueError(f"overlap[{d!r}]={ov} must be strictly less than tile_size[{d!r}]={self.tile_size[d]}")
        return self


# --- TileStitchSettings (top-level config) ---------------------------------


class TileStitchSettings(MyBaseModel):
    """User-facing config: tiling geometry + blend + the full ``wo rec`` schema.

    ``recon`` is the same ``ReconstructionSettings`` consumed by
    ``wo rec`` — including ``input_channel_names``,
    ``reconstruction_dimension`` (2 or 3), ``time_indices``, and one of
    ``phase`` / ``fluorescence`` (optionally with ``birefringence``). A
    tile-stitch run is therefore fully described by the YAML alone; the
    CLI only adds ``--device`` as a runtime override.

    ``schema_version`` is bumped on a backward-incompatible change.
    Downstream tools that read serialized configs should validate the
    version on load and refuse to proceed silently on mismatch.
    """

    schema_version: Literal["1"] = Field(
        default="1",
        description="schema version of this settings document (bumped on breaking changes)",
    )
    tile: TileSettings = Field(
        description="spatial tiling geometry (tile_size, overlap)",
    )
    blend: BlendSettings = Field(
        default_factory=BlendSettings,
        description="overlap-blend reduction kernel selection",
    )
    recon: ReconstructionSettings = Field(
        description="full ``wo rec`` reconstruction settings (channels, dims, modality block)",
    )


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
    device: "str | torch.device | None" = None,
) -> xr.Dataset:
    """Compute (or return cached) transfer function for the modality in
    ``settings.recon`` at the spatial shape implied by ``settings.tile.tile_size``.

    The reconstruction dimensionality is taken from
    ``settings.recon.reconstruction_dimension``. ``device`` may override
    ``settings.recon.device`` for the TF compute step (useful for
    distributed runs where the worker chooses its own device).

    **Worker-cache contract.** Distributed pipelines call this on each
    worker process once per task; the second and subsequent calls with
    identical ``(settings, device)`` return the cached ``xr.Dataset``
    without recomputing. The cache is a plain module-level dict — there
    is no eviction. Workers should call :func:`prepare_transfer_function`
    exactly once at the start of each task.
    """
    key = (
        settings.model_dump_json(),
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

    recon_dim = settings.recon.reconstruction_dimension
    name, modality_settings = select_recon_modality(settings.recon)
    if name == "phase":
        from waveorder.api import phase

        tf = phase.compute_transfer_function(sample, recon_dim=recon_dim, settings=modality_settings, device=device)
    elif name == "fluorescence":
        from waveorder.api import fluorescence

        tf = fluorescence.compute_transfer_function(
            sample, recon_dim=recon_dim, settings=modality_settings, device=device
        )
    elif name == "birefringence":
        from waveorder.api import birefringence

        tf = birefringence.compute_transfer_function(sample, modality_settings)
    else:  # pragma: no cover - exhaustive over select_recon_modality
        raise ValueError(f"unknown recon modality: {name!r}")

    _TF_CACHE[key] = tf
    return tf


def clear_transfer_function_cache() -> None:
    """Clear the process-local TF cache.

    Useful for tests; rarely needed in production. Distributed pipelines
    let the cache persist for the lifetime of the worker process.
    """
    _TF_CACHE.clear()
