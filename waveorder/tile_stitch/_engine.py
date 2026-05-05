"""Tile-stitching engine — pure-python single-process orchestrator.

This module is the algorithm reference. The biahub package builds a
distributed pipeline on top of the same primitives; this is the
pickling-free, dask-free path that a script can invoke directly.

Three primitives, plus one orchestrator:

* :class:`TileStitchPlan` — frozen dataclass holding the static plan
  (input tiles, output tiles, contributor map, schedule).
* :func:`build_plan` — derives :class:`TileStitchPlan` from a
  :class:`TileStitchSettings` and the input volume shape.
* :func:`reconstruct_tile` — apply the reconstruction function to a
  single input tile; returns a numpy ndarray.
* :func:`blend_output_tile` — accumulate contributions from N input
  tile reconstructions into one non-overlapping output tile.

* :func:`tile_stitch_reconstruction` — the end-to-end orchestrator that
  ties everything together for in-memory single-process execution.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import xarray as xr

from waveorder.tile_stitch.blend import Blend
from waveorder.tile_stitch.partition import (
    InputTile,
    OutputTile,
    generate_output_tiles,
    generate_tiles,
    input_tiles_for_output,
)
from waveorder.tile_stitch.scheduler import (
    bundle_inputs_by_cooccurrence,
    output_to_batches_map,
    schedule_coverage_greedy,
)

# `TileStitchSettings` lives in the public API surface; the engine
# imports it lazily to avoid a circular import (api.tile_stitch imports
# from this subpackage).


@dataclass(frozen=True, slots=True)
class TileStitchPlan:
    """Static plan: tiling geometry + contributor map + scheduling order.

    Built once at the start of a run and consumed by both the in-memory
    orchestrator (this module) and the distributed pipeline (biahub).

    Attributes
    ----------
    input_tiles : list[InputTile]
        Overlapping input tiles covering the full input volume.
    output_tiles : list[OutputTile]
        Non-overlapping output tiles partitioning the output volume.
        Each maps 1:1 to one output zarr chunk.
    output_to_inputs : dict[int, list[int]]
        For each output tile id, the input tile ids that contribute.
    input_order : list[int]
        Coverage-greedy ordering of input tile ids — used as a dask
        priority hint by distributed pipelines.
    input_batches : list[list[int]] or None
        Optional partition of input tiles into co-occurrence batches.
        ``None`` means no batching (each input is its own batch).
    output_to_batches : dict[int, list[int]] or None
        For each output tile, the batch indices that must complete
        before stitching. Mirrors ``output_to_inputs`` but at batch
        granularity. ``None`` when ``input_batches`` is None.
    tile_dims : tuple[str, ...]
        Spatial dim names in data order (e.g. ``("z", "y", "x")``).
    full_shape : dict[str, int]
        Per-dim size of the input/output volume.
    """

    input_tiles: list[InputTile]
    output_tiles: list[OutputTile]
    output_to_inputs: dict[int, list[int]]
    input_order: list[int]
    tile_dims: tuple[str, ...]
    full_shape: dict[str, int]
    input_batches: list[list[int]] | None = None
    output_to_batches: dict[int, list[int]] | None = None


def build_plan(
    czyx_data: xr.DataArray,
    settings,  # waveorder.api.tile_stitch.TileStitchSettings
    *,
    batch_size: int | None = None,
) -> TileStitchPlan:
    """Compute the static plan for a single (timepoint, channel) volume.

    Parameters
    ----------
    czyx_data : xr.DataArray
        Input volume with named spatial dims. ``settings.tile.tile_size``
        must reference a subset of these dim names.
    settings : TileStitchSettings
        User-facing tiling + blend + recon configuration.
    batch_size : int, optional
        When provided, partitions input tiles into co-occurrence batches
        of up to ``batch_size`` each (input-cohesive batched recon, the
        v6 / c0032 winner). When ``None``, no batching.

    Returns
    -------
    TileStitchPlan
    """
    input_tiles, tile_dims = generate_tiles(
        czyx_data,
        tile_size=dict(settings.tile.tile_size),
        overlap=dict(settings.tile.overlap),
    )
    full_shape = {d: int(czyx_data.sizes[d]) for d in tile_dims}
    output_tiles = generate_output_tiles(
        full_shape=full_shape,
        tile_size=dict(settings.tile.tile_size),
        tile_dims=tile_dims,
    )

    output_to_inputs: dict[int, list[int]] = {
        ot.tile_id: input_tiles_for_output(ot, input_tiles, tile_dims) for ot in output_tiles
    }
    input_order = schedule_coverage_greedy([it.tile_id for it in input_tiles], output_to_inputs)

    input_batches: list[list[int]] | None = None
    output_to_batches: dict[int, list[int]] | None = None
    if batch_size is not None and batch_size > 1:
        input_batches = bundle_inputs_by_cooccurrence(
            [it.tile_id for it in input_tiles],
            output_to_inputs,
            batch_size=batch_size,
        )
        output_to_batches = output_to_batches_map(input_batches, output_to_inputs)

    return TileStitchPlan(
        input_tiles=input_tiles,
        output_tiles=output_tiles,
        output_to_inputs=output_to_inputs,
        input_order=input_order,
        tile_dims=tile_dims,
        full_shape=full_shape,
        input_batches=input_batches,
        output_to_batches=output_to_batches,
    )


def reconstruct_tile(
    czyx_block: xr.DataArray,
    recon_fn: Callable[[xr.DataArray], xr.DataArray],
) -> np.ndarray:
    """Apply a bound reconstruction function to one input tile.

    ``recon_fn`` is the modality-specific entry point with transfer
    function + settings + device already bound; the engine stays
    modality-agnostic. Returns an ndarray (the xr.DataArray wrap is
    discarded; downstream blend math operates on raw arrays).

    Typical binding:

    .. code-block:: python

        from waveorder.api import phase

        recon_fn = lambda czyx: phase.apply_inverse_transfer_function(
            czyx,
            tf,
            recon_dim=3,
            settings=phase_settings,
            device="cpu",
        )
    """
    return np.asarray(recon_fn(czyx_block).values, dtype=np.float32)


def _intersection_slices(
    in_tile: InputTile,
    out_tile: OutputTile,
    tile_dims: tuple[str, ...],
) -> tuple[list[slice], list[slice]]:
    """Compute (input-local, output-local) slices for the bbox intersection."""
    in_local: list[slice] = []
    out_local: list[slice] = []
    for d in tile_dims:
        in_lo = in_tile.slices[d].start
        in_hi = in_tile.slices[d].stop
        ot_lo = out_tile.slices[d].start
        ot_hi = out_tile.slices[d].stop
        isect_lo = max(in_lo, ot_lo)
        isect_hi = min(in_hi, ot_hi)
        in_local.append(slice(isect_lo - in_lo, isect_hi - in_lo))
        out_local.append(slice(isect_lo - ot_lo, isect_hi - ot_lo))
    return in_local, out_local


def blend_output_tile(
    out_tile: OutputTile,
    contributors: list[tuple[InputTile, np.ndarray]],
    blend: Blend,
    *,
    leading_shape: tuple[int, ...] = (),
    tile_dims: tuple[str, ...],
    output_dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Accumulate input-tile contributions into one non-overlapping output tile.

    For each contributor, computes the rectangular intersection of its
    bbox with the output tile and accumulates IN-PLACE at the
    intersection-only slice. Per-contributor transient memory is
    intersection-sized, not output-sized — at 512³ tiles with 8
    contributors that's ~MB per task instead of ~GB.

    Two blend modes, dispatched by the shape of ``blend.init``'s return:

    * **Mean-style** (init returns 2-tuple ``(sum, weight)``): keeps
      ``accum_v`` (output-shape, fp64) + ``accum_w`` (out-spatial,
      fp64). Per contributor: ``accum_v[isect] += v * weight; accum_w[isect] += weight``.
      Finalize divides.
    * **Extremum-style** (init returns 1-tuple ``(values,)``): keeps
      ``accum_state`` (output-shape, fp64) initialized to
      ``blend.fill_value``. Per contributor: ``accum_state[isect] =
      blend.combine((accum_state[isect],), (v,))[0]``.

    Returns the finalized output cast to ``output_dtype``.

    Parameters
    ----------
    out_tile : OutputTile
    contributors : list[tuple[InputTile, np.ndarray]]
        ``(input_tile, recon_array)`` pairs. Contributors with
        zero-volume intersection are skipped silently.
    blend : Blend
    leading_shape : tuple[int, ...], default ``()``
    tile_dims : tuple[str, ...]
    output_dtype : numpy dtype, default ``np.float32``
    """
    out_spatial_shape = tuple(out_tile.slices[d].stop - out_tile.slices[d].start for d in tile_dims)
    output_shape = leading_shape + out_spatial_shape
    n_lead = len(leading_shape)

    if not contributors:
        return np.full(output_shape, blend.fill_value, dtype=output_dtype)

    # Probe init's return shape with a tiny sample to dispatch mode.
    _probe = blend.init(np.array([0.0, 1.0]), np.array([1.0, 1.0]))
    is_mean_style = len(_probe) == 2

    if is_mean_style:
        accum_v = np.zeros(output_shape, dtype=np.float64)
        accum_w = np.zeros(out_spatial_shape, dtype=np.float64)
        accum_state = None
    else:
        accum_v = None
        accum_w = None
        accum_state = np.full(output_shape, blend.fill_value, dtype=np.float64)

    for in_tile, recon in contributors:
        in_local, out_local = _intersection_slices(in_tile, out_tile, tile_dims)

        if any(s.stop <= s.start for s in in_local):
            continue

        in_idx = (slice(None),) * n_lead + tuple(in_local)
        out_idx = (slice(None),) * n_lead + tuple(out_local)

        kernel_full = blend.weight_kernel(in_tile.shape).astype(np.float64, copy=False)
        kernel_view = kernel_full[tuple(in_local)]
        v_view = recon[in_idx].astype(np.float64, copy=False)

        if is_mean_style:
            accum_v[out_idx] += v_view * kernel_view
            accum_w[tuple(out_local)] += kernel_view
        else:
            current = (accum_state[out_idx],)
            new = (v_view,)
            combined = blend.combine(current, new)
            accum_state[out_idx] = combined[0]

    if is_mean_style:
        with np.errstate(invalid="ignore", divide="ignore"):
            # accum_w is out-spatial only; broadcasts over leading dims.
            full_w = np.broadcast_to(accum_w, output_shape)
            result = np.where(full_w > 0, accum_v / full_w, blend.fill_value)
        return result.astype(output_dtype, copy=False)

    return accum_state.astype(output_dtype, copy=False)


def tile_stitch_reconstruction(
    czyx_data: xr.DataArray,
    settings,  # waveorder.api.tile_stitch.TileStitchSettings
    *,
    transfer_function: xr.Dataset,
    recon_dim: Literal[2, 3] = 3,
    device: str | torch.device | None = None,
) -> xr.DataArray:
    """End-to-end tile-stitching reconstruction (single-process, in-memory).

    Builds the plan, reconstructs each input tile with the modality
    selected by ``settings.recon.kind``, then blends contributions into
    each output tile and returns the assembled volume.

    For distributed execution, biahub uses :func:`build_plan` plus
    :func:`reconstruct_tile` and :func:`blend_output_tile` directly,
    skipping this orchestrator.

    Parameters
    ----------
    czyx_data : xr.DataArray
        Input volume.
    settings : TileStitchSettings
    transfer_function : xr.Dataset
        Pre-computed transfer function for the modality in
        ``settings.recon``.
    recon_dim : {2, 3}, default 3
    device : torch device, optional

    Returns
    -------
    xr.DataArray
        Output volume with the same dim names as the input.
    """
    from waveorder.api import birefringence, fluorescence, phase

    plan = build_plan(czyx_data, settings)
    blend_kernel = settings.blend.build()
    recon_kind = settings.recon.kind

    if recon_kind == "phase":
        recon_module = phase
    elif recon_kind == "fluorescence":
        recon_module = fluorescence
    elif recon_kind == "birefringence":
        recon_module = birefringence
    else:  # pragma: no cover - exhaustive over the discriminated union
        raise ValueError(f"unknown recon kind: {recon_kind!r}")

    def recon_fn(block: xr.DataArray) -> xr.DataArray:
        return recon_module.apply_inverse_transfer_function(
            block,
            transfer_function,
            recon_dim=recon_dim,
            settings=settings.recon,
            device=device,
        )

    # Stage A: reconstruct all input tiles, hold in dict for Stage B.
    leading_dims = tuple(d for d in czyx_data.dims if d not in plan.tile_dims)
    leading_shape_in = tuple(int(czyx_data.sizes[d]) for d in leading_dims)
    recon_by_tile: dict[int, np.ndarray] = {}
    for in_tile in plan.input_tiles:
        sub = czyx_data.isel({d: in_tile.slices[d] for d in plan.tile_dims})
        recon = reconstruct_tile(sub, recon_fn)
        recon_by_tile[in_tile.tile_id] = recon

    # Determine output leading shape from the first reconstructed tile
    # (recon may change leading dim sizes — e.g. channel reduction).
    first = next(iter(recon_by_tile.values()))
    output_leading_shape = first.shape[: first.ndim - len(plan.tile_dims)]

    # Stage B: blend each output tile.
    full_output_shape = output_leading_shape + tuple(plan.full_shape[d] for d in plan.tile_dims)
    output_arr = np.full(full_output_shape, blend_kernel.fill_value, dtype=np.float32)
    tiles_by_id = {it.tile_id: it for it in plan.input_tiles}

    for out_tile in plan.output_tiles:
        contributor_ids = plan.output_to_inputs.get(out_tile.tile_id, [])
        contributors = [(tiles_by_id[tid], recon_by_tile[tid]) for tid in contributor_ids]
        tile_result = blend_output_tile(
            out_tile,
            contributors,
            blend_kernel,
            leading_shape=output_leading_shape,
            tile_dims=plan.tile_dims,
        )
        n_lead = len(output_leading_shape)
        write_idx = (slice(None),) * n_lead + tuple(out_tile.slices[d] for d in plan.tile_dims)
        output_arr[write_idx] = tile_result

    # Wrap result with the same leading dim names; spatial dims keep input names.
    # Recon may change leading dims (e.g. channel reduction), so we guess from
    # the recon output: if dim count matches input, reuse names; otherwise
    # rebuild.
    output_dims: tuple[str, ...]
    if len(output_leading_shape) == len(leading_shape_in):
        output_dims = leading_dims + plan.tile_dims
    else:
        # Recon dropped/added a leading dim; fall back to generic names.
        output_dims = tuple(f"d{i}" for i in range(len(output_leading_shape))) + plan.tile_dims

    return xr.DataArray(output_arr, dims=output_dims)
