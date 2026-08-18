"""Per-subtile batched optimization of illumination tilt parameters
(zenith, azimuth, z-offset) using ``isotropic_thin_3d.reconstruct`` as
the forward model.

This module is intended for tilt-recon pipelines (e.g. OPS) that grid a
field of view into many small subtiles, each with its own per-subtile
tilt and focus offset that must be solved for jointly. The function
:func:`optimize_subtile_tilt_params` runs a single batched NAdam loop
across all subtiles, internally grouping by shape and focus offset so
the forward pass runs as one ``isotropic_thin_3d.reconstruct`` call per
group.

The function also exposes a *warmstart-skip* hook: when the caller
already has a high-confidence prior (e.g. from a universal warmstart
parquet built over many prior runs), the optimization can be bypassed
and the prior returned verbatim. The caller owns the skip decision —
the library just honors it. This is the algorithm hook that downstream
OPS code uses to implement per-position auto-skip routing and T-cache.

Usage modes
-----------
1. Cold start (``warmstart_params=None``):
       Optimize from the supplied init values.

2. Warmstart init (``warmstart_params`` provided,
   ``skip_optim_if_warmstart=False``):
       Use warmstart as Adam init, run ``n_iters`` of refinement.

3. Skip optim (``warmstart_params`` provided,
   ``skip_optim_if_warmstart=True``):
       Return warmstart directly, run 0 iters. Used by callers that
       trust the warmstart enough to bypass refinement.

Notes
-----
- Internally uses ``isotropic_thin_3d.reconstruct``, which already
  benefits from the closed-form 2×2 Tikhonov inverse when
  ``WAVEORDER_FAST_2D_TIKHONOV=1``.
- ``freeze_axes`` disables gradient on the corresponding parameter(s).
  Useful when the warmstart map provides reliable angle estimates and
  only z needs refinement — the 1-D z-only path is significantly faster.
- Convergence is iter-bounded (no early exit). For convergence-aware
  routing, populate ``warmstart_params`` and pass
  ``skip_optim_if_warmstart=True`` for positions you trust.
- Loss is computed in Fourier space (mid-band power) by default.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from waveorder.models import isotropic_thin_3d
from waveorder.optim.losses import (
    LossSettings,
    MidbandPowerLossSettings,
    build_loss_fn,
)


@dataclass
class TiltOptimResult:
    """Output of :func:`optimize_subtile_tilt_params`.

    Attributes
    ----------
    z_offsets : Tensor
        ``(B,)`` per-subtile z offset (in z-slice units, same convention
        as the ``z_init`` input).
    zeniths : Tensor
        ``(B,)`` per-subtile zenith angle, radians.
    azimuths : Tensor
        ``(B,)`` per-subtile azimuth angle, radians.
    final_loss : Tensor
        Scalar tensor with the summed final loss across subtiles. NaN
        if the loop hit a numerical failure before convergence.
    n_iters : int
        Actual NAdam iterations run. ``0`` if the call was a
        warmstart-skip (see :func:`optimize_subtile_tilt_params`).
    skipped : bool
        True if the optimization was bypassed via
        ``skip_optim_if_warmstart=True``.
    """

    z_offsets: Tensor
    zeniths: Tensor
    azimuths: Tensor
    final_loss: Tensor
    n_iters: int
    skipped: bool


def radial_blend_zenith_init(
    zen_formula: Tensor,
    grid_coords: Tensor,
    center: Optional[Tuple[float, float]] = None,
    r_max: Optional[float] = None,
) -> Tensor:
    """Smooth radial interpolation between 0 (at well center) and
    ``zen_formula`` (at edge).

    For low-NA tilt-recon (e.g. 5× track), well-edge subtiles converge
    to a different optimum than center subtiles when zenith is
    initialized at 0. Replacing ``zen_init = 0`` with a radial ramp
    recovers production-equivalent downstream segmentation. Validated
    on ops0154 + ops0153 cell counts within ±0.16 % of PROD even when
    phase Pearson is much lower than 1.0.

    Parameters
    ----------
    zen_formula : Tensor
        ``(B,)`` base zenith from the calibration formula (e.g.
        ``base + slope * r_tile``).
    grid_coords : Tensor
        ``(B, 2)`` per-subtile (row, col) coordinates on the well grid.
    center : tuple, optional
        ``(row_center, col_center)``. Defaults to the midpoint of the
        coordinate range in ``grid_coords``.
    r_max : float, optional
        Normalization radius. Defaults to the largest ``r_tile`` in the
        input.

    Returns
    -------
    Tensor
        ``(B,)`` blended zenith init, ``zen_formula * min(r_tile / r_max, 1)``.
    """
    if center is None:
        rmin = grid_coords.min(dim=0).values
        rmax = grid_coords.max(dim=0).values
        center = ((rmin[0] + rmax[0]) / 2.0, (rmin[1] + rmax[1]) / 2.0)
    cy, cx = float(center[0]), float(center[1])
    r = torch.sqrt(
        (grid_coords[:, 0] - cy) ** 2 + (grid_coords[:, 1] - cx) ** 2
    )
    if r_max is None:
        r_max = float(r.max().item())
    if r_max <= 0:
        return zen_formula.clone()
    blend = (r / r_max).clamp(0.0, 1.0)
    return zen_formula * blend


def _as_per_subtile(value, B: int, device, dtype=torch.float32) -> Tensor:
    """Broadcast a scalar or (B,) tensor to a contiguous (B,) float tensor."""
    if isinstance(value, Tensor):
        v = value.to(device=device, dtype=dtype)
        if v.numel() == 1:
            v = v.expand(B).contiguous()
        elif v.numel() != B:
            raise ValueError(
                f"Expected scalar or length-{B} tensor, got shape {tuple(v.shape)}"
            )
        return v.contiguous()
    return torch.full((B,), float(value), dtype=dtype, device=device)


def optimize_subtile_tilt_params(
    tiles: Sequence[Tensor],
    z_index: Tensor,
    tf_settings: dict,
    *,
    # Init
    zen_init: Union[float, Tensor] = 0.0,
    azi_init: Union[float, Tensor] = 0.0,
    z_init: Union[float, Tensor] = 0.0,
    focus_offsets: Optional[Sequence[float]] = None,
    # Optimizer
    n_iters: int = 8,
    freeze_axes: Sequence[Literal["zenith", "azimuth"]] = (),
    lr_z: float = 0.05,
    lr_zenith: float = 0.005,
    lr_azimuth: float = 0.01,
    # Warmstart / cache hook
    warmstart_params: Optional[TiltOptimResult] = None,
    skip_optim_if_warmstart: bool = False,
    # Algorithm
    regularization_strength: float = 1e-3,
    loss_settings: Optional[LossSettings] = None,
    reflect_pad: int = 16,
    pupil_steepness: float = 100.0,
) -> TiltOptimResult:
    """Batched per-subtile NAdam optimization of (zenith, azimuth, z) tilt.

    Parameters
    ----------
    tiles : sequence of Tensor
        ``B`` per-subtile brightfield Z-stacks, each shape ``(Z, y, x)``
        on the same CUDA (or CPU) device. Tiles of different ``(y, x)``
        shapes are allowed — they are grouped internally by shape before
        being batched into the forward call.
    z_index : Tensor
        ``(Z,)`` z-index offsets (typically ``-arange(Z) + Z // 2``) on
        the same device as ``tiles``.
    tf_settings : dict
        Transfer-function settings passed to
        ``isotropic_thin_3d.reconstruct``. Must include
        ``wavelength_illumination``, ``yx_pixel_size``,
        ``numerical_aperture_detection``,
        ``numerical_aperture_illumination``,
        ``index_of_refraction_media``, ``z_pixel_size``,
        ``invert_phase_contrast``. ``z_pixel_size`` and ``z_padding``
        are extracted for the z-position computation; the rest are
        forwarded to ``reconstruct``.
    zen_init, azi_init, z_init : float or Tensor
        Per-subtile init values. Scalars broadcast to all ``B`` subtiles.
        For per-subtile values, pass a ``(B,)`` tensor in the same order
        as ``tiles``.
    focus_offsets : sequence of float, optional
        ``(B,)`` per-subtile focus offset. Subtiles sharing the same
        focus_offset (after rounding to 1 decimal place by the caller)
        get a single ``z_positions`` tensor in the forward call, so the
        TF is computed once per group. If ``None``, all subtiles share
        a single z_positions computed from ``z_init``.
    n_iters : int
        Number of NAdam iterations.
    freeze_axes : sequence of {"zenith", "azimuth"}
        Axes to hold fixed at their init values (no gradient). When both
        angles are frozen, the optimization reduces to a 1-D shared-z
        search and is substantially faster.
    lr_z, lr_zenith, lr_azimuth : float
        Per-parameter learning rates. Defaults work for the OPS 5×/20×
        configurations.
    warmstart_params : TiltOptimResult, optional
        Prior optimization result to seed (or replace) this call.
    skip_optim_if_warmstart : bool
        If True and ``warmstart_params`` is provided, return the
        warmstart verbatim without running NAdam. Used by caller-side
        skip-opt / T-cache logic.
    regularization_strength : float
        Tikhonov regularization passed through to
        ``isotropic_thin_3d.reconstruct``.
    loss_settings : LossSettings, optional
        Loss configuration. Defaults to
        :class:`~waveorder.optim.losses.MidbandPowerLossSettings`.
    reflect_pad : int
        Reflect-pad pixels added on each side of every subtile before
        the forward pass; the reconstructed phase is then cropped back
        to the original ``(y, x)`` extent before the loss is computed.
    pupil_steepness : float
        Sigmoid steepness for the smooth pupil cutoff inside
        ``isotropic_thin_3d.reconstruct``.

    Returns
    -------
    TiltOptimResult
        Per-subtile optimized parameters. ``z_offsets`` is the final
        shared shift, broadcast across each subtile in its group.
    """
    if loss_settings is None:
        loss_settings = MidbandPowerLossSettings()
    if len(tiles) == 0:
        raise ValueError("Got 0 tiles — nothing to optimize.")

    B = len(tiles)
    device = tiles[0].device
    freeze_zenith = "zenith" in freeze_axes
    freeze_azimuth = "azimuth" in freeze_axes

    # ── Warmstart / skip path ──────────────────────────────────────────
    if warmstart_params is not None and skip_optim_if_warmstart:
        if warmstart_params.z_offsets.shape[0] != B:
            raise ValueError(
                f"warmstart has {warmstart_params.z_offsets.shape[0]} entries "
                f"but {B} tiles were passed"
            )
        return TiltOptimResult(
            z_offsets=warmstart_params.z_offsets.detach().clone(),
            zeniths=warmstart_params.zeniths.detach().clone(),
            azimuths=warmstart_params.azimuths.detach().clone(),
            final_loss=warmstart_params.final_loss.detach().clone(),
            n_iters=0,
            skipped=True,
        )

    # ── Pull TF settings apart ─────────────────────────────────────────
    tf_no_z = {
        k: v for k, v in tf_settings.items()
        if k not in ("z_pixel_size", "z_padding")
    }
    z_pixel_size = float(tf_settings["z_pixel_size"])

    # ── Build per-subtile init tensors ─────────────────────────────────
    if warmstart_params is not None:
        zen_full = warmstart_params.zeniths.detach().to(device, torch.float32)
        azi_full = warmstart_params.azimuths.detach().to(device, torch.float32)
        z_full = warmstart_params.z_offsets.detach().to(device, torch.float32)
    else:
        zen_full = _as_per_subtile(zen_init, B, device)
        azi_full = _as_per_subtile(azi_init, B, device)
        z_full = _as_per_subtile(z_init, B, device)

    if focus_offsets is None:
        focus_full = torch.zeros(B, dtype=torch.float32, device=device)
    else:
        focus_full = _as_per_subtile(focus_offsets, B, device)

    # Group by (rounded) focus offset → tiles sharing this focus offset
    # share the same z_positions tensor (TF computed once per group).
    # Then sub-group by tensor shape so we can stack into a single batched
    # forward call per shape.
    out_z = z_full.detach().clone()
    out_zen = zen_full.detach().clone()
    out_azi = azi_full.detach().clone()
    final_loss = torch.zeros((), dtype=torch.float32, device=device)
    iters_run = 0

    loss_fn = build_loss_fn(
        loss_settings,
        NA_det=tf_settings["numerical_aperture_detection"],
        wavelength=tf_settings["wavelength_illumination"],
        pixel_size=tf_settings["yx_pixel_size"],
    )

    groups: dict = defaultdict(list)
    for i in range(B):
        key = round(float(focus_full[i].item()), 1)
        groups[key].append(i)

    z_half = int(z_index.numel()) // 2

    for offset, group_idxs in groups.items():
        # Sub-group by shape
        shape_groups: dict = defaultdict(list)
        for i in group_idxs:
            shape_groups[tuple(tiles[i].shape)].append(i)

        for _shape, idxs in shape_groups.items():
            n = len(idxs)
            bzyx = torch.stack([tiles[i] for i in idxs])
            bzyx_pad = F.pad(bzyx, (reflect_pad,) * 4, mode="reflect")

            # Init param tensors (shared shift across the group's subtiles)
            z_p = torch.tensor(
                [(float(focus_full[i].item()) + float(z_full[i].item())) / 2.0
                 for i in idxs],
                dtype=torch.float32, device=device, requires_grad=True,
            )
            zen_p = torch.tensor(
                [float(zen_full[i].item()) for i in idxs],
                dtype=torch.float32, device=device,
                requires_grad=not freeze_zenith,
            )
            azi_p = torch.tensor(
                [float(azi_full[i].item()) for i in idxs],
                dtype=torch.float32, device=device,
                requires_grad=not freeze_azimuth,
            )

            param_groups = [{"params": [z_p], "lr": lr_z * 2}]
            if not freeze_zenith:
                param_groups.append({"params": [zen_p], "lr": lr_zenith * 2})
            if not freeze_azimuth:
                param_groups.append({"params": [azi_p], "lr": lr_azimuth * 2})
            optimizer = torch.optim.NAdam(param_groups)

            last_good = (z_p.detach().clone(), zen_p.detach().clone(),
                         azi_p.detach().clone())
            group_loss = torch.tensor(float("nan"), device=device)
            steps_for_group = 0
            for step in range(n_iters):
                optimizer.zero_grad()
                z_positions = (z_index + z_p.mean()) * z_pixel_size
                _, phase_byx = isotropic_thin_3d.reconstruct(
                    bzyx_pad, z_position_list=z_positions,
                    regularization_strength=regularization_strength,
                    tilt_angle_zenith=zen_p,
                    tilt_angle_azimuth=azi_p,
                    pupil_steepness=pupil_steepness, **tf_no_z,
                )
                phase_byx = phase_byx[
                    :, reflect_pad:-reflect_pad, reflect_pad:-reflect_pad,
                ]
                if torch.isnan(phase_byx).any():
                    break
                loss = torch.stack(
                    [loss_fn(phase_byx[b]) for b in range(n)]
                ).sum()
                if torch.isnan(loss):
                    break
                last_good = (z_p.detach().clone(), zen_p.detach().clone(),
                             azi_p.detach().clone())
                group_loss = loss.detach()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    z_p.clamp_(-z_half, z_half)
                steps_for_group += 1

            iters_run = max(iters_run, steps_for_group)
            final_loss = final_loss + group_loss if not torch.isnan(group_loss) else final_loss
            zg, zeng, azig = last_good
            for j, i in enumerate(idxs):
                out_z[i] = zg[j]
                out_zen[i] = zeng[j]
                out_azi[i] = azig[j]

    return TiltOptimResult(
        z_offsets=out_z,
        zeniths=out_zen,
        azimuths=out_azi,
        final_loss=final_loss,
        n_iters=iters_run,
        skipped=False,
    )
