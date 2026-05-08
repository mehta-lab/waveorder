"""Per-tile Zernike-coefficient recovery for shift-variant fluorescence.

Given a 3D fluorescence stack and a known sparse phantom (one Gaussian
bead per tile), this module fits a vector of Noll-indexed Zernike
coefficients per tile so the local forward model

    predicted_tile  =  bead_template  *  PSF(c_j)

best matches the measured tile under one of several loss functions
(``mse`` on the forward, or sharpness-on-Wiener-deconvolved-tile flavours
for blind-deconvolution-style fitting). The optimiser runs all tiles in
parallel via a batched gradient descent — gradients are independent per
tile, equivalent to running ``T`` separate optimisers.

After Adam (or another optimiser) converges, the module Wiener-
deconvolves each tile with its optimised OTF and stitches the result via
``waveorder.tile_stitch.blend_output_tile`` to produce a full-FOV
reconstruction.

Public entry points
-------------------
* :class:`RecoverySettings` — pydantic settings (loss, optimiser, LR
  schedule, regularisation, tile geometry).
* :class:`RecoveryResult` — recovered coefficients, per-tile loss
  history, stitched reconstruction.
* :func:`recover_zernikes` — single callable end-to-end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, PositiveInt
from torch import Tensor

from waveorder import optics
from waveorder.tile_stitch._engine import blend_output_tile
from waveorder.tile_stitch.blend import gaussian_mean
from waveorder.tile_stitch.partition import generate_output_tiles, generate_tiles, input_tiles_for_output
from waveorder.zernike import zernike_modes


class RecoverySettings(BaseModel):
    """Configuration for per-tile Zernike-coefficient recovery."""

    model_config = ConfigDict(extra="forbid")

    noll_indices: tuple[PositiveInt, ...] = Field(
        default=tuple(range(4, 16)),
        description="Noll-indexed Zernike modes to recover (default Z4..Z15)",
    )
    tile_size_yx: dict[str, PositiveInt] = Field(
        default_factory=lambda: {"y": 26, "x": 26},
        description="per-axis tile size in pixels for the recon partition",
    )
    tile_overlap_yx: dict[str, NonNegativeFloat] = Field(
        default_factory=lambda: {"y": 0.0, "x": 0.0},
        description="per-axis tile overlap in pixels",
    )
    bead_template_sigma_um: tuple[PositiveFloat, PositiveFloat, PositiveFloat] = Field(
        default=(0.15, 0.1, 0.1),
        description="(z, y, x) Gaussian sigma of the per-tile bead template, in microns",
    )
    loss: Literal["mse", "midband", "midband_3d", "tv", "laplacian_var", "normalized_var", "spectral_flatness"] = (
        "midband"
    )
    midband_fractions: tuple[float, float] = (0.2, 0.4)
    l1_strength: NonNegativeFloat = 0.0
    smooth_strength: NonNegativeFloat = 0.0
    scale_fit: bool = True
    optimizer: Literal["adam", "adamw", "nadam", "sgd", "lbfgs"] = "adam"
    lr_schedule: Literal["constant", "cosine", "step", "warmup"] = "constant"
    lr_per_mode: dict[int, PositiveFloat] = Field(
        default_factory=lambda: {
            4: 0.05,
            5: 0.05,
            6: 0.05,
            7: 0.03,
            8: 0.03,
            9: 0.03,
            10: 0.03,
            11: 0.02,
            12: 0.02,
            13: 0.02,
            14: 0.02,
            15: 0.02,
        },
        description="per-mode base learning rates (waves/step)",
    )
    lr_mult: PositiveFloat = 1.0
    n_iter: PositiveInt = 250
    wiener_regularization: PositiveFloat = 1.0e-3


@dataclass
class RecoveryResult:
    """Outputs of :func:`recover_zernikes`."""

    coefs: Tensor  # (T, M) recovered Zernike coefficients
    per_tile_loss_history: np.ndarray  # (n_iter, T)
    reconstruction_zyx: np.ndarray  # (Z, Y, X) stitched Wiener deconvolution
    deconv_per_tile: Tensor  # (T, Z, tY, tX)
    input_tile_bboxes: list[dict]  # one dict per input tile {"y": (lo, hi), "x": (lo, hi)}
    tile_centers_um: list[tuple[float, float]]  # (y_um, x_um) of each input tile center
    noll_indices: tuple[int, ...]
    settings: dict  # serialised RecoverySettings for traceability
    loss_history: list[float]


def recover_zernikes(
    sim_zyx: np.ndarray | Tensor,
    yx_pixel_size_um: float,
    z_pixel_size_um: float,
    wavelength_emission_um: float,
    numerical_aperture_detection: float,
    index_of_refraction_media: float,
    settings: RecoverySettings,
) -> RecoveryResult:
    """Run per-tile Zernike recovery + tile-stitched Wiener deconvolution.

    Parameters
    ----------
    sim_zyx : ndarray or Tensor
        3D fluorescence stack, shape ``(Z, Y, X)``.
    yx_pixel_size_um, z_pixel_size_um : float
        Voxel size in microns.
    wavelength_emission_um : float
        Emission wavelength in microns.
    numerical_aperture_detection : float
        Detection numerical aperture.
    index_of_refraction_media : float
        Refractive index of the immersion medium.
    settings : RecoverySettings
        Recovery configuration (loss, optimiser, regularisation, etc.).

    Returns
    -------
    RecoveryResult
    """
    if isinstance(sim_zyx, np.ndarray):
        sim_t = torch.from_numpy(sim_zyx).float()
    else:
        sim_t = sim_zyx.float()
    Z, Y, X = sim_t.shape

    sim_xr = xr.DataArray(sim_t.numpy(), dims=("z", "y", "x"))
    overlap = {k: int(v) for k, v in settings.tile_overlap_yx.items()}
    input_tiles, tile_dims = generate_tiles(sim_xr, dict(settings.tile_size_yx), overlap)
    output_tiles = generate_output_tiles({"y": Y, "x": X}, dict(settings.tile_size_yx), tile_dims)
    out_to_in = {ot.tile_id: input_tiles_for_output(ot, input_tiles, tile_dims) for ot in output_tiles}

    tile_centers_um = []
    for it in input_tiles:
        ys = it.slices["y"]
        xs = it.slices["x"]
        cy = (0.5 * (ys.start + ys.stop) - Y / 2) * yx_pixel_size_um
        cx = (0.5 * (xs.start + xs.stop) - X / 2) * yx_pixel_size_um
        tile_centers_um.append((float(cy), float(cx)))

    tiles = _stack_input_tiles(sim_t.numpy(), input_tiles)  # (T, Z, tY, tX)
    bead_template = _build_gaussian_bead_template(
        tiles.shape, settings.bead_template_sigma_um, (z_pixel_size_um, yx_pixel_size_um, yx_pixel_size_um)
    )

    pupil_amp, frr, theta = _build_pupil_grids(
        tiles.shape[-2:], yx_pixel_size_um, numerical_aperture_detection, wavelength_emission_um
    )
    z_position_list = torch.fft.ifftshift((torch.arange(tiles.shape[1]) - tiles.shape[1] // 2) * z_pixel_size_um)
    propagation_kernel = optics.generate_propagation_kernel(
        frr, pupil_amp, wavelength_emission_um / index_of_refraction_media, z_position_list
    )
    rho = (frr * wavelength_emission_um / numerical_aperture_detection).clamp(max=1.0)
    zernike_basis = zernike_modes(list(settings.noll_indices), rho, theta)
    bead_template_fft = torch.fft.fftn(bead_template, dim=(-3, -2, -1))
    midband_mask = _midband_mask(
        tiles.shape[-2:],
        yx_pixel_size_um,
        numerical_aperture_detection,
        wavelength_emission_um,
        settings.midband_fractions,
    )

    coefs, loss_history, per_tile_loss_history, deconv_tiles = _optimize(
        tiles=tiles,
        bead_template_fft=bead_template_fft,
        pupil_amp=pupil_amp,
        propagation_kernel=propagation_kernel,
        zernike_basis=zernike_basis,
        midband_mask=midband_mask,
        settings=settings,
    )

    deconv_volume = _stitch_with_blend(input_tiles, deconv_tiles, output_tiles, out_to_in, tile_dims, Z, Y, X)

    return RecoveryResult(
        coefs=coefs,
        per_tile_loss_history=per_tile_loss_history,
        reconstruction_zyx=deconv_volume,
        deconv_per_tile=deconv_tiles,
        input_tile_bboxes=[it.bbox() for it in input_tiles],
        tile_centers_um=tile_centers_um,
        noll_indices=tuple(settings.noll_indices),
        settings=settings.model_dump(),
        loss_history=loss_history,
    )


def _stack_input_tiles(sim: np.ndarray, input_tiles) -> Tensor:
    arrs = []
    for it in input_tiles:
        arrs.append(sim[:, it.slices["y"], it.slices["x"]])
    shapes = {a.shape for a in arrs}
    if len(shapes) > 1:
        Z = arrs[0].shape[0]
        maxY = max(a.shape[1] for a in arrs)
        maxX = max(a.shape[2] for a in arrs)
        padded = np.zeros((len(arrs), Z, maxY, maxX), dtype=np.float32)
        for i, a in enumerate(arrs):
            padded[i, :, : a.shape[1], : a.shape[2]] = a
        return torch.from_numpy(padded)
    return torch.from_numpy(np.stack(arrs, axis=0))


def _build_gaussian_bead_template(tile_shape, sigma_um, pix_um) -> Tensor:
    T, Z, Y, X = tile_shape
    sz, sy, sx = sigma_um
    pz, py, px = pix_um
    z = (torch.arange(Z) - Z // 2) * pz
    y = (torch.arange(Y) - Y // 2) * py
    x = (torch.arange(X) - X // 2) * px
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    bead = torch.exp(-(zz**2 / (2 * sz**2) + yy**2 / (2 * sy**2) + xx**2 / (2 * sx**2)))
    return bead.unsqueeze(0).expand(T, -1, -1, -1).contiguous()


def _build_pupil_grids(yx_shape, yx_pixel_size, NA_det, wavelength):
    fy = torch.fft.fftfreq(yx_shape[0], d=yx_pixel_size)
    fx = torch.fft.fftfreq(yx_shape[1], d=yx_pixel_size)
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    frr = torch.sqrt(fxx**2 + fyy**2)
    theta = torch.atan2(fyy, fxx)
    pupil_amp = optics.generate_pupil(frr, NA_det, wavelength)
    return pupil_amp, frr, theta


def _midband_mask(yx_shape, yx_pixel_size, NA_det, wavelength, midband_fractions):
    Y, X = yx_shape
    fy = torch.fft.fftshift(torch.fft.fftfreq(Y, d=yx_pixel_size))
    fx = torch.fft.fftshift(torch.fft.fftfreq(X, d=yx_pixel_size))
    fyy, fxx = torch.meshgrid(fy, fx, indexing="ij")
    frr = torch.sqrt(fxx**2 + fyy**2)
    cutoff = 2 * NA_det / wavelength
    inner, outer = midband_fractions
    return ((frr > cutoff * inner) & (frr < cutoff * outer)).float()


def _psf_from_coefs(coefs, pupil_amp, propagation_kernel, zernike_basis):
    phi = torch.einsum("tm,myx->tyx", coefs, zernike_basis)
    pupil = pupil_amp[None] * torch.exp(2j * math.pi * phi.to(torch.complex64))
    pupil_3d = propagation_kernel[None] * pupil[:, None, :, :]
    coherent = torch.fft.ifft2(pupil_3d, dim=(-2, -1))
    psf = torch.abs(coherent) ** 2
    return psf / psf.sum(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-12)


def _sharpness_loss_per_tile(deconv: Tensor, kind: str, midband_mask: Tensor) -> Tensor:
    T, Z, Y, X = deconv.shape
    if kind == "midband":
        slab = deconv[:, Z // 2]
        slab_fft = torch.fft.fftshift(torch.fft.fft2(slab), dim=(-2, -1))
        return -(torch.abs(slab_fft) ** 2 * midband_mask).mean(dim=(-2, -1))
    if kind == "midband_3d":
        d_fft = torch.fft.fftshift(torch.fft.fft2(deconv), dim=(-2, -1))
        return -((torch.abs(d_fft) ** 2) * midband_mask[None, None, :, :]).mean(dim=(-3, -2, -1))
    if kind == "tv":
        slab = deconv[:, Z // 2]
        gy = slab[:, 1:, :] - slab[:, :-1, :]
        gx = slab[:, :, 1:] - slab[:, :, :-1]
        return -(gy.abs().mean(dim=(-2, -1)) + gx.abs().mean(dim=(-2, -1)))
    if kind == "laplacian_var":
        slab = deconv[:, Z // 2]
        lap = -4 * slab
        lap[:, 1:, :] += slab[:, :-1, :]
        lap[:, :-1, :] += slab[:, 1:, :]
        lap[:, :, 1:] += slab[:, :, :-1]
        lap[:, :, :-1] += slab[:, :, 1:]
        return -(lap.var(dim=(-2, -1)))
    if kind == "normalized_var":
        slab = deconv[:, Z // 2]
        mean = slab.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        var = ((slab - mean) ** 2).mean(dim=(-2, -1))
        return -(var / mean.squeeze(-1).squeeze(-1) ** 2)
    if kind == "spectral_flatness":
        slab = deconv[:, Z // 2]
        fft = torch.fft.fftshift(torch.fft.fft2(slab), dim=(-2, -1))
        power = torch.abs(fft) ** 2 * midband_mask + 1e-12
        log_mean = torch.log(power).mean(dim=(-2, -1))
        arith_mean = power.mean(dim=(-2, -1)).clamp(min=1e-12)
        return -(torch.exp(log_mean) / arith_mean)
    raise ValueError(f"unknown sharpness kind: {kind!r}")


def _spatial_smoothness_pairs(grid_yx):
    ny, nx = grid_yx
    pairs: list[tuple[int, int]] = []
    for r in range(ny):
        for c in range(nx):
            t = r * nx + c
            if c + 1 < nx:
                pairs.append((t, t + 1))
            if r + 1 < ny:
                pairs.append((t, t + nx))
    return pairs


def _optimize(
    tiles: Tensor,
    bead_template_fft: Tensor,
    pupil_amp: Tensor,
    propagation_kernel: Tensor,
    zernike_basis: Tensor,
    midband_mask: Tensor,
    settings: RecoverySettings,
):
    T = tiles.shape[0]
    M = len(settings.noll_indices)
    n_iter = settings.n_iter

    coefs_per_mode = [torch.zeros(T, dtype=torch.float32, requires_grad=True) for _ in settings.noll_indices]
    base_lrs = [settings.lr_per_mode.get(int(j), 0.01) * settings.lr_mult for j in settings.noll_indices]
    param_groups = [{"params": [c], "lr": lr} for c, lr in zip(coefs_per_mode, base_lrs)]

    if settings.optimizer == "adam":
        optimizer = torch.optim.Adam(param_groups)
    elif settings.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-3)
    elif settings.optimizer == "nadam":
        optimizer = torch.optim.NAdam(param_groups)
    elif settings.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
    elif settings.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(coefs_per_mode, lr=max(base_lrs), max_iter=1)
    else:
        raise ValueError(settings.optimizer)

    def lr_at_step(step: int, base_lr: float) -> float:
        if settings.lr_schedule == "constant":
            return base_lr
        if settings.lr_schedule == "cosine":
            return 0.5 * base_lr * (1 + math.cos(math.pi * step / max(n_iter - 1, 1)))
        if settings.lr_schedule == "step":
            return base_lr * (0.5 ** (step // 100))
        if settings.lr_schedule == "warmup":
            warmup = max(1, n_iter // 10)
            return base_lr * ((step + 1) / warmup if step < warmup else 1.0)
        raise ValueError(settings.lr_schedule)

    grid_yx = (int(np.sqrt(T)), int(np.sqrt(T)))
    smooth_pairs = (
        _spatial_smoothness_pairs(grid_yx) if (settings.smooth_strength > 0 and grid_yx[0] * grid_yx[1] == T) else []
    )

    def per_tile_loss_for_coefs():
        coefs_local = torch.stack(coefs_per_mode, dim=1)
        psf = _psf_from_coefs(coefs_local, pupil_amp, propagation_kernel, zernike_basis)
        psf_fft = torch.fft.fftn(psf, dim=(-3, -2, -1))
        if settings.loss == "mse":
            predicted = torch.real(torch.fft.ifftn(bead_template_fft * psf_fft, dim=(-3, -2, -1)))
            if settings.scale_fit:
                scale = (predicted * tiles).sum(dim=(-3, -2, -1)) / (predicted * predicted).sum(dim=(-3, -2, -1)).clamp(
                    min=1e-20
                )
                predicted = predicted * scale[:, None, None, None]
            ptl = ((predicted - tiles) ** 2).mean(dim=(-3, -2, -1))
        else:
            otf_train = psf_fft
            inv = torch.conj(otf_train) / (torch.abs(otf_train) ** 2 + settings.wiener_regularization)
            data_fft = torch.fft.fftn(tiles, dim=(-3, -2, -1))
            deconv = torch.real(torch.fft.ifftn(data_fft * inv, dim=(-3, -2, -1)))
            ptl = _sharpness_loss_per_tile(deconv, settings.loss, midband_mask)
        return ptl, coefs_local

    loss_history: list[float] = []
    per_tile_loss_history: list[np.ndarray] = []
    for it in range(n_iter):
        if settings.optimizer != "lbfgs":
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = lr_at_step(it, base)

        if settings.optimizer == "lbfgs":
            captured: dict = {}

            def closure():
                optimizer.zero_grad()
                ptl, coefs_local = per_tile_loss_for_coefs()
                lossv = ptl.mean()
                if settings.l1_strength > 0:
                    lossv = lossv + settings.l1_strength * coefs_local.abs().mean()
                if settings.smooth_strength > 0 and smooth_pairs:
                    diffs = torch.stack([coefs_local[a] - coefs_local[b] for a, b in smooth_pairs])
                    lossv = lossv + settings.smooth_strength * (diffs**2).mean()
                lossv.backward()
                captured["ptl"] = ptl.detach()
                captured["loss"] = float(lossv.detach())
                return lossv

            optimizer.step(closure)
            per_tile_loss = captured["ptl"]
            loss_val = captured["loss"]
        else:
            optimizer.zero_grad()
            per_tile_loss, coefs_local = per_tile_loss_for_coefs()
            loss = per_tile_loss.mean()
            if settings.l1_strength > 0:
                loss = loss + settings.l1_strength * coefs_local.abs().mean()
            if settings.smooth_strength > 0 and smooth_pairs:
                diffs = torch.stack([coefs_local[a] - coefs_local[b] for a, b in smooth_pairs])
                loss = loss + settings.smooth_strength * (diffs**2).mean()
            loss.backward()
            optimizer.step()
            loss_val = float(loss)

        loss_history.append(loss_val)
        per_tile_loss_history.append(per_tile_loss.detach().numpy())

    coefs = torch.stack(coefs_per_mode, dim=1).detach()
    psf = _psf_from_coefs(coefs, pupil_amp, propagation_kernel, zernike_basis)
    psf_fft = torch.fft.fftn(psf, dim=(-3, -2, -1))
    inv = torch.conj(psf_fft) / (torch.abs(psf_fft) ** 2 + settings.wiener_regularization)
    data_fft = torch.fft.fftn(tiles, dim=(-3, -2, -1))
    deconv = torch.real(torch.fft.ifftn(data_fft * inv, dim=(-3, -2, -1)))

    return coefs, loss_history, np.stack(per_tile_loss_history), deconv


def _stitch_with_blend(input_tiles, per_tile_volume, output_tiles, out_to_in, tile_dims, Z, Y, X) -> np.ndarray:
    blend = gaussian_mean()
    tiles_by_id = {it.tile_id: it for it in input_tiles}
    arrays_by_id = {
        it.tile_id: per_tile_volume[i].detach().cpu().numpy().astype(np.float32) for i, it in enumerate(input_tiles)
    }
    output = np.zeros((Z, Y, X), dtype=np.float32)
    for ot in output_tiles:
        contributors = [(tiles_by_id[tid], arrays_by_id[tid]) for tid in out_to_in[ot.tile_id]]
        block = blend_output_tile(
            ot,
            contributors,
            blend,
            leading_shape=(Z,),
            tile_dims=tile_dims,
            output_dtype=np.float32,
        )
        output[:, ot.slices["y"], ot.slices["x"]] = block
    return output


@dataclass
class RecoveryMetrics:
    """Optional FoMs computed after recovery (when the truth is known)."""

    recovery_score: float
    per_mode_rmse: dict[str, float]
    per_mode_correlation: dict[str, float | None]
    truth_coefs: list[list[float]] = field(default_factory=list)
    recovered_coefs: list[list[float]] = field(default_factory=list)


def evaluate_recovery_against_truth(
    coefs: Tensor,
    truth_coefs: Tensor,
    noll_indices: tuple[int, ...],
) -> RecoveryMetrics:
    """Compute Zernike-recovery FoMs given a ground-truth coefficient grid.

    The ``recovery_score`` is

    .. math::

        1 - \\frac{\\|c_\\text{rec} - c_\\text{truth}\\|_F}{\\|c_\\text{truth}\\|_F},

    so 1 = perfect recovery, 0 = no better than zeros, negative = worse
    than zeros. Per-mode RMSE + Pearson correlation are also returned.
    """
    rec = coefs.numpy()
    tru = truth_coefs.numpy()
    err = float(np.sqrt(np.mean((rec - tru) ** 2)))
    norm = float(np.sqrt(np.mean(tru**2)))
    score = float("nan") if norm == 0 else 1.0 - err / norm

    per_mode_rmse: dict[str, float] = {}
    per_mode_correlation: dict[str, float | None] = {}
    for i, j in enumerate(noll_indices):
        rec_i = rec[:, i]
        tru_i = tru[:, i]
        per_mode_rmse[f"Z{j}"] = float(np.sqrt(np.mean((rec_i - tru_i) ** 2)))
        if tru_i.std() > 0:
            per_mode_correlation[f"Z{j}"] = float(np.corrcoef(rec_i, tru_i)[0, 1])
        else:
            per_mode_correlation[f"Z{j}"] = None

    return RecoveryMetrics(
        recovery_score=score,
        per_mode_rmse=per_mode_rmse,
        per_mode_correlation=per_mode_correlation,
        truth_coefs=tru.tolist(),
        recovered_coefs=rec.tolist(),
    )


def evaluate_truth_at_tile_centers(
    tile_centers_um: list[tuple[float, float]],
    field_extent_um: tuple[float, float],
    spatial_pupil_coefficients: dict[tuple[int, int, int], float],
    noll_indices: tuple[int, ...],
) -> Tensor:
    """Evaluate the ground-truth spatial polynomial at each tile center.

    Returns a ``(T, M)`` tensor matching the shape of ``coefs``.
    """
    half_y, half_x = field_extent_um
    j_to_idx = {j: i for i, j in enumerate(noll_indices)}
    truth = torch.zeros(len(tile_centers_um), len(noll_indices))
    for t, (y_um, x_um) in enumerate(tile_centers_um):
        y_norm = y_um / half_y
        x_norm = x_um / half_x
        for (j, m, n), c in spatial_pupil_coefficients.items():
            if j in j_to_idx:
                truth[t, j_to_idx[j]] += c * (x_norm**m) * (y_norm**n)
    return truth
