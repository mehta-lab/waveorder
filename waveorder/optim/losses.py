"""Loss functions for reconstruction parameter optimization.

All loss functions take a reconstruction tensor and optical parameters,
and return a scalar to **minimize**. Functions that measure image quality
(higher = better) are negated so the optimizer minimizes.

All losses are normalized to be tile-size-invariant and handle both
2D ``(Y, X)`` and 3D ``(Z, Y, X)`` input by computing per-slice and
averaging over Z.
"""

from __future__ import annotations

from typing import Literal, Union

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor


class _LossBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MidbandPowerLossSettings(_LossBaseModel):
    """Maximize power in a midband spatial frequency annulus.

    The annulus is defined by fractions of the optical cutoff frequency
    ``2 * NA_det / wavelength``. Good default for general-purpose
    autofocus and parameter optimization.

    Parameters
    ----------
    midband_fractions : list[float]
        Inner and outer fractions of the cutoff frequency.
    """

    type: Literal["midband_power"] = "midband_power"
    midband_fractions: list[float] = Field(
        default=[0.125, 0.25],
        description="inner/outer fractions of cutoff frequency",
    )


class TotalVariationLossSettings(_LossBaseModel):
    """Maximize total variation (mean of gradient magnitudes).

    Rewards sharp edges in the reconstruction. Operates in real space
    so it complements frequency-domain metrics. Always mean-normalized
    for tile-size invariance.
    """

    type: Literal["total_variation"] = "total_variation"


class LaplacianVarianceLossSettings(_LossBaseModel):
    """Maximize variance of the Laplacian (classic autofocus metric).

    The Laplacian highlights high-frequency content; its variance is
    highest for well-focused images. Produces a smooth loss landscape
    for defocus parameters. Already tile-size-invariant.
    """

    type: Literal["laplacian_variance"] = "laplacian_variance"


class NormalizedVarianceLossSettings(_LossBaseModel):
    """Maximize normalized variance: ``var(image) / mean(image)^2``.

    Rewards high contrast relative to brightness. Simple and
    differentiable. Already tile-size-invariant.
    """

    type: Literal["normalized_variance"] = "normalized_variance"


class SpectralFlatnessLossSettings(_LossBaseModel):
    """Maximize spectral flatness in the midband.

    Spectral flatness is the ratio of geometric to arithmetic mean of
    the power spectrum in the midband annulus. Rewards even distribution
    of power across frequencies. Already tile-size-invariant.

    Parameters
    ----------
    midband_fractions : list[float]
        Inner and outer fractions of the cutoff frequency.
    """

    type: Literal["spectral_flatness"] = "spectral_flatness"
    midband_fractions: list[float] = Field(
        default=[0.125, 0.25],
        description="inner/outer fractions of cutoff frequency",
    )


LossSettings = Union[
    MidbandPowerLossSettings,
    TotalVariationLossSettings,
    LaplacianVarianceLossSettings,
    NormalizedVarianceLossSettings,
    SpectralFlatnessLossSettings,
]


def build_loss_fn(
    loss_settings: LossSettings,
    NA_det: float,
    wavelength: float,
    pixel_size: float,
):
    """Create a loss function from settings and optical parameters.

    Parameters
    ----------
    loss_settings : LossSettings
        One of the loss settings classes.
    NA_det : float
        Detection numerical aperture.
    wavelength : float
        Illumination/emission wavelength.
    pixel_size : float
        Object-space pixel size.

    Returns
    -------
    callable
        Function that takes a 2D ``(Y, X)`` or 3D ``(Z, Y, X)``
        reconstruction tensor and returns a scalar loss to minimize.
    """
    if loss_settings.type == "midband_power":
        return _make_midband_power_loss(
            NA_det,
            wavelength,
            pixel_size,
            loss_settings.midband_fractions,
        )
    elif loss_settings.type == "total_variation":
        return _make_total_variation_loss()
    elif loss_settings.type == "laplacian_variance":
        return _make_laplacian_variance_loss()
    elif loss_settings.type == "normalized_variance":
        return _make_normalized_variance_loss()
    elif loss_settings.type == "spectral_flatness":
        return _make_spectral_flatness_loss(
            NA_det,
            wavelength,
            pixel_size,
            loss_settings.midband_fractions,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_settings.type}")


def _make_midband_power_loss(NA_det, wavelength, pixel_size, midband_fractions):
    from waveorder.focus import compute_midband_power

    def loss_fn(recon: Tensor) -> Tensor:
        power = compute_midband_power(
            recon,
            NA_det=NA_det,
            lambda_ill=wavelength,
            pixel_size=pixel_size,
            midband_fractions=midband_fractions,
        )
        # compute_midband_power sums over masked pixels; normalize
        # by mask size for tile-size invariance
        if recon.ndim == 2:
            Y, X = recon.shape
        else:
            _, Y, X = recon.shape

        import numpy as np

        from waveorder import util

        _, _, fxx, fyy = util.gen_coordinate((Y, X), pixel_size)
        frr = np.sqrt(fxx**2 + fyy**2)
        cutoff = 2 * NA_det / wavelength
        n_masked = ((frr > cutoff * midband_fractions[0]) & (frr < cutoff * midband_fractions[1])).sum()

        loss = -power / max(n_masked, 1)
        if loss.ndim > 0:
            loss = loss.mean()
        return loss

    return loss_fn


def _make_total_variation_loss():
    def _tv_2d(img: Tensor) -> Tensor:
        dy = img[1:, :] - img[:-1, :]
        dx = img[:, 1:] - img[:, :-1]
        return (dy.abs().mean() + dx.abs().mean()) / 2

    def loss_fn(recon: Tensor) -> Tensor:
        if recon.ndim == 2:
            return -_tv_2d(recon)
        # 3D: average TV over all slices
        tv = torch.stack([_tv_2d(recon[z]) for z in range(recon.shape[0])])
        return -tv.mean()

    return loss_fn


def _make_laplacian_variance_loss():
    def _lap_var_2d(img: Tensor) -> Tensor:
        kernel = (
            torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                dtype=img.dtype,
                device=img.device,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        padded = img.unsqueeze(0).unsqueeze(0)
        lap = torch.nn.functional.conv2d(padded, kernel, padding=1)
        return lap.var()

    def loss_fn(recon: Tensor) -> Tensor:
        if recon.ndim == 2:
            return -_lap_var_2d(recon)
        vals = torch.stack([_lap_var_2d(recon[z]) for z in range(recon.shape[0])])
        return -vals.mean()

    return loss_fn


def _make_normalized_variance_loss():
    def _nvar_2d(img: Tensor) -> Tensor:
        mean = img.mean()
        var = img.var()
        return var / (mean**2 + 1e-12)

    def loss_fn(recon: Tensor) -> Tensor:
        if recon.ndim == 2:
            return -_nvar_2d(recon)
        vals = torch.stack([_nvar_2d(recon[z]) for z in range(recon.shape[0])])
        return -vals.mean()

    return loss_fn


def _make_spectral_flatness_loss(NA_det, wavelength, pixel_size, midband_fractions):
    import numpy as np

    from waveorder import util

    def _flatness_2d(img: Tensor) -> Tensor:
        Y, X = img.shape
        device = img.device

        _, _, fxx, fyy = util.gen_coordinate((Y, X), pixel_size)
        frr = torch.tensor(np.sqrt(fxx**2 + fyy**2), device=device)
        cutoff = 2 * NA_det / wavelength
        mask = torch.logical_and(
            frr > cutoff * midband_fractions[0],
            frr < cutoff * midband_fractions[1],
        )

        power = torch.abs(torch.fft.fftn(img.float(), dim=(-2, -1)))
        midband_power = power[mask]

        log_mean = midband_power.log().mean()
        arith_mean = midband_power.mean()
        return log_mean.exp() / (arith_mean + 1e-12)

    def loss_fn(recon: Tensor) -> Tensor:
        if recon.ndim == 2:
            return -_flatness_2d(recon)
        vals = torch.stack([_flatness_2d(recon[z]) for z in range(recon.shape[0])])
        return -vals.mean()

    return loss_fn
