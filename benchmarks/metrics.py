"""Metric computation for benchmarks.

Three metric groups:
- image_quality (always): distribution stats + self-consistency metrics
- with_reference (when annotated reference available): ssim, mse, correlation
- with_phantom (when simulated phantom available): mse vs ground truth

Self-consistency metrics reuse the loss functions from
``waveorder.optim.losses``, which properly mask to the optical passband.
"""

from __future__ import annotations

import torch
from skimage.metrics import structural_similarity
from torch import Tensor
from torch.nn.functional import mse_loss

from waveorder.optim.losses import (
    MidbandPowerLossSettings,
    SpectralFlatnessLossSettings,
    TotalVariationLossSettings,
    build_loss_fn,
)

# --- image_quality: self-metrics (always computed) ---


def distribution_stats(volume: Tensor, n_bins: int = 50) -> dict:
    """Compute distribution stats: max, min, mean, and histogram.

    Parameters
    ----------
    volume : Tensor
        Input volume of any shape.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    dict
        Keys: max, min, mean, histogram (with bin_edges and counts).
    """
    v = volume.float().flatten()
    hist = torch.histogram(v.cpu(), bins=n_bins)
    return {
        "max": volume.max().item(),
        "min": volume.min().item(),
        "mean": volume.mean().item(),
        "histogram": {
            "bin_edges": hist.bin_edges.tolist(),
            "counts": hist.hist.tolist(),
        },
    }


# --- with_reference / with_phantom: comparison metrics ---


def comparison_metrics(recon: Tensor, target: Tensor) -> dict:
    """Compute MSE, SSIM, and Pearson correlation between two volumes.

    Parameters
    ----------
    recon : Tensor
        Reconstruction volume.
    target : Tensor
        Target volume (same shape). Can be an annotated reference
        or simulated phantom ground truth.

    Returns
    -------
    dict
        Keys: mse, ssim, correlation.
    """
    a, b = recon.float(), target.float()

    # Pearson correlation via torch.corrcoef
    corr_matrix = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))
    corr = corr_matrix[0, 1].item()

    # SSIM via scikit-image (windowed, proper constants)
    a_np, b_np = a.cpu().numpy(), b.cpu().numpy()
    data_range = b_np.max() - b_np.min()
    ssim_val = structural_similarity(a_np, b_np, data_range=data_range)

    return {
        "mse": mse_loss(a, b).item(),
        "ssim": float(ssim_val),
        "correlation": corr,
    }


# --- Aggregation ---


def compute_metrics(
    recon: Tensor,
    NA_det: float,
    wavelength: float,
    pixel_size: float,
    reference: Tensor | None = None,
    phantom: Tensor | None = None,
    n_bins: int = 50,
    midband_fractions: tuple[float, float] = (0.125, 0.25),
) -> dict:
    """Compute all applicable metric groups.

    Parameters
    ----------
    recon : Tensor
        Reconstruction volume, 2D (Y, X) or 3D (Z, Y, X).
    NA_det : float
        Detection numerical aperture.
    wavelength : float
        Illumination/emission wavelength (same units as pixel_size).
    pixel_size : float
        Object-space pixel size (same units as wavelength).
    reference : Tensor or None
        Annotated reference for with_reference comparison.
    phantom : Tensor or None
        Ground truth from simulated phantom for with_phantom comparison.
    n_bins : int
        Number of histogram bins.
    midband_fractions : tuple[float, float]
        Inner and outer fractions of the diffraction-limited cutoff
        frequency for midband power and spectral flatness.

    Returns
    -------
    dict
        Nested dict with image_quality, and optionally with_reference,
        with_phantom.
    """
    v = recon.float()

    # Reuse loss functions from waveorder.optim.losses — these mask to
    # the optical passband defined by NA_det / wavelength.
    # Losses are negated (for minimization), so we negate back.
    fracs = list(midband_fractions)
    mbp_fn = build_loss_fn(
        MidbandPowerLossSettings(midband_fractions=fracs),
        NA_det,
        wavelength,
        pixel_size,
    )
    sf_fn = build_loss_fn(
        SpectralFlatnessLossSettings(midband_fractions=fracs),
        NA_det,
        wavelength,
        pixel_size,
    )
    # TV is a real-space metric (optical params unused but required by API)
    tv_fn = build_loss_fn(TotalVariationLossSettings(), NA_det, wavelength, pixel_size)

    result = {
        "image_quality": {
            **distribution_stats(v, n_bins=n_bins),
            "midband_power": -mbp_fn(v).item(),
            "spectral_flatness": -sf_fn(v).item(),
            "total_variation": -tv_fn(v).item(),
        },
    }

    if reference is not None:
        result["with_reference"] = comparison_metrics(v, reference)

    if phantom is not None:
        result["with_phantom"] = comparison_metrics(v, phantom)

    return result
