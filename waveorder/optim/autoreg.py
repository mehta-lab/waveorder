"""Automatic selection of ``regularization_strength`` from the data.

A Tikhonov reconstruction ``H*/(|H|^2 + lambda)`` needs a regularization strength,
and the usual way to find one is to reconstruct at many values and look. This
module automates the looking: it sweeps lambda, scores each reconstruction, and
returns a pick plus everything needed to second-guess it.

Two rules are available, both ported from the CZ Biohub weight-search scripts:

``otsu_cnr``
    Contrast-to-noise ratio of an Otsu split of the local-variance map. Highest
    score wins. This is a per-reconstruction image-quality metric, in the same
    family as those in :mod:`waveorder.optim.losses`.

``l_curve``
    Corner of the trade-off curve between residual norm ``||Hx - y||`` and
    solution norm ``||x||``. This one is *not* a per-reconstruction score: its
    criterion is a property of the whole sweep, which is why this module sweeps
    rather than reusing the gradient-based machinery in
    :mod:`waveorder.optim.optimize`.

Caveats worth knowing before trusting a pick
--------------------------------------------
Neither rule is an oracle, and on real brightfield phase data both have been
observed to move a lot:

- The L-curves are often not L-shaped. A band-limited operator cannot fit
  out-of-band data, so the residual floors well above zero and the curve never
  flattens; the corner finder then locates a corner on what is nearly a line.
- The ``otsu_cnr`` pick shifts with the scoring crop and, less strongly, with
  the sweep endpoints.

Treat the result as a starting point, read the report, and check the warnings.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from torch import Tensor

from waveorder import sampling, util
from waveorder.reconstruct import tikhonov_regularized_inverse_filter

# Fixed rather than exposed as settings: these are implementation choices of the
# ported rules, not knobs a config is expected to turn. The crop that gets scored
# is reported so the choice stays visible.

# Side length of the square yx crop that is scored. Full frames are slow to sweep
# and their empty regions dilute the metrics; the whole Z range is always kept
# because the 3D transfer function needs it.
CROP_SIZE = 256

# Box-filter window for the local-variance map that otsu_cnr thresholds.
OTSU_WINDOW_SIZE = 3

_OTSU_NUM_BINS = 256
_LOCAL_VAR_CLIP_QUANTILE = 0.99
# torch.quantile rejects inputs beyond ~16M elements, which a full-frame volume
# exceeds; sample down to this many values instead.
_QUANTILE_MAX_ELEMENTS = 2**24
_L_CURVE_PROMINENCE_FRACTION = 0.10
_L_CURVE_PLATEAU_FRACTION = 0.90


class AutoRegularizationIgnoredWarning(UserWarning):
    """An ``auto_regularization`` block was dropped instead of being honoured."""


# The CLI suppresses UserWarning at startup to silence torch/CUDA noise; carve out
# this subclass, as PixelSizeMismatchWarning does, so a dropped sweep is not silent.
# "default" rather than "always" because a config is parsed at several points in the
# pipeline, and one copy of the message is the useful number.
warnings.filterwarnings("default", category=AutoRegularizationIgnoredWarning)


class AutoRegularizationSettings(BaseModel):
    """Sweep ``regularization_strength`` and pick a value, instead of hand-tuning it.

    Only valid for Tikhonov reconstructions of thick (3D) samples.
    """

    model_config = ConfigDict(extra="forbid")

    rule: Literal["otsu_cnr", "l_curve"] = Field(
        default="otsu_cnr",
        description="'otsu_cnr' maximizes local-variance contrast; 'l_curve' finds the trade-off corner",
    )
    search_min: float = Field(
        default=-6.0,
        description="lower log10 bound of the sweep, interpreted per search_scale",
    )
    search_max: float = Field(
        default=2.0,
        description="upper log10 bound of the sweep, interpreted per search_scale",
    )
    num_samples: PositiveInt = Field(default=25, description="regularization strengths to try")
    search_scale: Literal["relative", "absolute"] = Field(
        default="relative",
        description="'relative' reads the search bounds as log10(lambda / |H|^2max); 'absolute' as log10(lambda)",
    )
    report_path: Optional[str] = Field(
        default=None,
        description="write the full sweep (scores, norms, chosen index) to this JSON path; null = no report",
    )

    @model_validator(mode="after")
    def _validate_search(self):
        if not self.search_min < self.search_max:
            raise ValueError(f"search_min must be less than search_max, got {self.search_min} and {self.search_max}")
        # Curvature is a second derivative; the L-curve rule cannot run on fewer.
        if self.num_samples < 3:
            raise ValueError(f"num_samples must be at least 3, got {self.num_samples}")
        return self


@dataclass
class AutoRegResult:
    """Outcome of a regularization sweep, and the evidence behind it.

    Attributes
    ----------
    regularization_strength : float
        The pick, in the absolute units the setting takes.
    index : int
        Index of the pick within the swept arrays.
    rule : str
        Rule that chose it.
    regularization_strengths : np.ndarray
        Every strength that was scored, absolute units, ascending.
    transfer_function_peak_squared : float
        Peak ``|H|^2`` of the transfer function the sweep scored against.
    lambda_over_h2max : float
        ``regularization_strength / transfer_function_peak_squared``: how far the
        filter attenuates its own peak, and the scale-free way to compare picks.
    crop : tuple[int, int, int]
        ``(y0, x0, size)`` of the scored crop within the full frame.
    scores : np.ndarray
        ``otsu_cnr`` score per strength; empty when the rule was ``l_curve``.
    residual_norms : np.ndarray
        ``||Hx - y||`` per strength; empty when the rule was ``otsu_cnr``.
    solution_norms : np.ndarray
        ``||x||`` per strength; empty when the rule was ``otsu_cnr``.
    warnings : list[str]
        Non-fatal problems found while selecting, e.g. a pick on the sweep edge.
    """

    regularization_strength: float
    index: int
    rule: str
    regularization_strengths: np.ndarray
    transfer_function_peak_squared: float
    lambda_over_h2max: float
    crop: tuple[int, int, int]
    scores: np.ndarray = field(default_factory=lambda: np.array([]))
    residual_norms: np.ndarray = field(default_factory=lambda: np.array([]))
    solution_norms: np.ndarray = field(default_factory=lambda: np.array([]))
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serializable view, for ``report_path``."""

        def _list(a):
            return np.asarray(a).tolist()

        return {
            "rule": self.rule,
            "regularization_strength": self.regularization_strength,
            "index": self.index,
            "lambda_over_h2max": self.lambda_over_h2max,
            "transfer_function_peak_squared": self.transfer_function_peak_squared,
            "crop": {"y0": self.crop[0], "x0": self.crop[1], "size": self.crop[2]},
            "regularization_strengths": _list(self.regularization_strengths),
            "scores": _list(self.scores),
            "residual_norms": _list(self.residual_norms),
            "solution_norms": _list(self.solution_norms),
            "warnings": list(self.warnings),
        }


# --- otsu_cnr ---


def _local_variance(volumes: Tensor, window_size: int = OTSU_WINDOW_SIZE) -> Tensor:
    """Local variance map of each volume, computed slice-wise with a box filter.

    ``E[X^2] - E[X]^2`` over a ``window_size`` square, then clipped at the 99th
    percentile so a few hot voxels do not set the scale for the Otsu histogram.

    Parameters
    ----------
    volumes : Tensor
        ``(N, Z, Y, X)`` stack of reconstructions.
    window_size : int
        Side of the averaging window.

    Returns
    -------
    Tensor
        ``(N, Z, Y, X)`` local variances.
    """
    N, Z, Y, X = volumes.shape
    padding = window_size // 2

    flat = volumes.reshape(N * Z, 1, Y, X)
    # "replicate" matches scipy.ndimage.uniform_filter(mode="nearest").
    padded = F.pad(flat, [padding] * 4, mode="replicate")
    padded_sq = F.pad(flat**2, [padding] * 4, mode="replicate")

    mean_x = F.avg_pool2d(padded, kernel_size=window_size, stride=1)
    mean_x2 = F.avg_pool2d(padded_sq, kernel_size=window_size, stride=1)

    local_var = (mean_x2 - mean_x**2).clamp(min=0).reshape(N, Z, Y, X)

    per_volume = local_var.reshape(N, -1).float()
    # torch.quantile has a hard input-size limit; a strided sample of a variance
    # map estimates its 99th percentile closely enough to clip by.
    if per_volume.shape[1] > _QUANTILE_MAX_ELEMENTS:
        stride = per_volume.shape[1] // _QUANTILE_MAX_ELEMENTS + 1
        per_volume = per_volume[:, ::stride]
    high = torch.quantile(per_volume, _LOCAL_VAR_CLIP_QUANTILE, dim=-1)

    return torch.clamp(local_var, max=high.reshape(N, 1, 1, 1).expand_as(local_var))


def _otsu_cnr(flat: Tensor) -> Tensor:
    """Contrast-to-noise ratio across an Otsu threshold, batched.

    Each row is min-max normalized, histogrammed into 256 bins, split at the
    threshold maximizing between-class variance, and scored as
    ``|mu_signal - mu_background| / sigma_background``.

    Parameters
    ----------
    flat : Tensor
        ``(N, M)`` stack of flattened images.

    Returns
    -------
    Tensor
        ``(N,)`` CNR values.
    """
    N = flat.shape[0]

    vmin = flat.min(dim=-1, keepdim=True).values
    vmax = flat.max(dim=-1, keepdim=True).values
    normed = (flat - vmin) / (vmax - vmin).clamp(min=1e-12)

    bin_edges = torch.linspace(0.0, 1.0, _OTSU_NUM_BINS + 1, device=flat.device, dtype=flat.dtype)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_idx = torch.bucketize(normed, bin_edges[1:-1])
    hist = torch.zeros(N, _OTSU_NUM_BINS, device=flat.device, dtype=flat.dtype)
    hist.scatter_add_(1, bin_idx, torch.ones_like(normed))

    total = hist.sum(dim=-1, keepdim=True)
    cum_count = torch.cumsum(hist, dim=-1)
    cum_mean = torch.cumsum(hist * bin_centers.unsqueeze(0), dim=-1)
    global_mean = cum_mean[:, -1:]

    w0 = cum_count / total.clamp(min=1)
    w1 = 1.0 - w0
    mu0 = cum_mean / cum_count.clamp(min=1e-12)
    mu1 = (global_mean - cum_mean) / (total - cum_count).clamp(min=1e-12)

    between_var = w0 * w1 * (mu0 - mu1) ** 2
    # The extreme bins split nothing off, so exclude them as candidates.
    between_var[:, 0] = 0
    between_var[:, -1] = 0

    thresholds = bin_centers[torch.argmax(between_var, dim=-1)]

    signal_mask = (normed > thresholds.unsqueeze(1)).to(flat.dtype)
    bg_mask = 1.0 - signal_mask

    n_signal = signal_mask.sum(dim=-1).clamp(min=1)
    n_bg = bg_mask.sum(dim=-1).clamp(min=1)

    mu_signal = (normed * signal_mask).sum(dim=-1) / n_signal
    mu_bg = (normed * bg_mask).sum(dim=-1) / n_bg
    bg_sq = (normed**2 * bg_mask).sum(dim=-1) / n_bg
    sigma_bg = torch.sqrt((bg_sq - mu_bg**2).clamp(min=0)).clamp(min=1e-12)

    return torch.abs(mu_signal - mu_bg) / sigma_bg


def otsu_cnr_scores(recons: Tensor, window_size: int = OTSU_WINDOW_SIZE) -> Tensor:
    """Score every reconstruction in a sweep by Otsu CNR of its local variance.

    All reconstructions are scored on the *same* z slice — the one whose local
    variance is largest averaged over the sweep — so the scores stay comparable.
    Picking a slice per reconstruction would let the metric move for two reasons
    at once.

    Parameters
    ----------
    recons : Tensor
        ``(N, Z, Y, X)`` reconstructions, one per regularization strength.
    window_size : int
        Local-variance window.

    Returns
    -------
    Tensor
        ``(N,)`` scores; higher is better.
    """
    local_var = _local_variance(recons, window_size)
    best_z = torch.argmax(local_var.var(dim=(2, 3)).mean(dim=0))
    return _otsu_cnr(local_var[:, best_z].reshape(recons.shape[0], -1))


# --- l_curve ---


def compute_l_curve_norms(
    data_zyx: Tensor,
    transfer_function: Tensor,
    recons: Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Residual and solution norms for every reconstruction in a sweep.

    ``residual = ||real(ifft(fft(x) * H)) - y||`` and ``solution = ||x||``, the
    two axes of an L-curve.

    Parameters
    ----------
    data_zyx : Tensor
        ``(Z, Y, X)`` measurement, preprocessed exactly as the reconstruction was.
    transfer_function : Tensor
        ``(Z, Y, X)`` forward operator.
    recons : Tensor
        ``(N, Z, Y, X)`` reconstructions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(residual_norms, solution_norms)``, each ``(N,)``.
    """
    predicted = torch.fft.ifftn(
        torch.fft.fftn(recons, dim=(-3, -2, -1)) * transfer_function.unsqueeze(0),
        dim=(-3, -2, -1),
    ).real
    residual = predicted - data_zyx.unsqueeze(0)

    residual_norms = torch.linalg.vector_norm(residual, dim=(-3, -2, -1)).cpu().numpy()
    solution_norms = torch.linalg.vector_norm(recons, dim=(-3, -2, -1)).cpu().numpy()
    return residual_norms, solution_norms


def find_l_curve_corner(residual_norms: np.ndarray, solution_norms: np.ndarray) -> int:
    """Index of the L-curve corner, by curvature on range-normalized log axes.

    Both log axes are scaled to ``[0, 1]`` first so curvature is
    scale-independent and the two bends of an L become comparably sharp
    (Hansen 1992). The bends have opposite signed curvature, so peaks are
    sought in ``|kappa|``: with two prominent peaks the midpoint between them is
    returned, with one the centre of its plateau.

    Parameters
    ----------
    residual_norms, solution_norms : np.ndarray
        ``(N,)`` arrays from :func:`compute_l_curve_norms`.

    Returns
    -------
    int
        Index into the sweep.
    """
    if len(residual_norms) < 3:
        return 0

    x_raw = np.log(residual_norms)
    y_raw = np.log(solution_norms)

    def _normalize(a):
        span = a.max() - a.min()
        return (a - a.min()) / span if span > 1e-12 else a - a.min()

    x = uniform_filter1d(_normalize(x_raw), size=3)
    y = uniform_filter1d(_normalize(y_raw), size=3)

    t = np.arange(len(x))
    dx, dy = np.gradient(x, t), np.gradient(y, t)
    ddx, ddy = np.gradient(dx, t), np.gradient(dy, t)

    denominator = np.maximum((dx**2 + dy**2) ** 1.5, 1e-12)
    abs_curvature = np.abs((dx * ddy - dy * ddx) / denominator)

    max_abs_curv = np.max(abs_curvature)
    if max_abs_curv < 1e-12:
        return int(np.argmax(abs_curvature))

    peaks, properties = find_peaks(abs_curvature, prominence=0)
    if len(peaks) == 0:
        return int(np.argmax(abs_curvature))

    prominences = properties["prominences"]
    significant = prominences >= max_abs_curv * _L_CURVE_PROMINENCE_FRACTION

    if significant.sum() >= 2:
        # Two turning points: the corner of a real L sits between them.
        top2 = np.argsort(prominences[significant])[-2:]
        first, second = np.sort(peaks[significant][top2])
        return (int(first) + int(second)) // 2

    # One turning point: take the middle of the plateau around it, so a broad
    # flat maximum does not resolve to whichever sample happens to be highest.
    argmax_idx = int(np.argmax(abs_curvature))
    in_plateau = abs_curvature >= max_abs_curv * _L_CURVE_PLATEAU_FRACTION

    start = argmax_idx
    while start > 0 and in_plateau[start - 1]:
        start -= 1
    end = argmax_idx
    while end < len(abs_curvature) - 1 and in_plateau[end + 1]:
        end += 1

    return (start + end) // 2


# --- crop selection ---


def select_crop(zyx_data: Tensor, crop_size: int = CROP_SIZE) -> tuple[int, int, int]:
    """Locate the most structured square yx crop, on a grid of candidates.

    Variance stands in for structure: empty regions score near zero, so this
    lands on sample rather than background. Returns the full frame when it is
    already at or below ``crop_size``.

    Parameters
    ----------
    zyx_data : Tensor
        ``(Z, Y, X)`` volume.
    crop_size : int
        Side of the square crop.

    Returns
    -------
    tuple[int, int, int]
        ``(y0, x0, size)``.
    """
    _, Y, X = zyx_data.shape
    size = min(crop_size, Y, X)
    if size >= Y and size >= X:
        return 0, 0, size

    # Half-overlapping candidates, plus the last valid start in each axis so the
    # far edge of the frame is reachable when the stride does not divide it.
    step = max(size // 2, 1)

    def _starts(extent):
        last = extent - size
        return sorted({*range(0, last + 1, step), last})

    y_starts = torch.tensor(_starts(Y))
    x_starts = torch.tensor(_starts(X))

    # Every candidate's variance from one pair of integral images, rather than
    # re-reducing overlapping windows: a full frame has thousands of candidates.
    # float64 because the running sums of a 16-bit camera's counts over millions
    # of pixels leave float32 far behind.
    plane = zyx_data.double()
    sums = torch.cumsum(torch.cumsum(plane.sum(dim=0), dim=0), dim=1)
    squares = torch.cumsum(torch.cumsum((plane**2).sum(dim=0), dim=0), dim=1)
    sums = F.pad(sums, (1, 0, 1, 0))
    squares = F.pad(squares, (1, 0, 1, 0))

    def _window_totals(integral):
        top, bottom = y_starts.unsqueeze(1), (y_starts + size).unsqueeze(1)
        left, right = x_starts.unsqueeze(0), (x_starts + size).unsqueeze(0)
        return integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]

    count = zyx_data.shape[0] * size * size
    mean = _window_totals(sums) / count
    variance = _window_totals(squares) / count - mean**2

    flat = int(torch.argmax(variance))
    return int(y_starts[flat // len(x_starts)]), int(x_starts[flat % len(x_starts)]), size


def _inverse_filter(transfer_function: Tensor, strength: float, apodization_rolloff: float) -> Tensor:
    """Build the inverse filter the 3D models build, apodization included.

    Kept in step with ``phase_thick_3d`` and ``isotropic_fluorescent_thick_3d``:
    the sweep has to score the reconstruction that will actually be produced, and
    ``apodization_rolloff`` reshapes the filter after regularization.
    """
    inverse_filter = tikhonov_regularized_inverse_filter(transfer_function, strength)
    if apodization_rolloff > 0:
        window = sampling.raised_cosine_window(
            inverse_filter.shape[-2], apodization_rolloff, device=inverse_filter.device
        )[:, None] * sampling.raised_cosine_window(
            inverse_filter.shape[-1], apodization_rolloff, device=inverse_filter.device
        )
        inverse_filter = inverse_filter * window
    return inverse_filter


# --- the sweep ---


def preprocess_measurement(zyx_data: Tensor, contrast: str, z_padding: int) -> Tensor:
    """Apply the same preprocessing the model applies before its inverse filter.

    The L-curve residual is only meaningful against the data the reconstruction
    actually saw, so this has to track the models. Phase pads then intensity
    normalizes (:func:`waveorder.models.phase_thick_3d.apply_inverse_transfer_function`);
    fluorescence only pads
    (:func:`waveorder.models.isotropic_fluorescent_thick_3d.apply_inverse_transfer_function`).

    Parameters
    ----------
    zyx_data : Tensor
        ``(Z, Y, X)`` raw measurement.
    contrast : {"phase", "fluorescence"}
        Which model will reconstruct this.
    z_padding : int
        Slices padded onto each end of z.

    Returns
    -------
    Tensor
        ``(Z + 2 * z_padding, Y, X)`` preprocessed measurement.
    """
    padded = util.pad_zyx_along_z(zyx_data, z_padding)
    if contrast == "phase":
        return util.inten_normalization_3D(padded)
    if contrast == "fluorescence":
        return padded
    raise ValueError(f"contrast must be 'phase' or 'fluorescence', got {contrast!r}")


def select_regularization(
    zyx_data: Tensor,
    transfer_function: Tensor,
    settings: AutoRegularizationSettings,
    *,
    contrast: Literal["phase", "fluorescence"],
    z_padding: int = 0,
    crop: tuple[int, int, int] = (0, 0, 0),
    apodization_rolloff: float = 0.0,
) -> AutoRegResult:
    """Sweep regularization strengths and pick one.

    ``zyx_data`` and ``transfer_function`` must already agree in shape: pass the
    cropped volume together with a transfer function computed at that crop's
    shape.

    Parameters
    ----------
    zyx_data : Tensor
        ``(Z, Y, X)`` raw measurement to score against, already cropped.
    transfer_function : Tensor
        ``(Z + 2 * z_padding, Y, X)`` forward operator for that crop. For phase
        this is the real potential transfer function; for fluorescence, the OTF.
    settings : AutoRegularizationSettings
        Rule and sweep configuration.
    contrast : {"phase", "fluorescence"}
        Selects the preprocessing, which differs between the two models.
    z_padding : int
        Slices padded onto each end of z, from the transfer function settings.
    crop : tuple[int, int, int]
        ``(y0, x0, size)`` of ``zyx_data`` within the full frame, recorded in the
        result for provenance.
    apodization_rolloff : float
        Passed through to the inverse filter, so the sweep scores the same
        reconstruction the model will produce. By default 0.0 (no apodization).

    Returns
    -------
    AutoRegResult
        The pick and the sweep behind it.
    """
    messages: list[str] = []

    measurement = preprocess_measurement(zyx_data, contrast, z_padding)
    if measurement.shape != transfer_function.shape:
        raise ValueError(
            f"transfer function shape {tuple(transfer_function.shape)} does not match the "
            f"preprocessed data shape {tuple(measurement.shape)}; compute the transfer "
            f"function at the cropped shape"
        )

    # The crop's own peak anchors the sweep: |H|^2max is set by the optics and the
    # pixel size, not by the frame size, so it is the full frame's peak too.
    h2max = float((transfer_function.abs() ** 2).max())
    if h2max <= 0:
        raise ValueError("transfer function is identically zero; check the optical settings")

    powers = np.linspace(settings.search_min, settings.search_max, settings.num_samples)
    anchor = h2max if settings.search_scale == "relative" else 1.0
    strengths = (10.0**powers) * anchor

    measurement_fft = torch.fft.fftn(measurement, dim=(-3, -2, -1))
    recons = torch.stack(
        [
            torch.real(
                torch.fft.ifftn(
                    measurement_fft * _inverse_filter(transfer_function, float(strength), apodization_rolloff),
                    dim=(-3, -2, -1),
                )
            )
            for strength in strengths
        ]
    )

    scores = np.array([])
    residual_norms = np.array([])
    solution_norms = np.array([])

    if settings.rule == "otsu_cnr":
        unpadded = recons[:, z_padding : recons.shape[1] - z_padding] if z_padding else recons
        scores = otsu_cnr_scores(unpadded, OTSU_WINDOW_SIZE).cpu().numpy()
        index = int(np.argmax(scores))
    elif settings.rule == "l_curve":
        residual_norms, solution_norms = compute_l_curve_norms(measurement, transfer_function, recons)
        index = find_l_curve_corner(residual_norms, solution_norms)
    else:
        raise ValueError(f"unknown rule {settings.rule!r}")

    if index in (0, len(powers) - 1):
        edge = "lower" if index == 0 else "upper"
        messages.append(
            f"the {settings.rule} pick landed on the {edge} end of the sweep "
            f"(10^{powers[index]:g}), so the true optimum may lie outside the search bounds"
        )

    strength = float(strengths[index])
    return AutoRegResult(
        regularization_strength=strength,
        index=index,
        rule=settings.rule,
        regularization_strengths=strengths,
        transfer_function_peak_squared=h2max,
        lambda_over_h2max=strength / h2max,
        crop=crop,
        scores=scores,
        residual_norms=residual_norms,
        solution_norms=solution_norms,
        warnings=messages,
    )


def warn_all(result: AutoRegResult) -> None:
    """Re-emit a result's collected messages as :class:`UserWarning`."""
    for message in result.warnings:
        warnings.warn(message, UserWarning, stacklevel=2)
