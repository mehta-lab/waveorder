"""Evidence- and GCV-scored calibration of a shift-invariant fluorescence system.

Implements the MAPk estimator of the design document section 6 for the
Gaussian case: the score is the exact Whittle negative log evidence
(:func:`waveorder.calibrate.scores.whittle_evidence`) or the closed-form
GCV of the Wiener filter, plus the calibration penalty from the parameter
tolerances. The search runs the configured stages in order (``grid``,
``random``, ``gradient``), each starting from the previous stage's best
point, and the parameter covariance is the Laplace approximation (inverse
Hessian of the total score at the optimum).

Parameters are the free ``pupil_aberrations`` terms (Zernike coefficients
in waves). ``noise.gaussian.sigma: estimate`` and ``object.weight: auto``
add ``log sigma`` / ``log lambda`` as auxiliary variables of the gradient
stage; they are reported in the provenance, not in ``parameters``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from waveorder.calibrate.scores import whittle_evidence
from waveorder.calibrate.synthesis import fluorescence_otf
from waveorder.calibrate.tfzarr import TransferFunctionStore
from waveorder.calibrate.zernike_names import name_to_noll, noll_indices_up_to_order, noll_to_name
from waveorder.config.schema import (
    FreeParameter,
    GaussianNoise,
    SearchGradient,
    SearchGrid,
    SearchRandom,
    V4Config,
    free_parameters,
)


@dataclass
class ParameterSpec:
    """One free scalar parameter of the imaging system."""

    name: str  # canonical Zernike name, e.g. "astigmatism_0"
    nominal: float
    tolerance: float | str  # Gaussian prior std, or "free"
    bounds: tuple[float, float] | None


@dataclass
class EvidenceCalibrationResult:
    """Output of :func:`run_evidence_calibration`."""

    parameters: np.ndarray  # [1, P]
    parameter_names: list[str]
    parameters_covariance: np.ndarray  # [1, P, P] Laplace
    score_trace: list[float]  # per gradient iteration
    score_grid: np.ndarray | None  # [n_points, P + 1]: coordinates + score
    noise_sigma: float
    weight: float
    best_score: float
    extras: dict = field(default_factory=dict)


def gcv_score(
    data_fft: torch.Tensor,
    otf: torch.Tensor,
    regularization_strength: float | torch.Tensor,
) -> torch.Tensor:
    """Closed-form GCV of the Wiener filter ``A = |H|^2 / (|H|^2 + lambda)``.

    ``GCV = ||g - A g||^2 / (1 - tr(A) / N)^2`` with both terms diagonal per
    spatial frequency, so no trace estimation is needed. Independent of the
    noise level (Wahba).

    Parameters
    ----------
    data_fft : Tensor
        ``torch.fft.fftn(g, norm="ortho")``.
    otf : Tensor
        Transfer function, same shape, FFT order.
    regularization_strength : float or Tensor
        Tikhonov weight ``lambda``.

    Returns
    -------
    Tensor
        Scalar GCV score (lower is better).
    """
    h2 = torch.abs(otf) ** 2
    lam = regularization_strength
    shrink = lam / (h2 + lam)  # 1 - A, per frequency
    n = data_fft.numel()
    residual_power = (torch.abs(data_fft) ** 2 * shrink**2).sum()
    denominator = (shrink.sum() / n) ** 2
    return residual_power / denominator


def build_parameter_specs(cfg: V4Config) -> list[ParameterSpec]:
    """Free Zernike parameters of the configuration, in Noll order.

    Named entries override the ``zernike_up_to_order`` shorthand for their
    term (design document section 4.4).
    """
    aberrations = cfg.imaging_system.detection.pupil_aberrations
    if aberrations is None:
        raise ValueError("no pupil_aberrations block; nothing to estimate on the evidence path")

    by_noll: dict[int, ParameterSpec] = {}
    shorthand = aberrations.zernike_up_to_order
    if shorthand is not None:
        for j in noll_indices_up_to_order(shorthand.order):
            tolerance = shorthand.tolerance if shorthand.tolerance == "free" else float(shorthand.tolerance)
            by_noll[j] = ParameterSpec(noll_to_name(j), 0.0, tolerance, shorthand.bounds)
    for name, value in aberrations.terms.items():
        j = name_to_noll(name)
        if isinstance(value, FreeParameter):
            tolerance = value.tolerance if value.tolerance == "free" else float(value.tolerance)
            by_noll[j] = ParameterSpec(noll_to_name(j), value.nominal, tolerance, value.bounds)
        else:
            # Fixed named term: keep it out of the estimated set; nonzero
            # fixed terms are folded into the pupil as constants.
            by_noll.pop(j, None)

    non_zernike = {
        name for name in free_parameters(cfg.imaging_system) if not name.startswith("detection.pupil_aberrations")
    }
    if non_zernike:
        raise NotImplementedError(
            f"the evidence path estimates pupil aberrations only for now; also free: {sorted(non_zernike)}"
        )
    if not by_noll:
        raise ValueError("no pupil-aberration term has a tolerance")
    return [by_noll[j] for j in sorted(by_noll)]


def _fixed_coefficients(cfg: V4Config, estimated: list[ParameterSpec]) -> dict[int, float]:
    """Fixed nonzero Zernike terms, folded into every candidate pupil."""
    aberrations = cfg.imaging_system.detection.pupil_aberrations
    estimated_nolls = {name_to_noll(spec.name) for spec in estimated}
    fixed = {}
    for name, value in aberrations.terms.items():
        j = name_to_noll(name)
        if not isinstance(value, FreeParameter) and j not in estimated_nolls and float(value) != 0.0:
            fixed[j] = float(value)
    return fixed


def _stage_ranges(specs: list[ParameterSpec], centers: torch.Tensor) -> list[tuple[float, float]]:
    """Grid range per parameter: bounds, else center +/- 2 tolerance.

    ``centers`` are the (possibly ``-t``-moved) prior centers, so a grid
    stage after a ``-t`` prior searches around the prior parameters rather
    than the configuration's nominal values.
    """
    ranges = []
    for spec, center in zip(specs, centers.tolist()):
        if spec.bounds is not None:
            ranges.append(spec.bounds)
        elif spec.tolerance != "free":
            ranges.append((center - 2.0 * spec.tolerance, center + 2.0 * spec.tolerance))
        else:
            raise ValueError(f"grid/random stage needs bounds for 'free' parameter {spec.name!r}")
    return ranges


def run_evidence_calibration(
    cfg: V4Config,
    zyx_data: np.ndarray | torch.Tensor,
    gradient_lr: float = 0.01,
    prior_store: TransferFunctionStore | None = None,
    init_store: TransferFunctionStore | None = None,
) -> EvidenceCalibrationResult:
    """Estimate free pupil-aberration parameters by evidence or GCV.

    Parameters
    ----------
    cfg : V4Config
        Shift-invariant fluorescence configuration with free
        ``pupil_aberrations`` terms and ``solver.calibration.score`` in
        {``evidence``, ``gcv``}.
    zyx_data : ndarray or Tensor
        3D fluorescence stack ``(Z, Y, X)``.
    gradient_lr : float
        Adam step size for the gradient stage, in waves.
    prior_store : TransferFunctionStore, optional
        A ``tf.zarr`` passed with ``-t``: its ``parameters`` move the prior
        center (and the search start), and a non-zero
        ``parameters_covariance`` replaces the configuration's tolerances
        with a correlated Gaussian prior for the parameters it covers
        (design document section 3).
    init_store : TransferFunctionStore, optional
        A ``tf.zarr`` passed with ``--init``: sets the starting point only,
        without touching the prior. Takes precedence over ``prior_store``
        for initialization.

    Returns
    -------
    EvidenceCalibrationResult

    Notes
    -----
    The evidence score drops the DC (k = 0) term (``exclude_dc=True``):
    real data carries a constant background the Tikhonov prior does not
    model, so the reported score differs from the exact ``-log p(g | H)``
    by the DC term. Scores are comparable across candidate systems on the
    same data.

    ``parameters_covariance`` is the inverse Hessian of the total score at
    the optimum with ``sigma`` and ``lambda`` held fixed at their final
    values (a profile-curvature approximation); when ``sigma: estimate``
    or ``weight: auto`` is active the true marginal covariance is somewhat
    wider.
    """
    system = cfg.imaging_system
    if system.contrast != "fluorescence":
        raise NotImplementedError("the evidence path supports fluorescence contrast for now")
    if system.field_dependence != "shift_invariant":
        raise NotImplementedError("the evidence path is shift-invariant for now; tiled follows")
    calibration = cfg.solver.calibration
    score_name = calibration.score if calibration is not None else "evidence"
    if score_name not in ("evidence", "gcv"):
        raise ValueError(f"run_evidence_calibration handles score evidence|gcv, got {score_name}")

    specs = build_parameter_specs(cfg)
    names = [spec.name for spec in specs]
    nominals = torch.tensor([spec.nominal for spec in specs], dtype=torch.float64)

    # -t: move the prior center (and start) to the store's parameters; a
    # non-zero covariance replaces the per-parameter tolerances with a
    # correlated Gaussian prior for the parameters it covers. An all-zero
    # covariance block (e.g. from simulate) carries no uncertainty
    # information and leaves the configuration's tolerances in force.
    name_to_index = {name: i for i, name in enumerate(names)}
    covered: list[int] = []
    prior_precision: torch.Tensor | None = None
    if prior_store is not None:
        if prior_store.n_tiles != 1:
            raise NotImplementedError("a tiled tf.zarr prior (-t) is not supported on the evidence path yet")
        store_indices = []
        for store_index, store_name in enumerate(prior_store.parameter_names):
            index = name_to_index.get(store_name)
            if index is not None:
                nominals[index] = float(prior_store.parameters[0, store_index])
                store_indices.append(store_index)
                covered.append(index)
        if covered and prior_store.parameters_covariance is not None:
            block = prior_store.parameters_covariance[0][np.ix_(store_indices, store_indices)]
            if np.any(block):
                block_tensor = torch.as_tensor(block, dtype=torch.float64)
                try:
                    prior_precision = torch.linalg.inv(block_tensor)
                except torch.linalg.LinAlgError as error:
                    raise ValueError(
                        f"the -t parameters_covariance block over {len(store_indices)} covered "
                        f"parameters is singular and cannot serve as a Gaussian prior; "
                        f"re-estimate it, or drop the singular parameters from the store"
                    ) from error
    covariance_covered = covered if prior_precision is not None else []
    covered_set = set(covariance_covered)

    start = nominals.clone()
    if init_store is not None:
        if init_store.n_tiles != 1:
            raise NotImplementedError("a tiled --init store is not supported on the evidence path yet")
        for store_index, store_name in enumerate(init_store.parameter_names):
            index = name_to_index.get(store_name)
            if index is not None:
                start[index] = float(init_store.parameters[0, store_index])

    detection = system.detection
    if detection.wavelength is None:
        raise ValueError("imaging_system.detection.wavelength is required")
    yx = cfg.data.pixel_size.yx
    optics_kwargs = dict(
        zyx_shape=tuple(zyx_data.shape),
        yx_pixel_size=float(yx) if isinstance(yx, (int, float)) else float(yx.y),
        z_pixel_size=cfg.data.pixel_size.z,
        wavelength_emission=detection.wavelength,
        index_of_refraction_media=system.index_of_refraction_media,
        numerical_aperture_detection=(
            detection.numerical_aperture.nominal
            if isinstance(detection.numerical_aperture, FreeParameter)
            else detection.numerical_aperture
        ),
    )
    fixed_coefficients = _fixed_coefficients(cfg, specs)

    data = torch.as_tensor(np.asarray(zyx_data), dtype=torch.float32)
    data_fft = torch.fft.fftn(data, norm="ortho")

    # sigma and lambda: numbers, or jointly maximized ("estimate" / "auto").
    if cfg.noise.kind != "gaussian":
        raise NotImplementedError(f"the evidence path supports gaussian noise for now, got {cfg.noise.kind}")
    noise = cfg.noise.gaussian if cfg.noise.gaussian is not None else GaussianNoise()
    estimate_sigma = noise.sigma == "estimate"
    if estimate_sigma:
        sigma0 = float(data.std())  # refined by _profile_sigma_weight below
    elif isinstance(noise.sigma, (int, float)):
        sigma0 = float(noise.sigma)
    else:
        raise NotImplementedError("noise.gaussian.sigma from dark_frames is not implemented yet")

    auto_weight = cfg.object.weight == "auto"
    if auto_weight:
        weight0 = 1e-3
    elif isinstance(cfg.object.weight, (int, float)):
        weight0 = float(cfg.object.weight)
    else:
        raise ValueError(f"object.weight: {cfg.object.weight} is not usable on the evidence path")

    def total_score(values: torch.Tensor, log_sigma: torch.Tensor, log_weight: torch.Tensor) -> torch.Tensor:
        coefficients: dict[int, float | torch.Tensor] = dict(fixed_coefficients)
        for spec, value in zip(specs, values):
            coefficients[name_to_noll(spec.name)] = value
        otf = fluorescence_otf(zernike_coefficients=coefficients, **optics_kwargs)
        if score_name == "evidence":
            data_term = whittle_evidence(
                data_fft=data_fft,
                otf=otf,
                noise_sigma=torch.exp(log_sigma),
                regularization_strength=torch.exp(log_weight),
                exclude_dc=True,
            )
        else:
            data_term = gcv_score(data_fft, otf, torch.exp(log_weight))
        penalty = torch.zeros((), dtype=torch.float64)
        for i, spec in enumerate(specs):
            if i in covered_set:
                continue  # covered by the -t covariance's matrix term below
            if spec.tolerance != "free":
                penalty = penalty + (values[i] - nominals[i]) ** 2 / (2.0 * spec.tolerance**2)
        if prior_precision is not None:
            residual = values[covariance_covered] - nominals[covariance_covered]
            penalty = penalty + 0.5 * residual @ prior_precision @ residual
        return data_term + penalty

    def clamp(values: torch.Tensor) -> torch.Tensor:
        clamped = values.clone()
        for i, spec in enumerate(specs):
            if spec.bounds is not None:
                clamped[i] = clamped[i].clamp(spec.bounds[0], spec.bounds[1])
        return clamped

    current = start.clone()
    log_sigma = torch.tensor(math.log(sigma0), dtype=torch.float64)
    log_weight = torch.tensor(math.log(weight0), dtype=torch.float64)
    score_trace: list[float] = []
    score_grid: np.ndarray | None = None

    def profile_sigma_weight(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Coarse log-grid profile of (sigma, lambda) at fixed coefficients.

        A joint gradient from a bad (sigma, lambda) start settles into poor
        local optima (sigma absorbs the model mismatch before the
        coefficients can move), so the estimated nuisance scales are
        re-profiled before each gradient stage.
        """
        data_scale = float(data.std())
        sigma_candidates = [math.exp(float(log_sigma))] if not estimate_sigma else []
        weight_candidates = [math.exp(float(log_weight))] if not auto_weight else []
        if estimate_sigma:
            sigma_candidates = list(data_scale * np.geomspace(1e-3, 1.0, 13))
        if auto_weight:
            weight_candidates = list(np.geomspace(1e-6, 1.0, 13))
        best = (math.inf, float(log_sigma), float(log_weight))
        with torch.no_grad():
            for sigma_value in sigma_candidates:
                for weight_value in weight_candidates:
                    s = float(
                        total_score(
                            values,
                            torch.tensor(math.log(sigma_value), dtype=torch.float64),
                            torch.tensor(math.log(weight_value), dtype=torch.float64),
                        )
                    )
                    if s < best[0]:
                        best = (s, math.log(sigma_value), math.log(weight_value))
        return (
            torch.tensor(best[1], dtype=torch.float64),
            torch.tensor(best[2], dtype=torch.float64),
        )

    if estimate_sigma or auto_weight:
        log_sigma, log_weight = profile_sigma_weight(current)

    stages = calibration.search if calibration is not None else [SearchGradient.model_validate({"gradient": {}})]
    for stage in stages:
        if isinstance(stage, SearchGrid):
            ranges = _stage_ranges(specs, nominals)
            points = stage.grid.points
            axes = []
            for spec, (lo, hi) in zip(specs, ranges):
                n = points if isinstance(points, int) else points.get(spec.name, 7)
                axes.append(np.linspace(lo, hi, n))
            rows = []
            best = (None, math.inf)
            with torch.no_grad():
                for combo in itertools.product(*axes):
                    values = torch.tensor(combo, dtype=torch.float64)
                    s = float(total_score(values, log_sigma, log_weight))
                    rows.append([*combo, s])
                    if s < best[1]:
                        best = (values, s)
            current = best[0]
            score_grid = np.asarray(rows)
        elif isinstance(stage, SearchRandom):
            generator = torch.Generator().manual_seed(stage.random.seed)
            ranges = _stage_ranges(specs, nominals)
            best = (current, float(total_score(current, log_sigma, log_weight)))
            with torch.no_grad():
                for _ in range(stage.random.samples):
                    draw = current.clone()
                    for i, spec in enumerate(specs):
                        if spec.tolerance == "free":
                            lo, hi = ranges[i]
                            draw[i] = lo + (hi - lo) * torch.rand((), generator=generator, dtype=torch.float64)
                        else:
                            draw[i] = spec.nominal + spec.tolerance * torch.randn(
                                (), generator=generator, dtype=torch.float64
                            )
                    draw = clamp(draw)
                    s = float(total_score(draw, log_sigma, log_weight))
                    if s < best[1]:
                        best = (draw, s)
            current = best[0]
        elif isinstance(stage, SearchGradient):
            if stage.gradient.method != "split":
                raise NotImplementedError(f"gradient method {stage.gradient.method!r} (use split)")
            if estimate_sigma or auto_weight:
                log_sigma, log_weight = profile_sigma_weight(current)
            values = current.clone().requires_grad_(True)
            optimized = [values]
            if estimate_sigma and score_name == "evidence":
                log_sigma = log_sigma.clone().requires_grad_(True)
                optimized.append(log_sigma)
            if auto_weight:
                log_weight = log_weight.clone().requires_grad_(True)
                optimized.append(log_weight)
            optimizer = torch.optim.Adam(optimized, lr=gradient_lr)
            for _ in range(stage.gradient.max_iter):
                optimizer.zero_grad()
                s = total_score(values, log_sigma, log_weight)
                s.backward()
                optimizer.step()
                with torch.no_grad():
                    values.data = clamp(values.data)
                score_trace.append(float(s.detach()))
            current = values.detach()
            log_sigma = log_sigma.detach()
            log_weight = log_weight.detach()
        else:  # pragma: no cover - schema normalizes stages
            raise ValueError(f"unknown search stage {stage!r}")

    best_score = float(total_score(current, log_sigma, log_weight))

    # Laplace covariance: inverse Hessian of the total score at the optimum,
    # with sigma and lambda held fixed (profile curvature; see Notes above).
    def score_of(values: torch.Tensor) -> torch.Tensor:
        return total_score(values, log_sigma, log_weight)

    try:
        hessian = torch.autograd.functional.hessian(score_of, current)
        covariance = torch.linalg.inv(hessian).numpy()
    except RuntimeError:
        covariance = np.full((len(specs), len(specs)), np.nan)

    return EvidenceCalibrationResult(
        parameters=current.numpy()[None, :],
        parameter_names=names,
        parameters_covariance=covariance[None, :, :],
        score_trace=score_trace,
        score_grid=score_grid,
        noise_sigma=float(torch.exp(log_sigma)),
        weight=float(torch.exp(log_weight)),
        best_score=best_score,
        extras={"score": score_name},
    )
