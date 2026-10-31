"""Bead calibration: ``object: {prior: {beads: {diameter}}}`` (design example 4.3).

The object model is a sparse set of spheres of known diameter at unknown
subpixel positions with unknown amplitudes over a constant background::

    g = H * (sum_i a_i s(r - r_i)) + b + n,

where ``s`` is the sphere of the configured diameter. With this prior the
object step is point localization and ``solver.reconstruction`` is ignored:
positions, amplitudes, background, and the free imaging-system parameters
are fit jointly by maximum a posteriori (Gaussian noise), with the
calibration penalty from the parameter tolerances. The reported
``parameters_covariance`` is the Laplace curvature with respect to the
imaging-system parameters at the joint optimum, with the bead parameters
held fixed (a lower bound on the marginal covariance).

Everything runs in Fourier space and is differentiable: the beads enter as
phase ramps ``a_i exp(-2 pi i k . r_i)`` on the sphere's analytic form
factor, so subpixel positions are exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

from waveorder.calibrate.evidence_search import ParameterSpec, build_parameter_specs
from waveorder.calibrate.synthesis import fluorescence_otf
from waveorder.calibrate.zernike_names import name_to_noll
from waveorder.config.schema import (
    FreeParameter,
    GaussianNoise,
    PriorBeads,
    SearchGradient,
    V4Config,
)


@dataclass
class BeadCalibrationResult:
    """Output of :func:`run_bead_calibration`."""

    parameters: np.ndarray  # [1, P] imaging-system parameters (waves)
    parameter_names: list[str]
    parameters_covariance: np.ndarray  # [1, P, P] Laplace (beads held fixed)
    positions_zyx: np.ndarray  # [N, 3] fitted bead positions in voxels
    amplitudes: np.ndarray  # [N]
    background: float
    noise_sigma: float
    score_trace: list[float]
    best_score: float


def sphere_form_factor(
    zyx_shape: tuple[int, int, int],
    yx_pixel_size: float,
    z_pixel_size: float,
    diameter: float,
) -> torch.Tensor:
    """Fourier transform of a solid sphere, normalized to 1 at DC.

    Parameters
    ----------
    zyx_shape : tuple[int, int, int]
        Grid shape ``(Z, Y, X)``.
    yx_pixel_size, z_pixel_size : float
        Voxel sizes in um.
    diameter : float
        Sphere diameter in um.

    Returns
    -------
    Tensor
        Real array, FFT frequency order: ``3 (sin u - u cos u) / u^3`` with
        ``u = pi * diameter * |k|``.
    """
    fz = torch.fft.fftfreq(zyx_shape[0], d=z_pixel_size)
    fy = torch.fft.fftfreq(zyx_shape[1], d=yx_pixel_size)
    fx = torch.fft.fftfreq(zyx_shape[2], d=yx_pixel_size)
    kz, ky, kx = torch.meshgrid(fz, fy, fx, indexing="ij")
    u = math.pi * diameter * torch.sqrt(kz**2 + ky**2 + kx**2)
    small = u < 1e-4
    u_safe = torch.where(small, torch.ones_like(u), u)
    form = 3.0 * (torch.sin(u_safe) - u_safe * torch.cos(u_safe)) / u_safe**3
    return torch.where(small, torch.ones_like(u), form)


def detect_beads(
    zyx_data: torch.Tensor,
    max_beads: int = 50,
    min_separation: int = 5,
    snr_threshold: float = 6.0,
) -> torch.Tensor:
    """Initial bead positions: local maxima well above the background.

    Parameters
    ----------
    zyx_data : Tensor
        3D stack ``(Z, Y, X)``.
    max_beads : int
        Keep at most this many peaks, brightest first.
    min_separation : int
        Non-maximum suppression window in voxels.
    snr_threshold : float
        Peaks must exceed ``median + snr_threshold * MAD``.

    Returns
    -------
    Tensor
        ``[N, 3]`` integer ``(z, y, x)`` peak positions.
    """
    median = zyx_data.median()
    mad = (zyx_data - median).abs().median().clamp(min=1e-12)
    threshold = median + snr_threshold * 1.4826 * mad

    kernel = 2 * (min_separation // 2) + 1
    pooled = functional.max_pool3d(zyx_data[None, None], kernel_size=kernel, stride=1, padding=kernel // 2)[0, 0]
    peaks = (zyx_data == pooled) & (zyx_data > threshold)
    positions = peaks.nonzero()
    if positions.shape[0] == 0:
        raise ValueError(
            f"no beads found above median + {snr_threshold} * MAD; check the channel, or lower the threshold"
        )
    intensities = zyx_data[positions[:, 0], positions[:, 1], positions[:, 2]]
    order = torch.argsort(intensities, descending=True)[:max_beads]
    return positions[order]


def run_bead_calibration(
    cfg: V4Config,
    zyx_data: np.ndarray | torch.Tensor,
    gradient_lr: float = 0.01,
    max_beads: int = 50,
) -> BeadCalibrationResult:
    """Fit free imaging-system parameters to a bead volume.

    Parameters
    ----------
    cfg : V4Config
        Fluorescence configuration with ``object: {prior: {beads: ...}}``
        and at least one free ``pupil_aberrations`` term.
    zyx_data : ndarray or Tensor
        3D bead stack ``(Z, Y, X)``.
    gradient_lr : float
        Adam step size for the imaging-system parameters (waves). Bead
        positions use ``10 x`` this (voxels move faster than waves).
    max_beads : int
        At most this many beads are fit.

    Returns
    -------
    BeadCalibrationResult
    """
    if not isinstance(cfg.object.prior, PriorBeads):
        raise ValueError("run_bead_calibration requires object.prior: {beads: {diameter: ...}}")
    if cfg.imaging_system.contrast != "fluorescence":
        raise NotImplementedError("bead calibration supports fluorescence contrast")
    if cfg.imaging_system.field_dependence != "shift_invariant":
        raise NotImplementedError("bead calibration is shift-invariant for now")
    if cfg.noise.kind != "gaussian":
        raise NotImplementedError(f"bead calibration supports gaussian noise for now, got {cfg.noise.kind}")

    specs: list[ParameterSpec] = build_parameter_specs(cfg)
    names = [spec.name for spec in specs]

    detection = cfg.imaging_system.detection
    if detection.wavelength is None:
        raise ValueError("imaging_system.detection.wavelength is required")
    yx = cfg.data.pixel_size.yx
    yx_pixel_size = float(yx) if isinstance(yx, (int, float)) else float(yx.y)
    z_pixel_size = float(cfg.data.pixel_size.z)

    data = torch.as_tensor(np.asarray(zyx_data), dtype=torch.float32)
    zyx_shape = tuple(data.shape)

    optics_kwargs = dict(
        zyx_shape=zyx_shape,
        yx_pixel_size=yx_pixel_size,
        z_pixel_size=z_pixel_size,
        wavelength_emission=detection.wavelength,
        index_of_refraction_media=cfg.imaging_system.index_of_refraction_media,
        numerical_aperture_detection=(
            detection.numerical_aperture.nominal
            if isinstance(detection.numerical_aperture, FreeParameter)
            else detection.numerical_aperture
        ),
    )
    form_factor = sphere_form_factor(zyx_shape, yx_pixel_size, z_pixel_size, cfg.object.prior.beads.diameter)

    # Initialization from the data.
    peak_positions = detect_beads(data, max_beads=max_beads)
    background0 = float(data.median())
    amplitudes0 = (data[peak_positions[:, 0], peak_positions[:, 1], peak_positions[:, 2]] - background0).clamp(min=1e-6)

    noise = cfg.noise.gaussian if cfg.noise.gaussian is not None else GaussianNoise()
    estimate_sigma = noise.sigma == "estimate"
    sigma0 = float((data - data.median()).abs().median() * 1.4826) if estimate_sigma else float(noise.sigma)
    sigma0 = max(sigma0, 1e-6)

    # Frequency grids for the bead phase ramps (cycles per voxel).
    freqs = [torch.fft.fftfreq(n).to(torch.float64) for n in zyx_shape]
    grids = torch.meshgrid(*freqs, indexing="ij")

    values = torch.tensor([spec.nominal for spec in specs], dtype=torch.float64, requires_grad=True)
    positions = peak_positions.to(torch.float64).clone().requires_grad_(True)
    log_amplitudes = torch.log(amplitudes0.to(torch.float64)).clone().requires_grad_(True)
    background = torch.tensor(background0, dtype=torch.float64, requires_grad=True)
    log_sigma = torch.tensor(math.log(sigma0), dtype=torch.float64, requires_grad=estimate_sigma)

    n_voxels = data.numel()

    def total_score() -> torch.Tensor:
        coefficients = {name_to_noll(spec.name): value for spec, value in zip(specs, values)}
        otf = fluorescence_otf(zernike_coefficients=coefficients, **optics_kwargs)
        phase = (
            grids[0] * positions[:, 0, None, None, None]
            + grids[1] * positions[:, 1, None, None, None]
            + grids[2] * positions[:, 2, None, None, None]
        )
        ramps = torch.exp(-2j * math.pi * phase)
        object_fft = (torch.exp(log_amplitudes)[:, None, None, None] * ramps).sum(dim=0) * form_factor
        predicted = torch.real(torch.fft.ifftn(object_fft * otf)) + background
        sigma2 = torch.exp(2.0 * log_sigma)
        misfit = ((predicted - data) ** 2).sum() / (2.0 * sigma2) + 0.5 * n_voxels * torch.log(2.0 * math.pi * sigma2)
        penalty = torch.zeros((), dtype=torch.float64)
        for i, spec in enumerate(specs):
            if spec.tolerance != "free":
                penalty = penalty + (values[i] - spec.nominal) ** 2 / (2.0 * spec.tolerance**2)
        return misfit + penalty

    stages = cfg.solver.calibration.search if cfg.solver.calibration is not None else []
    max_iter = 200
    for stage in stages:
        if isinstance(stage, SearchGradient):
            max_iter = stage.gradient.max_iter
        else:
            raise NotImplementedError("bead calibration runs a single gradient stage for now")

    optimized = [
        {"params": [values], "lr": gradient_lr},
        {"params": [positions], "lr": 10.0 * gradient_lr},
        {"params": [log_amplitudes, background], "lr": 10.0 * gradient_lr},
    ]
    if estimate_sigma:
        optimized.append({"params": [log_sigma], "lr": gradient_lr})
    optimizer = torch.optim.Adam(optimized)

    score_trace: list[float] = []
    for _ in range(max_iter):
        optimizer.zero_grad()
        score = total_score()
        score.backward()
        optimizer.step()
        with torch.no_grad():
            for i, spec in enumerate(specs):
                if spec.bounds is not None:
                    values.data[i] = values.data[i].clamp(spec.bounds[0], spec.bounds[1])
        score_trace.append(float(score.detach()))

    best_score = score_trace[-1]

    def score_of_values(candidate: torch.Tensor) -> torch.Tensor:
        nonlocal values
        saved = values
        values = candidate
        try:
            return total_score()
        finally:
            values = saved

    final_values = values.detach()
    try:
        hessian = torch.autograd.functional.hessian(score_of_values, final_values)
        covariance = torch.linalg.inv(hessian).numpy()
    except RuntimeError:
        covariance = np.full((len(specs), len(specs)), np.nan)

    return BeadCalibrationResult(
        parameters=final_values.numpy()[None, :],
        parameter_names=names,
        parameters_covariance=covariance[None, :, :],
        positions_zyx=positions.detach().numpy(),
        amplitudes=torch.exp(log_amplitudes.detach()).numpy(),
        background=float(background.detach()),
        noise_sigma=float(torch.exp(log_sigma.detach())),
        score_trace=score_trace,
        best_score=best_score,
    )
