"""The 4.0 ``simulate`` verb: write ``data.zarr``, ``object.zarr``, ``tf.zarr``.

Minimal implementation of design document section 5: fluorescence
contrast, shift-invariant imaging system, ``gaussian_spectrum`` object
source, and Gaussian (``sigma`` or ``snr_db``) or Poisson (``photons``)
noise. A ``simulate`` configuration describes the *true* system: any
imaging-system number written with a numeric ``tolerance`` draws the truth
from that prior (clamped to its bounds) under the configuration's seed.

Outputs land in the standard formats: ``data.zarr`` and ``object.zarr``
are single-position HCS OME-Zarr stores (consumable by every verb), and
``tf.zarr`` holds the drawn free parameters, a zero covariance, the true
transfer function, and ``provenance: {kind: simulated, ...}``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from waveorder.calibrate.synthesis import fluorescence_otf
from waveorder.calibrate.tfzarr import write_tf_zarr
from waveorder.calibrate.zernike_names import name_to_noll, noll_indices_up_to_order, noll_to_name
from waveorder.config.schema import (
    FreeParameter,
    SourceGaussianSpectrum,
    V4Config,
)


def _draw_truth_coefficients(cfg: V4Config, generator: torch.Generator) -> dict[str, float]:
    """True Zernike coefficients: fixed values, and draws from numeric tolerances."""
    aberrations = cfg.imaging_system.detection.pupil_aberrations
    truth: dict[str, float] = {}
    if aberrations is None:
        return truth

    def draw(nominal: float, tolerance, bounds) -> float:
        if tolerance == "free" or tolerance is None:
            return float(nominal)
        value = nominal + float(tolerance) * float(torch.randn((), generator=generator))
        if bounds is not None:
            value = min(max(value, bounds[0]), bounds[1])
        return float(value)

    shorthand = aberrations.zernike_up_to_order
    if shorthand is not None:
        for j in noll_indices_up_to_order(shorthand.order):
            truth[noll_to_name(j)] = draw(0.0, shorthand.tolerance, shorthand.bounds)
    for name, value in aberrations.terms.items():
        canonical = noll_to_name(name_to_noll(name))
        if isinstance(value, FreeParameter):
            truth[canonical] = draw(value.nominal, value.tolerance, value.bounds)
        else:
            truth[canonical] = float(value)
    return truth


def _gaussian_spectrum_object(
    zyx_shape: tuple[int, int, int],
    exponent: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Zero-mean, unit-std Gaussian random field with power spectrum ``(|k|^2 + k0^2)^(-exponent/2)``."""
    freqs = [torch.fft.fftfreq(n) for n in zyx_shape]
    kz, ky, kx = torch.meshgrid(*freqs, indexing="ij")
    k2 = kz**2 + ky**2 + kx**2
    k0 = 1.0 / max(zyx_shape)
    amplitude = (k2 + k0**2) ** (-exponent / 4.0)
    white = torch.randn(zyx_shape, generator=generator)
    field = torch.real(torch.fft.ifftn(torch.fft.fftn(white) * amplitude))
    return (field - field.mean()) / field.std().clamp(min=1e-12)


def _write_position(path: Path, zyx: np.ndarray, channel_name: str, scale_zyx: tuple[float, float, float]) -> None:
    """Write one TCZYX = (1, 1, Z, Y, X) volume as a single-position HCS store."""
    from iohub.ngff import open_ome_zarr
    from iohub.ngff.models import TransformationMeta

    dataset = open_ome_zarr(path, layout="hcs", mode="w", channel_names=[channel_name])
    position = dataset.create_position("0", "0", "0")
    position.create_image(
        "0",
        zyx[None, None].astype(np.float32),
        transform=[TransformationMeta(type="scale", scale=[1.0, 1.0, *scale_zyx])],
    )
    dataset.close()


def run_simulation(cfg: V4Config, output_dirpath: Path) -> dict[str, float]:
    """Simulate data from the true imaging system of a 4.0 configuration.

    Parameters
    ----------
    cfg : V4Config
        Configuration with a ``simulate`` block (seed and shape), a
        fluorescence ``imaging_system``, an ``object.source``, and a noise
        model.
    output_dirpath : Path
        Directory receiving ``data.zarr``, ``object.zarr``, ``tf.zarr``.

    Returns
    -------
    dict[str, float]
        The drawn true Zernike coefficients (waves).
    """
    if cfg.simulate is None:
        raise ValueError("simulate requires a 'simulate: {seed, shape}' block (design section 5)")
    if cfg.imaging_system.contrast != "fluorescence":
        raise NotImplementedError("simulate supports fluorescence contrast for now")
    if cfg.imaging_system.field_dependence != "shift_invariant":
        raise NotImplementedError("simulate is shift-invariant for now")
    if not isinstance(cfg.object.source, SourceGaussianSpectrum):
        raise NotImplementedError("simulate supports object.source: {gaussian_spectrum: {exponent}} for now")

    detection = cfg.imaging_system.detection
    if detection.wavelength is None:
        raise ValueError("imaging_system.detection.wavelength is required")
    yx = cfg.data.pixel_size.yx
    yx_pixel_size = float(yx) if isinstance(yx, (int, float)) else float(yx.y)
    z_pixel_size = float(cfg.data.pixel_size.z)

    zyx_shape = tuple(cfg.simulate.shape)
    generator = torch.Generator().manual_seed(cfg.simulate.seed)

    truth = _draw_truth_coefficients(cfg, generator)
    numerical_aperture = detection.numerical_aperture
    if isinstance(numerical_aperture, FreeParameter):
        raise NotImplementedError("drawing a true numerical aperture is not supported yet; use a plain number")

    otf = fluorescence_otf(
        zyx_shape=zyx_shape,
        yx_pixel_size=yx_pixel_size,
        z_pixel_size=z_pixel_size,
        wavelength_emission=detection.wavelength,
        index_of_refraction_media=cfg.imaging_system.index_of_refraction_media,
        numerical_aperture_detection=numerical_aperture,
        zernike_coefficients=truth,
    )

    exponent = cfg.object.source.gaussian_spectrum.exponent
    zyx_object = _gaussian_spectrum_object(zyx_shape, exponent, generator)
    clean = torch.real(torch.fft.ifftn(torch.fft.fftn(zyx_object) * otf))

    if cfg.noise.kind == "gaussian":
        noise = cfg.noise.gaussian
        if noise.snr_db is not None:
            sigma = float(clean.std()) / (10.0 ** (noise.snr_db / 20.0))
        elif isinstance(noise.sigma, (int, float)):
            sigma = float(noise.sigma)
        else:
            raise ValueError("simulation needs noise.gaussian.sigma as a number, or snr_db")
        data = clean + sigma * torch.randn(zyx_shape, generator=generator)
        noise_description = {"kind": "gaussian", "sigma": sigma}
    elif cfg.noise.kind == "poisson":
        noise = cfg.noise.poisson
        if noise.photons is None:
            raise ValueError("simulation needs noise.poisson.photons")
        nonneg = clean - clean.min()
        scale = noise.photons / float(nonneg.mean().clamp(min=1e-12))
        data = torch.poisson(nonneg * scale, generator=generator)
        noise_description = {"kind": "poisson", "photons": float(noise.photons), "scale": scale}
    else:
        raise NotImplementedError(f"simulate does not support noise: {cfg.noise.kind} yet")

    output_dirpath = Path(output_dirpath)
    output_dirpath.mkdir(parents=True, exist_ok=True)
    channel_name = cfg.data.input_channel_names[0]
    scale_zyx = (z_pixel_size, yx_pixel_size, yx_pixel_size)
    _write_position(output_dirpath / "data.zarr", data.numpy(), channel_name, scale_zyx)
    _write_position(output_dirpath / "object.zarr", zyx_object.numpy(), "object", scale_zyx)

    # tf.zarr: P = the parameters with a tolerance; fixed terms live in provenance.
    aberrations = cfg.imaging_system.detection.pupil_aberrations
    free_names: list[str] = []
    if aberrations is not None:
        if aberrations.zernike_up_to_order is not None:
            free_names.extend(noll_to_name(j) for j in noll_indices_up_to_order(aberrations.zernike_up_to_order.order))
        for name, value in aberrations.terms.items():
            canonical = noll_to_name(name_to_noll(name))
            if isinstance(value, FreeParameter) and canonical not in free_names:
                free_names.append(canonical)
    free_names.sort(key=name_to_noll)
    parameters = np.array([[truth[name] for name in free_names]], dtype=np.float64)

    write_tf_zarr(
        output_dirpath / "tf.zarr",
        parameters=parameters,
        parameter_names=free_names,
        tile_grid={
            "size_yx": [int(zyx_shape[1]), int(zyx_shape[2])],
            "overlap_yx": [0, 0],
            "centers_yx": [[0.0, 0.0]],
        },
        provenance={
            "kind": "simulated",
            "seed": cfg.simulate.seed,
            "true_coefficients_waves": truth,
            "noise": noise_description,
        },
        transfer_function=otf.numpy().astype(np.complex64),
    )
    return truth
