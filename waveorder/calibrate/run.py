"""Map a 4.0 configuration onto the existing calibration numerics.

This is the engine behind ``wo calibrate-tf`` at the "fold the existing
blind paths into the configuration" stage (design document section 9,
step 2):

* ``pupil_aberrations`` entries with a tolerance, ``score: sharpness`` and a
  gradient search run the existing per-tile Zernike recovery
  (:func:`waveorder.api.shift_variant_recovery.recover_zernikes`).
  ``field_dependence: shift_invariant`` is the same recovery with a single
  tile spanning the field of view; ``tiled`` uses the configured tile grid.
* Scores other than ``sharpness``, numeric tolerances, and grid/random
  search stages belong to later steps and raise ``NotImplementedError``
  with a pointer to what is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from waveorder.api.shift_variant_recovery import RecoverySettings, recover_zernikes
from waveorder.calibrate.zernike_names import (
    name_to_noll,
    noll_indices_up_to_order,
    noll_to_name,
)
from waveorder.config.schema import (
    FreeParameter,
    SearchGradient,
    V4Config,
    free_parameters,
    nominal_of,
)

#: Map from the 3.x sharpness loss names to the recovery's loss names.
_LOSS_NAME_MAP = {
    "midband_power": "midband",
    "total_variation": "tv",
    "laplacian_variance": "laplacian_var",
    "normalized_variance": "normalized_var",
    "spectral_flatness": "spectral_flatness",
}


@dataclass
class CalibrationResult:
    """Output of :func:`run_zernike_calibration`, ready for ``tf.zarr``."""

    parameters: np.ndarray  # [N_tiles, P] Zernike coefficients in waves
    parameter_names: list[str]
    tile_grid: dict[str, Any]
    reconstruction_zyx: np.ndarray  # stitched Wiener deconvolution
    loss_history: list[float]
    noll_indices: list[int]
    recovery_settings: dict


def zernike_calibration_requested(cfg: V4Config) -> bool:
    """Whether the configuration frees any pupil-aberration term."""
    aberrations = cfg.imaging_system.detection.pupil_aberrations
    if aberrations is None:
        return False
    if aberrations.zernike_up_to_order is not None:
        return True
    return any(isinstance(value, FreeParameter) for value in aberrations.terms.values())


def _not_yet(reason: str) -> NotImplementedError:
    return NotImplementedError(f"calibrate-tf does not support this yet: {reason}")


def _resolve_noll_indices(cfg: V4Config) -> list[int]:
    """Noll indices to estimate, from the shorthand and the named terms."""
    aberrations = cfg.imaging_system.detection.pupil_aberrations
    indices: set[int] = set()
    if aberrations.zernike_up_to_order is not None:
        shorthand = aberrations.zernike_up_to_order
        if shorthand.tolerance != "free":
            raise _not_yet(
                "numeric tolerances on pupil aberrations (the sharpness score has no "
                "calibration penalty; use tolerance: free until the evidence score lands)"
            )
        indices.update(noll_indices_up_to_order(shorthand.order))
    for name, value in aberrations.terms.items():
        if isinstance(value, FreeParameter):
            if value.tolerance != "free":
                raise _not_yet(f"numeric tolerance on pupil aberration {name!r}")
            if value.nominal != 0.0:
                raise _not_yet(f"nonzero nominal on free pupil aberration {name!r} (the recovery starts at 0)")
            indices.add(name_to_noll(name))
        elif float(value) != 0.0:
            raise _not_yet(f"fixed nonzero pupil aberration {name!r}")
    return sorted(indices)


def build_recovery_settings(cfg: V4Config, yx_shape: tuple[int, int]) -> tuple[RecoverySettings, list[int]]:
    """Translate a 4.0 configuration into :class:`RecoverySettings`.

    Parameters
    ----------
    cfg : V4Config
        Configuration with at least one free pupil-aberration term.
    yx_shape : tuple[int, int]
        ``(Y, X)`` shape of the data, used for the shift-invariant
        single-tile grid.

    Returns
    -------
    settings : RecoverySettings
        Settings for :func:`waveorder.api.shift_variant_recovery.recover_zernikes`.
    noll_indices : list[int]
        The estimated Noll indices, in ``parameters`` column order.
    """
    system = cfg.imaging_system
    if system.contrast != "fluorescence":
        raise _not_yet("pupil-aberration calibration for phase contrast")
    if cfg.object.dimension != 3:
        raise _not_yet("pupil-aberration calibration for 2D (thin) objects")

    # Other free imaging-system numbers run through the 3.x optimizer via
    # `wo reconstruct`; mixing them with pupil aberrations in one calibration
    # is a later step.
    non_zernike = {
        name: p for name, p in free_parameters(system).items() if not name.startswith("detection.pupil_aberrations")
    }
    if non_zernike:
        raise _not_yet(f"estimating {sorted(non_zernike)} jointly with pupil aberrations")

    calibration = cfg.solver.calibration
    score = calibration.score if calibration is not None else "sharpness"
    if score != "sharpness":
        raise _not_yet(f"score: {score} (evidence/gcv/sure land with design step 4; use sharpness)")

    max_iter = 250
    if calibration is not None:
        if len(calibration.search) != 1 or not isinstance(calibration.search[0], SearchGradient):
            raise _not_yet("search stages other than a single gradient stage")
        max_iter = calibration.search[0].gradient.max_iter
        if calibration.final_estimate != "best_H":
            raise _not_yet("final_estimate: average_near_best_H")

    loss = "midband"
    midband_fractions = (0.125, 0.25)
    optimizer = "adam"
    advanced = calibration.advanced if calibration is not None else None
    if advanced is not None:
        if advanced.loss is not None:
            loss_type = advanced.loss.get("type", "midband_power")
            loss = _LOSS_NAME_MAP.get(loss_type)
            if loss is None:
                raise _not_yet(f"sharpness loss {loss_type!r}")
            midband_fractions = tuple(advanced.loss.get("midband_fractions", midband_fractions))
        if advanced.optimizer is not None:
            if advanced.optimizer in ("nelder_mead", "grid_search"):
                raise _not_yet(f"optimizer {advanced.optimizer!r} for pupil aberrations")
            optimizer = advanced.optimizer

    if cfg.object.prior != "tikhonov" or cfg.reconstruction_kind != "closed_form":
        raise _not_yet("pupil-aberration calibration scores a Tikhonov closed-form reconstruction")
    weight = cfg.object.weight
    if not isinstance(weight, (int, float)):
        raise _not_yet(f"object.weight: {weight} during calibration (use a number)")

    if system.field_dependence == "shift_invariant":
        tile_size_yx = {"y": int(yx_shape[0]), "x": int(yx_shape[1])}
        tile_overlap_yx = {"y": 0.0, "x": 0.0}
    else:
        tiled = system.field_dependence.tiled
        tile_size_yx = {"y": int(tiled.size_yx[0]), "x": int(tiled.size_yx[1])}
        tile_overlap_yx = {"y": float(tiled.overlap_yx[0]), "x": float(tiled.overlap_yx[1])}

    noll_indices = _resolve_noll_indices(cfg)
    if not noll_indices:
        raise ValueError("nothing to estimate: no pupil-aberration term has a tolerance")

    settings = RecoverySettings(
        noll_indices=tuple(noll_indices),
        tile_size_yx=tile_size_yx,
        tile_overlap_yx=tile_overlap_yx,
        loss=loss,
        midband_fractions=midband_fractions,
        optimizer=optimizer,
        n_iter=max_iter,
        wiener_regularization=float(weight),
    )
    return settings, noll_indices


def run_zernike_calibration(cfg: V4Config, zyx_data: np.ndarray | torch.Tensor) -> CalibrationResult:
    """Estimate pupil-aberration parameters from one volume.

    Parameters
    ----------
    cfg : V4Config
        Configuration with free pupil-aberration terms
        (:func:`zernike_calibration_requested` is True).
    zyx_data : ndarray or Tensor
        3D fluorescence stack ``(Z, Y, X)``.

    Returns
    -------
    CalibrationResult
        Per-tile Zernike coefficients (waves), tile grid, and the stitched
        Wiener reconstruction the sharpness score was evaluated on.
    """
    zyx_shape = tuple(zyx_data.shape)
    settings, noll_indices = build_recovery_settings(cfg, zyx_shape[1:])

    system = cfg.imaging_system
    detection = system.detection
    if detection.wavelength is None:
        raise ValueError("imaging_system.detection.wavelength is required for fluorescence calibration")

    yx = cfg.data.pixel_size.yx
    yx_pixel_size = float(yx) if isinstance(yx, (int, float)) else float(yx.y)

    result = recover_zernikes(
        zyx_data,
        yx_pixel_size_um=yx_pixel_size,
        z_pixel_size_um=cfg.data.pixel_size.z,
        wavelength_emission_um=detection.wavelength,
        numerical_aperture_detection=nominal_of(detection.numerical_aperture),
        index_of_refraction_media=system.index_of_refraction_media,
        settings=settings,
    )

    tile_grid = {
        "size_yx": [settings.tile_size_yx["y"], settings.tile_size_yx["x"]],
        "overlap_yx": [settings.tile_overlap_yx["y"], settings.tile_overlap_yx["x"]],
        "centers_yx": [[float(y), float(x)] for (y, x) in result.tile_centers_um],
    }
    return CalibrationResult(
        parameters=result.coefs.numpy().astype(np.float64),
        parameter_names=[noll_to_name(j) for j in noll_indices],
        tile_grid=tile_grid,
        reconstruction_zyx=result.reconstruction_zyx,
        loss_history=result.loss_history,
        noll_indices=noll_indices,
        recovery_settings=result.settings,
    )
