"""Deterministic migration between 3.x and 4.0 configurations.

Three public entry points:

* :func:`detect_version` — a configuration without ``version`` is 3.x.
* :func:`migrate_3x_dict` — 3.x dict -> :class:`~waveorder.config.schema.V4Config`,
  deterministic and lossless for phase and fluorescence configurations.
* :func:`v4_to_legacy` — 4.0 -> the 3.x :class:`ReconstructionSettings`
  that selects the same code path with the same numerical parameters.

The round trip ``v4_to_legacy(migrate_3x_dict(cfg))`` reproduces the
directly-parsed 3.x settings field for field (enforced by unit tests and,
array by array, by the benchmark regression suite).

Birefringence configurations are not migratable: WaveOrder 4.0 defers the
birefringence refactor, so they continue to run as 3.x without a
deprecation warning.
"""

from __future__ import annotations

from typing import Any

import yaml

from waveorder.cli.settings import OptimizationSettings, ReconstructionSettings
from waveorder.config import schema
from waveorder.optim._types import OptimizableFloat

#: Default learning rate used when a 4.0 free parameter has no recorded
#: step size (matches ``OptimizableFloat.learning_rate``).
_DEFAULT_LEARNING_RATE = 0.01


class MigrationError(ValueError):
    """A 3.x configuration that cannot be expressed in the 4.0 schema."""


class NotRunnableInLegacyError(ValueError):
    """A 4.0 configuration that has no 3.x-equivalent execution path yet."""


def detect_version(raw: dict) -> int:
    """Schema version of a raw configuration dict; 3 when ``version`` is absent."""
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a YAML mapping, got {type(raw).__name__} (empty file?)")
    return int(raw.get("version", 3))


# --- 3.x -> 4.0 ------------------------------------------------------------


def _convert_calibratable(value: Any, dotted_name: str, learning_rates: dict[str, float]) -> Any:
    """Convert a 3.x number-or-OptimizableFloat into a 4.0 calibratable value.

    An optimizable float with a positive learning rate becomes a
    :class:`~waveorder.config.schema.FreeParameter` with a flat prior, and
    its learning rate is recorded under ``calibration.advanced.learning_rates``.
    A zero learning rate is a fixed value and becomes a plain float.
    """
    if isinstance(value, OptimizableFloat):
        if value.is_optimizable:
            learning_rates[dotted_name] = value.learning_rate
            return schema.FreeParameter(nominal=value.initial_value, tolerance="free")
        return value.initial_value
    return value


def migrate_3x_dict(raw: dict) -> schema.V4Config:
    """Migrate a 3.x configuration dict to a 4.0 configuration.

    Parameters
    ----------
    raw : dict
        A parsed 3.x YAML configuration.

    Returns
    -------
    V4Config
        The equivalent 4.0 configuration.

    Raises
    ------
    MigrationError
        For birefringence configurations, which stay 3.x in this release.
    """
    settings = ReconstructionSettings(**raw)
    return migrate_3x_settings(settings)


def migrate_3x_settings(settings: ReconstructionSettings) -> schema.V4Config:
    """Migrate parsed 3.x settings to a 4.0 configuration. See :func:`migrate_3x_dict`."""
    if settings.birefringence is not None:
        raise MigrationError(
            "birefringence configurations are not migratable to 4.0 yet; they continue to run in the 3.x format"
        )

    contrast = "phase" if settings.phase is not None else "fluorescence"
    block = settings.phase if settings.phase is not None else settings.fluorescence
    tf = block.transfer_function
    ai = block.apply_inverse

    learning_rates: dict[str, float] = {}

    detection: dict[str, Any] = {
        "numerical_aperture": _convert_calibratable(
            tf.numerical_aperture_detection, "detection.numerical_aperture", learning_rates
        ),
        "z_focus_offset": _convert_calibratable(tf.z_focus_offset, "detection.z_focus_offset", learning_rates),
    }
    illumination: dict[str, Any] = {
        "tilt": {
            "zenith": _convert_calibratable(tf.tilt_angle_zenith, "illumination.tilt.zenith", learning_rates),
            "azimuth": _convert_calibratable(tf.tilt_angle_azimuth, "illumination.tilt.azimuth", learning_rates),
        },
    }
    if contrast == "phase":
        illumination["wavelength"] = tf.wavelength_illumination
        illumination["numerical_aperture"] = _convert_calibratable(
            tf.numerical_aperture_illumination, "illumination.numerical_aperture", learning_rates
        )
        illumination["invert_phase_contrast"] = tf.invert_phase_contrast
    else:
        detection["wavelength"] = tf.wavelength_emission
        if tf.confocal_pinhole_diameter is not None:
            detection["pinhole"] = {"diameter_um": tf.confocal_pinhole_diameter}

    # The 3.x reconstruction_algorithm splits into (object.prior, noise,
    # solver.reconstruction).
    noise: dict[str, Any] = {"gaussian": {"sigma": 1.0}}
    algorithm = ai.reconstruction_algorithm
    if algorithm == "Tikhonov":
        prior: Any = "tikhonov"
        reconstruction: Any = "closed_form"
    elif algorithm == "TV":
        prior = "tv"
        reconstruction = {"admm": {"iterations": ai.TV_iterations, "rho": ai.TV_rho_strength}}
    elif algorithm in ("RL", "RLGC"):
        prior = "none"
        noise = {"poisson": {}}
        rl = ai.rl
        reconstruction = {
            "richardson_lucy": {
                "iterations": rl.iterations,
                "background": rl.background,
                "stopping_tolerance": rl.stopping_tolerance,
                "back_projector": rl.back_projector,
                "bp_alpha": rl.bp_alpha,
                "bp_beta": rl.bp_beta,
                "gradient_consensus": algorithm == "RLGC",
            }
        }
    else:  # pragma: no cover - 3.x schema has no other values
        raise MigrationError(f"unknown reconstruction_algorithm {algorithm!r}")

    weight: Any = ai.regularization_strength
    if ai.auto_regularization is not None:
        weight = {"auto": ai.auto_regularization.model_dump()}

    calibration = None
    if settings.optimization is not None or learning_rates:
        opt = settings.optimization
        advanced: dict[str, Any] = {"learning_rates": learning_rates}
        search = [{"gradient": {"max_iter": opt.max_iterations if opt else 200}}]
        if opt is not None:
            advanced.update(
                {
                    "optimizer": opt.method,
                    "convergence_tol": opt.convergence_tol,
                    "convergence_patience": opt.convergence_patience,
                    "use_gradients": opt.use_gradients,
                    "grid_points": opt.grid_points,
                    "loss": opt.loss.model_dump(),
                    "log_dir": opt.log_dir,
                }
            )
        calibration = {
            "score": "sharpness",
            "search": search,
            "advanced": advanced,
        }

    return schema.V4Config(
        version=4,
        data={
            "input_channel_names": settings.input_channel_names,
            "time_indices": settings.time_indices,
            "pixel_size": {"yx": tf.yx_pixel_size, "z": tf.z_pixel_size},
        },
        imaging_system={
            "contrast": contrast,
            "index_of_refraction_media": tf.index_of_refraction_media,
            "detection": detection,
            "illumination": illumination,
        },
        object={
            "dimension": settings.reconstruction_dimension,
            "prior": prior,
            "weight": weight,
        },
        noise=noise,
        solver={
            "padding": {"z": tf.z_padding},
            "reconstruction": reconstruction,
            "calibration": calibration,
            "apodization_rolloff": ai.apodization_rolloff,
            "device": settings.device,
        },
    )


# --- 4.0 -> 3.x ------------------------------------------------------------


def _restore_calibratable(value: Any, dotted_name: str, learning_rates: dict[str, float]) -> Any:
    """Inverse of :func:`_convert_calibratable`."""
    if isinstance(value, schema.FreeParameter):
        if value.tolerance != "free":
            raise NotRunnableInLegacyError(
                f"parameter {dotted_name!r} has a numeric tolerance; "
                "numeric tolerances are not supported by the 3.x execution path yet"
            )
        return {
            "initial_value": value.nominal,
            "learning_rate": learning_rates.get(dotted_name, _DEFAULT_LEARNING_RATE),
        }
    return value


def _require_legacy_supported(cfg: schema.V4Config) -> None:
    """Reject 4.0-only features that have no 3.x execution equivalent yet."""

    def unsupported(reason: str) -> NotRunnableInLegacyError:
        return NotRunnableInLegacyError(f"not runnable on the 3.x execution path yet: {reason}")

    if cfg.imaging_system.field_dependence != "shift_invariant":
        raise unsupported("imaging_system.field_dependence: tiled")
    # Free parameters with no calibration block would silently fall through
    # to the 3.x sharpness optimizer, although the design's default score is
    # evidence. Fail loudly instead of guessing the user's intent.
    if schema.free_parameters(cfg.imaging_system) and cfg.solver.calibration is None:
        raise NotRunnableInLegacyError(
            "imaging_system has free parameters (tolerances) but no solver.calibration "
            "block. The 4.0 default score (evidence) runs through 'wo calibrate-tf'; to "
            "use the 3.x sharpness optimizer through 'wo reconstruct', add "
            "solver.calibration: {score: sharpness, search: [{gradient: {max_iter: N}}]}."
        )
    detection = cfg.imaging_system.detection
    if detection.pupil_aberrations is not None:
        raise unsupported("imaging_system.detection.pupil_aberrations (estimate them with 'wo calibrate-tf')")
    if detection.pinhole == "point" or (
        isinstance(detection.pinhole, schema.Pinhole) and detection.pinhole.diameter_au is not None
    ):
        raise unsupported("detection.pinhole: point / diameter_au (use diameter_um)")
    if not (isinstance(cfg.object.prior, str) and cfg.object.prior in ("tikhonov", "tv", "none")):
        raise unsupported(f"object.prior: {cfg.object.prior}")
    if cfg.object.weight == "auto":
        raise unsupported("object.weight: auto (bare); the migrated 3.x sweep is weight: {auto: {...}}")
    if cfg.object.constraints:
        raise unsupported(f"object.constraints: {cfg.object.constraints}")
    if cfg.reconstruction_kind == "fista":
        raise unsupported("solver.reconstruction: fista")
    if cfg.solver.padding.yx not in (0, (0, 0)):
        raise unsupported("solver.padding.yx != 0")
    calibration = cfg.solver.calibration
    if calibration is not None:
        if calibration.score != "sharpness":
            raise unsupported(f"solver.calibration.score: {calibration.score}")
        if calibration.final_estimate != "best_H":
            raise unsupported("solver.calibration.final_estimate: average_near_best_H")
        if len(calibration.search) != 1 or not isinstance(calibration.search[0], schema.SearchGradient):
            raise unsupported("solver.calibration.search stages other than a single gradient stage")


def v4_to_legacy(cfg: schema.V4Config) -> ReconstructionSettings:
    """Convert a 4.0 configuration to the equivalent 3.x settings.

    Parameters
    ----------
    cfg : V4Config
        A 4.0 configuration limited to features with a 3.x execution path.

    Returns
    -------
    ReconstructionSettings
        3.x settings selecting the same code path with the same numbers.

    Raises
    ------
    NotRunnableInLegacyError
        When the configuration uses 4.0-only features (tiled field
        dependence, pupil aberrations, new priors/scores/stages, ...).
    """
    _require_legacy_supported(cfg)

    system = cfg.imaging_system
    detection = system.detection
    illumination = system.illumination
    calibration = cfg.solver.calibration
    learning_rates = (
        calibration.advanced.learning_rates if calibration is not None and calibration.advanced is not None else {}
    )

    tf: dict[str, Any] = {
        "yx_pixel_size": cfg.data.pixel_size.yx,
        "z_pixel_size": cfg.data.pixel_size.z,
        "z_padding": cfg.solver.padding.z,
        "z_focus_offset": _restore_calibratable(detection.z_focus_offset, "detection.z_focus_offset", learning_rates),
        "index_of_refraction_media": system.index_of_refraction_media,
        "numerical_aperture_detection": _restore_calibratable(
            detection.numerical_aperture, "detection.numerical_aperture", learning_rates
        ),
        "tilt_angle_zenith": _restore_calibratable(
            illumination.tilt.zenith, "illumination.tilt.zenith", learning_rates
        ),
        "tilt_angle_azimuth": _restore_calibratable(
            illumination.tilt.azimuth, "illumination.tilt.azimuth", learning_rates
        ),
    }
    if system.contrast == "phase":
        tf["wavelength_illumination"] = illumination.wavelength if illumination.wavelength is not None else 0.532
        tf["numerical_aperture_illumination"] = _restore_calibratable(
            illumination.numerical_aperture if illumination.numerical_aperture is not None else 0.9,
            "illumination.numerical_aperture",
            learning_rates,
        )
        tf["invert_phase_contrast"] = illumination.invert_phase_contrast
    else:
        tf["wavelength_emission"] = detection.wavelength if detection.wavelength is not None else 0.532
        if isinstance(detection.pinhole, schema.Pinhole):
            diameter = detection.pinhole.diameter_um
            if isinstance(diameter, schema.FreeParameter):
                raise NotRunnableInLegacyError("a calibratable pinhole diameter is not supported by 3.x")
            tf["confocal_pinhole_diameter"] = diameter

    apply_inverse: dict[str, Any] = {
        "apodization_rolloff": cfg.solver.apodization_rolloff,
    }
    prior = cfg.object.prior
    recon = cfg.solver.reconstruction
    if prior == "tikhonov":
        if cfg.reconstruction_kind != "closed_form":
            raise NotRunnableInLegacyError(
                f"object.prior: tikhonov runs with solver.reconstruction: closed_form in 3.x, "
                f"got {cfg.reconstruction_kind}"
            )
        apply_inverse["reconstruction_algorithm"] = "Tikhonov"
    elif prior == "tv":
        if not isinstance(recon, schema.ReconstructionAdmm):
            raise NotRunnableInLegacyError("object.prior: tv requires solver.reconstruction: admm in 3.x")
        apply_inverse["reconstruction_algorithm"] = "TV"
        apply_inverse["TV_rho_strength"] = recon.admm.rho
        apply_inverse["TV_iterations"] = recon.admm.iterations
    elif prior == "none":
        if not isinstance(recon, schema.ReconstructionRichardsonLucy):
            raise NotRunnableInLegacyError("object.prior: none requires solver.reconstruction: richardson_lucy")
        if system.contrast != "fluorescence":
            raise NotRunnableInLegacyError("richardson_lucy is fluorescence-only in 3.x")
        rl = recon.richardson_lucy
        apply_inverse["reconstruction_algorithm"] = "RLGC" if rl.gradient_consensus else "RL"
        apply_inverse["rl"] = {
            "iterations": rl.iterations,
            "background": rl.background,
            "stopping_tolerance": rl.stopping_tolerance,
            "back_projector": rl.back_projector,
            "bp_alpha": rl.bp_alpha,
            "bp_beta": rl.bp_beta,
        }

    weight = cfg.object.weight
    if isinstance(weight, schema.WeightAuto):
        apply_inverse["auto_regularization"] = weight.auto.model_dump()
    else:
        apply_inverse["regularization_strength"] = weight

    optimization = None
    if calibration is not None and calibration.advanced is not None and calibration.advanced.optimizer is not None:
        advanced = calibration.advanced
        gradient_stage = calibration.search[0]
        optimization = OptimizationSettings(
            max_iterations=gradient_stage.gradient.max_iter,
            method=advanced.optimizer,
            convergence_tol=advanced.convergence_tol,
            convergence_patience=advanced.convergence_patience,
            use_gradients=advanced.use_gradients,
            grid_points=advanced.grid_points,
            loss=advanced.loss if advanced.loss is not None else {"type": "midband_power"},
            log_dir=advanced.log_dir,
        )

    contrast_settings = {"transfer_function": tf, "apply_inverse": apply_inverse}
    return ReconstructionSettings(
        input_channel_names=cfg.data.input_channel_names,
        time_indices=cfg.data.time_indices,
        reconstruction_dimension=cfg.object.dimension,
        device=cfg.solver.device,
        optimization=optimization,
        **{system.contrast: contrast_settings},
    )


# --- YAML helpers -----------------------------------------------------------


def v4_to_yaml_text(cfg: schema.V4Config) -> str:
    """Render a 4.0 configuration as YAML, dropping null-valued fields."""

    def prune(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: prune(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [prune(v) for v in value]
        return value

    return yaml.safe_dump(prune(cfg.model_dump()), default_flow_style=False, sort_keys=False)
