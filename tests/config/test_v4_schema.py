"""The 4.0 schema parses the design document's examples and enforces its rules."""

import pytest
import yaml

from waveorder.config.schema import (
    FreeParameter,
    SearchGradient,
    SearchGrid,
    V4Config,
    free_parameters,
)

# Design document example 4.1: blind, tiled, evidence-scored phase reconstruction.
EXAMPLE_4_1 = """
version: 4

data:
  input_channel_names: [BF]
  time_indices: all
  pixel_size: {yx: 0.1, z: 0.25}

imaging_system:
  contrast: phase
  field_dependence:
    tiled: {size_yx: [256, 256], overlap_yx: [32, 32], tile_blend: linear}
  index_of_refraction_media: 1.3
  detection:
    numerical_aperture: {nominal: 1.2, tolerance: 0.05, bounds: [1.0, 1.3]}
    pupil_aberrations:
      zernike_up_to_order: {order: 4, tolerance: 0.02}
      spherical: {nominal: 0.05, tolerance: 0.02}
  illumination:
    wavelength: 0.532
    numerical_aperture: 0.9
    tilt: {zenith: {nominal: 0.0, tolerance: 0.02}, azimuth: 0.0}
    invert_phase_contrast: false

object:
  dimension: 3
  prior: tikhonov
  weight: auto
  constraints: []

noise:
  gaussian: {sigma: estimate}

solver:
  padding: {z: 5, yx: 0}
  reconstruction: closed_form
  calibration:
    score: evidence
    search:
      - grid: {points: 7}
      - gradient: {max_iter: 200}
    final_estimate: best_H

outputs: [reconstruction, transfer_function, psf, parameters, parameters_covariance,
          score_trace, score_grid, weight_trace, residual, tile_map]
"""

# Design document example 4.2: the same system, non-blind, as 3.x ran it.
EXAMPLE_4_2 = """
version: 4
data:
  input_channel_names: [BF]
  pixel_size: {yx: 0.1, z: 0.25}
imaging_system:
  contrast: phase
  index_of_refraction_media: 1.3
  detection: {numerical_aperture: 1.2}
  illumination: {wavelength: 0.532, numerical_aperture: 0.9}
object: {prior: tikhonov, weight: 1e-3}
"""

# Design document example 4.3: bead calibration (fluorescence).
EXAMPLE_4_3 = """
version: 4
data:
  input_channel_names: [GFP]
  pixel_size: {yx: 0.1, z: 0.25}
imaging_system:
  contrast: fluorescence
  field_dependence: shift_invariant
  index_of_refraction_media: 1.3
  detection:
    wavelength: 0.520
    numerical_aperture: {nominal: 1.2, tolerance: free, bounds: [1.0, 1.3]}
    pinhole: none
    pupil_aberrations: {zernike_up_to_order: {order: 4, tolerance: free}}
object:
  prior: {beads: {diameter: 0.2}}
noise: {poisson_gaussian: {gain: 2.1, offset: 100, sigma: estimate}}
solver:
  calibration: {score: evidence, search: [{gradient: {max_iter: 100}}]}
outputs: [parameters, parameters_covariance, transfer_function, psf]
"""


def _parse(text: str) -> V4Config:
    return V4Config(**yaml.safe_load(text))


def test_example_4_1_parses():
    cfg = _parse(EXAMPLE_4_1)
    assert cfg.imaging_system.field_dependence.tiled.size_yx == (256, 256)
    na = cfg.imaging_system.detection.numerical_aperture
    assert isinstance(na, FreeParameter)
    assert na.tolerance == 0.05
    assert na.bounds == (1.0, 1.3)
    aberrations = cfg.imaging_system.detection.pupil_aberrations
    assert aberrations.zernike_up_to_order.order == 4
    assert aberrations.terms["spherical"].nominal == 0.05
    assert isinstance(cfg.solver.calibration.search[0], SearchGrid)
    assert isinstance(cfg.solver.calibration.search[1], SearchGradient)
    assert cfg.noise.gaussian.sigma == "estimate"
    assert cfg.object.weight == "auto"


def test_example_4_2_parses_and_is_fixed():
    cfg = _parse(EXAMPLE_4_2)
    assert free_parameters(cfg.imaging_system) == {}
    assert cfg.solver.reconstruction == "closed_form"
    assert cfg.noise.kind == "gaussian"
    assert cfg.noise.gaussian.sigma == 1.0  # default reproduces 3.x
    assert cfg.solver.padding.z == 0  # 3.x z_padding default
    assert cfg.solver.calibration is None


def test_example_4_3_parses():
    cfg = _parse(EXAMPLE_4_3)
    assert cfg.object.prior.beads.diameter == 0.2
    assert cfg.noise.kind == "poisson_gaussian"
    params = free_parameters(cfg.imaging_system)
    assert "detection.numerical_aperture" in params
    assert "detection.pupil_aberrations.zernike_up_to_order" in params


def test_free_parameters_collects_named_terms():
    cfg = _parse(EXAMPLE_4_1)
    params = free_parameters(cfg.imaging_system)
    assert params["detection.pupil_aberrations.spherical"].tolerance == 0.02
    assert params["illumination.tilt.zenith"].tolerance == 0.02
    assert "illumination.tilt.azimuth" not in params


def test_nominal_outside_bounds_rejected():
    with pytest.raises(ValueError, match="outside bounds"):
        FreeParameter(nominal=2.0, tolerance=0.1, bounds=(0.0, 1.0))


def test_na_above_index_rejected():
    bad = yaml.safe_load(EXAMPLE_4_2)
    bad["imaging_system"]["detection"]["numerical_aperture"] = 1.4
    with pytest.raises(ValueError, match="index_of_refraction_media"):
        V4Config(**bad)


def test_unknown_zernike_name_rejected():
    bad = yaml.safe_load(EXAMPLE_4_3)
    bad["imaging_system"]["detection"]["pupil_aberrations"] = {"comatose": 0.1}
    with pytest.raises(ValueError, match="Zernike"):
        V4Config(**bad)


def test_grid_stage_with_unbounded_free_parameter_rejected():
    bad = yaml.safe_load(EXAMPLE_4_3)
    bad["imaging_system"]["detection"]["numerical_aperture"] = {"nominal": 1.2, "tolerance": "free"}
    bad["imaging_system"]["detection"].pop("pupil_aberrations")
    bad["solver"]["calibration"]["search"] = [{"grid": {"points": 5}}]
    with pytest.raises(ValueError, match="bounds"):
        V4Config(**bad)


def test_rl_requires_poisson_and_no_prior():
    raw = yaml.safe_load(EXAMPLE_4_2)
    raw["solver"] = {"reconstruction": {"richardson_lucy": {"iterations": 10}}}
    with pytest.raises(ValueError, match="richardson_lucy"):
        V4Config(**raw)
    raw["object"] = {"prior": "none"}
    raw["noise"] = {"poisson": {}}
    cfg = V4Config(**raw)
    assert cfg.reconstruction_kind == "richardson_lucy"


def test_prior_none_with_closed_form_rejected():
    raw = yaml.safe_load(EXAMPLE_4_2)
    raw["object"] = {"prior": "none"}
    with pytest.raises(ValueError, match="unregularized"):
        V4Config(**raw)


def test_weight_auto_requires_evidence_family_score():
    raw = yaml.safe_load(EXAMPLE_4_3)
    raw["object"] = {"prior": "tikhonov", "weight": "auto"}
    raw["solver"]["calibration"]["score"] = "sharpness"
    with pytest.raises(ValueError, match="weight: auto"):
        V4Config(**raw)


def test_extra_fields_rejected():
    raw = yaml.safe_load(EXAMPLE_4_2)
    raw["imaging_system"]["detection"]["numerical_apertur"] = 1.0
    with pytest.raises(ValueError):
        V4Config(**raw)
