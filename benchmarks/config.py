"""Experiment YAML loading, validation, and config management.

An experiment consists of named cases, each specifying a phantom,
reconstruction config, and case type. Cases can inherit shared
``base_phantom`` and ``base_config`` from the experiment level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator


class PhantomConfig(BaseModel):
    """Configuration for phantom generation.

    Parameters
    ----------
    function : str
        Name of the phantom function ("single_bead" or "random_beads").
    shape : tuple[int, int, int]
        Volume shape (Z, Y, X) in pixels.
    pixel_sizes : tuple[float, float, float]
        (z, y, x) pixel sizes in um.
    """

    model_config = ConfigDict(extra="allow")

    function: Literal["single_bead", "random_beads", "grid_beads", "grid_beads_gaussian"]
    shape: tuple[PositiveInt, PositiveInt, PositiveInt] = (64, 128, 128)
    pixel_sizes: tuple[PositiveFloat, PositiveFloat, PositiveFloat] = (0.25, 0.1, 0.1)


class SimulationConfig(BaseModel):
    """Optional forward-model override for a synthetic case.

    By default a synthetic case uses the shift-invariant
    ``isotropic_fluorescent_thick_3d`` / ``phase_thick_3d`` simulators.
    Set ``forward_model: shift_variant`` to use the spatial-polynomial
    pupil simulator from
    :mod:`waveorder.models.shift_variant_fluorescent_3d`.

    Parameters
    ----------
    forward_model : str
        ``"shift_invariant"`` (default) or ``"shift_variant"``.
    spatial_pupil_coefficients : dict
        Mapping ``"j_m_n": c`` (waves, RMS) defining the spatial
        polynomial pupil.
    n_tiles_yx : tuple[int, int]
        Number of partition tiles along ``(Y, X)`` for the shift-variant
        forward model.
    """

    model_config = ConfigDict(extra="forbid")

    forward_model: Literal["shift_invariant", "shift_variant"] = "shift_invariant"
    spatial_pupil_coefficients: dict[str, float] = Field(default_factory=dict)
    n_tiles_yx: tuple[PositiveInt, PositiveInt] = (8, 8)


class RecoveryConfig(BaseModel):
    """Optional shift-variant Zernike recovery, executed in place of ``wo rec``.

    When set on a synthetic case, the runner replaces the standard
    inverse-transfer-function reconstruction with the per-tile Zernike
    optimisation from :mod:`waveorder.api.shift_variant_recovery`. The
    case writes ``recovered_coefs.npy``, ``truth_coefs.npy``, and a
    ``zernike_recovery`` block into ``metrics.json`` containing the
    recovery_score FoM and per-mode RMSE/correlation.

    Tile geometry mirrors ``waveorder.tile_stitch``; the sim's
    ``n_tiles_yx`` is independent — the recovery's ``tile_size_yx`` is
    aligned with the bead grid.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    noll_indices: list[PositiveInt] = Field(default_factory=lambda: list(range(4, 16)))
    tile_size_yx: dict[str, PositiveInt] = Field(default_factory=lambda: {"y": 26, "x": 26})
    tile_overlap_yx: dict[str, int] = Field(default_factory=lambda: {"y": 0, "x": 0})
    bead_template_sigma_um: tuple[PositiveFloat, PositiveFloat, PositiveFloat] = (0.15, 0.1, 0.1)
    loss: Literal["mse", "midband", "midband_3d", "tv", "laplacian_var", "normalized_var", "spectral_flatness"] = (
        "midband"
    )
    midband_fractions: tuple[float, float] = (0.2, 0.4)
    l1_strength: float = 0.0
    smooth_strength: float = 0.0
    scale_fit: bool = True
    optimizer: Literal["adam", "adamw", "nadam", "sgd", "lbfgs"] = "sgd"
    lr_schedule: Literal["constant", "cosine", "step", "warmup"] = "constant"
    lr_mult: PositiveFloat = 1.0
    n_iter: PositiveInt = 250
    wiener_regularization: PositiveFloat = 1.0e-3


class ReferenceBound(BaseModel):
    """One-sided or two-sided bound on a referenced value.

    Used to gate benchmark regressions. Each key in a case's
    ``reference`` dict is a dotted path into the computed metrics (e.g.
    ``image_quality.midband_power``, ``with_phantom.ssim``) or into the
    optimizer's final parameters under the ``parameter.`` prefix (e.g.
    ``parameter.z_focus_offset``). At least one of ``min`` or ``max``
    must be provided.
    """

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _validate(self):
        if self.min is None and self.max is None:
            raise ValueError("ReferenceBound requires at least one of 'min' or 'max'")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("ReferenceBound 'min' must be <= 'max'")
        return self


class CropConfig(BaseModel):
    """Optional bbox to crop the input zarr before reconstruction.

    Each axis is ``[start, stop]`` in pixel indices. Omitted axes take
    the full range. Only ``hpc`` cases use this; synthetic cases already
    generate data at the requested shape.
    """

    model_config = ConfigDict(extra="forbid")

    z: tuple[int, int] | None = None
    y: tuple[int, int] | None = None
    x: tuple[int, int] | None = None

    def slices(self) -> tuple[slice, slice, slice]:
        """Return ``(z_slice, y_slice, x_slice)`` ready to index a ZYX array."""

        def _sl(r):
            return slice(r[0], r[1]) if r else slice(None)

        return _sl(self.z), _sl(self.y), _sl(self.x)


class CaseConfig(BaseModel):
    """Configuration for a single benchmark case."""

    model_config = ConfigDict(extra="allow")

    type: Literal["synthetic", "hpc"] = "synthetic"
    phantom: PhantomConfig | None = None
    config: str | None = None
    overrides: dict[str, Any] | None = None
    input: str | None = None
    position: str | None = None
    crop: CropConfig | None = None
    reference: dict[str, ReferenceBound] | None = None
    simulation: SimulationConfig | None = None
    recovery: RecoveryConfig | None = None


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration.

    Parameters
    ----------
    name : str
        Experiment name (used in output directory naming).
    base_phantom : PhantomConfig or None
        Default phantom inherited by cases that don't define their own.
    base_config : str or None
        Path to base reconstruction YAML. Cases inherit and can override.
    cases : dict[str, CaseConfig]
        Named benchmark cases.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    base_phantom: PhantomConfig | None = None
    base_config: str | None = None
    cases: dict[str, CaseConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def inherit_base_phantom(self):
        """Copy base_phantom into cases that don't define their own."""
        if self.base_phantom is not None:
            for case in self.cases.values():
                if case.phantom is None:
                    case.phantom = self.base_phantom.model_copy(deep=True)
        return self

    @model_validator(mode="after")
    def inherit_base_config(self):
        """Copy base_config into cases that don't define their own."""
        if self.base_config is not None:
            for case in self.cases.values():
                if case.config is None:
                    case.config = self.base_config
        return self


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the experiment YAML.

    Returns
    -------
    ExperimentConfig
        Validated experiment configuration.
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(**raw)


def resolve_recon_config(case: CaseConfig, experiment_dir: Path) -> dict:
    """Resolve the reconstruction config for a case.

    Loads the base config YAML and applies any overrides.

    Parameters
    ----------
    case : CaseConfig
        Case configuration.
    experiment_dir : Path
        Directory of the experiment YAML (for resolving relative paths).

    Returns
    -------
    dict
        Resolved reconstruction config dict.
    """
    if case.config is None:
        return {}

    config_path = Path(case.config)
    if not config_path.is_absolute():
        config_path = experiment_dir / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if case.overrides:
        for dotted_key, value in case.overrides.items():
            _deep_set(config, dotted_key.split("."), value)

    return config


def infer_modality(recon_config: dict) -> str:
    """Return ``"phase"`` if the recon config has a phase block, else ``"fluorescence"``."""
    return "phase" if "phase" in recon_config else "fluorescence"


def _deep_set(d: dict, keys: list[str], value: Any):
    """Set a nested dict value by key path.

    Parameters
    ----------
    d : dict
        Target dict (modified in-place).
    keys : list[str]
        Key path, e.g. ["phase", "apply_inverse", "regularization_strength"].
    value : Any
        Value to set.
    """
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
