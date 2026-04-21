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

    function: Literal["single_bead", "random_beads"]
    shape: tuple[PositiveInt, PositiveInt, PositiveInt] = (64, 128, 128)
    pixel_sizes: tuple[PositiveFloat, PositiveFloat, PositiveFloat] = (0.25, 0.1, 0.1)


class ReferenceParameter(BaseModel):
    """Expected optimized value for a single parameter, with tolerance.

    Used by cases that run parameter optimization to gate regressions —
    if the optimizer drifts beyond ``tolerance``, the benchmark flags it.
    """

    model_config = ConfigDict(extra="forbid")

    value: float
    tolerance: PositiveFloat


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
    reference_parameters: dict[str, ReferenceParameter] | None = None


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
