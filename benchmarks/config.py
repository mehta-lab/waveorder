"""Experiment YAML loading and config management."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_experiment(path: str | Path) -> dict:
    """Load an experiment YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the experiment YAML.

    Returns
    -------
    dict
        Parsed experiment with resolved cases.
    """
    path = Path(path)
    with open(path) as f:
        experiment = yaml.safe_load(f)

    base_phantom = experiment.get("base_phantom")
    base_config = experiment.get("base_config")

    for name, case in experiment.get("cases", {}).items():
        # Inherit base_phantom if case doesn't define its own
        if "phantom" not in case and base_phantom is not None:
            case["phantom"] = copy.deepcopy(base_phantom)

        # Inherit base_config and apply overrides
        if "config" not in case and base_config is not None:
            case["config"] = base_config
        if "overrides" in case and base_config is not None:
            merged = _load_and_merge(base_config, case["overrides"], path.parent)
            case["_resolved_config"] = merged

    return experiment


def _load_and_merge(config_path: str, overrides: dict, base_dir: Path) -> dict:
    """Load a YAML config and deep-merge overrides into it.

    Parameters
    ----------
    config_path : str
        Path to the base config YAML (resolved relative to base_dir).
    overrides : dict
        Dot-separated keys mapped to override values.
    base_dir : Path
        Directory for resolving relative config paths.

    Returns
    -------
    dict
        Deep-merged config.
    """
    resolved_path = Path(config_path)
    if not resolved_path.is_absolute():
        resolved_path = base_dir / resolved_path

    with open(resolved_path) as f:
        config = yaml.safe_load(f)

    for dotted_key, value in overrides.items():
        _deep_set(config, dotted_key.split("."), value)

    return config


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
