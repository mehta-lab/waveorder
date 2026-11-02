"""Version-aware loading of reconstruction configurations.

Every CLI verb loads its configuration through
:func:`load_reconstruction_settings`:

* a configuration with ``version: 4`` is parsed against the 4.0 schema and
  converted to the 3.x settings that select the same code path;
* a configuration without ``version`` is 3.x: it runs as before, and (when
  it is migratable) the 4.0 equivalent is printed once per process with a
  deprecation warning, per the 4.0 design document section 2.

Sunset policy (design document section 2): 3.x configurations are accepted
with the deprecation warning for two releases (4.0 and 4.1); after that
they are rejected with a message pointing to ``waveorder migrate-config``.
Birefringence configurations stay 3.x this release and are exempt.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import yaml

from waveorder.cli.settings import ReconstructionSettings
from waveorder.config.migrate import (
    MigrationError,
    detect_version,
    migrate_3x_settings,
    v4_to_legacy,
    v4_to_yaml_text,
)
from waveorder.config.schema import V4Config

_already_warned_paths: set[str] = set()


def load_v4_config(config_path: Path | str) -> V4Config:
    """Load a ``version: 4`` configuration file against the 4.0 schema."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if detect_version(raw) < 4:
        raise ValueError(f"{config_path} is a 3.x configuration; run 'waveorder migrate-config' first")
    return V4Config(**raw)


def load_reconstruction_settings(config_path: Path | str) -> ReconstructionSettings:
    """Load a 3.x or 4.0 configuration file into 3.x execution settings.

    Parameters
    ----------
    config_path : Path or str
        Path to the YAML configuration.

    Returns
    -------
    ReconstructionSettings
        Settings driving the existing execution paths. For a 4.0
        configuration these are derived by
        :func:`waveorder.config.migrate.v4_to_legacy`; for a 3.x
        configuration they are parsed directly (and the 4.0 equivalent is
        printed once with a deprecation warning).
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if detect_version(raw) >= 4:
        return v4_to_legacy(V4Config(**raw))

    settings = ReconstructionSettings(**raw)
    _warn_once_about_3x(settings, str(config_path))
    return settings


def _warn_once_about_3x(settings: ReconstructionSettings, config_path: str) -> None:
    """Print the 4.0 equivalent of a migratable 3.x configuration, once per path."""
    if config_path in _already_warned_paths:
        return
    _already_warned_paths.add(config_path)
    try:
        migrated = migrate_3x_settings(settings)
    except MigrationError:
        # Birefringence configurations stay 3.x in this release; no warning.
        return
    warnings.warn(
        f"{config_path} is a 3.x configuration; it still runs, but 3.x acceptance ends "
        f"two releases from now (after 4.1), when it will be rejected. Convert it with: "
        f"waveorder migrate-config {config_path}",
        DeprecationWarning,
        stacklevel=3,
    )
    print(f"The 4.0 equivalent of {config_path} is:\n\n{v4_to_yaml_text(migrated)}")
