"""WaveOrder 4.0 configuration: schema, migration, and loading.

The 4.0 configuration regroups a reconstruction into five blocks (``data``,
``imaging_system``, ``object``, ``noise``, ``solver``) plus ``outputs``, with
a required ``version`` field. A configuration without ``version`` is 3.x and
is migrated mechanically; see :mod:`waveorder.config.migrate`.
"""

from waveorder.config.loader import (
    load_reconstruction_settings,
    load_v4_config,
)
from waveorder.config.migrate import (
    MigrationError,
    NotRunnableInLegacyError,
    detect_version,
    migrate_3x_dict,
    migrate_3x_settings,
    v4_to_legacy,
    v4_to_yaml_text,
)
from waveorder.config.schema import (
    Calibration,
    DataSettings,
    Detection,
    FreeParameter,
    Illumination,
    ImagingSystem,
    NoiseModel,
    ObjectSettings,
    PupilAberrations,
    SolverSettings,
    V4Config,
    free_parameters,
    is_free,
    nominal_of,
)

__all__ = [
    "Calibration",
    "MigrationError",
    "NotRunnableInLegacyError",
    "load_reconstruction_settings",
    "load_v4_config",
    "migrate_3x_settings",
    "v4_to_yaml_text",
    "DataSettings",
    "Detection",
    "FreeParameter",
    "Illumination",
    "ImagingSystem",
    "NoiseModel",
    "ObjectSettings",
    "PupilAberrations",
    "SolverSettings",
    "V4Config",
    "detect_version",
    "free_parameters",
    "is_free",
    "migrate_3x_dict",
    "nominal_of",
    "v4_to_legacy",
]
