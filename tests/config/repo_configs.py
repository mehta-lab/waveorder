"""Every reconstruction configuration YAML committed to the repository.

The design document (section 2) requires the regression tests to cover
every 3.x configuration in the repository and in the published examples.
This module is the single source of that list;
``test_migrate.test_every_repo_config_is_listed`` guards it against new
configuration directories appearing without coverage.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

#: Directories holding reconstruction configuration files.
CONFIG_DIRS = (
    REPO_ROOT / "docs/examples/cli/configs",
    REPO_ROOT / "docs/examples/optimization",
    REPO_ROOT / "benchmarks/configs",
)

ALL_CONFIG_PATHS = sorted(path for config_dir in CONFIG_DIRS for path in config_dir.glob("*.yml"))

#: Birefringence stays 3.x this release (design document section 2).
BIREFRINGENCE_CONFIG_PATHS = [p for p in ALL_CONFIG_PATHS if "birefringence" in p.name]

#: Tile-stitch configurations wrap a reconstruction block in stitching
#: parameters; they are not plain ``wo rec`` configurations.
TILE_STITCH_CONFIG_PATHS = [p for p in ALL_CONFIG_PATHS if "tile-stitch" in p.name]

#: Every 3.x phase / fluorescence configuration in the repository.
MIGRATABLE_CONFIG_PATHS = [
    p for p in ALL_CONFIG_PATHS if p not in BIREFRINGENCE_CONFIG_PATHS and p not in TILE_STITCH_CONFIG_PATHS
]
