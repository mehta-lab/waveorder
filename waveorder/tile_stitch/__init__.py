"""Tile-stitching engine — pure-python primitives for partition/blend/recon.

This subpackage holds the implementation details (blend kernels, partition
geometry, scheduler ordering, recon engine). The user-facing surface lives
in ``waveorder.api.tile_stitch``. Distributed orchestration (dask, cluster
lifecycle) is intentionally *not* here — that lives in the biahub package.
"""

from waveorder.tile_stitch._engine import (
    TileStitchPlan,
    blend_output_tile,
    build_plan,
    reconstruct_tile,
    tile_stitch_reconstruction,
)
from waveorder.tile_stitch.blend import (
    Blend,
    clipped_mean,
    gaussian_mean,
    max_blend,
    min_blend,
    uniform_mean,
)
from waveorder.tile_stitch.partition import (
    InputTile,
    OutputTile,
    generate_output_tiles,
    generate_tiles,
    input_tiles_for_output,
)
from waveorder.tile_stitch.scheduler import (
    bundle_inputs_by_cooccurrence,
    output_to_batches_map,
    schedule_coverage_greedy,
)

__all__ = [
    "Blend",
    "InputTile",
    "OutputTile",
    "TileStitchPlan",
    "blend_output_tile",
    "build_plan",
    "bundle_inputs_by_cooccurrence",
    "clipped_mean",
    "gaussian_mean",
    "generate_output_tiles",
    "generate_tiles",
    "input_tiles_for_output",
    "max_blend",
    "min_blend",
    "output_to_batches_map",
    "reconstruct_tile",
    "schedule_coverage_greedy",
    "tile_stitch_reconstruction",
    "uniform_mean",
]
