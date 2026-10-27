#!/bin/bash
# Reconstruct in overlapping YX tiles and blend them back into one volume.
# configs/tile-stitch_phase_3d.yml holds the tiling and the blend, plus a
# `recon` block with the same schema as configs/phase_3d.yml.
wo sim \
  -c ./configs/phase_3d.yml \
  -o ./phase_data.zarr

wo tile-stitch \
  -i ./phase_data.zarr/0/0/0 \
  -c ./configs/tile-stitch_phase_3d.yml \
  -o ./phase_3d_tile_stitch.zarr

wo view ./phase_data.zarr ./phase_3d_tile_stitch.zarr
