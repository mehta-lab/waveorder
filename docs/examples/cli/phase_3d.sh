#!/bin/bash
# `configs/phase_3d.yml` carries an `auto_regularization` block, so `wo rec` sweeps
# `regularization_strength` before reconstructing, prints its pick, and freezes it
# into `configs/phase_3d_autoreg.yml` for reproducible reruns. Set the block to
# `null` to use `regularization_strength` as written, or set its `report_path` to
# dump the whole sweep as JSON. Neither rule is an oracle: read the pick.
wo sim \
  -c ./configs/phase_3d.yml \
  -o ./phase_data.zarr

wo rec \
  -i ./phase_data.zarr \
  -c ./configs/phase_3d.yml \
  -o ./phase_3d_recon.zarr

wo view ./phase_data.zarr ./phase_3d_recon.zarr
