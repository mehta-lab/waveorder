#!/bin/bash
# To choose regularization_strength from the data instead of using the value in
# the config, replace `auto_regularization: null` in configs/phase_3d.yml with the
# commented-out block beneath it.
wo sim \
  -c ./configs/phase_3d.yml \
  -o ./phase_data.zarr

wo rec \
  -i ./phase_data.zarr \
  -c ./configs/phase_3d.yml \
  -o ./phase_3d_recon.zarr

wo view ./phase_data.zarr ./phase_3d_recon.zarr
