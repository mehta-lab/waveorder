#!/bin/bash
wo sim \
  -c ./configs/phase_3d.yml \
  -o ./phase_data.zarr

wo rec \
  -i ./phase_data.zarr \
  -c ./configs/phase_3d.yml \
  -o ./phase_3d_recon.zarr

wo view ./phase_data.zarr ./phase_3d_recon.zarr
