#!/bin/bash
wo sim \
  -c ./configs/phase_2d.yml \
  -o ./phase_2d_data.zarr

wo rec \
  -i ./phase_2d_data.zarr \
  -c ./configs/phase_2d.yml \
  -o ./phase_2d_recon.zarr

wo view ./phase_2d_data.zarr ./phase_2d_recon.zarr
