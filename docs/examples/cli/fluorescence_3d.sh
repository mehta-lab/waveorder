#!/bin/bash
wo sim \
  -c ./configs/fluorescence_3d.yml \
  -o ./fluorescence_data.zarr

wo rec \
  -i ./fluorescence_data.zarr \
  -c ./configs/fluorescence_3d.yml \
  -o ./fluorescence_3d_recon.zarr

wo view ./fluorescence_data.zarr ./fluorescence_3d_recon.zarr
