#!/bin/bash
wo sim \
  -c ./configs/fluorescence_2d.yml \
  -o ./fluorescence_2d_data.zarr

wo rec \
  -i ./fluorescence_2d_data.zarr \
  -c ./configs/fluorescence_2d.yml \
  -o ./fluorescence_2d_recon.zarr

wo view ./fluorescence_2d_data.zarr ./fluorescence_2d_recon.zarr
