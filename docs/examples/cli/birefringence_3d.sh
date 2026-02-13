#!/bin/bash
wo sim \
  -c ./configs/birefringence_3d.yml \
  -o ./birefringence_data.zarr

wo rec \
  -i ./birefringence_data.zarr \
  -c ./configs/birefringence_3d.yml \
  -o ./birefringence_3d_recon.zarr

wo view ./birefringence_data.zarr ./birefringence_3d_recon.zarr
