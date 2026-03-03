#!/bin/bash
# NOTE: Work in progress. Pipeline runs end-to-end but physics still being validated.
wo sim \
  -c ./configs/birefringence-and-phase_3d.yml \
  -o ./birefringence_and_phase_data.zarr

wo rec \
  -i ./birefringence_and_phase_data.zarr \
  -c ./configs/birefringence-and-phase_3d.yml \
  -o ./birefringence_and_phase_3d_recon.zarr

wo view ./birefringence_and_phase_data.zarr ./birefringence_and_phase_3d_recon.zarr
