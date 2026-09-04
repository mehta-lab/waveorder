#!/bin/bash
# Reconstruct 3D phase with `regularization_strength` chosen from the data.
#
# `wo rec` sweeps the strength after computing the transfer function, writes the
# value it picked into `phase_3d_autoreg_autoreg.yml`, and dumps the whole sweep
# to `autoreg_report.json`. Read the report before trusting the pick: neither
# rule is an oracle, and `wo rec` warns when the pick lands on a sweep endpoint.
wo sim \
  -c ./configs/phase_3d_autoreg.yml \
  -o ./phase_autoreg_data.zarr

wo rec \
  -i ./phase_autoreg_data.zarr \
  -c ./configs/phase_3d_autoreg.yml \
  -o ./phase_3d_autoreg_recon.zarr

wo view ./phase_autoreg_data.zarr ./phase_3d_autoreg_recon.zarr
