#!/bin/bash
#SBATCH --job-name=wo-bench-regression
#SBATCH --partition=gpu
#SBATCH --constraint=intel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=wo-bench-%j.log

# Run the full regression suite on the Biohub HPC.
#
# Usage (from a checkout with `wo` + bench group on PATH, see
# benchmarks/README.md):
#
#   sbatch benchmarks/run_regression.sbatch
#
# Runs land in $WAVEORDER_BENCH_OUTPUT (shared default below). The
# intel constraint keeps baselines hardware-comparable across runs.

export WAVEORDER_BENCH_OUTPUT=${WAVEORDER_BENCH_OUTPUT:-/hpc/projects/waveorder/benchmarks}

wo bm run --scope all
