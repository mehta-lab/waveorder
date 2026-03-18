#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=quality-cmp
#SBATCH --output=docs/examples/gpu-benchmarks/results/quality_h100_%j.out
#SBATCH --error=docs/examples/gpu-benchmarks/results/quality_h100_%j.out

cd /hpc/mydata/sricharan.varra/repos/waveorder.io-cuda-streaming/docs/examples/gpu-benchmarks

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "---"

uv run python compare_recon_quality.py \
    --device cuda:0 \
    --tile-size 128 \
    --n-tiles 16 \
    --opt-iterations 50 \
    --save-npz results/quality_comparison_h100.npz
