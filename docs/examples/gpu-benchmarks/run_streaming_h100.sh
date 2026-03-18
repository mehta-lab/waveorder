#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=stream-h100
#SBATCH --output=docs/examples/gpu-benchmarks/results/streaming_h100_%j.out
#SBATCH --error=docs/examples/gpu-benchmarks/results/streaming_h100_%j.out

cd /hpc/mydata/sricharan.varra/repos/waveorder.io-cuda-streaming/docs/examples/gpu-benchmarks

nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv,noheader
echo "CPUs: $(nproc)"
echo "---"

# 3x3 grid centered on 029029, 50 iterations
uv run python benchmark_stream_optimize.py \
    --device cuda:0 \
    --tile-size 128 \
    --batch-size 16 \
    --opt-iterations 50
