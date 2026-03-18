#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=test-stream
#SBATCH --output=docs/examples/gpu-benchmarks/results/test_stream_h100_%j.out
#SBATCH --error=docs/examples/gpu-benchmarks/results/test_stream_h100_%j.out

cd /hpc/mydata/sricharan.varra/repos/waveorder.io-cuda-streaming/docs/examples/gpu-benchmarks

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "---"

uv run python test_stream_optimize.py \
    --device cuda:0 \
    --opt-iterations 10 \
    --tile-size 128 \
    --batch-size 16
