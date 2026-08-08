#!/bin/bash
#SBATCH --job-name=tper-bench-h100
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu-h100-80g
#SBATCH --gpus=1
#SBATCH --output=logs/%x-%j.out

module load scicomp-pytorch-env
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results logs
nvidia-smi --query-gpu=name,driver_version --format=csv | tee results/gpu_h100.txt
python bench/bench_sampling.py --out results/micro_h100.csv
