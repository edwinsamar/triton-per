#!/bin/bash
#SBATCH --job-name=tper-bench-mi210
#SBATCH --partition=gpu-amd
#SBATCH --gpus=mi210:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/%x-%j.out

# AMD path: no module and no compiled extension; a plain PyTorch ROCm
# wheel environment suffices (pip install torch --index-url
# https://download.pytorch.org/whl/rocmX.Y ships the Triton AMD backend;
# the device string remains "cuda" under ROCm). torchrl backends are
# omitted: its CUDA extension does not build under HIP.
source "${ROCM_VENV:-$HOME/venvs/rocm-torch}/bin/activate"
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results logs

python bench/bench_sampling.py \
  --backends triton,triton-unfused,torch \
  --capacities 100000,1000000,10000000 \
  --batch-sizes 256,1024,4096 \
  --obs-dims 32,256 \
  --iters 200 --warmup 50 \
  --out results/micro_mi210.csv
