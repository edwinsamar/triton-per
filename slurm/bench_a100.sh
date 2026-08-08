#!/bin/bash
#SBATCH --job-name=tper-bench-a100
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100:1
#SBATCH --output=logs/%x-%j.out

module load scicomp-pytorch-env
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results logs

nvidia-smi --query-gpu=name,driver_version --format=csv | tee results/gpu_a100.txt
python -c "import torch, triton; print(torch.__version__, triton.__version__)"

python bench/bench_sampling.py --out results/micro_a100.csv
python bench/profile_utd.py --backend cpu-sumtree --utd 32 --trace results/trace_cpu_utd32.json
python bench/profile_utd.py --backend triton      --utd 32 --trace results/trace_triton_utd32.json
