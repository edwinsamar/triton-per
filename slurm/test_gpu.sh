#!/bin/bash
#SBATCH --job-name=tper-tests
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=v100:1
#SBATCH --output=logs/%x-%j.out

module load scicomp-pytorch-env
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
python -m pytest tests/ -v
