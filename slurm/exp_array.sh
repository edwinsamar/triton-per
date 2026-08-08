#!/bin/bash
# End-to-end SAC/REDQ sweep as a slurm job array.
# Grid: 3 tasks x 2 backends x 2 UTD x 3 seeds = 36 runs (array indices 0-35).
# Submit:  sbatch slurm/exp_array.sh
#SBATCH --job-name=tper-e2e
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100:1
#SBATCH --array=0-35
#SBATCH --output=logs/%x-%A_%a.out

module load scicomp-pytorch-env
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

ENVS=(HalfCheetah-v5 Walker2d-v5 Ant-v5)
BACKENDS=(triton cpu-sumtree)
UTDS=(1 20)
SEEDS=(1 2 3)

i=$SLURM_ARRAY_TASK_ID
env=${ENVS[$(( i % 3 ))]};      i=$(( i / 3 ))
backend=${BACKENDS[$(( i % 2 ))]}; i=$(( i / 2 ))
utd=${UTDS[$(( i % 2 ))]};      i=$(( i / 2 ))
seed=${SEEDS[$(( i % 3 ))]}

numq=2; [ "$utd" = "20" ] && numq=10   # UTD 20 -> REDQ config

python experiments/sac.py \
    --env-id "$env" --backend "$backend" --utd "$utd" \
    --num-q "$numq" --seed "$seed" --total-steps 100000
