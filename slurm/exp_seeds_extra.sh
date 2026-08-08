#!/bin/bash
# Equivalence-strengthening runs: seeds 4-6 for both backends at UTD 20
# (indices 0-17), plus a torch-backend control arm at seeds 1-3 (indices
# 18-26). The control shares the CPU reference's exact sampling math on GPU,
# so it separates "kernel distribution effect" from seed noise.
#SBATCH --job-name=tper-e2e-x
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=a100:1
#SBATCH --array=0-26
#SBATCH --output=logs/%x-%A_%a.out

module load scicomp-pytorch-env
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

ENVS=(HalfCheetah-v5 Walker2d-v5 Ant-v5)
i=$SLURM_ARRAY_TASK_ID

if [ "$i" -lt 18 ]; then
    env=${ENVS[$(( i % 3 ))]};  i=$(( i / 3 ))
    backend=$([ $(( i % 2 )) = 0 ] && echo triton || echo cpu-sumtree); i=$(( i / 2 ))
    seed=$(( i % 3 + 4 ))
else
    j=$(( i - 18 ))
    env=${ENVS[$(( j % 3 ))]}
    backend=torch
    seed=$(( j / 3 + 1 ))
fi

python experiments/sac.py \
    --env-id "$env" --backend "$backend" --utd 20 --num-q 10 \
    --seed "$seed" --total-steps 100000
