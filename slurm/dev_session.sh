#!/bin/bash
# Interactive dev session on a V100 (plentiful -> short queues).
# Usage: bash slurm/dev_session.sh
srun --time=04:00:00 --mem=16G --cpus-per-task=4 --gpus=v100:1 --pty bash
