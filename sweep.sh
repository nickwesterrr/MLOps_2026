#!/bin/bash
#SBATCH --job-name=pcam_hparam_sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --partition=gpu_course
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-17

module load 2025
source .venv/bin/activate

# Search space
LRS=(0.001 0.0001 0.00001)
BS=(32 64 128)
OPTS=("adam" "sgd")

# Index calculation
LR_IDX=$((SLURM_ARRAY_TASK_ID % 3))
BS_IDX=$(((SLURM_ARRAY_TASK_ID / 3) % 3))
OPT_IDX=$((SLURM_ARRAY_TASK_ID / 9))

python experiments/train.py \
    --config experiments/configs/train_config.yaml \
    --lr ${LRS[$LR_IDX]} \
    --batch_size ${BS[$BS_IDX]} \
    --optimizer ${OPTS[$OPT_IDX]}