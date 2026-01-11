#!/bin/bash
# This is a training script for the GPU partition
#SBATCH --job-name=gpu_training_script
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=gpu_course
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

# Load environment modules
module load 2025

# Activate virtual environment
source $HOME/MLOps_2026/.venv/bin/activate

python $HOME/MLOps_2026/experiments/train.py --config experiments/configs/train_config.yaml
