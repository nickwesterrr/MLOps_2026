#!/bin/bash
#SBATCH --job-name=MLOps_Train
#SBATCH --output=gpu_training_%j.out
#SBATCH --error=gpu_training_error_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_course
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# 1. Load modules
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

# 2. Activate Virtual Environment
# Make sure this points to your CORRECT venv folder
source ../.venv/bin/activate

# 3. Debug info
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 4. Run Python
# The python script will detect TMPDIR automatically (see Step 2)
python experiments/train.py --config experiments/configs/train_config_seed44.yaml