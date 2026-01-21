#!/bin/bash
#SBATCH --job-name=plot_eda_script
#SBATCH --partition=thin_course
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

# Load environment modules
module load 2025

# Activate virtual environment
source $HOME/MLOps_2026/.venv/bin/activate

python $HOME/MLOps_2026/scripts/plotting/plot_eda.py
