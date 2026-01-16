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

set -euo pipefail

# Always run from the directory you submitted from (repo root)
cd "${SLURM_SUBMIT_DIR}"

# 1) Load modules
module load 2025
module load Python/3.13.1-GCCcore-14.2.0

# 2) Block ~/.local site-packages
export PYTHONNOUSERSITE=1

# 3) Ensure your src/ is importable
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${PYTHONPATH:-}"

# 4) Use venv Python explicitly (DO NOT rely on `python` from modules)
VENV_PY="${SLURM_SUBMIT_DIR}/.venv/bin/python"

# 5) Debug
echo "PWD=$(pwd)"
echo "VENV_PY=${VENV_PY}"
"${VENV_PY}" -c "import sys; print('PY_EXE=', sys.executable)"
"${VENV_PY}" -c "import numpy as np; print('NUMPY=', np.__version__, 'FROM', np.__file__)"

# 6) Run training
"${VENV_PY}" experiments/train.py --config experiments/configs/train_config.yaml
