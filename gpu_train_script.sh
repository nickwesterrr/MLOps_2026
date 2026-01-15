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
source .venv/bin/activate

# ======================================================
# STAP 1: COPY DATA TO LOCAL SCRATCH ($TMPDIR)
# ======================================================
echo "------------------------------------------------"
echo "Starting data copy to Local Scratch ($TMPDIR)..."

# 1. Create the directory on the fast local SSD
mkdir -p "$TMPDIR/pcam"

# 2. Copy the files from the slow network drive to the fast SSD
# This might take 30-60 seconds, but saves HOURS of training time.
cp /scratch-shared/scur2282/data/*.h5 "$TMPDIR/pcam/"

echo "Data copy finished. Files in $TMPDIR/pcam:"
ls -lh "$TMPDIR/pcam"
echo "------------------------------------------------"
# ======================================================

# 3. Debug info
echo "Running on host: $(hostname)"
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 4. Run Python
# The python script will detect TMPDIR automatically (see Step 2)
python experiments/train.py --config experiments/configs/train_config.yaml