# MLOps 2026 – PCAM MLP (Reproducible Training & Inference)

This repository contains a small MLP pipeline for PatchCamelyon (PCAM) binary classification, with reproducible training and a pinned checkpoint for quick inference.

---

## Installation

### Option A (recommended): install from `pyproject.toml`
```bash
# From repo root
python -m venv .venv
source .venv/bin/activate

# Install package + dependencies
pip install -e .
```

### Option B: install from `requirements.txt`
```bash
# From repo root
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### Verify setup
```bash
pytest -q
```

---

## Data setup

The default training config expects the dataset at:
- `data_path: "./data/"` (see `experiments/configs/train_config.yaml`)

1) Create the folder:
```bash
mkdir -p data
```

2) Place (or symlink) the PCAM H5 files into `./data/` with the exact filenames below:
- `camelyonpatch_level_2_split_train_x.h5`
- `camelyonpatch_level_2_split_train_y.h5`
- `camelyonpatch_level_2_split_valid_x.h5`
- `camelyonpatch_level_2_split_valid_y.h5`

(Optional, if you later run explicit test evaluation)
- `camelyonpatch_level_2_split_test_x.h5`
- `camelyonpatch_level_2_split_test_y.h5`

---

## Training

Train the model using the provided configuration:

```bash
python experiments/train.py --config experiments/configs/train_config.yaml
```

Outputs (logs/checkpoints) are written under:
- `./experiments/results/`

---

## Expected performance

Champion checkpoint for Q9 is committed to:
- `checkpoints/best_model.pt`

### Inference proof (example)
Running inference on one validation batch (first 5 samples) produced:
- True labels: `[1 1 1 1 1]`
- Predictions: `[1 1 1 1 1]`
- Tumor probabilities: `[0.94037294 0.50822943 0.81582975 0.8192233  0.75301254]`
- Matches in first 5: `5/5`
- Runtime: ~13.6s


---

## Inference

Inference entrypoint:
- `scripts/inference.py`

Pinned checkpoint:
- `checkpoints/best_model.pt`

Run inference (loads the checkpoint, builds the model from the YAML config, and prints predictions/probabilities for a few validation samples):

```bash
python scripts/inference.py checkpoints/best_model.pt experiments/configs/train_config.yaml
```

---

## Offline handover (USB checklist)

If a teammate needs to run this on a cluster with no internet access, copy these files/folders:

Required:
- `checkpoints/best_model.pt` (pinned checkpoint)
- `scripts/inference.py` (inference entrypoint)
- `experiments/configs/train_config.yaml` (exact configuration used)
- `src/ml_core/` (all Python source code for the `ml_core` package)
- `pyproject.toml` and/or `requirements.txt` (dependency list)
- `data/` containing the PCAM `.h5` files with the exact filenames listed above

Recommended (for full reproducibility / retraining evidence):
- `experiments/train.py` (training entrypoint)
- Any logs/checkpoints you want to preserve under `experiments/results/`

If there is no internet access, also bring one of:
- A prebuilt environment (e.g., a copied `.venv` if compatible with the target system), or
- Offline Python wheels for all dependencies (a “wheelhouse”), matching the target Python version/OS.

---

## 📂 Project Structure

```text
.
├── src/ml_core/          # The Source Code (Library)
│   ├── data/             # Data loaders and transformations
│   ├── models/           # PyTorch model architectures
│   ├── solver/           # Trainer class and loops
│   └── utils/            # Loggers and experiment trackers
├── experiments/          # The Laboratory
│   ├── configs/          # YAML files for hyperparameters
│   ├── results/          # Checkpoints and logs (Auto-generated)
│   └── train.py          # Entry point for training
├── scripts/              # Helper scripts (plotting, etc)
├── tests/                # Unit tests for QA
├── pyproject.toml        # Config for Tools (Ruff, Pytest)
└── setup.py              # Package installation script
```
