from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from ml_core.data.loader import get_dataloaders


def main():
    config_path = PROJECT_ROOT / "experiments" / "configs" / "train_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_loader, _ = get_dataloaders(config)

    labels_seen = []

    for _, y in train_loader:
        labels_seen.extend(y.numpy())

    labels_seen = np.array(labels_seen, dtype=int)
    counts = np.bincount(labels_seen)

    plt.figure(figsize=(7, 5))
    plt.bar(
        ["Normal (0)", "Tumor (1)"],
        counts,
        color=["purple", "gold"],
    )
    plt.ylabel("Count")
    plt.title("Class Distribution of the PCAM Training Set")
    plt.tight_layout()

    output_dir = PROJECT_ROOT / "assets" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / "pcam_class_balance.png")
    plt.close()


if __name__ == "__main__":
    main()
