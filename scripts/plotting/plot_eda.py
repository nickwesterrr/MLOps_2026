import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from ml_core.data.loader import get_dataloaders  # noqa: E402
from ml_core.utils import load_config  # noqa: E402


def main():
    config_path = PROJECT_ROOT / "experiments" / "configs" / "train_config.yaml"
    config = load_config(config_path)

    train_loader, _ = get_dataloaders(config)

    output_dir = PROJECT_ROOT / "assets" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # CLass Balance
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

    plt.savefig(output_dir / "pcam_class_balance.png")
    plt.close()

    # Imaage Mean Intensity
    image_means = []

    for x, _ in train_loader:
        x_denorm = (x * 0.5) + 0.5
        batch_means = x_denorm.mean(dim=[1, 2, 3]).numpy()
        image_means.extend(batch_means)

    image_means = np.array(image_means) * 255

    plt.figure(figsize=(7, 5))
    plt.hist(
        image_means,
        bins=100,
        color="teal",
        edgecolor="black",
    )

    plt.axvline(0, color="red", linestyle="--", label="Black threshold")
    plt.axvline(255, color="red", linestyle="--", label="White threshold")

    plt.xlabel("Mean Pixel Value (0-255)")
    plt.ylabel("Count")
    plt.title("Distribution of Image Mean Intensities (All Images)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pcam_image_mean_intensity.png")
    plt.close()


if __name__ == "__main__":
    main()
