from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from ml_core.utils import load_config


def main():
    # Load config to get original number of epochs
    config_path = PROJECT_ROOT / "experiments" / "configs" / "train_config.yaml"
    config = load_config(str(config_path))
    original_epochs = config["training"]["epochs"]
    
    tracker_path = PROJECT_ROOT / "experiments" / "trainer_tracker.pt"
    tracker = torch.load(tracker_path)

    train_loss_per_step = tracker["train_loss_per_step"]
    train_loss_per_epoch = tracker["train_loss"][:3]
    val_loss_per_epoch = tracker["val_loss"][:3]

    plt.figure(figsize=(10, 5))

    # Training loss per epoch
    plt.plot(
        train_loss_per_step,
        label="Training loss per step",
        alpha=0.3,
    )

    # Validation loss per epoch
    steps_per_epoch = len(train_loss_per_step) // original_epochs
    epoch_steps =  []
    for i in range(3):  # Only plot first 3 epochs
        epoch_steps.append(i * steps_per_epoch)

    plt.plot(
        epoch_steps,
        val_loss_per_epoch,
        label="Validation loss per epoch",
    )

    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.title("Training and validation loss over time")
    plt.legend()
    plt.tight_layout()

    output_dir = PROJECT_ROOT / "assets" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "training_validation_loss.png")
    plt.close()


if __name__ == "__main__":
    main()