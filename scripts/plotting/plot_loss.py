from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


def main():
    tracker_path = PROJECT_ROOT / "trainer_tracker.pt"
    tracker = torch.load(tracker_path)

    train_loss_per_epoch = tracker["train_loss"][:3]
    val_loss_per_epoch = tracker["val_loss"][:3]

    epochs = [1, 2, 3]

    plt.figure(figsize=(8, 5))

    # Training loss per epoch
    plt.plot(
        epochs,
        train_loss_per_epoch,
        label="Training loss per epoch",
    )

    plt.plot(
        epochs,
        val_loss_per_epoch,
        label="Validation loss per epoch",
    )

    plt.xticks(epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss of the first 3 epochs")
    plt.legend()
    plt.tight_layout()

    output_dir = PROJECT_ROOT / "assets" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "training_validation_loss.png")
    plt.close()


if __name__ == "__main__":
    main()