import sys
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

def main():
    # 1. Point to the JSON file from your latest run (run_2_hanna)
    # You can change 'run_2_hanna' to whichever folder you want to plot
    tracker_path = PROJECT_ROOT / "experiments" / "results" / "results_run_1_1.4" / "pcam_mlp_baseline_history.json"
    
    if not tracker_path.exists():
        print(f"Error: Could not find file at {tracker_path}")
        print("Check if the folder name 'run_2_hanna' matches your actual results folder.")
        return

    # 2. Load JSON instead of Torch
    with open(tracker_path, "r") as f:
        tracker = json.load(f)

    # 3. Extract data
    train_loss = tracker["train_loss"]
    val_loss = tracker["val_loss"]
    
    num_epochs = len(train_loss)
    epochs = range(1, num_epochs + 1)

    # 4. Plot
    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_loss, 'b-o', label="Training loss")
    plt.plot(epochs, val_loss, 'r-o', label="Validation loss")

    plt.xticks(epochs)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss ({num_epochs} epochs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 5. Save output
    output_dir = PROJECT_ROOT / "assets" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "training_validation_loss.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    main()