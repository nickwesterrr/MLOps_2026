import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Learning Rate Schedule")
    parser.add_argument("--file", type=Path, required=True, help="Path to a history.json file")
    parser.add_argument("--output", type=Path, default="assets/plots/q4_learning_rate.png", help="Output path")
    return parser.parse_args()

def main():
    args = parse_args()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    if not args.file.exists():
        print(f"Error: File {args.file} not found.")
        return

    with open(args.file, "r") as f:
        data = json.load(f)

    if "learning_rates" not in data:
        print("Error: 'learning_rates' key missing in JSON.")
        return

    lrs = data["learning_rates"]
    epochs = range(1, len(lrs) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lrs, marker='o', linestyle='-', color='orange', linewidth=2)
    plt.title("Learning Rate Schedule (StepLR)")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale("log")  # Log scale makes the steps look cleaner
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved to: {args.output}")

if __name__ == "__main__":
    main()