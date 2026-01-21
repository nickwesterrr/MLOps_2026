import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training metrics from history.json"
    )
    parser.add_argument(
        "--input_file", type=Path, required=True, help="Path to history.json"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=None, help="Folder to save the plots"
    )
    return parser.parse_args()


def load_data(file_path: Path) -> Dict[str, Any]:
    """Loads the history data from JSON."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def setup_style():
    """Sets a professional plotting style."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def safe_plot(ax, data, key, label, color, marker=None, linestyle="-"):
    """
    Helper to safely plot data even if lengths mismatch between metrics.
    """
    if key in data and len(data[key]) > 0:
        y_values = data[key]
        # Generate X-axis based on the specific length of THIS metric
        x_values = range(1, len(y_values) + 1)
        sns.lineplot(
            x=x_values,
            y=y_values,
            ax=ax,
            label=label,
            color=color,
            marker=marker,
            linestyle=linestyle,
        )
        return True
    return False


def plot_metrics(data: Dict[str, Any], output_path: Optional[Path]):
    """
    Generate and save plots for Question 4 and Question 5.
    """
    if data is None:
        return

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Experiment Results: Q4 & Q5 Metrics", fontsize=16)

    # --- Plot 1: Internal Dynamics (Gradient Norms) - Q4 ---
    if "grad_norms" in data and len(data["grad_norms"]) > 0:
        grad_norms = data["grad_norms"]
        steps = range(len(grad_norms))
        sns.lineplot(
            x=steps, y=grad_norms, ax=axes[0, 0], color="purple", alpha=0.7, linewidth=1
        )
        axes[0, 0].set_title("Internal Dynamics: Gradient Norms (Q4)")
        axes[0, 0].set_xlabel("Logging Step (x10)")
        axes[0, 0].set_ylabel("L2 Norm")
        axes[0, 0].set_yscale("log")
    else:
        axes[0, 0].text(0.5, 0.5, "No Gradient Norm Data", ha="center")

    # --- Plot 2: Learning Rate Schedule - Q4 ---
    safe_plot(axes[0, 1], data, "learning_rates", "Learning Rate", "orange", marker="o")
    axes[0, 1].set_title("Learning Rate Schedule (Q4)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("LR")

    # --- Plot 3: Loss Curves ---
    # We plot these individually to handle potential length mismatches
    safe_plot(axes[1, 0], data, "train_loss", "Train Loss", "blue", marker="o")
    safe_plot(axes[1, 0], data, "val_loss", "Val Loss", "orange", marker="o")
    axes[1, 0].set_title("Loss Curves")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")

    # --- Plot 4: Validation Metrics (Q5) ---
    safe_plot(axes[1, 1], data, "val_f2_score", "F2-Score", "green", marker="s")
    safe_plot(axes[1, 1], data, "val_accuracy", "Accuracy", "blue", linestyle="--")
    safe_plot(axes[1, 1], data, "val_roc_auc", "ROC-AUC", "red", linestyle="--")

    axes[1, 1].set_title("Validation Metrics (Q5)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")

    plt.tight_layout()

    # Save logic
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        save_file = output_path / "metrics_summary.png"
        plt.savefig(save_file, dpi=300)
        print(f"Plots saved to: {save_file}")
    else:
        input_path = Path(parse_args().input_file)
        save_file = input_path.parent / "metrics_summary.png"
        plt.savefig(save_file, dpi=300)
        print(f"Plots saved to: {save_file}")


def main():
    args = parse_args()
    setup_style()

    try:
        data = load_data(args.input_file)
        plot_metrics(data, args.output_dir)
    except Exception as e:
        print(f"Error plotting metrics: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
