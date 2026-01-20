import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Gradient Norms across multiple runs.")
    parser.add_argument("--files", nargs="+", type=Path, required=True, 
                        help="List of history.json files to compare (e.g., run1/history.json run2/history.json)")
    parser.add_argument("--labels", nargs="+", type=str, default=None,
                        help="Labels for the legend (e.g. Seed 42, Seed 43, Seed 44). Must match number of files.")
    parser.add_argument("--output", type=Path, default="assets/plots/gradient_comparison.png",
                        help="Path to save the output image")
    parser.add_argument("--max_steps", type=int, default=None, 
                        help="Optional: Limit to first N steps (e.g. to zoom in on first few epochs)")
    return parser.parse_args()

def load_grad_norms(file_path):
    if not file_path.exists():
        print(f"Warning: File not found {file_path}")
        return None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return data.get("grad_norms", [])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    args = parse_args()
    sns.set_theme(style="whitegrid", context="paper")
    
    # Validate labels
    if args.labels and len(args.labels) != len(args.files):
        print("Error: Number of labels must match number of files.")
        return

    plt.figure(figsize=(12, 6))
    
    # Plot each file
    palette = sns.color_palette("bright", n_colors=len(args.files))
    
    for i, file_path in enumerate(args.files):
        norms = load_grad_norms(file_path)
        
        if norms and len(norms) > 0:
            if args.max_steps:
                norms = norms[:args.max_steps]
            
            label = args.labels[i] if args.labels else f"Run {i+1}"
            
            # Gebruik de kleur uit het palet en zet alpha iets lager voor overlap
            plt.plot(norms, label=label, color=palette[i], alpha=0.8, linewidth=1.2)

    plt.title("Internal Dynamics: Gradient Norm Comparison (Question 4.1)")
    plt.xlabel("Logging Step (x10)")
    plt.ylabel("Gradient Norm (L2)")
    plt.yscale("log")  # Log scale is usually better for gradients
    plt.legend()
    plt.tight_layout()
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300)
    print(f"Comparison plot saved to: {args.output}")

if __name__ == "__main__":
    main()