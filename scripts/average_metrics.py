import argparse
import json
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=Path, required=True, help="List of history.json files")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Store max/final values for each seed
    accuracies = []
    f2_scores = []
    roc_aucs = []

    print(f"{'File':<40} | {'Acc':<8} | {'F2':<8} | {'AUC':<8}")
    print("-" * 75)

    for f_path in args.files:
        with open(f_path, "r") as f:
            data = json.load(f)
        
        # We take the value from the LAST epoch (or you could take the max)
        # Let's use the max F2 score epoch as the "Best Model"
        best_f2_idx = np.argmax(data["val_f2_score"])
        
        acc = data["val_accuracy"][best_f2_idx]
        f2 = data["val_f2_score"][best_f2_idx]
        auc = data["val_roc_auc"][best_f2_idx]

        accuracies.append(acc)
        f2_scores.append(f2)
        roc_aucs.append(auc)

        print(f"{f_path.name:<40} | {acc:.4f}   | {f2:.4f}   | {auc:.4f}")

    print("-" * 75)
    print(f"{'AVERAGE':<40} | {np.mean(accuracies):.4f}   | {np.mean(f2_scores):.4f}   | {np.mean(roc_aucs):.4f}")
    print(f"{'STD DEV':<40} | {np.std(accuracies):.4f}   | {np.std(f2_scores):.4f}   | {np.std(roc_aucs):.4f}")

if __name__ == "__main__":
    main()