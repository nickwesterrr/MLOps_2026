import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, fbeta_score

from ml_core.models import MLP
from ml_core.data import get_dataloaders
from ml_core.utils import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Run Error Analysis and Slicing")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best model checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=Path, default="assets/plots", help="Where to save images")
    return parser.parse_args()

def unnormalize(tensor):
    """Helper to convert tensor back to plot-able image."""
    # Assuming data is [0, 1] float. If normalized with mean/std, inverse it here.
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img, 0, 1)

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Model & Data
    config = load_config(str(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MLP(
        input_shape=config["data"]["input_shape"],
        hidden_units=config["model"]["hidden_units"],
        dropout_rate=config["model"]["dropout_rate"],
        num_classes=config["model"]["num_classes"],
    ).to(device)
    
    # Load weights (weights_only=False to allow config loading if needed)
    print(f"Loading model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    _, val_loader = get_dataloaders(config)

    # 2. Inference Loop (Collect Data)
    all_images = []
    all_labels = []
    all_preds = []
    all_intensities = [] # For Slicing

    print("Running inference on validation set...")
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            
            # Store data for analysis
            # We move to CPU immediately to save GPU memory
            all_images.append(x.cpu())
            all_labels.append(y.cpu())
            all_preds.append(preds.cpu())
            
            # Calculate mean intensity per image (Slice Metric)
            # Shape: [Batch, 3, 96, 96] -> mean over [1,2,3] -> [Batch]
            intensities = x.mean(dim=(1, 2, 3)).cpu()
            all_intensities.append(intensities)

    # Concatenate all batches
    images = torch.cat(all_images)
    labels = torch.cat(all_labels).numpy()
    preds = torch.cat(all_preds).numpy()
    intensities = torch.cat(all_intensities).numpy()

    print(f"Total samples analyzed: {len(labels)}")

    # ---------------------------------------------------------
    # PART A: QUALITATIVE ANALYSIS (Visualizing Errors)
    # ---------------------------------------------------------
    false_positives = np.where((preds == 1) & (labels == 0))[0]
    false_negatives = np.where((preds == 0) & (labels == 1))[0]
    
    print(f"Found {len(false_positives)} False Positives")
    print(f"Found {len(false_negatives)} False Negatives")

    def plot_errors(indices, title, filename):
        if len(indices) == 0:
            print(f"No {title} found to plot.")
            return
            
        # Pick top 5 (or fewer if less exist)
        n_plot = min(5, len(indices))
        indices_to_plot = indices[:n_plot] # Take first 5 found
        
        fig, axes = plt.subplots(1, n_plot, figsize=(15, 4))
        if n_plot == 1: axes = [axes]
        
        for i, idx in enumerate(indices_to_plot):
            img = unnormalize(images[idx])
            axes[i].imshow(img)
            axes[i].set_title(f"True: {labels[idx]} | Pred: {preds[idx]}\nIntensity: {intensities[idx]:.2f}")
            axes[i].axis("off")
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(args.output_dir / filename)
        print(f"Saved {filename}")

    plot_errors(false_positives, "False Positives (Predicted Tumor, Actually Healthy)", "q6_false_positives.png")
    plot_errors(false_negatives, "False Negatives (Predicted Healthy, Actually Tumor)", "q6_false_negatives.png")

    # ---------------------------------------------------------
    # PART B: QUANTITATIVE SLICING (Intensity Slice)
    # ---------------------------------------------------------
    # Slice Definition:
    # "Dark Slice" (Dense Tissue/Artifacts) = Intensity < 0.4
    # "Bright Slice" (Glass/Fat) = Intensity > 0.7
    
    # You can adjust these thresholds based on the histogram if needed
    threshold_dark = 0.35
    threshold_bright = 0.75
    
    dark_mask = intensities < threshold_dark
    bright_mask = intensities > threshold_bright
    
    print("\n--- Slicing Analysis ---")
    
    # Global Metrics
    global_acc = accuracy_score(labels, preds)
    global_f2 = fbeta_score(labels, preds, beta=2.0)
    print(f"Global Performance:  Acc: {global_acc:.4f} | F2: {global_f2:.4f}")

    # Dark Slice Metrics
    if np.sum(dark_mask) > 0:
        dark_acc = accuracy_score(labels[dark_mask], preds[dark_mask])
        dark_f2 = fbeta_score(labels[dark_mask], preds[dark_mask], beta=2.0)
        print(f"Dark Slice (N={np.sum(dark_mask)}):  Acc: {dark_acc:.4f} | F2: {dark_f2:.4f}")
    else:
        print("Dark Slice is empty (Try adjusting threshold)")

    # Bright Slice Metrics
    if np.sum(bright_mask) > 0:
        bright_acc = accuracy_score(labels[bright_mask], preds[bright_mask])
        bright_f2 = fbeta_score(labels[bright_mask], preds[bright_mask], beta=2.0)
        print(f"Bright Slice (N={np.sum(bright_mask)}): Acc: {bright_acc:.4f} | F2: {bright_f2:.4f}")
    else:
        print("Bright Slice is empty (Try adjusting threshold)")

    # Plot Histogram of intensities to justify slice
    plt.figure(figsize=(8, 4))
    sns.histplot(intensities, bins=50, kde=True)
    plt.axvline(threshold_dark, color='r', linestyle='--', label='Dark Threshold')
    plt.axvline(threshold_bright, color='g', linestyle='--', label='Bright Threshold')
    plt.title("Distribution of Image Intensities (Slice Definition)")
    plt.xlabel("Mean Pixel Intensity")
    plt.legend()
    plt.savefig(args.output_dir / "q6_slice_histogram.png")
    print("Saved q6_slice_histogram.png")

if __name__ == "__main__":
    main()