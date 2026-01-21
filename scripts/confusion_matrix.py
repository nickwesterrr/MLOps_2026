from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.utils import load_config

# 1. Setup and Load Champion Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config("experiments/configs/train_config.yaml")

# 2. Get Validation Data
_, val_loader = get_dataloaders(config)

# 3. Initialize Champion Model
model = MLP(
    input_shape=config["data"]["input_shape"],
    hidden_units=config["model"]["hidden_units"],
    num_classes=config["model"]["num_classes"],
).to(device)

# 4. Load the Best Checkpoint
checkpoint_path = "experiments/results/sweep_lr0.0001_bs32_optsgd/checkpoint_epoch_5.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 5. Collect Predictions and Labels
all_preds = []
all_labels = []

with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# 6. Generate and Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Tumor"],
    yticklabels=["Normal", "Tumor"],
)
plt.title("Confusion Matrix: Champion Model")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")

# Create directory if it doesn't exist
plot_dir = Path("assets/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# Save the plot
save_path = plot_dir / "confusion_matrix_champion.png"
plt.savefig(save_path, bbox_inches="tight", dpi=300)
print(f"Confusion Matrix saved to: {save_path}")

# Close the plot
plt.close()
