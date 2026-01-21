import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.utils import load_config
from pathlib import Path

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config("experiments/configs/train_config.yaml")
_, val_loader = get_dataloaders(config)

# 2. Load model (Champion: sweep_lr0.0001_bs32_optsgd)
model = MLP(
    input_shape=config["data"]["input_shape"], 
    hidden_units=config["model"]["hidden_units"], 
    num_classes=config["model"]["num_classes"]
).to(device)

checkpoint = torch.load("experiments/results/sweep_lr0.0001_bs32_optsgd/checkpoint_epoch_5.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_probs = []
all_labels = []

# 3. Collect predictions
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of 'Tumor'
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# 4. Calculate and plot PR-curve
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
pr_auc = auc(recall, precision)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR AUC = {pr_auc:.3f}')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve: Champion Model')
plt.legend(loc="lower left")
plt.grid(alpha=0.3)

# 5. Save
plot_dir = Path("assets/plots")
plot_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_dir / "pr_curve_champion.png", dpi=300, bbox_inches='tight')
print(f"PR-curve saved to: {plot_dir}/pr_curve_champion.png")