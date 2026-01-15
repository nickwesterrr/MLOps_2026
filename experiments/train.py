import argparse
from pathlib import Path
import torch
import os
import torch.nn as nn  # <--- Nodig voor CrossEntropyLoss
import torch.optim as optim

from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.solver import Trainer
from ml_core.utils import load_config, seed_everything

def main(args):
    # 1. Load Config
    config = load_config(args.config)

    # --- REPRODUCIBILITY START ---
    seed = config.get("seed", 42)
    seed_everything(seed)
    print(f"Global Random Seed set to: {seed}")
    # --- REPRODUCIBILITY END ---

    # ======================================================
    # STAP 2: AUTOMATICALLY USE LOCAL SCRATCH IF AVAILABLE
    # ======================================================
    if "TMPDIR" in os.environ:
        # Construct the path to where the .sh script copied the data
        local_path = Path(os.environ["TMPDIR"]) / "pcam"
        
        # Verify the data is actually there
        if local_path.exists() and any(local_path.iterdir()):
            print(f"🚀 SPEED BOOST: Switching data path to Local Scratch: {local_path}")
            # Override the slow path from config.yaml with the fast local path
            config["data"]["data_path"] = str(local_path)
        else:
            print("⚠️ WARNING: TMPDIR detected but no data found. Did the copy in .sh fail?")
    # ======================================================

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Data
    train_loader, val_loader = get_dataloaders(config)

    # 4. Model
    model = MLP(
        input_shape=config["data"]["input_shape"],
        hidden_units=config["model"]["hidden_units"],
        dropout_rate=config["model"]["dropout_rate"],
        num_classes=config["model"]["num_classes"],
    )

    # 5. Optimizer
    lr = config["training"]["learning_rate"]
    opt_name = config["training"].get("optimizer", "sgd").lower()

    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type in config: {opt_name}")

    print(f"Initialized optimizer: {opt_name} with lr: {lr}")

    # 6. Loss Function (Criterion) <--- NIEUW: Toegevoegd
    criterion = nn.CrossEntropyLoss()

    # 7. Trainer Setup <--- AANGEPAST: Loaders en criterion hier meegeven
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader, # Toegevoegd aan init
        val_loader=val_loader,     # Toegevoegd aan init
        criterion=criterion,       # Toegevoegd aan init
        config=config,
        device=device,
    )

    # 8. Start Training <--- AANGEPAST: Geen argumenten meer (zitten nu in self)
    # Als dit nog steeds fout gaat, check dan src/ml_core/solver.py hoe fit() eruit ziet.
    trainer.fit()

    # 9. Save Results
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    exp_name = config.get("experiment_name", "experiment")
    save_path = save_dir / f"{exp_name}_tracker.pt"

    torch.save(trainer.tracker, save_path)
    print(f"Saved experiment results to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    main(args)