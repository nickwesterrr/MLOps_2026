import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim

from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.solver import Trainer
from ml_core.utils import load_config, seed_everything


def main(args):
    # 1) Load config
    config = load_config(args.config)

    # 2) Reproducibility
    seed = config.get("seed", 42)
    seed_everything(seed)
    print(f"Global Random Seed set to: {seed}")

    # 3) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4) Data
    train_loader, val_loader = get_dataloaders(config)

    # 5) Model
    model = MLP(
        input_shape=config["data"]["input_shape"],
        hidden_units=config["model"]["hidden_units"],
        dropout_rate=config["model"]["dropout_rate"],
        num_classes=config["model"]["num_classes"],
    )

    # 6) Optimizer
    lr = config["training"]["learning_rate"]
    opt_name = config["training"].get("optimizer", "sgd").lower()

    if opt_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type in config: {opt_name}")

    print(f"Initialized optimizer: {opt_name} with lr: {lr}")

    # 7) Trainer (match src/ml_core/solver/trainer.py signature)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=str(device),
    )

    # 8) Fit expects loaders as args (per jouw Trainer.fit(train_loader, val_loader))
    trainer.fit(train_loader, val_loader)

    # 9) Save outputs
    save_dir = Path(config["training"].get("save_dir", "experiments/results"))
    save_dir.mkdir(parents=True, exist_ok=True)

    exp_name = config.get("experiment_name", "experiment")

    # 9a) Save tracker as .pt
    pt_path = save_dir / f"{exp_name}_tracker.pt"
    torch.save(trainer.tracker, pt_path)
    print(f"Saved experiment results to: {pt_path}")

    # 9b) Save history as .json (exactly what the assignment expects)
    # Ensure JSON-serializable floats
    history = {
        "train_loss": [float(x) for x in trainer.tracker.get("train_loss", [])],
        "val_loss": [float(x) for x in trainer.tracker.get("val_loss", [])],
    }

    json_path = save_dir / f"{exp_name}_history.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved history to: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    main(args)
