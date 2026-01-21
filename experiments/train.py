import argparse

import torch
import torch.nn as nn  # <--- Nodig voor CrossEntropyLoss
import torch.optim as optim

from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.solver import Trainer
from ml_core.utils import load_config, seed_everything
from ml_core.utils.tracker import ExperimentTracker


def main(args):
    # 1. Load Config
    config = load_config(args.config)

    # Override config with command-line args if provided
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.optimizer:
        config["training"]["optimizer"] = args.optimizer

    # Dynamically set experiment name based on hyperparameters
    lr_val = config["training"]["learning_rate"]
    bs_val = config["data"]["batch_size"]
    opt_val = config["training"]["optimizer"]
    config["experiment_name"] = f"sweep_lr{lr_val}_bs{bs_val}_opt{opt_val}"

    # --- REPRODUCIBILITY START ---
    seed = config.get("seed")
    seed_everything(seed)
    print(f"Global Random Seed set to: {seed}")
    # --- REPRODUCIBILITY END ---

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

    # 5. Optimizer (Dynamic Implementation for Q3)
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

    exp_name = config.get("experiment_name", "experiment")
    tracker = ExperimentTracker(experiment_name=exp_name, config=config)

    # 7. Trainer Setup <--- AANGEPAST: Loaders en criterion hier meegeven
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,  # Toegevoegd aan init
        val_loader=val_loader,  # Toegevoegd aan init
        criterion=criterion,  # Toegevoegd aan init
        config=config,
        device=device,
        tracker=tracker,
    )

    # 8. Start Training <--- AANGEPAST: Geen argumenten meer (zitten nu in self)
    trainer.fit()

    # 9. Save Results
    print(f"Training complete. Saving results in: {tracker.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")

    # Hyperparameter overrides for grid search
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument(
        "--optimizer", type=str, choices=["adam", "sgd"], help="Override optimizer"
    )

    args = parser.parse_args()
    main(args)
