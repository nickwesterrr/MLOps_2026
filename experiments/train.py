import argparse

import torch
import torch.optim as optim

from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.solver import Trainer
from ml_core.utils import load_config

# logger = setup_logger("Experiment_Runner")


def main(args):
    # 1. Load Config & Set Seed
    config = load_config(args.config)

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
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    # 6. Trainer & Fit
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
    )
    
    trainer.fit(train_loader, val_loader)

    torch.save(trainer.tracker, "trainer_tracker.pt")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    main(args)