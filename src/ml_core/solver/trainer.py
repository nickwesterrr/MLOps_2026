from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device

        # TODO: Define Loss Function (Criterion)
        self.criterion = nn.CrossEntropyLoss()

        # TODO: Initialize ExperimentTracker
        self.tracker = {
            "train_loss": [],
            "val_loss": [],
            "train_loss_per_step": [],
        }

        # TODO: Initialize metric calculation (like accuracy/f1-score) if needed

    def train_epoch(
        self, dataloader: DataLoader, epoch_idx: int
    ) -> float:
        self.model.train()

        # TODO: Implement Training Loop
        # 1. Iterate over dataloader
        # 2. Move data to device
        # 3. Forward pass, Calculate Loss
        # 4. Backward pass, Optimizer step
        # 5. Track metrics (Loss, Accuracy, F1)

        total_loss = 0.0
        n_batches = 0

        log_after_steps = self.config["training"]["log_after_steps"]

        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(x)
            loss = self.criterion(output, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (step) % log_after_steps == 0:
                self.tracker["train_loss_per_step"].append(loss.item())
            
        return total_loss / n_batches

    @torch.no_grad()
    def validate(
        self, dataloader: DataLoader, epoch_idx: int
    ) -> float:
        self.model.eval()

        # TODO: Implement Validation Loop
        # Remember: No gradients needed here
        total_loss = 0.0
        n_batches = 0

        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            output = self.model(x)
            loss = self.criterion(output, y)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        # TODO: Save model state, optimizer state, and config
        pass

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # TODO: Call train_epoch and validate
            # TODO: Log metrics to tracker
            # TODO: Save checkpoints
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)

            self.tracker["train_loss"].append(train_loss)
            self.tracker["val_loss"].append(val_loss)

            print(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )


# Remember to handle the trackers properly
