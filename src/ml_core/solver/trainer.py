cat <<EOF > src/ml_core/solver/trainer.py
from typing import Any, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

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
        self.criterion = nn.CrossEntropyLoss()
        self.tracker = {
            "train_loss": [],
            "val_loss": [],
            "train_loss_per_step": [],
        }

    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        log_after_steps = self.config["training"].get("log_after_steps", 10)

        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            if step % log_after_steps == 0:
                self.tracker["train_loss_per_step"].append(loss.item())
        
        return total_loss / n_batches if n_batches > 0 else 0.0

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch_idx: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.criterion(output, y)
            total_loss += loss.item()
            n_batches += 1
        return total_loss / n_batches if n_batches > 0 else 0.0

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        save_dir = Path(self.config["training"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config
        }
        torch.save(checkpoint, filename)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]
        print(f"Starting training for {epochs} epochs on {self.device}...")
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)
            self.tracker["train_loss"].append(train_loss)
            self.tracker["val_loss"].append(val_loss)
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.save_checkpoint(epoch, val_loss)