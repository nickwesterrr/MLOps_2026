from typing import Any, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.device = device
        
        # --- Q4: Tracker ---
        self.tracker = {
            "train_loss": [],
            "val_loss": [],
            "train_loss_per_step": [],
            "grad_norms": [],
            "learning_rates": []
        }

        # --- Q4: Scheduler ---
        self.scheduler = None
        scheduler_type = self.config["training"].get("scheduler", None)
        if scheduler_type == "step":
            step_size = self.config["training"].get("step_size", 5)
            gamma = self.config["training"].get("gamma", 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )

    def train_epoch(self, epoch_idx: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        log_after_steps = self.config["training"].get("log_after_steps", 10)

        for step, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()

            # --- Q4: Gradient Norm Logic ---
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if step % log_after_steps == 0:
                self.tracker["train_loss_per_step"].append(loss.item())
                self.tracker["grad_norms"].append(total_norm)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches if n_batches > 0 else 0.0

    @torch.no_grad()
    def validate(self, epoch_idx: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for x, y in self.val_loader:
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

    def save_history(self) -> None:
        save_dir = Path(self.config["training"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "history.json", "w") as f:
            json.dump(self.tracker, f, indent=4)

    def fit(self) -> None:
        epochs = self.config["training"]["epochs"]
        print(f"Starting training for {epochs} epochs on {self.device}...")
        
        for epoch in range(epochs):
            # --- Q4: Log LR ---
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.tracker["learning_rates"].append(current_lr)
            
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            # --- Q4: Step Scheduler ---
            if self.scheduler:
                self.scheduler.step()

            self.tracker["train_loss"].append(train_loss)
            self.tracker["val_loss"].append(val_loss)
            
            print(f"Epoch [{epoch+1}/{epochs}] - LR: {current_lr:.6f} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.save_checkpoint(epoch, val_loss)
        
        self.save_history()
