import csv
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

# TODO: Add TensorBoard Support
from torch.utils.tensorboard import SummaryWriter


class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        base_dir: str = "experiments/results",
    ):
        self.run_dir = Path(base_dir) / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Save config to yaml in run_dir
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        # Log git commit hash
        try:
            self.git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            )
        except Exception:
            self.git_hash = "Not a git repository"

        # Add Hash to Config
        config["git_commit"] = self.git_hash

        # Save Config to yaml
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

        # Log environment info
        with open(self.run_dir / "requirements.txt", "w") as f:
            f.write(
                subprocess.check_output(
                    [sys.executable, "-m", "pip", "freeze"]
                ).decode()
            )

        # Initialize TensorBoard Writer
        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        self.writer.add_text("Git Commit Hash", self.git_hash)
        self.writer.add_text("Config", str(config))

        # Initialize CSV
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # Header (TODO: add the rest of things we want to track, loss, gradients, accuracy etc.)
        self.metrics_keys = [
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_f2_score",
            "val_roc_auc",
            "grad_norm",
            "lr",
        ]
        self.csv_writer.writerow(["epoch"] + self.metrics_keys)

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Writes metrics to CSV (and TensorBoard).
        """
        # TODO: Write other useful metrics to CSV
        row = [epoch] + [metrics.get(k, 0.0) for k in self.metrics_keys]
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        # TODO: Log to TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, epoch)

    def get_checkpoint_path(self, filename: str) -> str:
        return str(self.run_dir / filename)

    def close(self, final_metrics: Dict[str, float] = None):
        if final_metrics:
            hparam_dict = {
                "hparam/lr": self.config["training"]["learning_rate"],
                "hparam/batch_size": self.config["data"]["batch_size"],
                "hparam/optimizer": self.config["training"]["optimizer"],
            }

            self.writer.add_hparams(hparam_dict, final_metrics)

        self.csv_file.close()
        self.writer.close()
