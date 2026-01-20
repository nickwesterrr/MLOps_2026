import csv
from pathlib import Path
from typing import Any, Dict

import yaml
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

        self.config_path = self.run_dir / "config.yaml"
        with self.config_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(config, config_file, sort_keys=False)

        self.writer = SummaryWriter(log_dir=str(self.run_dir))
        self.writer.add_text("config", yaml.safe_dump(config, sort_keys=False))

        # Initialize CSV
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["epoch"])
        self.csv_file.flush()
        self.metric_keys = None

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Writes metrics to CSV (and TensorBoard).
        """
        if self.metric_keys is None:
            self.metric_keys = sorted(metrics.keys())
            self.csv_writer.writerow(["epoch", *self.metric_keys])
            self.csv_file.flush()

        self.csv_writer.writerow([epoch, *[metrics.get(key) for key in self.metric_keys]])
        self.csv_file.flush()

        for name, value in metrics.items():
            if value is not None:
                self.writer.add_scalar(name, value, epoch)

    def get_checkpoint_path(self, filename: str) -> str:
        return str(self.run_dir / filename)

    def close(self):
        self.csv_file.close()
        self.writer.close()
