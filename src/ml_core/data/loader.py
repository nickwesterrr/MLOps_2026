from pathlib import Path
from typing import Dict, Tuple
import random  # <--- Toegevoegd voor seeding

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .pcam import PCAMDataset

def seed_worker(worker_id):
"""
Ensures that each DataLoader worker has a deterministic random seed.
"""
worker_seed = torch.initial_seed() % 2**32
np.random.seed(worker_seed)
random.seed(worker_seed)

def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
"""
Factory function to create Train and Validation DataLoaders
using pre-split H5 files.
"""
data_cfg = config["data"]
base_path = Path(data_cfg["data_path"])

# --- REPRODUCIBILITY SETUP ---
# Haal de seed op uit de config (fallback naar 42 als hij mist)
seed = config.get("seed", 42)

# Maak een generator voor deterministische shuffling en sampling
g = torch.Generator()
g.manual_seed(seed)
# -----------------------------

# TODO: Define Transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

# TODO: Define Paths for X and Y (train and val)
train_x = base_path / "camelyonpatch_level_2_split_train_x.h5"
train_y = base_path / "camelyonpatch_level_2_split_train_y.h5"

val_x = base_path / "camelyonpatch_level_2_split_valid_x.h5"
val_y = base_path / "camelyonpatch_level_2_split_valid_y.h5"

# TODO: Instantiate PCAMDataset for train and val
train_ds = PCAMDataset(
    x_path=str(train_x),
    y_path=str(train_y),
    transform=train_transform
)

val_ds = PCAMDataset(
    x_path=str(val_x),
    y_path=str(val_y),
    transform=val_transform
)

# Add WeightedRandomSampler
with h5py.File(train_y, "r") as f:
    labels = f["y"][:].reshape(-1)
    labels = labels[train_ds.indices]  # Apply filtering if any

class_counts = np.bincount(labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[labels]

sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights).double(),
    num_samples=len(sample_weights),
    replacement=True,
    generator=g  # <--- Belangrijk: Generator meegeven voor determinisme
)

# TODO: Create DataLoaders
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=data_cfg["batch_size"],
    sampler=sampler,
    num_workers=data_cfg["num_workers"],
    worker_init_fn=seed_worker, # <--- Belangrijk: Worker seeds fixen
    generator=g                 # <--- Belangrijk: Shuffle volgorde fixen
)

val_loader = DataLoader(
    dataset=val_ds,
    batch_size=data_cfg["batch_size"],
    shuffle=False,
    num_workers=data_cfg["num_workers"],
    worker_init_fn=seed_worker, # <--- Ook voor val (zekerheid)
    generator=g
)

return train_loader, val_loader
