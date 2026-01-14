import h5py
import numpy as np
from pathlib import Path
import yaml

# Laad config om het pad te vinden
with open("experiments/configs/train_config.yaml", "r") as f: # Pas pad aan als je config anders heet
    config = yaml.safe_load(f)

base_path = Path(config["data"]["data_path"])
files = {
    "Train X": base_path / "camelyonpatch_level_2_split_train_x.h5",
    "Train Y": base_path / "camelyonpatch_level_2_split_train_y.h5",
    "Valid X": base_path / "camelyonpatch_level_2_split_valid_x.h5",
}

print(f"{'File':<10} | {'Shape':<20} | {'Dtype':<10} | {'Size (GB)':<10}")
print("-" * 60)

for name, p in files.items():
    if p.exists():
        with h5py.File(p, "r") as f:
            # We nemen aan dat de dataset keys 'x' of 'y' heten
            key = 'x' if 'x' in str(p) else 'y'
            dset = f[key]
            
            # Bereken grootte in GB
            size_gb = p.stat().st_size / (1024**3)
            
            print(f"{name:<10} | {str(dset.shape):<20} | {str(dset.dtype):<10} | {size_gb:.2f}")
    else:
        print(f"{name:<10} | NOT FOUND")