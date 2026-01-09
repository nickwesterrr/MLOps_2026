from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.
    """

    def __init__(
            self,
            x_path: str,
            y_path: str,
            transform: Optional[Callable] = None,
            filter_data: bool = False,
    ):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform
        self.filter_data = filter_data

        # TODO: Initialize dataset
        # 1. Check if files exist
        # 2. Open h5 files in read mode
        if not self.x_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(
                f"PCAM files not found at {self.x_path} or {self.y_path}"
            )
        
        # Open in read mode (lazy loading with H5)
        self.x_data = h5py.File(self.x_path, "r")["x"]
        self.y_data = h5py.File(self.y_path, "r")["y"]

        if filter_data:
            self.indices = self._filer_indices()
        else:
            self.indices = list(range(len(self.x_data)))
    
    def _filer_indices(self) -> list:
        valid = []
        for i in range(len(self.x_data)):
            img = self.x_data[i]
            mean = img.mean()
            if mean > 0 and mean < 255:
                valid.append(i)
        return valid

    def __len__(self) -> int:
        # TODO: Return length of dataset
        # The dataloader will know hence how many batches to create
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement data retrieval
        # 1. Read data at idx
        # 2. Convert to uint8 (for PIL compatibility if using transforms)
        # 3. Apply transforms if they exist
        # 4. Return tensor image and label (as long)
        
        # Read specific index
        real_idx = self.indices[idx]
        image = self.x_data[real_idx]
        label = self.y_data[real_idx][0]

        # Ensure uint8 for PIL compatibility
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        image = np.clip(image, 0, 255).astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        
        # CrossEntropyLoss requires Long (int64)
        return image, torch.tensor(label, dtype=torch.long).squeeze()
