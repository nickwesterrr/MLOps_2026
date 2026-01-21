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
        
        # FIX: We openen de files hier NIET meer. We zetten ze op None.
        self.x_data = None
        self.y_data = None

        if not self.x_path.exists() or not self.y_path.exists():
            raise FileNotFoundError(f"Files not found: {self.x_path}, {self.y_path}")

        # Voor filtering moeten we de file even tijdelijk openen (alleen in main process)
        # Of we vertrouwen erop dat de indices gewoon 0..N zijn als we niet filteren.
        with h5py.File(self.x_path, "r") as f:
             self.dataset_len = len(f["x"])
        
        # Als filter_data True is, moeten we dit slimmer doen, 
        # maar voor nu gaan we ervan uit dat we alle data gebruiken.
        # (Filtering is traag om vooraf te doen op 150k images zonder index file)
        self.indices = list(range(self.dataset_len))
        if filter_data:
            kept = []
            with h5py.File(self.x_path, "r") as f:
                x = f["x"]
                for i in range(self.dataset_len):
                    img = x[i]
                    m = float(np.nan_to_num(img, nan=0.0).mean())

                    is_black = m <= 1.0
                    is_white = m >= 254.0

                    if not (is_black or is_white):
                        kept.append(i)

            self.indices = kept

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # FIX: Lazy loading. Pas openen als de worker de data nodig heeft.
        if self.x_data is None:
            self.x_data = h5py.File(self.x_path, "r")["x"]
            self.y_data = h5py.File(self.y_path, "r")["y"]

        # Handle float32 vs uint8 issue (Poisoning fix)
        # Nan_to_num is soms nodig als er corruptie is, anders is clip genoeg
        real_idx = self.indices[idx]
        image = self.x_data[real_idx]
        label = self.y_data[real_idx][0]

        # FIX: Haal NaNs weg VOORDAT je clipt
        image = np.nan_to_num(image, nan=0.0)  # <--- Voeg deze regel toe
        image = np.clip(image, 0, 255).astype(np.uint8)

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long).squeeze()