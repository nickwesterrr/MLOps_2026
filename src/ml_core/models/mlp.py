from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        hidden_units: List[int],
        num_classes: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        # TODO: Build the MLP architecture
        # If you are up to the task, explore other architectures or model types
        # Hint: Flatten -> [Linear -> ReLU -> Dropout] * N_layers -> Linear
        input_dim = 1
        for d in input_shape:
            input_dim *= d

        hidden_dim1, hidden_dim2 = hidden_units

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass
        return self.net(x)
