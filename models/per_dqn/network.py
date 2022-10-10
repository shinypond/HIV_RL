import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, in_dim: int, nf: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, nf), 
            nn.ReLU(),
            nn.Linear(nf, nf), 
            nn.ReLU(), 
            nn.Linear(nf, nf), 
            nn.ReLU(), 
            nn.Linear(nf, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)