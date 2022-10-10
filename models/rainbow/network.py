import torch
import torch.nn as nn
import torch.nn.functional as F
from .noisy_layer import NoisyLinear


class RainbowDQN(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        nf: int,
        support: torch.Tensor,
        dropout: float = 0.,
    ):
        '''Initialization.'''
        super().__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # Set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, nf), 
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(nf, nf), 
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )
        
        # Set advantage layer
        self.adv_layer = nn.Sequential(
            NoisyLinear(nf, nf),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            NoisyLinear(nf, out_dim * atom_size),
        )

        # Set value layer
        self.val_layer = nn.Sequential(
            NoisyLinear(nf, nf),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            NoisyLinear(nf, atom_size),
        )

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Forward method implementation.'''
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        '''Get distribution for atoms.'''
        h = self.feature_layer(x)
        adv = self.adv_layer(h).view(-1, self.out_dim, self.atom_size)
        val = self.val_layer(h).view(-1, 1, self.atom_size)
        q_atoms = val + adv - adv.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist

    def init_weights(self) -> None:
        '''Initialize all trainable parameters.'''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(1e-3)
            elif isinstance(m, NoisyLinear):
                m.reset_parameters()
                m.reset_noise()
    
    def reset_noise(self) -> None:
        '''Reset all noise in noisy layers.'''
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()