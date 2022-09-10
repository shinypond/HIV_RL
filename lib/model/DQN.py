import torch
import torch.nn as nn


def get_norm(name):
    if name.lower() == 'layernorm':
        norm = nn.LayerNorm
    elif name.lower() == 'batchnorm':
        norm = nn.BatchNorm1d
    else:
        raise ValueError(f'Normalization {name} is not yet supported.')
    return norm


class LinearBlock(nn.Module):
    def __init__(self, cfg, in_dim, out_dim):
        super().__init__()
        normalization = get_norm(cfg.model.normalization)
        self.main = nn.Sequential(
            normalization(in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, in_dim),
            normalization(in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, is_residual: bool = False):
        if is_residual:
            return x + self.main(x)
        else:
            return self.main(x)
        

class DQN(nn.Module):
    def __init__(self, cfg, final_dim):
        super().__init__()
        n_layers = cfg.model.n_layers
        nf = cfg.model.nf

        self.main = nn.ModuleList([nn.Linear(6, nf)]) # 6 : dim of states

        for i in range(n_layers - 1):
            if i < n_layers - 2:
                in_dim = out_dim = nf
            else:
                in_dim = nf
                out_dim = final_dim

            # Append a linear block
            self.main.append(
                LinearBlock(cfg, in_dim, out_dim)
            )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor):
        out = self.main[0](torch.log(x + 1e-10))
        for i in range(1, len(self.main)):
            if i < len(self.main) - 1:
                out = self.main[i](out, is_residual=True)
            else:
                out = self.main[i](out, is_residual=False)
        return out
