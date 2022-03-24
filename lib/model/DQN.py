import torch
import torch.nn as nn


def _get_normalization(name):
    if name.lower() == 'layernorm':
        norm = nn.LayerNorm
    elif name.lower() == 'batchnorm':
        norm = nn.BatchNorm1d
    else:
        raise ValueError(f'Normalization {name} is not yet supported.')
    return norm


def _get_activation(name):
    if name.lower() == 'silu':
        act = nn.SiLU()
    elif name.lower() == 'relu':
        act = nn.ReLU()
    elif name.lower() == 'leakyrelu':
        act = nn.LeakyReLU()
    else:
        raise ValueError(f'Activation {name} is not yet supported.')
    return act


class LinearBlock(nn.Module):
    def __init__(self, config, in_dim, out_dim):
        super().__init__()
        normalization = _get_normalization(config.model.normalization)
        activation = _get_activation(config.model.activation)
        self.main = nn.Sequential(
            normalization(in_dim),
            activation,
            nn.Linear(in_dim, in_dim),
            normalization(in_dim),
            activation,
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, is_residual: bool = False):
        if is_residual:
            return self.main(x) + x
        else:
            return self.main(x)
        

class DQN(nn.Module):
    def __init__(self, config, final_dim):
        super().__init__()
        n_layers = config.model.n_layers
        nf = config.model.nf

        self.main = nn.ModuleList([nn.Linear(6, nf)]) # 6 : dim of states

        for i in range(n_layers - 1):
            if i < n_layers - 2:
                in_dim = out_dim = nf
            else:
                in_dim = nf
                out_dim = final_dim

            # Append a linear block
            self.main.append(
                LinearBlock(config, in_dim, out_dim)
            )
        
    def forward(self, x: torch.Tensor):
        out = self.main[0](torch.log(x + 1e-10))
        for i in range(1, len(self.main)):
            if i < len(self.main) - 1:
                out = self.main[i](out, is_residual=True)
            else:
                out = self.main[i](out, is_residual=False)
        return out


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
