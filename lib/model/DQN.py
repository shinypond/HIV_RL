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


class DQN(nn.Module):
    def __init__(self, config, out_dim):
        super(DQN, self).__init__()
        n_layers = config.model.n_layers
        nf = config.model.nf

        modules = [nn.Linear(6, nf)]
        for i in range(n_layers - 1):
            normalization = _get_normalization(config.model.normalization)
            activation = _get_activation(config.model.activation)
            modules.append(normalization(nf))
            modules.append(activation)
            if i < n_layers - 2:
                modules.append(nn.Linear(nf, nf))
            else:
                modules.append(nn.Linear(nf, out_dim))

        self.main = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.main(torch.log(x + 1e-10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
