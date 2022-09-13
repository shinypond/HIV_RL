from functools import partial
import numpy as np
from scipy.integrate import solve_ivp
import torch
from numba import njit
from .constants import *


@njit(cache=True)
def ode_ftn(t, x, B):
    x = x.reshape(B, -1)
    dx = np.zeros_like(x)
    dx[:, 0] = lmbd1 - d1 * x[:, 0] - (1 - x[:, 6]) * k1 * x[:, 4] * x[:, 0]
    dx[:, 1] = lmbd2 - d2 * x[:, 1] - (1 - f * x[:, 6]) * k2 * x[:, 4] * x[:, 1]
    dx[:, 2] = (1 - x[:, 6]) * k1 * x[:, 4] * x[:, 0] - delta * x[:, 2] - m1 * x[:, 5] * x[:, 2]
    dx[:, 3] = (1 - f * x[:, 6]) * k2 * x[:, 4] * x[:, 1] - delta * x[:, 3] - m2 * x[:, 5] * x[:, 3]
    dx[:, 4] = (1 - x[:, 7]) * N_T * delta * (x[:, 2] + x[:, 3]) - c * x[:, 4] - \
        ((1 - x[:, 6]) * rho1 * k1 * x[:, 0] + (1 - f * x[:, 6]) * rho2 * k2 * x[:, 1]) * x[:, 4]
    _I = x[:, 2] + x[:, 3]
    _E_first = b_E * _I / (_I + K_b + 1e-16) * x[:, 5]
    _E_second = d_E * _I / (_I + K_d + 1e-16) * x[:, 5]
    dx[:, 5] = lmbd_E + _E_first - _E_second - delta_E * x[:, 5]
    # Original
    # dx[:, 8] = -(Q * x[:, 4] + R1 * x[:, 6] ** 2 + R2 * x[:, 7] ** 2 - S * x[:, 5])
    # Time-equipped (final weighting function is excluded here)
    dx[:, 8] = -(R1 * x[:, 6] ** 2 + R2 * x[:, 7] ** 2)
    dx = dx.reshape(-1)
    return dx


class HIV:
    def __init__(self, cfg):
        self.scaler = cfg.reward.scaler
        self.max_step = cfg.train.max_step

    def step(self, state, action, t):
        B = state.shape[0] # batch size
        x_state = state.clone()
        x_action = action.clone()
        x_reward = torch.zeros(B, 1)
        x = torch.cat([x_state, x_action, x_reward], dim=-1).numpy()
        x = x.reshape(-1)

        sol = solve_ivp(
            partial(ode_ftn, B=B), (0, 1), x, rtol=1e-4, atol=1e-4, method=solve_ode_method,
        )
        y = torch.tensor(sol.y[:, -1], dtype=torch.float32)
        y = y.reshape(B, -1)
        next_state = y[:, :6] 
        reward = y[:, -1]
        # Time-equipped (final weighting function)
        if t == self.max_step - 1:
            reward += -(Q * (y[:, 4] - V_TARGET) ** 2 + S * (y[:, 5] - E_TARGET) ** 2)
        is_done = torch.Tensor([t == self.max_step - 1]).repeat(reward.shape[0])

        return reward / self.scaler, next_state, is_done 

