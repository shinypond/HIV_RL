import numpy as np
from scipy.integrate import solve_ivp
import torch


class HIV:
    def __init__(self, config):
        for p in config.dynamics.__dict__.keys():
            setattr(self, p, getattr(config.dynamics, p))
        for p in config.reward.__dict__.keys():
            setattr(self, p, getattr(config.reward, p))

    def step(self, state, action):
        B = state.shape[0] # batch size
        x_state = state.clone()
        x_action = action.clone()
        x_reward = torch.zeros(B, 1)
        x = torch.cat([x_state, x_action, x_reward], dim=-1).numpy()
        x = x.reshape(-1)

        def ode_ftn(t, x):
            x = x.reshape(B, -1)
            dx = np.zeros_like(x)
            dx[:, 0] = self.lmbd1 - self.d1 * x[:, 0] - (1 - x[:, 6]) * self.k1 * x[:, 4] * x[:, 0]
            dx[:, 1] = self.lmbd2 - self.d2 * x[:, 1] - (1 - self.f * x[:, 6]) * self.k2 * x[:, 4] * x[:, 1]
            dx[:, 2] = (1 - x[:, 6]) * self.k1 * x[:, 4] * x[:, 0] - self.delta * x[:, 2] - self.m1 * x[:, 5] * x[:, 2]
            dx[:, 3] = (1 - self.f * x[:, 6]) * self.k2 * x[:, 4] * x[:, 1] - self.delta * x[:, 3] - self.m2 * x[:, 5] * x[:, 3]
            dx[:, 4] = (1 - x[:, 7]) * self.N_T * self.delta * (x[:, 2] + x[:, 3]) - self.c * x[:, 4] - \
                ((1 - x[:, 6]) * self.rho1 * self.k1 * x[:, 0] + (1 - self.f * x[:, 6]) * self.rho2 * self.k2 * x[:, 1]) * x[:, 4]
            _I = x[:, 2] + x[:, 3]
            _E_first = self.b_E * _I / (_I + self.K_b + 1e-16) * x[:, 5]
            _E_second = self.d_E * _I / (_I + self.K_d + 1e-16) * x[:, 5]
            dx[:, 5] = self.lmbd_E + _E_first - _E_second - self.delta_E * x[:, 5]
            dx[:, 8] = -(self.Q * x[:, 4] + self.R1 * x[:, 6] ** 2 + self.R2 * x[:, 7] ** 2 - self.S * x[:, 5])
            dx = dx.reshape(-1)
            return dx

        sol = solve_ivp(
            ode_ftn, (0, 1), x, rtol=1e-5, atol=1e-5, method=self.method,
        )
        y = torch.tensor(sol.y[:, -1], dtype=torch.float32)
        y = y.reshape(B, -1)
        next_state = y[:, :6] 
        reward = y[:, -1]

        return reward / self.scaler, next_state


        # def ode_ftn(t, x):
        #     dx = np.zeros_like(x)
        #     dx[0] = self.lmbd1 - self.d1 * x[0] - (1 - x[6]) * self.k1 * x[4] * x[0]
        #     dx[1] = self.lmbd2 - self.d2 * x[1] - (1 - self.f * x[6]) * self.k2 * x[4] * x[1]
        #     dx[2] = (1 - x[6]) * self.k1 * x[4] * x[0] - self.delta * x[2] - self.m1 * x[5] * x[2]
        #     dx[3] = (1 - self.f * x[6]) * self.k2 * x[4] * x[1] - self.delta * x[3] - self.m2 * x[5] * x[3]
        #     dx[4] = (1 - x[7]) * self.N_T * self.delta * (x[2] + x[3]) - self.c * x[4] - \
        #         ((1 - x[6]) * self.rho1 * self.k1 * x[0] + (1 - self.f * x[6]) * self.rho2 * self.k2 * x[1]) * x[4]
        #     _I = x[2] + x[3]
        #     _E_first = self.b_E * _I / (_I + self.K_b + 1e-12) * x[5]
        #     _E_second = self.d_E * _I / (_I + self.K_d + 1e-12) * x[5]
        #     dx[5] = self.lmbd_E + _E_first - _E_second - self.delta_E * x[5]
        #     dx[8] = -(self.Q * x[4] + self.R1 * x[6] ** 2 + self.R2 * x[7] ** 2 - self.S * x[5])
        #     return dx