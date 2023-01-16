from typing import *
import copy
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
import gym
from gym import spaces
from gym.envs.registration import register
from .constants_v0 import *


MAX_DAYS = 600
INT_TREATMENT = 1
MAX_EPISODE_STEPS = MAX_DAYS // INT_TREATMENT


def make_HIV_env() -> gym.Env:
    register(
        id='hiv-v0',
        entry_point='envs.hiv_v0:HIV_Dynamics',
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    hiv_env = gym.make('hiv-v0')
    hiv_env.reset()
    return hiv_env


class HIV_Dynamics(gym.Env):
    '''Gym Environment with HIV Infection Dynamics'''
    metadata = {'render.modes': ['human']}
    def __init__(self) -> None:
        self.action_space = spaces.Discrete(4)
        self.controls = self.make_controls()
        self.observation_space = spaces.Box(
            low=0.,
            high=1.0e+10,
            shape=(6,),
            dtype=np.float32,
        )
        self.reset()

    def reset(self) -> Tuple[np.ndarray, dict]:
        super().reset()
        self.state = np.array(init_state, dtype=np.float32)
        self.time = 0
        return self.state, {}

    def make_controls(self) -> np.ndarray:
        eps = 1e-12
        a1 = np.arange(min_a1, max_a1 + eps, interval_a1)
        a2 = np.arange(min_a2, max_a2 + eps, interval_a2)
        x, y = np.meshgrid(a1, a2)
        controls = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1, dtype=np.float32)
        return controls
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        self.state, reward = ode_step(self.state, self.controls[action])
        reward = reward / reward_scaler
        self.time += 1
        done = True if self.time > self.spec.max_episode_steps else False
        return self.state, reward, done, False, {}


def ode_step(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    reward = np.zeros((1,), dtype=np.float32)
    x = np.concatenate([state, control, reward], axis=0) # shape: (9,)
    sol = solve_ivp(_ode_ftn, (0, INT_TREATMENT), x, t_eval=np.array([1.]), rtol=1e-5, atol=1e-5, method='RK45')
    y = sol.y[:, 0]
    assert x.shape == y.shape
    next_state = y[:6]
    reward = y[-1]  
    return next_state, reward


@njit(cache=True)
def _ode_ftn(t: float, x: np.ndarray) -> np.ndarray:
    dx = np.zeros((9,), dtype=np.float32)
    dx[0] = lmbd1 - d1 * x[0] - (1 - x[6]) * k1 * x[4] * x[0]
    dx[1] = lmbd2 - d2 * x[1] - (1 - f * x[6]) * k2 * x[4] * x[1]
    dx[2] = (1 - x[6]) * k1 * x[4] * x[0] - delta * x[2] - m1 * x[5] * x[2]
    dx[3] = (1 - f * x[6]) * k2 * x[4] * x[1] - delta * x[3] - m2 * x[5] * x[3]
    dx[4] = (1 - x[7]) * n_T * delta * (x[2] + x[3]) - c * x[4] - \
        ((1 - x[6]) * rho1 * k1 * x[0] + (1 - f * x[6]) * rho2 * k2 * x[1]) * x[4]
    _I = x[2] + x[3]
    _E_first = b_E * _I / (_I + K_b + 1e-16) * x[5]
    _E_second = d_E * _I / (_I + K_d + 1e-16) * x[5]
    dx[5] = lmbd_E + _E_first - _E_second - delta_E * x[5]
    dx[8] = -(Q * x[4] + R1 * x[6] ** 2 + R2 * x[7] ** 2 - S * x[5])
    return dx

