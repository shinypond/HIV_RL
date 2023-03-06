from typing import *
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
import gym
from gym import spaces
from gym.envs.registration import register
from .constants_v1 import *


def make_HIV_env(**kwargs) -> gym.Env:
    register(
        id='hiv-v1',
        entry_point='envs.hiv_v1:HIV_Dynamics',
        kwargs=kwargs,
    )
    hiv_env = gym.make('hiv-v1')
    hiv_env.reset()
    return hiv_env


class HIV_Dynamics(gym.Env):
    '''Gym Environment with HIV Infection Dynamics (with log10)'''
    metadata = {'render.modes': ['human']}
    def __init__(
        self,
        max_days: int,
        treatment_days: int,
        reward_scaler: float,
        init_state: Optional[np.ndarray] = None,
        is_test: bool = False,
    ) -> None:
        self.max_days = max_days
        self.treatment_days = treatment_days
        self.max_episode_steps = max_days // treatment_days
        self.reward_scaler = reward_scaler
        if init_state is not None:
            self.init_state = init_state
        else:
            self.init_state = INIT_STATE
        self.is_test = is_test

        self.t_interval = (0, treatment_days)
        self.t_eval = np.array([treatment_days,])

        self.action_space = spaces.Discrete(4)
        self.controls = self.make_controls()
        self.observation_space = spaces.Box(
            low=-20.,
            high=20,
            shape=(6,),
            dtype=np.float32,
        )
        self.reset()

    def reset(self) -> Tuple[np.ndarray, dict]:
        super().reset()
        self.state = self.init_state
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
        self.state, reward, intermediate_sol = self.ode_step(self.state, self.controls[action])
        reward = reward / self.reward_scaler
        self.time += 1
        done = True if self.time > self.max_episode_steps else False
        return self.state, reward, done, False, {'intermediate_sol': intermediate_sol}

    def ode_step(self, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        reward = np.zeros((1,), dtype=np.float32)
        x = np.concatenate([state, control, reward], axis=0) # shape: (9,)
        sol = solve_ivp(_ode_ftn, self.t_interval, x, t_eval=self.t_eval, rtol=1e-6, atol=1e-6, method='RK45')
        y = sol.y[:, -1]
        if self.is_test and len(self.t_eval) > 1:
            intermediate_sol = sol.y[:, :-1]
        else:
            intermediate_sol = None
        assert x.shape == y.shape
        next_state = y[:6]
        reward = y[-1]  
        return next_state, reward, intermediate_sol


log_10 = np.log(10)
@njit(cache=True)
def _ode_ftn(t: float, x: np.ndarray) -> np.ndarray:
    dx = np.zeros((9,), dtype=np.float32)
    _x0 = 10 ** x[0]
    _x1 = 10 ** x[1]
    _x2 = 10 ** x[2]
    _x3 = 10 ** x[3]
    _x4 = 10 ** x[4]
    _x5 = 10 ** x[5]

    dx[0] = lmbd1 - d1 * _x0 - (1 - x[6]) * k1 * _x4 * _x0 
    dx[0] *= 10 ** (-x[0]) / log_10
    dx[1] = lmbd2 - d2 * _x1 - (1 - f * x[6]) * k2 * _x4 * _x1
    dx[1] *= 10 ** (-x[1]) / log_10
    dx[2] = (1 - x[6]) * k1 * _x4 * _x0 - delta * _x2 - m1 * _x5 * _x2
    dx[2] *= 10 ** (-x[2]) / log_10
    dx[3] = (1 - f * x[6]) * k2 * _x4 * _x1 - delta * _x3 - m2 * _x5 * _x3
    dx[3] *= 10 ** (-x[3]) / log_10
    dx[4] = (1 - x[7]) * n_T * delta * (_x2 + _x3) - c * _x4 - \
        ((1 - x[6]) * rho1 * k1 * _x0 + (1 - f * x[6]) * rho2 * k2 * _x1) * _x4
    dx[4] *= 10 ** (-x[4]) / log_10
    _I = _x2 + _x3
    _E_first = b_E * _I / (_I + K_b + 1e-16) * _x5 
    _E_second = d_E * _I / (_I + K_d + 1e-16) * _x5
    dx[5] = lmbd_E + _E_first - _E_second - delta_E * _x5
    dx[5] *= 10 ** (-x[5]) / log_10
    dx[8] = -(Q * _x4 + R1 * x[6] ** 2 + R2 * x[7] ** 2 - S * _x5)
    return dx

