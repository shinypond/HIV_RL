from typing import *
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import ray
import gym
from envs.hiv_v1 import make_HIV_env
from configs import cfg


def plot_E_hist(num_episodes: int = 2000):
    os.makedirs('assets', exist_ok=True)
    env = make_HIV_env(is_test=True, **cfg['env'])
    max_steps = env.max_episode_steps
    immunes = []
    tasks = []
    start = datetime.now()
    for _ in range(num_episodes):
        tasks.append(_ray_dynamics.remote(env, max_steps))
    for task in tasks:
        result = ray.get(task)
        immunes.append(result)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(immunes, bins=100)
    ax.set_xlim(1.3, 5.7)
    ax.set_xticks(np.arange(1.5, 6.0, 0.5))
    ax.text(
        3.0, 10, 'Healthy Equilibrium', size=15,
        bbox={'facecolor': 'white', 'edgecolor': 'red', 'boxstyle': 'round', 'alpha': 1.0},
    )
    ax.arrow(
        3.5, 10, 2, -10,
        alpha=0.7, fc='black', ec=None, shape='full',
        width=0.1, head_width=0.2, head_length=0.2,
    )
    ax.set_xlabel(r'Final $\log_{10}(E)$ at Day 600')
    ax.set_ylabel('Count')
    fig.savefig(
        './assets/E_hist.png',
        bbox_inches='tight',
        pad_inches=0.2,
    )
    print('Elapsed time: ', datetime.now() - start)
    return


@ray.remote
def _ray_dynamics(env: gym.Env, max_steps: int):
    state = env.reset()[0]
    for _ in range(max_steps):
        action = np.random.randint(4)
        next_state, _, _, _, info = env.step(action)
        state = next_state
    return state[5]


ray.init(num_cpus=30)
plot_E_hist()
ray.shutdown()