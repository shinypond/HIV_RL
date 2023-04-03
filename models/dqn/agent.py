from typing import *
import os
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .memory import ReplayBuffer, PrioritizedReplayBuffer
from .network import Network 
from envs.hiv_v1 import make_HIV_env
from configs import treatment_days


class DQNAgent:
    def __init__(
        self, 
        # General parameters
        log_dir: str,
        ckpt_dir: str,
        load_ckpt: bool,
        writer: SummaryWriter,
        # Environment parameters
        max_days: int = 600,
        treatment_days: int = 1,
        reward_scaler: float = 1e+8,
        # Training parameters
        memory_size: int = int(1e6),
        batch_size: int = 2048,
        lr: float = 2e-4,
        l2_reg: float = 0.,
        grad_clip: float = 1000.,
        target_update: int = 3000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 1 / 200,
        decay_option: str = 'logistic',
        discount_factor: float = 0.99,
        n_train: int = 1,
        # Network parameters
        hidden_dim: int = 1024,
        # PER parameters
        per: bool = True,
        alpha: float = 0.2,
        beta: float = 0.6,
        beta_increment_per_sampling: float = 0.000005,
        prior_eps: float = 1e-6,
        # Double DQN
        double_dqn: bool = False,
    ):
        '''Initialization.
        Args:
            log_dir (str): Logging directory path (root for the experiment)
            ckpt_dir (str): checkpoint directory path
            load_ckpt (bool): Load checkpoint? or not?
            writer (SummaryWriter): Tensorboard SummaryWriter

            max_days (int): Time length of one episode (days)
            treatment_days (int): Treatment interval days 
            reward_scaler (float): Scaling factor for the instantaneous rewards

            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            lr (float): learning rate
            l2_reg (float) : L2-regularization (weight decay)
            grad_clip (float) : gradient clipping
            target_update (int): period for target model's hard update
            max_epsilon (float): Maximum value of epsilon
            min_epsilon (float): Minimum value of epsilon
            epsilon_decay (float): Epsilon decaying rate
            decay_option (str): Epsilon decaying schedule option (`linear`, `logistic`)
            discount_factor (float): Discounting factor
            n_train (int): Number of training per each step

            hidden_dim (int): hidden dimension in network

            per (bool): If true, PER is activated. Otherwise, the replay buffer of original DQN is used.
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            beta_increment_per_sampling (float): to increase beta per each sampling
            prior_eps (float): guarantees every transition can be sampled

            double_dqn (bool): Activate dqn or not
        '''
        
        # Make environments
        self.max_days = max_days
        self.treatment_days = treatment_days
        self.reward_scaler = reward_scaler

        # (0) Prepare initial
        UNHEALTHY_STEADY_INIT_STATE = np.log10(np.array([163573, 5, 11945, 46, 63919, 24], dtype=np.float32))
        HIGH_T_LOW_V_INIT_STATE = np.log10(np.array([1.0e+6, 3198, 1.0e-4, 1.0e-4, 1, 10], dtype=np.float32))
        HIGH_T_HIGH_V_INIT_STATE = np.log10(np.array([1.0e+6, 3198, 1.0e-4, 1.0e-4, 1000000, 10], dtype=np.float32))
        LOW_T_HIGH_V_INIT_STATE = np.log10(np.array([1000, 10, 10000, 100, 1000000, 10], dtype=np.float32))

        # (1) Make Envs
        self.envs = {
            'train': self.make_env(UNHEALTHY_STEADY_INIT_STATE),
            'HTLV': self.make_env(HIGH_T_LOW_V_INIT_STATE),
            'HTHV': self.make_env(HIGH_T_HIGH_V_INIT_STATE),
            'LTHV': self.make_env(LOW_T_HIGH_V_INIT_STATE),
        }
        obs_dim = self.envs['train'].observation_space.shape[0]
        action_dim = self.envs['train'].action_space.n

        # Parameters
        self.writer = writer
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.target_update = target_update
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.decay_option = decay_option
        self.discount_factor = discount_factor
        self.n_train = n_train

        # Device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'device: {self.device}')
        
        # PER memory
        self.per = per
        if per:
            self.prior_eps = prior_eps
            self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha, beta, beta_increment_per_sampling)
        else:
            self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)

        # Double DQN
        self.double_dqn = double_dqn

        # Networks: DQN, DQN_target
        dqn_config = dict(
            in_dim = obs_dim,
            nf = hidden_dim,
            out_dim = action_dim,
        )
        self.dqn = Network(**dqn_config).to(self.device)
        self.dqn_target = Network(**dqn_config).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=lr, weight_decay=l2_reg)

        # Mode: train / test
        self.is_test = False
        self.max_cum_reward = -1.
        
        # Record (archive train / test results)
        self.record = [] 

        # Initial episode (default: 1)
        self.init_episode = 1

        # Benchmark (no_drug, full_drug for each environment)
        logging.info('Computing Benchmark... (Wait for few seconds)')
        self.bm_info = {
            'no_drug': {'states': {}, 'actions': {}, 'rewards': {}},
            'full_drug': {'states': {}, 'actions': {}, 'rewards': {}},
        }
        for opt in self.bm_info.keys():
            for name, _env in self.envs.items():
                _states, _actions, _rewards = self._test(_env, opt)
                self.bm_info[opt]['states'][name] = _states 
                self.bm_info[opt]['actions'][name] = _actions
                self.bm_info[opt]['rewards'][name] = _rewards
        logging.info('Done!')

        if load_ckpt:
            self.load_ckpt()

    def make_env(self, init_state: Optional[np.ndarray] = None) -> gym.Env:
        env = make_HIV_env(
            max_days = self.max_days,
            treatment_days = self.treatment_days,
            reward_scaler = self.reward_scaler,
            init_state = init_state,
        )
        return env

    def save_ckpt(self, episode: int, path: str) -> None:
        if self.per:
            _memory = _gather_per_buffer_attr(self.memory)
        else:
            _memory = _gather_replay_buffer_attr(self.memory)
        ckpt = dict(
            episode = episode,
            dqn = self.dqn.state_dict(),
            dqn_target = self.dqn_target.state_dict(),
            optim = self.optimizer.state_dict(),
            memory = _memory,
        )
        torch.save(ckpt, path)

    def load_ckpt(self) -> None:
        ckpt = torch.load(os.path.join(self.ckpt_dir, 'ckpt.pt'))
        self.init_episode = ckpt['episode'] + 1
        self.dqn.load_state_dict(ckpt['dqn'])
        self.dqn_target.load_state_dict(ckpt['dqn_target'])
        self.optimizer.load_state_dict(ckpt['optim'])
        for key, value in ckpt['memory'].items():
            if key not in ['sum_tree', 'min_tree']:
                setattr(self.memory, key, value)
            else:
                tree = getattr(self.memory, key)
                setattr(tree, 'capacity', value['capacity'])
                setattr(tree, 'tree', value['tree'])
                
        logging.info(f'Success: Checkpoint loaded (start from Episode {self.init_episode})!')

    def select_action(self, state: np.ndarray) -> int:
        '''Select an action from the input state.'''
        # epsilon greedy policy (only for training)
        if self.epsilon > np.random.random() and not self.is_test:
            selected_action = self.envs['train'].action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        return selected_action

    def step(self, env: gym.Env, action: int) -> Tuple[np.ndarray, np.float64, bool, Optional[np.ndarray]]:
        '''Take an action and return the response of the env.'''
        next_state, reward, done, _, info = env.step(action)
        return next_state, reward, done, info['intermediate_sol']

    def update_model(self) -> torch.Tensor:
        '''Update the model by gradient descent.'''
        if self.per:
            # PER needs beta to calculate weights
            samples = self.memory.sample_batch()
            weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
            indices = samples["indices"]
        else:
            # Vanilla DQN does not require any weights
            samples = self.memory.sample_batch()
        
        # PER: importance sampling before average
        elementwise_loss = self._compute_dqn_loss(samples)
        if self.per:
            loss = torch.mean(elementwise_loss * weights)
        else:
            loss = torch.mean(elementwise_loss)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), self.grad_clip)
        self.optimizer.step()
        
        if self.per:
            # PER: update priorities
            loss_for_prior = elementwise_loss.squeeze().detach().cpu().numpy()
            new_priorities = loss_for_prior + self.prior_eps
            self.memory.update_priorities(indices, new_priorities)

        return loss.item()
        
    def train(self, max_episodes: int, log_freq: int, test_freq: int, save_freq: int, img_dir: str) -> None:
        '''Train the agent.'''
        self.is_test = False

        max_steps = self.envs['train'].max_episode_steps
        update_cnt = 0
        start = datetime.now()

        for episode in range(self.init_episode, max_episodes+1):
            state = self.envs['train'].reset()[0]
            losses = []
            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.step(self.envs['train'], action)
                transition = [state, action, reward, next_state, done]
                self.memory.store(*transition)
                state = next_state

                # If training is available:
                if len(self.memory) >= self.batch_size:
                    for _ in range(self.n_train):
                        loss = self.update_model()
                    losses.append(loss)
                    self.writer.add_scalar('loss', loss, update_cnt)
                    update_cnt += 1

                    # epsilon decaying
                    if self.decay_option == 'linear':
                        self.epsilon = max(
                            self.min_epsilon, self.epsilon - (
                                self.max_epsilon - self.min_epsilon
                            ) * self.epsilon_decay
                        )
                    elif self.decay_option == 'logistic':
                        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * \
                            sigmoid(1 / self.epsilon_decay - episode)

                    # Target network update
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                # If episode ends:
                if done:
                    break

            avg_step_train_loss = np.array(losses).sum() * self.batch_size / max_steps

            # Test
            if test_freq > 0 and episode % test_freq == 0:
                last_treatment_day, max_E, last_E, cum_reward = self.test(episode, img_dir)
                self.record.append({
                    'episode': episode,
                    'last_treatment_day': last_treatment_day,
                    'max_E': max_E,
                    'last_E': last_E,
                    'cum_reward': cum_reward,
                    'train_loss': avg_step_train_loss,
                })
                self._save_record_df()

                # Logging
                if log_freq > 0 and episode % log_freq == 0:
                    self._track_results(
                        episode,
                        datetime.now() - start,
                        train_loss = avg_step_train_loss,
                        max_E = max_E,
                        last_E = last_E,
                        cum_reward = cum_reward,
                    )

            # Save
            if save_freq > 0 and episode % save_freq == 0:
                path = os.path.join(self.ckpt_dir, 'ckpt.pt')
                self.save_ckpt(episode, path)
                
        self.envs['train'].close()
                
    def test(self, episode: int, img_dir: str) -> Tuple[int, float, float, float]:
        '''Test the agent (computation & plotting)'''

        # Compute state/action/reward sequence for train env
        _states, _actions, _rewards = self._test(self.envs['train'], 'policy')
        states = {'train': _states}
        actions = {'train': _actions}
        rewards = {'train': _rewards}

        # cum_reward = rewards['train'].sum() # Original total reward
        cum_reward = discounted_sum(rewards['train'], self.discount_factor) # total discounted reward
        if cum_reward > max(1e+0, self.max_cum_reward):

            # Compute state/action/reward sequence for other envs, too
            for name, _env in self.envs.items():
                if name == 'train': continue
                states[name], actions[name], rewards[name] = self._test(_env, 'policy')

            # FIGURE 1 (6-states & 2-actions - one figure per each env)
            for env_name in self.envs.keys():
                self._plot_6_states_2_actions(
                    episode, img_dir,
                    states[env_name], actions[env_name],
                    self.bm_info['no_drug']['states'][env_name],
                    self.bm_info['full_drug']['states'][env_name],
                    env_name,
                )

            # FIGURE 2 (V, E phase-plane - one figure for all envs)
            self._plot_VE_phase_plane(episode, img_dir, states, actions)

        for env in self.envs.values():
            env.close()

        self.dqn.train()

        last_a1_day = get_last_treatment_day(actions['train'][:, 0])
        last_a2_day = get_last_treatment_day(actions['train'][:, 1])
        last_treatment_day = max(last_a1_day, last_a2_day) * (self.max_days // len(actions['train']))
        max_E = 10 ** (states['train'][:, 5].max())
        last_E = 10 ** (states['train'][-1, 5])
        if cum_reward > self.max_cum_reward:
            self.max_cum_reward = cum_reward

        return last_treatment_day, max_E, last_E, cum_reward

    def _test(self, env: gym.Env, mode: str = 'policy') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''Test the agent (dynamics propagation only)'''
        assert (mode in ['policy', 'no_drug', 'full_drug']) 
        self.is_test = True
        self.dqn.eval()

        max_steps = env.max_episode_steps
        states = []
        actions = []
        rewards = []

        with torch.no_grad():
            state = env.reset()[0]
            for _ in range(max_steps):
                if mode == 'policy':
                    action = self.select_action(state)
                elif mode == 'no_drug':
                    action = 0
                elif mode == 'full_drug':
                    action = 3
                next_state, reward, _, intermediate_sol = self.step(env, action)
                _action = env.controls[action].reshape(1, -1)
                _reward = np.array([reward,]) * env.reward_scaler
                if intermediate_sol is not None: # i.e, treatment days > 1
                    intermediate_states = intermediate_sol[:6, :].transpose()
                    _state = np.concatenate([state.reshape(1, -1), intermediate_states], axis=0)
                    _action = np.repeat(_action, _state.shape[0], axis=0)
                else: # i.e, treatment days = 1 
                    _state = state.reshape(1, -1)
                states.append(_state)
                actions.append(_action)
                rewards.append(_reward)
                state = next_state

        states = np.concatenate(states, axis=0, dtype=np.float32) # shape (N, 6)
        actions = np.concatenate(actions, axis=0, dtype=np.float32) # shape (N, 2)
        rewards = np.concatenate(rewards, axis=0, dtype=np.float32).reshape(-1, 1) # shape (N//T, 1)
        self.is_test = False
        return states, actions, rewards

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        '''Return categorical dqn loss.'''
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        curr_q_value = self.dqn(state).gather(1, action)
        if not self.double_dqn:
            next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        else:
            next_q_value = self.dqn_target(next_state).gather(
                1, self.dqn(next_state).argmax(dim=1, keepdim=True)
            ).detach()
        mask = 1 - done
        target = (reward + self.discount_factor * next_q_value * mask).to(self.device)

        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction='none')

        return elementwise_loss

    def _target_hard_update(self):
        '''Hard update: target <- local.'''
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _track_results(
        self,
        episodes: int,
        elapsed_time: timedelta,
        train_loss: float,
        max_E: float,
        last_E: float,
        cum_reward: float,
    ):
        elapsed_time = str(timedelta(seconds=elapsed_time.seconds))
        logging.info(
            f'Epi {episodes:>4d} | {elapsed_time} | LastE {last_E:8.1f} | CumR {cum_reward:.3e} | '\
            f'Loss (Train) {train_loss:.2e} | Buffer {self.memory.size}')

    def _save_record_df(self):
        '''Save self.record as a pandas dataframe.'''
        df = pd.DataFrame(self.record).set_index('episode')
        df.to_csv(os.path.join(self.log_dir, 'records.csv'))

    def _plot_6_states_2_actions(
        self,
        episode: int,
        img_dir: str,
        policy_states: np.ndarray,
        policy_actions: np.ndarray,
        no_drug_states: np.ndarray,
        full_drug_states: np.ndarray,
        env_name: str,
    ) -> None:
        '''Draw a figure with 6-states and 2-actions (our policy & no drug & full drug)'''
        fig = plt.figure(figsize=(14, 18))
        plt.axis('off')
        state_names = [
            r'$\log_{10}(T_{1}$)', r'$\log_{10}(T_{2})$',
            r'$\log_{10}(T_{1}^{*})$', r'$\log_{10}(T_{2}^{*})$',
            r'$\log_{10}(V)$', r'$\log_{10}(E)$',
        ]
        action_names = [
            rf'RTI $\epsilon_{1}$', rf'PI $\epsilon_{2}$',
        ]
        axis_t = np.arange(policy_states.shape[0]) * treatment_days
        label_fontdict = {
            'size': 13,
        }

        for i in range(6):
            ax = fig.add_subplot(4, 2, i+1)
            ax.plot(axis_t, policy_states[:, i], label='ours', color='crimson', linewidth=2)
            ax.plot(axis_t, no_drug_states[:, i], label='no drug', color='royalblue', linewidth=2, linestyle='--')
            ax.plot(axis_t, full_drug_states[:, i], label='full drug', color='black', linewidth=2, linestyle='-.')
            ax.set_xlabel('Days', labelpad=0.8, fontdict=label_fontdict)
            ax.set_ylabel(state_names[i], labelpad=0.5, fontdict=label_fontdict)
            if i == 0:
                ax.set_ylim(min(4.8, policy_states[:, i].min() - 0.2), 6)

        last_a1_day = get_last_treatment_day(policy_actions[:, 0])
        last_a2_day = get_last_treatment_day(policy_actions[:, 1])
        last_treatment_day = max(last_a1_day, last_a2_day) * (self.max_days // len(policy_actions))
        for i in range(2):
            ax = fig.add_subplot(4, 2, i+7)
            if last_treatment_day < 550:
                if last_a1_day >= last_a2_day:
                    if i == 0:
                        ax.text(last_a1_day * treatment_days, -0.07, f'Day {last_a1_day * treatment_days}')
                else:
                    if i == 1:
                        ax.text(last_a2_day * treatment_days, -0.03, f'Day {last_a2_day * treatment_days}')
            _a = np.repeat(policy_actions[:, i], treatment_days, axis=0)
            ax.plot(np.arange(policy_states.shape[0] * treatment_days), _a, color='forestgreen', linewidth=2)
            if i == 0:
                ax.set_ylim(0.7 * (-0.2), 0.7 * 1.2)
                ax.set_yticks([0.0, 0.7])
            else:
                ax.set_ylim(0.3 * (-0.2), 0.3 * 1.2)
                ax.set_yticks([0.0, 0.3])
            ax.set_xlabel('Days', labelpad=0.8, fontdict=label_fontdict)
            ax.set_ylabel(action_names[i], labelpad=0.5, fontdict=label_fontdict)

        fig.savefig(
            os.path.join(img_dir, f'Epi{episode}_{env_name}_{last_treatment_day}.png'),
            bbox_inches='tight',
            pad_inches=0.2,
        )
        return

    def _plot_VE_phase_plane(
        self,
        episode: int,
        img_dir: str,
        states: Dict[str, np.ndarray],
        actions: Dict[str, np.ndarray],
    ) -> None:
        '''Draw a figure of logV - logE phase plane for each environment'''
        label_fontdict = {
            'size': 13,
        }
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        meta_info = {
            'train': dict(color='navy', alpha=0.8, label='train (initial: unhealthy steady state)'),
            'HTLV': dict(color='forestgreen', alpha=0.8, label='test (initial: early infection with one virus)'),
            'HTHV': dict(color='darkorange', alpha=0.8, label=r'test (initial: early infection with $10^6$ virus)'),
            'LTHV': dict(color='indianred', alpha=0.8, label=r'test (initial: small T-cells with $10^6$ virus)'),
        }
        init_labels = ['A', 'B', 'C', 'D']
        for i, (env_name, kwargs) in enumerate(meta_info.items()):
            _s = states[env_name]
            x = _s[:, 4] # log(V)
            y = _s[:, 0] # log(T1)
            z = _s[:, 5] # log(E) 
            ax.plot(x, y, z, **kwargs)
            ax.scatter(x[0], y[0], z[0], color='black', marker='o', s=70)
            ax.text(x[0], y[0], z[0] - 0.4, init_labels[i], fontdict=dict(size=13,))

        # End point (only for training env)
        ax.scatter(
            states['train'][-1, 4], states['train'][-1, 0], states['train'][-1, 5],
            color='red', marker='*', s=120,
        )
        ax.text(
            states['train'][-1, 4], states['train'][-1, 0], states['train'][-1, 5] + 0.4,
            'End', fontdict=dict(size=14,),
        )

        ax.view_init(15, 45)
        ax.set_xlabel(r'$\log_{10}(V)$', labelpad=2, fontdict=label_fontdict)
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylabel(r'$\log_{10}(T_{1})$', labelpad=2, fontdict=label_fontdict)
        ax.set_ylim(3, 7)
        ax.set_zlabel(r'$\log_{10}(E)$', labelpad=2, fontdict=label_fontdict)
        ax.set_zlim(0, 6.5)
        ax.legend(loc='upper right')
        fig.savefig(
            os.path.join(img_dir, f'Epi{episode}_VE.png'),
            bbox_inches='tight',
            pad_inches=0.2,
        )
        return


def _gather_replay_buffer_attr(memory: Optional[ReplayBuffer]) -> dict:
    if memory is None:
        return {}
    replay_buffer_keys = [
        'obs_buf', 'next_obs_buf', 'acts_buf', 'rews_buf', 'done_buf',
        'max_size', 'batch_size', 'ptr', 'size',
    ]
    result = {key: getattr(memory, key) for key in replay_buffer_keys}
    return result


def _gather_per_buffer_attr(memory: Optional[PrioritizedReplayBuffer]) -> dict:
    if memory is None:
        return {}
    per_buffer_keys = [
        'obs_buf', 'next_obs_buf', 'acts_buf', 'rews_buf', 'done_buf',
        'max_size', 'batch_size', 'ptr', 'size',
        'max_priority', 'tree_ptr', 'alpha',
    ]
    result = {key: getattr(memory, key) for key in per_buffer_keys}
    result['sum_tree'] = dict(
        capacity = memory.sum_tree.capacity,
        tree = memory.sum_tree.tree,
    )
    result['min_tree'] = dict(
        capacity = memory.min_tree.capacity,
        tree = memory.min_tree.tree,
    )
    return result


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


@njit(cache=True)
def get_last_treatment_day(action: np.ndarray) -> int:
    '''Find the last treatment day (i.e, nonzero actions) for a given action sequence.'''
    n = len(action)
    for i in range(n-1, -1, -1):
        if action[i] != 0:
            return i + 1
    return 0


@njit(cache=True)
def discounted_sum(rewards: np.ndarray, discount_factor: float = 0.99) -> float:
    _sum = 0.
    _factor = 1.
    for r in rewards[:, 0]:
        _sum += r * _factor
        _factor *= discount_factor 
    return _sum