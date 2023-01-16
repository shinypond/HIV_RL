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


class DQNAgent:
    def __init__(
        self, 
        # General parameters
        env: gym.Env,
        test_env: gym.Env,
        writer: SummaryWriter,
        log_dir: str,
        ckpt_dir: str,
        load_ckpt: bool,
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
        # Network parameters
        hidden_dim: int = 1024,
        # PER parameters
        per: bool = True,
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Double DQN
        double_dqn: bool = False,
    ):
        '''Initialization.
        Args:
            env (gym.Env): openAI Gym environment
            test_env (gym.Env): Test environment
            writer (SummaryWriter): Tensorboard SummaryWriter
            log_dir (str): Logging directory path (root for the experiment)
            ckpt_dir (str): checkpoint directory path
            load_ckpt (bool): Load checkpoint? or not?

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

            hidden_dim (int): hidden dimension in network

            per (bool): If true, PER is activated. Otherwise, the replay buffer of original DQN is used.
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled

            double_dqn (bool): Activate dqn or not
        '''
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        self.test_env = test_env
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

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'device: {self.device}')
        
        # PER memory
        self.per = per
        if per:
            self.beta = beta
            self.prior_eps = prior_eps
            self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha=alpha)
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

        # Record (archive train / test results)
        self.record = [] 

        # Initial episode (default: 1)
        self.init_episode = 1

        if load_ckpt:
            self.load_ckpt()

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
                
        logging.info(f'Success: Checkpoint loaded (start from Episode {self.init_episode}!')

    def select_action(self, state: np.ndarray) -> int:
        '''Select an action from the input state.'''
        # epsilon greedy policy (only for training)
        if self.epsilon > np.random.random() and not self.is_test:
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        return selected_action

    def step(self, action: int) -> Tuple[np.ndarray, np.float64, bool, Optional[np.ndarray]]:
        '''Take an action and return the response of the env.'''
        if not self.is_test:
            next_state, reward, done, _, info = self.env.step(action)
        else:
            next_state, reward, done, _, info = self.test_env.step(action)
        return next_state, reward, done, info['intermediate_sol']

    def update_model(self) -> torch.Tensor:
        '''Update the model by gradient descent.'''
        if self.per:
            # PER needs beta to calculate weights
            samples = self.memory.sample_batch(self.beta)
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

        max_steps = self.env.max_episode_steps
        update_cnt = 0
        start = datetime.now()

        for episode in range(self.init_episode, max_episodes+1):
            state = self.env.reset()[0]
            losses = []
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                transition = [state, action, reward, next_state, done]
                self.memory.store(*transition)
                state = next_state

                # PER
                fraction = min(step / max_steps, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

                # If training is available:
                if len(self.memory) >= self.batch_size:
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
                self.is_test = False
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
                    )

            # Save
            if save_freq > 0 and episode % save_freq == 0:
                path = os.path.join(self.ckpt_dir, 'ckpt.pt')
                self.save_ckpt(episode, path)
                
        self.env.close()
                
    def test(self, episode: int, img_dir: str) -> Tuple[int, float, float, float]:
        '''Test the agent.'''
        self.is_test = True
        self.dqn.eval()

        max_steps = self.test_env.max_episode_steps
        states = []
        actions = []
        rewards = []

        with torch.no_grad():
            state = self.test_env.reset()[0]
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, _, intermediate_sol = self.step(action)
                _action = self.test_env.controls[action].reshape(1, -1)
                _reward = np.array([reward,]) * self.test_env.reward_scaler
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

        # Plotting
        last_treatment_day = get_last_treatment_day(actions) * (self.test_env.max_days // len(actions))
        max_E = 10 ** (states[:, 5].max())
        last_E = 10 ** (states[-1, 5])
        cum_reward = rewards.sum() # Not scaled. Original total discounted reward itself.
        if max_E > 0:
            fig = plt.figure(figsize=(16, 10))
            plt.title(
                f'Episode {episode} | Last Treatment: Day {last_treatment_day} | '\
                f'E: Max {int(max_E)} Last {int(last_E)} | Total Discounted Reward {cum_reward:.4e}')
            plt.axis('off')
            legends = ['T1', 'T2', 'T1I', 'T2I', 'V', 'E', 'a1', 'a2', 'reward/1e6']
            axis_t = np.arange(states.shape[0])
            axis_t_r = np.arange(rewards.shape[0]) * (states.shape[0] / rewards.shape[0])

            for i in range(6):
                ax = fig.add_subplot(3, 3, i+1)
                ax.plot(axis_t, states[:, i], label=legends[i])
                ax.legend()
                ax.grid()

            for i in range(2):
                ax = fig.add_subplot(3, 3, i+7)
                ax.plot(axis_t, actions[:, i], label=legends[i+6], color='r')
                ax.legend()
                ax.grid()

            ax = fig.add_subplot(3, 3, 9)
            ax.plot(axis_t_r, rewards/1e6, label=legends[8], color='g')
            ax.legend()
            ax.grid()

            success = '_success' if max_E > 300000 else ''
            fig.savefig(
                os.path.join(img_dir, f'Epi_{episode}_{last_treatment_day}{success}.png'),
                bbox_inches='tight',
                pad_inches=0.2,
            )

        self.dqn.train()
        self.test_env.close()
        return last_treatment_day, max_E, last_E, cum_reward

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
    ):
        elapsed_time = str(timedelta(seconds=elapsed_time.seconds))
        logging.info(
            f'Epi {episodes:>4d} | {elapsed_time} | MaxE {max_E:8.1f} | LastE {last_E:8.1f} | '\
            f'Loss (Train) {train_loss:.2e} | Buffer {self.memory.size}')

    def _save_record_df(self):
        '''Save self.record as a pandas dataframe.'''
        df = pd.DataFrame(self.record).set_index('episode')
        df.to_csv(os.path.join(self.log_dir, 'records.csv'))


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
def get_last_treatment_day(actions: np.ndarray) -> int:
    '''Find the last treatment day (i.e, nonzero actions) for a given action sequence.'''
    n = len(actions)
    for i in range(n-1, -1, -1):
        if actions[i, 0] != 0 or actions[i, 1] != 0:
            return i + 1
    return 0