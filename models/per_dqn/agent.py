from typing import *
import os
import logging
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .memory import PrioritizedReplayBuffer
from .network import Network 


class PERDQNAgent:
    def __init__(
        self, 
        # General parameters
        env: gym.Env,
        test_env: gym.Env,
        writer: SummaryWriter,
        ckpt_dir: str,
        load_ckpt: bool,
        # Training parameters
        memory_size: int,
        batch_size: int,
        lr: float,
        l2_reg: float = 1e-6,
        grad_clip: float = 10.,
        target_update: int = 100,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        epsilon_decay: float = 1 / 2000,
        # Network parameters
        hidden_dim: int = 256,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
    ):
        '''Initialization.
        Args:
            env (gym.Env): openAI Gym environment
            test_env (gym.Env): Test environment
            writer (SummaryWriter): Tensorboard SummaryWriter
            ckpt_dir (str): checkpoint directory path
            load_ckpt (bool): Load checkpoint? or not?

            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            lr (float): learning rate
            l2_reg (float) : L2-regularization (weight decay)
            target_update (int): period for target model's hard update

            hidden_dim (int): hidden dimension in network
            dropout (float): dropout rate in network

            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
        '''
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.env = env
        self.test_env = test_env
        self.writer = writer
        self.ckpt_dir = ckpt_dir
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.target_update = target_update
        self.epsilon = max_epsilon
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'device: {self.device}')
        
        # PER memory
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha=alpha)

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

        # Transition to store in memory
        self.transition = list()
        
        # Mode: train / test
        self.is_test = False

        # Initial episode (default: 1)
        self.init_episode = 1

        if load_ckpt:
            self.load_ckpt()

    def save_ckpt(self, episode: int, path: str) -> None:
        ckpt = dict(
            episode = episode,
            dqn = self.dqn.state_dict(),
            dqn_target = self.dqn_target.state_dict(),
            optim = self.optimizer.state_dict(),
            memory = _gather_per_buffer_attr(self.memory),
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
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: int) -> Tuple[np.ndarray, np.float64, bool]:
        '''Take an action and return the response of the env.'''
        if not self.is_test:
            next_state, reward, done, _, _ = self.env.step(action)
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
        
        else:
            next_state, reward, done, _, _ = self.test_env.step(action)
    
        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        '''Update the model by gradient descent.'''
        t1 = datetime.now()
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]
        
        # PER: importance sampling before average
        t2 = datetime.now()
        elementwise_loss = self._compute_dqn_loss(samples)
        loss = torch.mean(elementwise_loss * weights)
        t3 = datetime.now()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.dqn.parameters(), self.grad_clip)
        self.optimizer.step()
        t4 = datetime.now()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.squeeze().detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        t5 = datetime.now()
        
        sec = lambda x: x.microseconds
        #logging.info(f'{sec(t2-t1)} {sec(t3-t2)} {sec(t4-t3)} {sec(t5-t4)}')

        return loss.item()
        
    def train(self, max_episodes: int, log_freq: int, test_freq: int, save_freq: int, img_dir: str) -> None:
        '''Train the agent.'''
        self.is_test = False

        max_steps = self.env.spec.max_episode_steps
        update_cnt = 0
        start = datetime.now()

        for episode in range(self.init_episode, max_episodes+1):
            state = self.env.reset()[0]
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                state = next_state

                # PER
                fraction = min(step / max_steps, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

                # If training is available:
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()
                    self.writer.add_scalar('loss', loss, update_cnt)
                    update_cnt += 1

                    # linearly decrease epsilon
                    self.epsilon = max(
                        self.min_epsilon, self.epsilon - (
                            self.max_epsilon - self.min_epsilon
                        ) * self.epsilon_decay
                    )

                    # Target network update
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                # If episode ends:
                if done:
                    break

            # Logging
            if log_freq > 0 and episode % log_freq == 0:
                self._track_results(
                    episode,
                    datetime.now() - start,
                    immune_effectors=state[5],
                )

            # Test
            if test_freq > 0 and episode % test_freq == 0:
                self.test(episode, img_dir)
                self.is_test = False

            # Save
            if save_freq > 0 and episode % save_freq == 0:
                path = os.path.join(self.ckpt_dir, 'ckpt.pt')
                self.save_ckpt(episode, path)
                
        self.env.close()
                
    def test(self, episode: int, img_dir: str) -> None:
        '''Test the agent.'''
        self.is_test = True
        self.dqn.eval()

        max_steps = self.test_env.spec.max_episode_steps
        states = []
        actions = []
        rewards = []

        with torch.no_grad():
            state = self.test_env.reset()[0]
            for _ in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done = self.step(action)
                states.append(state.reshape(1, -1))
                actions.append(self.test_env.controls[action].reshape(1, -1))
                rewards.append(reward)
                state = next_state
                if done:
                    break

        states = np.concatenate(states, axis=0, dtype=np.float32) # shape (600, 6)
        actions = np.concatenate(actions, axis=0, dtype=np.float32) # shape (600, 2)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1) # shape (600, 1)

        # Plotting
        fig = plt.figure(figsize=(16, 10))
        cum_reward = rewards.sum()
        plt.title(f'Episode {episode} | Cumulative Reward {cum_reward:.2f}')
        plt.axis('off')
        axis_t = np.arange(0, max_steps)
        legends = ['T1', 'T2', 'T1I', 'T2I', 'V', 'E', 'a1', 'a2', 'reward']

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
        ax.plot(axis_t, rewards, label=legends[8], color='g')
        ax.legend()
        ax.grid()

        fig.savefig(os.path.join(img_dir, f'Epi_{episode}.png'))

        self.dqn.train()
        self.test_env.close()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        '''Return categorical dqn loss.'''
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + next_q_value * mask).to(self.device)

        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction='none')

        return elementwise_loss

    def _target_hard_update(self):
        '''Hard update: target <- local.'''
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _track_results(
        self,
        episodes: int,
        elapsed_time: timedelta,
        immune_effectors: float,
    ):
        elapsed_time = str(timedelta(seconds=elapsed_time.seconds))
        logging.info(f'Epi {episodes:>4d} | {elapsed_time} | E {immune_effectors:.2f} | buffer {self.memory.size}')


def _gather_per_buffer_attr(memory: Optional[PrioritizedReplayBuffer]) -> Optional[dict]:
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
