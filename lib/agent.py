import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .env.hiv_dynamics import HIV
from .model.DQN import DQN, init_weights
from .utils import load_ckpt, save_ckpt
from .memory.per import Memory


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = torch.tensor([]) # shape : N x (6+1+1+6)
        self.position = 0
        self.zero = torch.tensor([[0.] * 14])
        
    def push(self, state, action_idx, reward, next_state):
        if self.memory.shape[0] == 0:
            self.memory = self.zero
        elif self.memory.shape[0] < self.capacity:
            self.memory = torch.cat([self.memory, self.zero], dim=0)
        self.memory[self.position, :] = torch.cat(
            [state, action_idx.unsqueeze(0), reward.unsqueeze(0), next_state], dim=0
        )
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return self.memory[np.random.choice(self.memory.shape[0], batch_size), :]
    
    def __len__(self):
        return self.memory.shape[0]
        

class AGENT:
    def __init__(self, config):
        self.make_action_set(config.action)
        self.env = HIV(config)

    def make_action_set(self, constraint):
        self.action_set = []
        min_a1 = constraint.min_a1
        max_a1 = constraint.max_a1
        int_a1 = constraint.interval_a1
        min_a2 = constraint.min_a2
        max_a2 = constraint.max_a2
        int_a2 = constraint.interval_a2

        for i in np.arange(min_a1, max_a1 + int_a1, int_a1):
            for j in np.arange(min_a2, max_a2 + int_a2, int_a2):
                self.action_set.append([i, j])
        self.action_set = torch.tensor(self.action_set, dtype=torch.float32)

    def choose_action(self, state, policy_net, eps=1.0):

        # State shape: (B, 6)
        B = state.shape[0]
        assert eps >= 0 and eps <= 1
        if np.random.rand(1)[0] < eps:
            idx = torch.tensor(
                np.random.randint(len(self.action_set), size=(B)), dtype=torch.int64
            )
        else:
            with torch.no_grad():
                policy_net.eval()
                pred = policy_net(state)
                policy_net.train()
            idx = torch.argmax(pred, dim=1)

        # action, action_idx shape: (B,)
        action = self.action_set[idx]
        action_idx = idx.clone().cpu().type(torch.int64)
        return action, action_idx

    def train_Q(self, config, info):

        device = config.device

        # Sample mini-batch from the memory
        batch, idxs, is_weights = info['memory'].sample(config.train.batch_size)
        _state = batch[:, :6].to(device)
        _action_idx = batch[:, [6]].to(device).type(torch.int64)
        _reward = batch[:, [7]].to(device)
        _next_state = batch[:, 8:].to(device)

        # Update priority
        policy_pred = info['policy_net'](_state).gather(1, _action_idx)
        target_best = torch.max(info['target_net'](_next_state), dim=1, keepdim=True).values
        errors = torch.abs(_reward + config.train.discount * target_best - policy_pred).detach().cpu().numpy()
        info['memory'].update(idxs, errors)

        # Compute loss
        loss = F.mse_loss(_reward + config.train.discount * target_best, policy_pred)
        loss = loss * torch.FloatTensor(is_weights).to(device)
        max_loss = loss.max()
        loss = loss.mean()

        # Update the Q-network
        info['optimizer'].zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(info['policy_net'].parameters(), config.train.grad_clip)
        info['optimizer'].step()

        # Update the target Q-network
        for target_param, policy_param in zip(info['target_net'].parameters(), info['policy_net'].parameters()):
            target_param = (1 - config.train.soft_update) * target_param + config.train.soft_update * policy_param

        return loss, max_loss

    def train(self, config, logdir, resume=True):
        tb_dir = os.path.join(logdir, 'tensorboard')
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)

        device = config.device
        policy_net = DQN(config, len(self.action_set)).to(device)
        target_net = DQN(config, len(self.action_set)).to(device)
        optimizer = optim.Adam(policy_net.parameters(), lr=config.train.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.max_episode)

        policy_net.apply(init_weights)
        policy_net.train()
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        memory = Memory(config.train.replay_buffer_size)

        info = {
            'policy_net': policy_net,
            'target_net': target_net,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'memory': memory,
            'episode': 0,
        }

        ckpt_dir = os.path.join(logdir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        if resume:
            info = load_ckpt(ckpt_dir, info, device, train=True)

        start_episode = info['episode']

        log_freq = config.train.log_freq
        init_state = config.dynamics.init_state
        init_state = torch.tensor(init_state).float()
        eps_start = config.train.eps_start
        eps_end = config.train.eps_end
        eps_decay = config.train.eps_decay
        dyn_batch_size = config.dynamics.batch_size

        for episode in range(start_episode, config.train.max_episode):

            state = torch.cat([init_state.unsqueeze(0)] * dyn_batch_size, dim=0)
            eps = eps_end + (eps_start - eps_end) * np.exp(-1. * episode / eps_decay)

            for t in range(config.train.max_step):

                # Choose and Execute an action
                action, action_idx = self.choose_action(state.to(device), info['policy_net'], eps)
                reward, next_state = self.env.step(state, action)

                # Compute TD error of this < s, a, r, s' >
                with torch.no_grad():
                    info['policy_net'].eval()
                    policy_pred = info['policy_net'](state.to(device))
                    policy_pred = policy_pred.gather(1, action_idx.unsqueeze(1).to(device))
                    target_best = torch.max(info['target_net'](next_state.to(device)), dim=1, keepdim=True).values
                    error = torch.abs(
                        reward.unsqueeze(1).to(device) + config.train.discount * target_best - policy_pred
                    ).cpu().numpy()
                    info['policy_net'].train()

                # Push the sample into the replay memory
                sample = torch.cat(
                    [state, action_idx.unsqueeze(1), reward.unsqueeze(1), next_state], dim=1
                )
                info['memory'].add(error, sample)

                # State <- Next state
                state = next_state

                loss, max_loss = self.train_Q(config, info)

                step = episode * config.train.max_step + t + 1
                writer.add_scalar('mean loss', loss.item(), step)
                writer.add_scalar('max loss', max_loss.item(), step)

                # Logging
                if t % log_freq == log_freq - 1:
                    logging.info(f'epi {episode} step {t+1} loss {loss.item():.3e} max_loss {max_loss.item():.3e}')

            info['episode'] = episode + 1
            info['scheduler'].step()

            # Target update
            if info['episode'] % config.train.target_update == 0:
                info['target_net'].load_state_dict(info['policy_net'].state_dict())

            # Save checkpoint (save as ckpt.pt - overwritten)
            if info['episode'] % config.train.save_freq == 0 or info['episode'] == config.train.max_episode:
                save_ckpt(ckpt_dir, info, archive=False)

            # Evaluate
            if info['episode'] % config.train.eval_freq == 0:
                self.eval(config, logdir)

            # Archive checkpoint (save as ckpt_123.pt)
            if info['episode'] % config.train.archive_freq == 0 or info['episode'] == config.train.max_episode:
                save_ckpt(ckpt_dir, info, archive=True)

    def eval(self, config, logdir, ckpt_num=None):
        device = config.device
        policy_net = DQN(config, len(self.action_set)).to(device)
        info = {
            'policy_net': policy_net,
            'episode': 0,
        }
        ckpt_dir = os.path.join(logdir, 'checkpoints')
        info = load_ckpt(ckpt_dir, info, device, train=False, ckpt_num=ckpt_num)
        info['policy_net'].eval()
        init_state = config.dynamics.init_state
        init_state = torch.tensor(init_state).float()

        state = init_state.unsqueeze(0)
        states = []
        actions = []
        rewards = []

        for t in range(config.train.max_step):
            best_action_idx = torch.argmax(info['policy_net'](state.to(device)), dim=1)
            action = self.action_set[best_action_idx]
            # action = torch.tensor([0.0, 0.0])
            reward, next_state = self.env.step(state, action)

            states.append(state)
            actions.append(action)
            rewards.append(reward.unsqueeze(0))

            state = next_state

        states = torch.cat(states, dim=0).numpy()
        actions = torch.cat(actions, dim=0).numpy()
        rewards = torch.cat(rewards, dim=0).numpy()

        cum_rewards = rewards.sum()

        # np.savez(
        #     './actions.npz',
        #     action=actions,
        # )
        # np.savez(
        #     './rewards.npz',
        #     reward=rewards,
        # )

        fig = plt.figure(figsize=(16, 10))
        scaler = config.reward.scaler
        plt.title(f'Episode {info["episode"]} | Cumulative reward {cum_rewards * scaler:.5e}')
        plt.axis('off')
        axis_t = np.arange(0, config.train.max_step)
        legends = ['T1', 'T2', 'T1I', 'T2I', 'V', 'E', 'a1', 'a2', 'reward']

        # states
        for i in range(6):
            ax = fig.add_subplot(3, 3, i+1)
            ax.plot(axis_t, states[:, i], label=legends[i])
            ax.legend()
            ax.grid()

        # actions
        for i in range(2):
            ax = fig.add_subplot(3, 3, i+7)
            ax.plot(axis_t, actions[:, i], label=legends[i+6], color='r')
            ax.legend()
            ax.grid()

        # rewards
        ax = fig.add_subplot(3, 3, 9)
        ax.plot(axis_t, rewards, label=legends[-1], color='g')
        ax.legend()
        ax.grid()

        evaldir = os.path.join(logdir, 'eval')
        os.makedirs(evaldir, exist_ok=True)
        fig.savefig(os.path.join(evaldir, f'result_{info["episode"]}.png'))
        plt.close()



