from typing import *
import numpy as np
import torch


class PrioritizedReplayBuffer:
    e = 0.01
    a = 1.0 # 0.6
    beta = 0.4

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.priorities = np.zeros(shape=(capacity,), dtype=np.float32)
        self.data = np.zeros(shape=(capacity, 15), dtype=np.float32)
        self.idx = 0
        self.max_idx = 0
        self.is_full = False

    def get_priority(self, error) -> np.ndarray:
        return (np.abs(error) + self.e) ** self.a

    def add(self, error: np.ndarray, sample: np.ndarray) -> None:
        B = sample.shape[0]
        assert self.capacity % B == 0
        p = self.get_priority(error)
        self.priorities[self.idx:self.idx+B] = p.reshape(-1)
        self.data[self.idx:self.idx+B] = sample
        self.idx += B
        if not self.is_full:
            self.max_idx = self.idx
        if self.idx >= self.capacity:
            self.idx = 0 
            self.is_full = True

    def sample(self, n: int) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        total_idxs = np.arange(self.max_idx)
        p = self.priorities[:self.max_idx] / self.priorities[:self.max_idx].sum()
        valid = np.where(p > 2 / self.max_idx)[0] # Focus on not small probabilities
        if len(valid) == 0:
            valid = total_idxs
        total_idxs = total_idxs[valid]
        p = p[valid]
        p /= p.sum()
        batch_idxs = np.random.choice(total_idxs, n, p=p)
        batch = self.data[batch_idxs]
        batch_priorities = self.priorities[batch_idxs]
        batch_priorities /= batch_priorities.sum()
        is_weight = np.power(self.max_idx * batch_priorities, -self.beta)
        is_weight /= is_weight.max()
        return torch.from_numpy(batch), batch_idxs, is_weight

    def update(self, idxs: np.ndarray, error: np.ndarray) -> None:
        p = self.get_priority(error)
        self.priorities[idxs] = p.reshape(-1)


# class PrioritizedReplayBuffer:
#     e = 0.01
#     a = 1.0 # 0.6
#     beta = 0.4
#     # beta_increment_per_sampling = 0.001

#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.priorities = np.array([], dtype=np.float32)
#         self.data = None

#     def _get_priority(self, error):
#         return (np.abs(error) + self.e) ** self.a

#     def add(self, error, sample):
#         p = self._get_priority(error)
#         self.priorities = np.append(self.priorities, p)
#         if self.data is None:
#             self.data = sample.clone()
#         else:
#             self.data = torch.cat([self.data.cpu(), sample.clone()], dim=0)

#     def sample(self, n):
#         data_idxs = np.arange(self.data.shape[0])
#         p = self.priorities / self.priorities.sum()
        
#         batch_idxs = np.random.choice(data_idxs, n, p=p)

#         batch = self.data[batch_idxs]

#         batch_priorities = self.priorities[batch_idxs]
#         batch_priorities /= batch_priorities.sum()
#         is_weight = np.power(self.data.shape[0] * batch_priorities, -self.beta)
#         is_weight /= is_weight.max()

#         return batch, batch_idxs, is_weight

#     def update(self, idxs, errors):
#         p = self._get_priority(errors)
#         self.priorities[idxs] = p.reshape(-1)
