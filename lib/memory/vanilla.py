import torch
import numpy as np


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
        
