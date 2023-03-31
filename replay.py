import random
import torch
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Returns a tensor for each of the Transition elements state, action, next_state, and reward
        and a mask tensor of final states
        """
        batch = random.sample(self.memory, batch_size)
        # Convert list of tuples to list of tuple elements
        state_batch, action_batch, next_state_batch, reward_batch = zip(*batch)
        # Convert to tensors
        final_state_mask = torch.tensor(list(map(lambda s: s is None, next_state_batch)), dtype=bool)
        state_batch = torch.tensor(np.array(state_batch), dtype=float)
        action_batch = torch.tensor(action_batch, dtype=int).unsqueeze(1)
        next_state_batch = torch.tensor(list(filter(lambda s: s is not None, next_state_batch)), dtype=float)
        reward_batch = torch.tensor(reward_batch, dtype=float).unsqueeze(1)
        return  state_batch, action_batch, next_state_batch, reward_batch, final_state_mask

    def __len__(self):
        return len(self.memory)