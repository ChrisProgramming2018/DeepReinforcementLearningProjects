""" Replay buffer for experience without PER """
from collections import namedtuple, deque
import random
import numpy as np
import torch


class ReplayBuffer(object):
    """ Standard replay buffer to store and sample experience """
    def __init__(self, buffer_size, batch_size, seed=0, device="cpu"):
        """ Init all vars
        Args:
           param1: (int) buffer_size
           param2: (int) batch_size
           param3: (int) seed
           param4: (string) device
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """ Adds experience to the buffer
        Args:
           param1: (numpy.array) state
           param2: (int) action
           param3: (int) reward
           param4: (torch.tensor) next_state
           param5: (numpy.array) done
        """
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)

    def sample(self):
        """ Returns a experience of the batch_size
        Return (state, action, reward, next_state, done)
        """
        experience = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None])).astype(np.uint8).float().to(self.device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """ Returns the current size of the buffer
        Return buffer size
        """
        return len(self.memory)
