""" Replay Buffer to save the experience of the agent """

from collections import deque, namedtuple
import random
import torch
import numpy as np

class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, config):
        """Initialize a ReplayBuffer object.

        Args:
            param1: ()
        """
        self.memory = deque(maxlen=config.memory_capacity)  # internal memory (deque)
        self.batch_size = config.batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = config.device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def size(self):
        """Return the current size of internal memory."""
        return len(self.memory)
