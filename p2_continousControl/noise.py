""" Implemetation of the Ornstein-Uhlenbeck process to add noise to actions to explore """
import copy
import numpy as np


class OUNoise(object):
    """Ornstein-Uhlenbeck process."""
    def __init__(self, config, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mean = mu * np.ones(config.action_dim)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(config.seed)
        self.state = None
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mean)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        delta_x = self.theta * (self.mean - x) + self.sigma * np.array([np.random.random() for _ in range(len(x))])
        self.state = x + delta_x
        return self.state
