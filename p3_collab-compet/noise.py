""" Ornstein-Uhlenbeck process  """
import copy
import random
import numpy as np

class OUNoise(object):
    """ Implementation of the ORU process """
    def __init__(self, size, seed, n=0, theta=0.15, sigma=0.2):
        """
        set member variables
        Args:
            param1:
            param1:
            param1:
            param1:
        """
        self.n = n *np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        self.state = None

    def reset(self):
        """ set the state noise to the mean of n """
        self.state = copy.copy(self.n)

    def sample(self):
        """ add noise to the internal state and return it
        Return noise sample
        """
        state_x = self.state
        delta_x = self.theta * (self.n - state_x) + self.sigma * np.array([random.random() for _ in range(len(state_x))])
        self.state = state_x + delta_x
        return self.state
