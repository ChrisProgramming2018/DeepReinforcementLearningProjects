""" Replay Buffer to save the experience of the agent """
import numpy as np


class Replay(object):
    """  Replay Buffer    """
    def __init__(self, memory_size, batch_size):
        """ set member variables
        Args:
            param1: (int) memory_size
            param2: (int) batch_size
        """
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0

    def feed(self, experience):
        """
            saves  state reward action next_state
        Args:
           param1: (tuple) experience
        """
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        """
        saves a batch of experience
        Args:
           param1: (tuple) experience
        """
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        """
        returns experience of batch_size
        Args:
           param1: (int) batch_size
        Return: experience of the amount  batch_size
        """
        if self.empty():
            return None
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        batch_data = list(map(lambda x: np.asarray(x), zip(*sampled_data)))
        return batch_data

    def size(self):
        """
        Return: size of Replay buffer
        """
        return len(self.data)

    def empty(self):
        """

        Return: True if replay buffer is empty
        """
        if self.data:
            return False
        return True
