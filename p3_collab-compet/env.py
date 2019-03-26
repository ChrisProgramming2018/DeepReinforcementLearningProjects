""" environment """

import numpy as np
from unityagents import UnityEnvironment


class Env(object):
    """ tennis environment from unity """
    def __init__(self):
        """ init memeber variables   """
        self.env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64', no_graphics=True)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.env_info.vector_observations.shape[1]
        self.num_agents = len(self.env_info.agents)

    def reset(self):
        """ resets the environment set the rewards to zero
        Returns: states and scores array
        """
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        states = env_info.vector_observations
        scores = np.zeros(self.num_agents)
        return states, scores

    def step(self, actions):
        """ takes the actions to take a step in the env
            to get the reward and next_states.
            dones bool if true the episode is over

        Args:
            param1: (numpy)
        Return: next_states , rewards, dones
        """
        env_info = self.env.step(actions)[self.brain_name]
        rewards = env_info.rewards
        next_states = env_info.vector_observations
        dones = env_info.local_done

        return next_states, rewards, dones
