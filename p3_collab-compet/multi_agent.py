""" multi DDPG Agents  """
import  numpy as np
from agent import DDPGAgent


class MultiAgent(object):
    """ represent the two agents """
    def __init__(self, arg, memory):
        """
        Args:
            param1: (arg) command line arguments parameter
            param2: (ReplayBuffer) saves experience
        """
        self.memory = memory
        self.discount = arg.discount
        self.batch_size = arg.batch_size
        self.update_every = arg.update_every
        self.ddpg_agents = [DDPGAgent(arg, memory) for _ in range(arg.num_agents)]
        self.t_step = 0

    def reset(self):
        """ resets the noise for exploration for the 2 agents """
        for agent in self.ddpg_agents:
            agent.noise_reset()

    def act(self, all_states, explore=True):
        """  ueses the actor network from each agent to choose the actions
             according to the policy
        Args:
           param1:(numpy) all_states
        Return: actions from all agents
        """
        actions = [agent.act(np.expand_dims(states, axis=0), explore) for agent, states in zip(self.ddpg_agents, all_states)]
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """  learns the optimal policy for the every given update step
        Args:
            param1: (numpy) states
            param2: (numpy) actions
            param3: (numpy) rewards
            param4: (numpy) next_states
            param5: (numpy) dones
        """

        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        self.t_step = self.t_step + 1
        if self.t_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                for agent in self.ddpg_agents:
                    agent.learn(self.memory.sample())
