""" multi DDPG Agents  """
import  numpy as np
from agent import DDPGAgent


class MultiAgent(object):
    """ """
    def __init__(self, arg, memory):
        """

        """
        self.memory = memory
        self.discount = arg.discount
        self.batch_size = arg.batch_size
        self.update_every = arg.update_every
        self.ddpg_agents = [DDPGAgent(arg, memory) for _ in range(arg.num_agents)]
        self.t_step = 0

    def reset(self):
        """ """
        for agent in self.ddpg_agents:
            agent.noise_reset()
    
    def act(self, all_states):
        """ 
        Args:
           param1:() all_states
        Return: actions from all agents
        """
        actions = [agent.act(np.expand_dims(states, axis=0)) for agent, states in zip(self.ddpg_agents, all_states)]
        return actions

    def step(self,states, actions, rewards, next_states, dones):
        """ 
        
        Args:
            param1: 
            param2: 
            param3: 
            param4: 
            param5: 
        """

        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = self.t_step + 1
        if self.t_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                for agent in self.ddpg_agents:
                    agent.learn(self.memory.sample())
