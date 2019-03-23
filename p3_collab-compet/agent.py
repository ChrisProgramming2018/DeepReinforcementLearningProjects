"""DDPG Agent """
import random
import numpy as np
import torch
import torch.nn.functional as F
import copy

from noise import OUNoise

class DDPGAgent(object):
    """ """
    def __init__(self, config):
        """

        Args:
            param1: (config) 
        """
        self.config = config
        self.device = config.device
        self.seed = config.seed
        self.gamma = config.gamma
        # Actor online and target network
        self.actor_online = Actor(config.action_size, config.state_size, config.actor_hidden_units, config.seed).to(config.device)
        self.actor_target = Actor(config.action_size, config.state_size, config.actor_hidden_units, config.seed).to(config.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_online.parameters(), lr=config.actor_learning_rate)

        # Critic

        self.critic_online = Critic(config.action_size, config.state_size, config.critic_hidden_units, config.seed).to(config.device)
        self.critic_target = Critic(config.action_size, config.state_size, config.critic_hidden_units, config.seed).to(config.device)
        self.critic_optimizer = torch.optim.Adam(self.critic_online.parameters(), lr=config.critic_learning_rate)

        self.softe_update(self.actor_online, self.actor_target)
        self.softe_update(self.critic_online, self.critic_target)

        self.noise = OUNoise(config.action_size, config.seed)

        self.memory = ReplayBuffer(config.buffer_size, config.batch_size, config.seed, config.device)

    def noise_reset(self):
        """ set the  """
        self.noise.reset()

    def act(self, states):
        """ 
        Args:
           param1: (numpy) states
        
        Return: actions between -1 and 1
        """

        states = torch.from_numpy(states).float().to(self.device)
        self.actor_online.eval()  # deactivate backprob for inference 
        with torch.no_grad():
            actions = self.actor_online(states).cpu().data.numpy()
        self.actor_online.train()
        actions =  actions + self.noise.sample()
        return np.clip(actions, -1, 1)

    def learn(self, experience):
        """


        Args:
            param1: (tuple) experience (s, a, r, s, done)
        """
        states, actions, rewards, next_states, dones = experience

        actor_actions = self.actor_target(next_states)   # use actor_tar network take action from next states
        
        Q_target_next = self.critic_target(next_states, actor_actions) # evalute the actions taken from actor

        Q_target = reward +( gamma * Q_target_next * ( 1 - dones))  # bellman equation

        Q_expected = self.critic_online(states, actions)  

        #---------------------------------- update critic ------------------------------------------------------
        critic_loss = F.mse_loss(Q_expected, Q_target) # compute the loss critic online
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #---------------------------------- update actor ------------------------------------------------------
        pred_actions = self.actor_online(states) 
        actor_loss = -self.critic_online(states, pred_actions).mean()  # compute loss actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()                                          # minimize loss
        self.actor_optimizer.step()

        #---------------------------------- update target networks ------------------------------------------------------
        self.soft_update(self, online_model, target_model, tau=1):
            """

            Args:
                param1:
                par
            """


