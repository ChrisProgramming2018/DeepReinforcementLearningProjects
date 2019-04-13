""" implementation of the DDPG algorithm  """
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Actor, Critic
from noise import OUNoise
from memory import ReplayBuffer




class DDPGAgent(object):
    """ class of the DDPG Agent """
    def __init__(self, config):
        """Initialize an Agent object.

        Args:
            param1: (config)
        """

        self.state_size = config.state_dim
        self.action_size = config.action_dim
        self.seed = np.random.seed(config.seed)
        self.n_agents = config.n_agents
        self.batch_size = config.batch_size
        self.tau = config.tau
        self.gamma = config.gamma
        self.device = config.device
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(config).to(config.device)
        self.actor_target = Actor(config).to(config.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(config).to(config.device)
        self.critic_target = Critic(config).to(config.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.lr_critic)

        # Noise process
        self.noise = OUNoise(config)

        # Replay memory
        self.memory = ReplayBuffer(config)
        #self.timesteps = 0

    def act(self, states, epsilon, add_noise=True):
        """ Given a list of states for each agent it returns the actions to be
        taken by each agent based on the current policy.
        Returns a numpy array of shape [n_agents, n_actions]
        NOTE: clips actions to be between -1, 1
        Args:
            states:    (torch) states
            epsilon: (float)
            add_noise: (bool) add noise to the actions
        """
        states = torch.from_numpy(states).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise and epsilon > np.random.random():
            actions += [self.noise.sample() for _ in range(self.n_agents)]
        return np.clip(actions, -1, 1)

    def reset_noise(self):
        """ reset noise"""
        self.noise.reset()

    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
        actor_target(state) -> action
        critic_target(state, action) -> Q-value
        """
        if self.batch_size > self.memory.size():
            return
        states, actions, rewards, next_states, dones = self.memory.sample()

        # ---------------------------- update critic ----------------------------

        # Get predicted next-state actions and Q values from target model

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.

        Args:
         param1: (torch network) local_model
         param2: (torch network) target_model
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
