"""
Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)
Permission given to modify the code as long as you keep this
declaration at the top

I modified the file to full fill my purpose

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from envs import tensor

class DDPGAgent(object):
    """ use the DDPG from the research paper """
    def __init__(self, config):
        """

        Args:
           param1 (config): config
        """
        self.config = config
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None
        self.episode_reward = 0
        self.episode_rewards = []

    def soft_update(self, target, src):
        """

        Args:
            param1 (torch): target
            param2 (torch): src
        """
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) + param * self.config.target_network_mix)


    def learn(self, replay_buffer):
        """ actor critic  """
        
        experiences = replay_buffer.sample()
        states, actions, rewards, next_states, terminals = experiences
        #states = states.squeeze(1)
        #actions = actions.squeeze(1)
        rewards = tensor(rewards)
        #rewards = rewards.unsqueeze(1)
        #next_states = next_states.squeeze(1)
        terminals = tensor(terminals)

        phi_next = self.target_network.feature(next_states)
        a_next = self.target_network.actor(phi_next)
        q_next = self.target_network.critic(phi_next, a_next)
        q_next = self.config.discount * q_next * (1 - terminals)
        q_next.add_(rewards)
        q_next = q_next.detach()
        phi = self.network.feature(states)
        q = self.network.critic(phi, tensor(actions))
        critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()
     
        # improve the crtic network weights
        self.network.zero_grad()
        critic_loss.backward()
        self.network.critic_opt.step()
        
        # actor gets actions
        phi = self.network.feature(states)
        action = self.network.actor(phi)
        policy_loss = -self.network.critic(phi.detach(), action).mean()

        # improve the actor network weights
        self.network.zero_grad()
        policy_loss.backward()
        self.network.actor_opt.step()

        # copy the weigths from online(network) to the target network
        self.soft_update(self.target_network, self.network)


    def save(self, filename):
        """ save the network weights to the given path
        Args:
            param1 (string): filename
        """
        torch.save(self.network.state_dict(), filename)

    def load(self, filename):
        """ loads the network from the filename
        Args:
            param1 (string):
        """
        state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)


class FCBody(nn.Module):
    """ Fully connected layer"""
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        """

        Args:
           param1 (int): state_dim
           param2 (tuple): hidden_units
           param3 (torch.activationfunction): gate
        """
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        """

        Args:
            param1 (torch): x state
        """
        for layer in self.layers:
            x = self.gate(layer(x))
        return x

def layer_init(layer, w_scale=1.0):
    """

    Args:
        param1 (torch layer) : layer
        param2 (float)
    """
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class TwoLayerFCBodyWithAction(nn.Module):
    """ special layer the network """
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        """

        Args:
           param1 (int): state_dim
           param2 (tuple): hidden_units
           param3 (torch.activationfunction): gate
        """
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        """

        Args:
           param1 (x)

        Return:
        """
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi
