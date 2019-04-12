""" Actor Crtic Implementation in pytorch  """
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """ Actor Network implementaion """
    def __init__(self, config):
        """
        Args:
           param1:  (int): Dimension of each state
        """
        super(Actor, self).__init__()
        self.leak = config.leak
        self.seed = torch.manual_seed(config.seed)
        self.hdl1 = nn.Linear(config.state_dim, config.hdl1)
        self.hdl2 = nn.Linear(config.hdl1, config.hdl2)
        self.hdl3 = nn.Linear(config.hdl2, config.action_dim)
        self.batch_norm = nn.BatchNorm1d(config.state_dim)
        self.reset_weights()

    def reset_weights(self):
        """ Initilaize the weights  """
        torch.nn.init.kaiming_normal_(self.hdl1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.hdl2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.hdl3.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        """actor (policy) network  maps states to actions.

        Args:
            param1: (torch) state
        """
        state = self.batch_norm(state)
        x = F.leaky_relu(self.hdl1(state), negative_slope=self.leak)
        x = F.leaky_relu(self.hdl2(x), negative_slope=self.leak)
        x = torch.tanh(self.hdl3(x))
        return x


class Critic(nn.Module):
    """ Critic Network implementaion """
    def __init__(self, config):
        """Initialize parameters and build model.

        Args:
            param1: (config)
        """
        self.leak = config.leak
        self.seed = torch.manual_seed(config.seed)
        self.batch_norm = nn.BatchNorm1d(config.state_dim)
        self.hdl1 = nn.Linear(config.state_dim, config.hdl1)
        self.hdl2 = nn.Linear(config.hdl1 + config.action_dim, config.hdl2)
        self.hdl3 = nn.Linear(config.hdl2, config.hdl3)
        self.hdl4 = nn.Linear(config.hdl3, 1)
        self.reset_weights()

    def reset_weights(self):
        """ Initilaize the weights

        """
        torch.nn.init.kaiming_normal_(self.hdl1.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.kaiming_normal_(self.hdl2.weight.data, a=self.leak, mode='fan_in')
        torch.nn.init.uniform_(self.hdl3.weight.data, -3e-3, 3e-3)

    def forward(self, states, actions):
        """ Build a critic (value) network that maps (state, action) pairs -> Q-values.

        Args:
            param1: (torch) states
            param2: (torch) actions

        """
        states = self.batch_norm(states)
        x = F.leaky_relu(self.hdl1(states), negative_slope=self.leak)
        x = torch.cat((x, actions), dim=1)
        x = F.leaky_relu(self.hdl2(x), negative_slope=self.leak)
        x = F.leaky_relu(self.hdl3(x), negative_slope=self.leak)
        x = self.hdl4(x)
        return x
