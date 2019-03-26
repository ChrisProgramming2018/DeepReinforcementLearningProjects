""" pytroch implemtation of actor and critic """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
    """
    Args:
       param1: (torch) layer

    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Critic(nn.Module):
    """ Model for evalute the value of a state and action """
    def __init__(self, action_size, state_size, hidden_units, seed, gate=F.relu, dropout=0.2):
        """ Creates the layer of the network
        Args:
           param1: (int) action dim
           param2: (int) state dim
           param3: (array) hidden layer nodes
           param4: (float) seed
           param5: (func) action dim
           param6: (float) between 0 and 1 prob. disconnet
        """
        super(Critic, self).__init__()
        self.gate = gate
        self.seed = torch.manual_seed(seed)
        self.dropout = nn.Dropout(p=dropout)
        self.normalizer = nn.BatchNorm1d(state_size)
        self.layers = nn.ModuleList()
        dims = (state_size, ) + hidden_units
        counter = 0
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            if counter == 1:
                self.layers.append(nn.Linear(dim_in + action_size, dim_out))
            else:
                self.layers.append(nn.Linear(dim_in, dim_out))
            counter = counter + 1
        self.output = nn.Linear(dims[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        """ set the weight values to uniform dist """
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, actions, states):
        """ Maps the state and actions to Q-values

        Args:
            param1: (torch) batch of states
            param2: (torch) batch of actions
        """
        normal_states = self.normalizer(states)
        normal_states = self.gate(self.layers[0](normal_states))
        normal_state = torch.cat((normal_states, actions), dim=1)
        for layer in self.layers[1:]:
            normal_state = self.gate(layer(normal_state))
        return self.output(self.dropout(normal_state))


class Actor(nn.Module):
    """ Model for the policy  """
    def __init__(self, action_size, state_size, hidden_units, seed, gate=F.relu, final_gate=F.tanh):
        """
        Args:
           param1: (int) action dim
           param2: (int) state dim
           param3: (array) hidden layer nodes
           param4: (float) seed
           param5: (func) activation function
           param6: (func) last layer activation function
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.normalizer = nn.BatchNorm1d(state_size)
        self.gate = gate
        self.final_gate = final_gate
        dims = (state_size, ) + hidden_units
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.output = nn.Linear(dims[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """ set the weight values to uniform dist """
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        """ maps the state to the action policy
        Args:
            param1: (torch) states
        """
        normal_states = self.normalizer(states)
        for layer in self.layers:
            normal_states = self.gate(layer(normal_states))
        return self.final_gate(self.output(normal_states))
