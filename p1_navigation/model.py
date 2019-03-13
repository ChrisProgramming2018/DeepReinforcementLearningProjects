""" pytorch model Noisy layers, Duelling and distributione
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class NoisyLinear(nn.Module):
    """ Factorised NoisyLinear layers with bias """
    def __init__(self, in_features, out_features, std_init=0.4):
        """
        Args:
            param1 (int) in_features size of state
            param2 (int) out_features action ssize
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """ reset the values of the weights """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        """
        Args:
            param1 (int): size max value of the random noise
        Return:
            the layer values
        """
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """ resets the noise to the weights and bias """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """
        Args:
           param1(int) : input
        Return
             if train is set add noise for exporation
             else without the noise
        """
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    """
       main neural network use duelling architecture
    """
    def __init__(self, args, state_size, action_space):
        """
        Args:
            param1 (args):  argscommand line argument
            param2 (int): state_size
            param3 (int): action_space
        """
        super().__init__()
        self.atoms = args.atoms
        self.action_space = action_space

        self.fc_h_v = NoisyLinear(state_size, args.hidden_size_1, std_init=args.noisy_std)
        self.fc_h1_v = NoisyLinear(args.hidden_size_1, args.hidden_size_2, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(state_size, args.hidden_size_1, std_init=args.noisy_std)
        self.fc_h1_a = NoisyLinear(args.hidden_size_1, args.hidden_size_2, std_init=args.noisy_std)
        self.fc_z_v = NoisyLinear(args.hidden_size_2, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size_2, action_space * self.atoms, std_init=args.noisy_std)

    def forward(self, x, log=False):
        """ forward path for the network
        Args:
           param1 (torch) state
           param2 (bool) log
        Return:
           if log is True Log probabilities with action over second dimension
           else probabilities with action over second dimension
        """
        v = self.fc_z_v(F.relu(self.fc_h1_v(F.relu(self.fc_h_v(x)))))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h1_a(F.relu(self.fc_h_a(x)))))  # Advantage stream
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

    def reset_noise(self):
        """ for only the fully connected layer 'fc' """
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()



class QNetwork(nn.Module):
    """ Standard NN for the DQN """
    def __init__(self, state_size, action_size, hidden_size1, hidden_size2, seed=1):

        """Initialize parameters and build model.
        Args:
            param1 (int): Dimension of each state
            param2 (int): Dimension of each action
            param3 (int): Nodes of the hidden layer 1
            param4 (int): Nodes of the hidden layer 2
            param5 (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.model = nn.Sequential(nn.Linear(state_size, hidden_size1), nn.ReLU(), nn.Linear(hidden_size1, hidden_size2), nn.ReLU(), nn.Linear(hidden_size2, action_size))


    def forward(self, state):
        """" forward path for the NN
        Args:
            param1 (torch_tensor): current state as input
        Return:
        """
        return self.model.forward(state)
