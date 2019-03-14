""" Neural Network Actor and Critic  """
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from ddpg_agent import layer_init

class ActorCriticNet(nn.Module):
    """ Body for the ActorCritic Net  """
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        """ init all member variables

        Args:
            param1: (int) state_dim
            param2: (int) action_dim
            param3: () phi_body
            param4: () actor_body
            param5: () critic_body

        """
        super(ActorCriticNet, self).__init__()
        if phi_body is None:
            phi_body = DummyBody(state_dim)
        if actor_body is None:
            actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None:
            critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())


class DeterministicActorCriticNet(nn.Module):
    """ DAC netork  """
    def __init__(self, state_dim, action_dim, actor_opt_fn, critic_opt_fn, phi_body=None, actor_body=None, critic_body=None):
        """  agent that use a NN for the crit
        Args:
            param1: (int) state_dim
            param2: (int) action_dim
            param3: () phi_body
            param4: () actor_body
        """
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.to(torch.device('cpu'))

    def forward(self, obs):
        """ overwritte the given forward function

        Args:
            param1: () obs
        Return: action
        """
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        """
        evaluates a given state
        Args:
            param1: (torch) obs
        Return: prediction of network
        """
        obs = tensor(obs)
        return self.network.phi_body(obs)

    def actor(self, phi):
        """
        use critic network to get the action
        Args:
           param1: () phi

        Return:
        """
        return F.tanh(self.network.fc_action(self.network.actor_body(phi)))

    def critic(self, phi, a):
        """ use critic to evalaute the state value function

        Args:
            param1: () phi
            param2: () a
        Return: prediction of the critic
        """
        return self.network.fc_critic(self.network.critic_body(phi, a))

class DummyBody(nn.Module):
    """ Dummy body for the case no other body is given """
    def __init__(self, state_dim):
        """  init member varibales

        Args:
            param1: (int) state_dim

        """
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        """ place holder
        Args:
           param1: (torch) x
        Return: x
        """
        return x

def tensor(array):
    """ ensure that the state is torch tensor

    Args:
       param1: (numpy)  tensor

    Return: tensor
    """
    if isinstance(array, torch.Tensor):
        return array
    return torch.tensor(array, device=Config.DEVICE, dtype=torch.float32)
