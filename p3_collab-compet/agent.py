"""DDPG Agent """
import numpy as np
import torch
import torch.nn.functional as F
from model import Actor, Critic
from noise import OUNoise

class DDPGAgent(object):
    """ deep deterministic policy gradient """
    def __init__(self, args, memory):
        """ set actor and crtic network and hyperparameter

        Args:
            param1: (args) commandline arguments
            param2: (ReplayBuffer) experience stored
        """
        self.args = args
        self.device = args.device
        self.seed = args.seed
        self.gamma = args.discount
        self.tau = args.tau
        # Actor online and target network
        self.actor_online = Actor(args.action_size, args.state_size, args.actor_hidden_units, args.seed).to(args.device)
        self.actor_target = Actor(args.action_size, args.state_size, args.actor_hidden_units, args.seed).to(args.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_online.parameters(), lr=args.actor_learning_rate)

        # Critic
        self.critic_online = Critic(args.action_size, args.state_size, args.critic_hidden_units, args.seed).to(args.device)
        self.critic_target = Critic(args.action_size, args.state_size, args.critic_hidden_units, args.seed).to(args.device)
        self.critic_optimizer = torch.optim.Adam(self.critic_online.parameters(), lr=args.critic_learning_rate)

        self.soft_update(self.actor_online, self.actor_target)
        self.soft_update(self.critic_online, self.critic_target)

        self.noise = OUNoise(args.action_size, args.seed)
        self.memory = memory

    def noise_reset(self):
        """ set the  """
        self.noise.reset()

    def act(self, states, explore=True):
        """ actor maps the states with the policy to actions
            and noise to explore
        Args:
           param1: (numpy) states
           param2: (Boole) true of add noise

        Return: actions between -1 and 1
        """
        states = torch.from_numpy(states).float().to(self.device)
        self.actor_online.eval()  # deactivate backprob for inference
        with torch.no_grad():
            actions = self.actor_online(states).cpu().data.numpy()
        self.actor_online.train()
        if explore:
            actions = actions + self.noise.sample()
        return np.clip(actions, -1, 1)

    def learn(self, experience):
        """ Update the actor and critic weights to improve policy
        Args:
            param1: (tuple) experience (s, a, r, s, done)
        """

        states, actions, rewards, next_states, dones = experience

        actor_actions = self.actor_target(next_states)   # use actor_tar network take action from next states
        q_target_next = self.critic_target(actor_actions, next_states) # evalute the actions taken from actor

        q_target = rewards +(self.gamma * q_target_next * (1 - dones))  # bellman equation

        q_expected = self.critic_online(actions, states)

        #---------------------------------- update critic ------------------------------------------------------
        critic_loss = F.mse_loss(q_expected, q_target) # compute the loss critic online
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #---------------------------------- update actor ------------------------------------------------------
        pred_actions = self.actor_online(states)
        actor_loss = -self.critic_online(pred_actions, states).mean()  # compute loss actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()                                          # minimize loss
        self.actor_optimizer.step()

        #---------------------------------- update target networks ------------------------------------------------------
        self.soft_update(self.critic_online, self.critic_target, self.tau)
        self.soft_update(self.actor_online, self.actor_target, self.tau)


    def soft_update(self, online_model, target_model, tau=1):
        """ copy weights from online model to the target model with tau
        Args:
            param1:(model) online_model
            param2:(model) target model
            param3:(float) update param between 0 and 1
        """
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data.copy_(tau*online_param.data + (1.0-tau)*target_param.data)
