""" agent to learn Q Function and act according to it
    use Neural Network to approximate it
"""
import os
import numpy as np
import torch
from torch import optim
from model import DQN


class Agent(object):
    """ all improvments from Rainbow research work
    """
    def __init__(self, args, state_size, action_size):
        """
        Args:
           param1 (args): args
           param2 (int): args
           param3 (int): args
        """
        self.action_size = action_size
        self.state_size = state_size
        self.atoms = args.atoms
        self.V_min = args.V_min
        self.V_max = args.V_max
        self.device = args.device
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=self.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount

        self.online_net = DQN(args, self.state_size, self.action_size).to(device=args.device)
        if args.model and os.path.isfile(args.model):
            # Always load tensors onto CPU by default, will shift to GPU if necessary
            self.online_net.load_state_dict(torch.load(args.model, map_location='cpu'))
        self.online_net.train()

        self.target_net = DQN(args, self.state_size, self.action_size).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=args.lr, eps=args.adam_eps)

    def reset_noise(self):
        """ resets noisy weights in all linear layers """
        self.online_net.reset_noise()

    def act(self, state):
        """
          acts greedy(max) based on a single state
          Args:
             param1 (int) : state
        """
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0).to(self.device)) * self.support).sum(2).argmax(1).item()

    def act_e_greedy(self, state, epsilon=0.001):
        """ acts with epsilon greedy policy
            epsilon exploration vs exploitation traide off
        Args:
            param1(int): state
            param2(float): epsilon
        Return : action int number between 0 and 4
        """
        return np.random.randint(0, self.action_size) if np.random.random() < epsilon else self.act(state)

    def learn(self, mem):
        """ uses samples with the given batch size to improve the Q function
        Args:
            param1 (Experince Replay Buffer) : mem
        """
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, *; theta online)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; theat online)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, *; theta online)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, *; theat online))
            argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a;  theat online))]
            self.target_net.reset_noise()  # Sample new target net noise
            pns = self.target_net(next_states)  # Probabilities p(s_t+n,  ; theata target)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; theat online))]; theat target)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (discoit ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.V_min, max=self.V_max)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.V_min) / self.delta_z  # b = (Tz - Vmin) / delta z
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        self.optimizer.step()

        mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions
        self.soft_update()

    def soft_update(self, tau=1e-3):
        """ swaps the network weights from the online to the target

        Args:
           param1 (float): tau
        """
        for target_param, local_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def update_target_net(self):
        """ copy the model weights from the online to the target network """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path):
        """ save the model weights to a file
        Args:
           param1 (string): pathname
        """
        torch.save(self.online_net.state_dict(), os.path.join(path, 'model.pth'))

    def evaluate_q(self, state):
        """ Evaluates Q-value based on single state
        """
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
        """
        activates the backprob. layers for the online network
        """
        self.online_net.train()

    def eval(self):
        """ invoke the eval from the online network
            deactivates the backprob
            layers like dropout will work in eval model instead
        """
        self.online_net.eval()
