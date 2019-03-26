""" watch smart agents play against each other """
import argparse
import torch
import numpy as np
from memory import ReplayBuffer
from env import Env
from multi_agent import MultiAgent

def main(args):
    """
    Args:
        param1: (args) parameter from command line
    """
    memory = ReplayBuffer(args.memory_capacity, args.batch_size)
    env = Env(args.no_graphic, train_mode=False, seed=args.seed)
    args.num_agents = env.num_agents
    args.action_size = env.action_size
    args.state_size = env.state_size

    multi_agents = MultiAgent(args, memory)
    checkpoint_actor = torch.load('checkpoint_actor.pth')
    multi_agents.ddpg_agents[0].actor_online.load_state_dict(checkpoint_actor)
    multi_agents.ddpg_agents[1].actor_online.load_state_dict(checkpoint_actor)

    checkpoint_critic = torch.load('checkpoint_critic.pth')
    multi_agents.ddpg_agents[0].critic_online.load_state_dict(checkpoint_critic)
    multi_agents.ddpg_agents[1].critic_online.load_state_dict(checkpoint_critic)

    multi_agents.reset()
    states, scores = env.reset()
    steps = 0
    while True:
        steps = steps + 1
        actions = multi_agents.act(states, False)
        next_states, rewards, dones = env.step(actions)
        states = next_states
        scores += rewards
        if np.any(dones):                                  # exit loop if episode finished
            print(steps, scores)
            break




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MultiAgent')
    parser.add_argument('--seed', type=int, default=2, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--actor_hidden_units', type=int, default=(512, 256), metavar='SIZE', help='Network hidden size')
    parser.add_argument('--critic_hidden_units', type=int, default=(512, 256), metavar='SIZE', help='Network hidden size')
    parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--update_every', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
    parser.add_argument('--discount', type=float, default=0.99, metavar='gamma', help='Discount factor')
    parser.add_argument('--tau', type=float, default=1e-3, metavar='gamma', help='Discount factor')
    parser.add_argument('--actor-learning-rate', type=float, default=1e-4, metavar='n', help='Learning rate')
    parser.add_argument('--critic-learning-rate', type=float, default=3e-4, metavar='n', help='Learning rate')
    parser.add_argument('--weight-decay', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=512, metavar='SIZE', help='Batch size')
    parser.add_argument('--model_num', default=0)
    parser.add_argument('--n_episodes', default=500)
    parser.add_argument('--env_num', default=1)
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--no-graphic', default="True")

    main(parser.parse_args())
