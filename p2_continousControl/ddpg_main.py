""" main file to train agent in the env """
from collections import deque
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from unityagents import UnityEnvironment

from envs import OrnsteinUhlenbeckProcess
from envs import LinearSchedule
from config import Config
from ac_model import DeterministicActorCriticNet
from memory import Replay
from ddpg_agent import DDPGAgent
from ddpg_agent import FCBody
from ddpg_agent import TwoLayerFCBodyWithAction



def main(arg):
    """

    Args:
        param1: (args)
        param2: (config)

    """
    env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64', no_graphics=True)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    states = env_info.vector_observations

    print('Size of each action:', brain.vector_action_space_size)
    print(states.shape[1])

    config = Config()
    config.state_dim = states.shape[1]
    config.action_dim = brain.vector_action_space_size


    config = set_config(config, arg)
    agent = DDPGAgent(config)
    agent.random_process = config.random_process_fn()
    env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
    state = env_info.vector_observations                  # get the current state (for each agent)
    t_0 = time.time()
    n_episodes = 1500
    train_every = 10
    total_steps = 0
    scores_window = deque(maxlen=100)  # last 100 scores
    scores = []
    for i_episode in range(1, n_episodes+1):
        episode_reward = 0
        env_info = env.reset(train_mode=True)[brain_name]
        agent.random_process.reset_states()
        while True:
            total_steps += 1
            action = agent.network(state)
            action = action.cpu().detach().numpy()
            action += agent.random_process.sample()
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards[0]
            done = np.array(env_info.local_done)
            episode_reward += reward
            agent.replay.feed([state, action, reward, next_state, done.astype(np.uint8)])
            state = next_state
            if  total_steps % train_every == 0:
                agent.learn()
            if done:
                scores.append(episode_reward)
                scores_window.append(episode_reward)
                print('\rEpisode {}\t Average Score: {:.2f} , Score: {:.2f} Time: {:.2f}'.format(i_episode, np.mean(scores_window), episode_reward, time.time() - t_0))
                break
            if np.mean(scores_window) >= 30:
                print('total steps: ', total_steps)
                agent.save('smart')
                print("save smart agent")
                return  scores


def save_and_plot(score, model_num):
    """ saves the result of the training into the given file
    Args:
        param1 (list): score
        param2 (int):
    """
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(score)), score)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores.png')
    # plt.show()

    df = pd.DataFrame({'episode':np.arange(len(score)), 'score':score})
    df.set_index('episode', inplace=True)
    df.to_csv('scores.csv'.format(model_num))



def set_config(config, args):
    """
    Args:
       param1: (args): args
    Return config
    """
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    config.save_interval = 10000
    config.discounti = args.discount
    config.network_fn = lambda: DeterministicActorCriticNet(config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, (args.hidden_size1, args.hidden_size2), gate=F.relu),
            critic_body=TwoLayerFCBodyWithAction(config.state_dim, config.action_dim, (args.hidden_size1, args.hidden_size2), gate=F.relu), actor_opt_fn=lambda params: torch.optim.Adam(params, lr=args.lr), critic_opt_fn=lambda params: torch.optim.Adam(params, lr=args.lr))
    config.replay_fn = lambda: Replay(memory_size=int(20000), batch_size=args.batch_size)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(size=(config.action_dim, ), std=LinearSchedule(0.2))
    config.min_memory_size = 10000
    config.target_network_mix = 1e-3
    config.DEVICE = 'cuda:0'

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG')
    parser.add_argument('--hidden-size1', type=int, default=400, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--hidden-size2', type=int, default=300, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='mue', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='eps', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=128, metavar='SIZE', help='Batch size')
    parser.add_argument('--discount', type=float, default=0.99, metavar='gamma', help='Discount factor')
    parser.add_argument('--n_episodes', default=500)
    parser.add_argument('--model_num', default=0)
    arg = parser.parse_args()
    s = main(arg)
    save_and_plot(s, arg.model_num)
