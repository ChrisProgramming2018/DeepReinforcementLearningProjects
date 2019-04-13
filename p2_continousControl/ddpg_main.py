""" main file to train agent in the env """  
import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import deque
from unityagents import UnityEnvironment
  
from config import Config
from ddpg_agent import DDPGAgent




def main(arg):
    """

     Args:
         param1: (args)
    """
    env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64', no_graphics=True, seed=arg.seed)
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
    
    epsilon = arg.epsilon
    epsilon_min = arg.epsilon_min
    epsilon_decay = arg.epsilon_decay
    config = Config()
    config.state_dim = states.shape[1]
    config.action_dim = brain.vector_action_space_size
    config.n_agents = num_agents
    set_config(config, arg)
    t_0 = time.time()
    n_episodes = arg.n_episodes
    train_every = arg.train_every
    scores_window = deque(maxlen=100)  # last 100 scores
    agent = DDPGAgent(config)
    scores = []
    print("Start training")
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations                  # get the current state (for each agent)
        agent.reset_noise()
        episode_reward = np.zeros(num_agents)
        for t in range(arg.t_max):
            actions = agent.act(states, epsilon)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            episode_reward += np.array(env_info.rewards)       # update the score (for each agent)
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.memory.add(state, action, reward, next_state, done)
            states = next_states
            if t % train_every == 0:
                for _ in range(arg.repeat_learning):
                    agent.learn()
            
            if np.any(dones):
                break
        epsilon = epsilon * epsilon_decay   
        epsilon = max(epsilon_min, epsilon)
        scores_window.append(np.mean(episode_reward))
        scores.append(np.mean(episode_reward))
        duration = time.time() - t_0
        sec = duration % 60
        minutes = duration // 60
        print('\rEpisode {}\t Average Score all: {:.2f} , Score: {:.2f} Time: min {:.2f} sec: {}'.format(i_episode, np.mean(scores_window), np.mean(episode_reward), minutes, sec))
        if np.mean(scores_window) >= 30:
            print("Enviroment solved save smart agent")
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    return scores



def set_config(config, args):
    """
    
    Args:
       param1: (args): args
       Return config
    """
    config.gamma = args.discount
    config.tau = args.tau
    config.hdl1 = args.hdl1
    config.hdl2 = args.hdl2
    config.hdl3 = args.hdl3
    config.lr_actor = args.lr_actor
    config.lr_critic = args.lr_critic
    config.batch_size = args.batch_size
    config.weight_decay = args.weight_decay
    config.seed = args.seed
    config.leak = args.leak
    config.memory_capacity = args.memory_capacity
    config.repeat_learning = args.repeat_learning
    config.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_and_plot(score, model_num=1):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG')
    parser.add_argument('--hdl1', type=int, default=256, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--hdl2', type=int, default=128, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--hdl3', type=int, default=128, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--lr-actor', type=float, default=1e-4, metavar='mue', help='Learning rate')
    parser.add_argument('--lr-critic', type=float, default=3e-4, metavar='mue', help='Learning rate')
    parser.add_argument('--weight-decay', default=0.0001, metavar='eps', help='weight_dacay')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='SIZE', help='Batch size')
    parser.add_argument('--train-every', default=20)
    parser.add_argument('--epsilon', default=1.0)
    parser.add_argument('--epsilon-min', default=0.005)
    parser.add_argument('--epsilon-decay', default=0.97)
    parser.add_argument('--discount', type=float, default=0.99, metavar='gamma', help='Discount factor')
    parser.add_argument('--n_episodes', default=500)
    parser.add_argument('--repeat-learning', type=int, default=10)
    parser.add_argument('--t-max', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tau', type=float, default=1e-3)
    parser.add_argument('--leak', type=float, default=0.01)
    parser.add_argument('--model_num', default=0)
    arg = parser.parse_args()
    array_scores = main(arg)
    save_and_plot(array_scores)
