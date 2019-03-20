""" main file to train agent in the env """
import argparse
import sys
import os
import gym
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from collections import deque
from unityagents import UnityEnvironment
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.atari_wrappers import FrameStack as FrameStack_
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv, VecEnv

from envs import Task
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
    action_size = brain.vector_action_space_size
    print(states.shape[1])

    config = Config()
    config.state_dim = states.shape[1]
    config.action_dim = brain.vector_action_space_size


    con = set_config(config, arg) 
    #agent = DDPGAgent(config)
    #agent.random_process = config.random_process_fn()
    t0 = time.time()
    n_episodes = arg.n_episodes
    learn_updates = arg.learn_updates
    train_every = arg.train_every
    total_steps = 0
    scores_window = deque(maxlen=100)  # last 100 scores
    all_agents = [[]for x in range(num_agents)] 
    agents_rewards = [scores_window for x in range(num_agents)]
    agents = [DDPGAgent(config) for _ in range(num_agents)]
    replay_buffer = Replay(memory_size=arg.memory_capacity, batch_size=arg.batch_size)

    # fill replay buffer to minimum
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations                  # get the current state (for each agent)
    for agent in agents:
        agent.random_process.reset_states()
    percent = config.min_memory_size / 100
    T = 0
    while True:
        T = T + 20
        print("In Progress", T /percent, end='\r')
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1 
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            replay_buffer.feed([state, action, reward, next_state, int(done)])
        states = next_states
        if replay_buffer.size() >= config.min_memory_size:
            break
        if np.any(dones):
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations                  # get the current state (for each agent)
            agent.random_process.reset_states()



    print("Train")
    for i_episode in range(1, n_episodes+1):
        scores = np.zeros(num_agents)     
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations # get the current state (for each agent)
        for agent in agents:
            agent.random_process.reset_states()
        scores = np.zeros(num_agents) 
        while True:
            total_steps +=1
            actions = []
            for state, agent in zip(states, agents):
                action = (agent.network(state))
                action = action.cpu().detach().numpy()
                action += agent.random_process.sample()
                actions.append(action)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                replay_buffer.feed([state, action, reward, next_state, int(done)])
            states = next_states
            if  total_steps % train_every == 0:
                for _ in range(learn_updates):
                    for agent in agents:
                        agent.learn(replay_buffer)
            if np.any(dones):
                for score, one_agent, agent_reward in zip(scores, all_agents, agents_rewards):
                    one_agent.append(score)
                    agent_reward.append(score)
                scores_window.append(np.mean(scores))
                print('\rEpisode {}\t Average Score all: {:.2f} , Score: {:.2f} Time: {:.2f}'.format(i_episode,
                            np.mean(scores_window),
                            np.mean(scores), time.time() - t0))
                break
            if np.mean(agents_rewards[0]) >= 30:
                print('total steps: ', total_steps)
                agent.save('smart')
                print("save smart agent")
                return  all_agents
    return  all_agents
            

def save_and_plot(scores, model_num):
    """ saves the result of the training into the given file
    Args:
        param1 (list): score
        param2 (int):
    """
    for i, score in enumerate(scores):
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(np.arange(len(score)), score)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.savefig('results/model-{}/scores{}.png'.format(model_num, i))
        
        df = pd.DataFrame({'episode':np.arange(len(score)), 'score':score})
        df.set_index('episode', inplace=True)
        df.to_csv('results/model-{}/scores{}.csv'.format(model_num, i))



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
    config.discount= args.discount
    config.network_fn = lambda: DeterministicActorCriticNet(
            config.state_dim, config.action_dim,
            actor_body=FCBody(config.state_dim, (args.hidden_size1,
                args.hidden_size2), gate=F.relu),
            critic_body=TwoLayerFCBodyWithAction(config.state_dim,
                config.action_dim, (args.hidden_size1, args.hidden_size2), gate=F.relu),
            actor_opt_fn=lambda params: torch.optim.Adam(params, lr=args.lr),
            critic_opt_fn=lambda params: torch.optim.Adam(params, lr=args.lr))
    config.replay_fn = lambda: Replay(memory_size=args.memory_capacity,
            batch_size=args.batch_size)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(size=(config.action_dim, ), std=LinearSchedule(0.2))
    config.min_memory_size = 10000
    config.target_network_mix = 1e-3
    config.DEVICE = 'cuda:0'
     
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPGg')
    parser.add_argument('--hidden-size1', type=int, default=256, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--hidden-size2', type=int, default=128, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--memory-capacity', type=int, default=int(1e8), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='mue', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='eps', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=128*4, metavar='SIZE', help='Batch size')
    parser.add_argument('--train_every', type=int, default=20)
    parser.add_argument('--discount', type=float, default=0.99, metavar='gamma', help='Discount factor') 
    parser.add_argument('--n_episodes', type=int, default=300)
    parser.add_argument('--learn_updates', type=int, default=10)
    parser.add_argument('--model_num', default=0)
    arg = parser.parse_args()
    v_args = vars(parser.parse_args())
    for key, value in v_args.items():
        exec(f'{key} = {value}')
    for k, v in v_args.items():
        print(' ' * 26 + k + ': ' + str(v))

    os.system(f'mkdir -p results/model-{model_num}')
    with open(f'results/model-{model_num}/training_params.json', 'w') as outfile:
        json.dump(v_args, outfile)
    sol = main(arg)
    model_num = arg.model_num
    save_and_plot(sol, model_num)
