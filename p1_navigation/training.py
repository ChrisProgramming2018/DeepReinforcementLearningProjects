""" main file to train the agent  """
from __future__ import print_function
import argparse
import json
import os
from collections import  deque
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from agent import Agent
from memory import ReplayMemory
from unityagents import UnityEnvironment
from utils import save_checkpoint
matplotlib.use('Agg')
sns.set()

def main(args):
    """ Load the environment and train agent to solve

    Args:
       param1 (args): args
    """
    path = "Banana_Linux/Banana.x86_64"
    env = UnityEnvironment(file_name=path, no_graphics=True, worker_id=int(args.env_num))
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset()[brain_name]

    # Environment
    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)
    args.device = 'cpu'
    dqn = Agent(args, state_size, action_size)
    mem = ReplayMemory(args, args.memory_capacity)
    priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)

    scores = []                             # list containing scores from each episode
    scores_window = deque(maxlen=100)       # last 100 scores
    eps = 1.0
    eps_end = 0.01
    eps_decay = args.eps_decay

    # Construct validation memory
    val_mem = ReplayMemory(args, args.evaluation_size)
    T, done = 0, True
    percent = args.evaluation_size / 100
    while T < args.evaluation_size:
        print("In Progress {}".format(T /percent), end='\r')
        if done:
            env_info = env.reset(train_mode=True)[brain_name] # reset the environment
            state = env_info.vector_observations[0]
            done = False
        action = np.random.choice(action_size)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]
        val_mem.append(torch.from_numpy(state), action, reward, done)
        state = next_state
        T += 1

    # Training loop
    dqn.train()
    T, done = 0, True
    n_episodes = args.n_episodes
    mem = val_mem
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]
        state = torch.from_numpy(state)
        state = state.float()
        score = 0
        for T in range(args.T_max):
            action = dqn.act_e_greedy(state.float(), eps)  # Choose an action greedily (with noisy weights)
            env_info = env.step(action)[brain_name]
            next_state = torch.from_numpy(env_info.vector_observations[0])   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]
            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight beta  to 1
            if T % args.target_update == 0:
                dqn.learn(mem)  # Train with n-step distributional double-Q learning


            mem.append(state, action, reward, done)  # Append transition to memory
            score += reward
            state = next_state
            if done:
                break

            if i_episode % args.replay_frequency == 0:
                dqn.reset_noise()  # Draw a new set of noisy weights

        if np.mean(scores_window) >= 13:
            print('Solved problem after {} episodes'.format(i_episode - 100))
            print('save model of smart agent :) ')
            model_dir = 'models/'
            os.system('mkdir -p models/{model_num}')
            state = {'epoch': i_episode, 'state_dict': dqn.qnetwork_local.state_dict(), 'optim_dict' : dqn.optimizer.state_dict()}
            save_checkpoint(state, is_best=True, checkpoint=model_dir+str(args.model_num))
            text = 'Solved problem after {} episodes'.format(i_episode - 100)
            with open("results.txt", "a") as myfile:
                myfile.write(text)
                myfile.write(" parameters : " + str(args.eps_decay))
                myfile.write(" " + str(args.priority_exponent))
                myfile.write(" " + str(args.priority_weight))
                myfile.write(" " + str(args.multi_step))
                myfile.write('\n')

            break

        scores_window.append(score)       # save most recent score
        scores.append(score)
        # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f} eps : {}'.format(i_episode, np.mean(scores_window), eps), end="")
    return scores


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
    plt.savefig('results/model-{}/scores.png'.format(model_num))
    # plt.show()

    df = pd.DataFrame({'episode':np.arange(len(score)), 'score':score})
    df.set_index('episode', inplace=True)
    df.to_csv('results/model-{}/scores.csv'.format(model_num))

    os.system('cp model.py results/model-{}/'.format(model_num))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--T-max', type=int, default=int(1000), metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--history-length', type=int, default=1, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--hidden-size-1', type=int, default=128, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--hidden-size-2', type=int, default=64, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='sigma', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=10, metavar='k', help='Frequency of sampling from memory')
    parser.add_argument('--priority-exponent', type=float, default=0.8, metavar='omega', help='Prioritised experience replay exponent (originally denoted alpha)')
    parser.add_argument('--priority-weight', type=float, default=0.8, metavar='beata', help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--multi-step', type=int, default=7, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='gamma', help='Discount factor')
    parser.add_argument('--target-update', type=int, default=int(4), metavar='tau', help='Number of steps after which to update target network')
    parser.add_argument('--reward-clip', type=int, default=0, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='mue', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='eps', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=64, metavar='SIZE', help='Batch size')
    parser.add_argument('--learn-start', type=int, default=int(800), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=1000, metavar='STEPS', help='Number of training steps between evaluations')
    parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
    parser.add_argument('--evaluation-size', type=int, default=50000, metavar='N', help='Number of transitions to use for validating Q')
    parser.add_argument('--model_num', default=0)
    parser.add_argument('--eps_decay', type=float, default=0.99)
    parser.add_argument('--n_episodes', default=2000)
    parser.add_argument('--env_num', default=1)
    v_args = vars(parser.parse_args())
    arg = parser.parse_args()
    for k, v in v_args.items():
        print(' ' * 26 + k + ': ' + str(v))
    PATH = 'results/model-{}'.format(arg.model_num)
    os.system('mkdir -p '+ PATH)
    with open(PATH + '/training_params.json', 'w') as outfile:
        json.dump(v_args, outfile)
    s = main(arg)
    save_and_plot(s, arg.model_num)
