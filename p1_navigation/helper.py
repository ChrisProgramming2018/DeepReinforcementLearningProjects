""" contains support functions to train agent and plot and save results """
from collections import  deque
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from utils import save_checkpoint




def dqn(agent, env, brain_name, mem, args, eps_start=0.7, eps_end=0.01, eps_decay=0.995):
    """Training the agent and saveing the weights of the NN.


    Args:
        param1: (agent) act and learn
        param2: (env) : environment to interact with (state, rewards)
        param3: (brain_name) : name of the enviromn
        param4: (mem): PER buffer
        param5: (args): commandline arguments
        param6: (float): eps start
        param7: (float): minimum value of epsilon
        param8: (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                             # list containing scores from each episode
    scores_window = deque(maxlen=100)       # last 100 scores
    eps = eps_start                   # initialize epsilon
    priority_weight_increase = (1 - args.priority_weight) / (args.max_t - args.learn_start)
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    print('create exampels ', args.evaluation_size)
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0]
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
        mem.append(torch.from_numpy(state), action, reward, done)
        state = next_state
        T += 1
    print("Start training")
    for i_episode in range(1, args.n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]
        state = torch.from_numpy(state).float()
        score = 0
        for T in range(args.max_t):
            action = agent.act_e_greedy(state.float(), eps)  # Choose an action greedily (with noisy weights)
            env_info = env.step(action)[brain_name]
            next_state = torch.from_numpy(env_info.vector_observations[0])   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]
            if args.reward_clip > 0:
                reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
            mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight beta  to 1
            if T % args.target_update == 0:
                agent.learn(mem)  # Train with n-step distributional double-Q learning
            mem.append(state, action, reward, done)  # Append transition to memory
            score += reward
            state = next_state
            if done:
                break

            if i_episode % args.replay_frequency == 0 and args.noise:
                agent.reset_noise()  # Draw a new set of noisy weights


        if np.mean(scores_window) >= 13:
            model_dir = 'models-'
            print('Solved problem after {} episodes'.format(i_episode - 100))
            print('save model of smart agent :) ')
            state = {'epoch': i_episode, 'state_dict': agent.qnetwork_local.state_dict(), 'optim_dict' : agent.optimizer.state_dict()}
            save_checkpoint(state, is_best=True, checkpoint=model_dir+str(args.model_num))
            return scores

        scores_window.append(score)       # save most recent score
        scores.append(score)
        # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f} eps : {}'.format(i_episode, np.mean(scores_window), eps), end="")
    return scores




def save_and_plot(scores, args, model_num):
    """ saves the result and parameter of the model

    Args:
       param1: (list) results of the agent
       param2: (args) parameter of the agent
       param3: (int)  model number

    """
    os.system('mkdir -p results/model-{}'.format(model_num))
    with open('results/model-{model_num}/training_params.json', 'w') as outfile:
        json.dump(vars(args), outfile)

    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('results/model-{}/scores.png'.format(model_num))
    # plt.show()

    df = pd.DataFrame({'episode':np.arange(len(scores)), 'score':scores})
    df.set_index('episode', inplace=True)
    df.to_csv('results/model-{}/scores.csv'.format(model_num))
    os.system('cp model.py results/model-{}/'.format(model_num))
