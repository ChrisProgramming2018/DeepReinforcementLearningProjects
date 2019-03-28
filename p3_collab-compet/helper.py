""" Implementation helper function for ddpg  """
from collections import deque
import time
import numpy as np
import matplotlib.pyplot as plt
import torch



def ddpg(env, multi_agents, n_episodes=3000, max_t=2000):
    """  use the env to train the agents until it reach the
          goal score or run out of episodes return array of score of
          each episode
    Args:
        param1: (Env) enviromnent
        param2: (multi_agents)
        param3: (int) episodes
        param3: (int) steps each episode
    Return: scores of all episodes
    """

    overall_scorces = []
    average_score = []
    agent_1_scores = []
    agent_2_scores = []
    score_window = deque(maxlen=100) # saves last 100 rewards
    t_0 = time.time()
    for i_episode in range(1, n_episodes + 1):
        multi_agents.reset()
        states, scores = env.reset()
        for _ in range(max_t):
            actions = multi_agents.act(states)
            next_states, rewards, dones = env.step(actions)
            multi_agents.step(states, actions, rewards, next_states, dones)
            scores = scores + rewards
            states = next_states
        agent_1_scores.append(scores[0])
        agent_2_scores.append(scores[1])
        avg_score = np.mean(scores)
        score_window.append(avg_score)
        overall_scorces.append(avg_score)
        average_score.append(np.mean(score_window))

        print('\r Epoisode {}\t Score: {:.2f}, Average Score: {:.2f}  Time: {:.2f} buffer {}'.format(i_episode, avg_score, np.mean(score_window), time.time() - t_0, len(multi_agents.memory)))
        if np.mean(score_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-5, np.mean(score_window)))
            print("Save Agensts")
            torch.save(multi_agents.ddpg_agents[0].actor_online.state_dict(), 'checkpoint_actor.pth')
            torch.save(multi_agents.ddpg_agents[0].critic_online.state_dict(), 'checkpoint_critic.pth')
            break
    return overall_scorces, average_score, agent_1_scores, agent_2_scores



def plot_score(scores, average, a1, a2, size=10):
    """ creates an png file from the score and saves same dir
    Args:
        param1:(list) scores of the episode
    """
    plt.rcParams["font.size"] = size
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label='score')
    plt.plot(np.arange(len(average)), average, label='average score')
    plt.plot(np.arange(len(a1)), a1, label='score agent_1')
    plt.plot(np.arange(len(a2)), a2, label='score agent_2')
    plt.legend()
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores.png')
    plt.show()
