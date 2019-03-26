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

        avg_score = np.mean(scores)
        score_window.append(avg_score)
        overall_scorces.append(avg_score)

        print('\r Epoisode {}\t Score: {:.2f}, Average Score: {:.2f}  Time: {:.2f} buffer {}'.format(i_episode, avg_score, np.mean(score_window), time.time() - t_0, len(multi_agents.memory)))
        if np.mean(score_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-5, np.mean(score_window)))
            print("Save Agensts")
            torch.save(multi_agents.ddpg_agents[0].actor_online.state_dict(), 'checkpoint_actor.pth')
            torch.save(multi_agents.ddpg_agents[0].critic_online.state_dict(), 'checkpoint_critic.pth')
            break
    return overall_scorces



def plot_score(scores):
    """ creates an png file from the score and saves same dir
    Args:
        param1:(list) scores of the episode
    """
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores.png')
    plt.show()
