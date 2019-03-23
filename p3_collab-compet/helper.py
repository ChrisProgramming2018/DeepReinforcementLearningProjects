""" Implementation of ddpg  """
import numpy as np
import matplotlib.pyplot as plt





def ddpg(env, ma, n_episodes=3000, max_t=2000):
    """ 
    
    
    """

    overall_scorces = []
    score_window = deque(max_len(100)) # saves last 100 rewards
    t0 = time.time()

    for i_episode in range(1, n_episodes + 1):

        ma.reset()
        states, scores = env.reset()
        for _ in range(max_t):
            actions = ma.act(states)
            next_states, actions, rewards, dones = env.step(action) 
            ma.step(states, actions, rewards, next_states, dones)
            scores = scores + rewards
            states = next_states

        avg_score = np.mean(scores)
        score_window.append(avg_score)
        overall_scorces.append(avg_score)

        print('\r Epoisode {}\t Score: {:.2f}, Average Score: {:.2f}  Time: {:.2f}'.format(i_episode, np.mean(scores), np.mean(score_window), time.time() - t0))
        if np.mean(score_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-5, np.mean(scores_window)))
            print("Save Agensts"
            torch.save(ma.ddpg_agents[0].actor_online.state_dict(), 'checkpoint_actor.pth')
            torch.save(ma.ddpg_agents[0].critic_online.state_dict(), 'checkpoint_critic.pth')
            break
    return overall_scorces



def plot_score(scores):
    """   
    Args:
        param1:(list) scores of the episode 
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
