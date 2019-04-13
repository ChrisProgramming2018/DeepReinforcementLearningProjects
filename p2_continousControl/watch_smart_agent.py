""" show the smart agents solving the task """
import torch
import numpy as np
from unityagents import UnityEnvironment
from config import Config
from ddpg_agent import Agent


def main(path=''):
    """ show the environment controlled by the 20 smart agents
    Args:
       param1: (string) pathname for saved network weights

    """
    env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64', no_graphics=True)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    config = Config()
    config.discount = 0.99
    config.state_dim = states.shape[1]
    config.action_dim = brain.vector_action_space_size
    action_size = config.action_dim
    state_size = config.state_dim
    seed = 42

    agent = Agent(state_size=state_size, action_size=action_size, n_agents=num_agents, seed=seed)

    agent.actor_local.load_state_dict(torch.load(path + 'checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load(path + 'checkpoint_critic.pth'))
    for _ in range(3):
        episode_reward = []
        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        total_steps = 0
        while True:
            total_steps += 1
            actions = agent.act(states, False)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            reward = env_info.rewards
            done = np.array(env_info.local_done)
            episode_reward.append(np.mean(reward))
            scores += reward
            states = next_states
            if np.any(done):
                print("total steps", total_steps)
                print(sum(episode_reward))
                print('average: ', np.mean(scores))
                print('min: ', np.min(np.array(episode_reward)))
                print('max: ', np.max(np.array(episode_reward)))
                break


if __name__ == "__main__":
    main()
