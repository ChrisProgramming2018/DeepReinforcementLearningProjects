""" loads the trained network weights to solve the enviroment """
import argparse
import time
import torch
from utils import load_checkpoint
from unityagents import UnityEnvironment
from agent import Agent


def main(args):
    """ shows the agent collecting the bananas
          and print the score to the terminal
    Args:
        param1 (args) : args
    """
    # get the default brain
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset()[brain_name]

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

    dqn = Agent(args, state_size, action_size)

    # load the weights of the trained  Neural Network
    load_checkpoint('dqn_rainbow_agent.pth.tar', dqn.online_net)
    for _ in range(1):
        env_info = env.reset()[brain_name] # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        for _ in range(2000):
            action = dqn.act(torch.from_numpy(state).float())
            env_info = env.step(action)[brain_name]
            state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]
            score += reward
            time.sleep(0.05)
            if done:
                print("score is : {} after the episode".format(score))
                break
    env.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--T-max', type=int, default=int(1000), metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--history-length', type=int, default=1, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--hidden-size', type=int, default=128, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='sigmar', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=10, metavar='k', help='Frequency of sampling from memory')
    parser.add_argument('--priority-exponent', type=float, default=0.8, metavar='omega', help='Prioritised experience replay exponent(originally denoted alpha)')
    parser.add_argument('--priority-weight', type=float, default=0.8, metavar='beta', help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--multi-step', type=int, default=7, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='gamma', help='Discount factor')
    parser.add_argument('--target-update', type=int, default=int(4), metavar='tau', help='Number of steps after which to update target network')
    parser.add_argument('--reward-clip', type=int, default=0, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='meu', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='eps', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=64, metavar='SIZE', help='Batch size')
    parser.add_argument('--learn-start', type=int, default=int(800), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=1000, metavar='STEPS', help='Number of training steps between evaluations')
    parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
    parser.add_argument('--evaluation-size', type=int, default=50000, metavar='N', help='Number of transitions to use for validating Q')
    parser.add_argument('--model_num', default=0)
    parser.add_argument('--eps_decay', type=float, default=0.990)
    parser.add_argument('--n_episodes', default=500)
    parser.add_argument('--env_num', default=1)
    parser.add_argument('--device', type=str, default='cpu')
    arg = parser.parse_args()
    main(arg)
