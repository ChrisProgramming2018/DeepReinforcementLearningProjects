"""  compares 3 different agent and creates the score plot for each  """
import argparse
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from agent import Agent
from duelling_agent import Duelling_DDQNAgent
from ddqn_agent import Double_DQNAgent
from helper import dqn, save_and_plot
from memory import ReplayMemory


def main(args):
    ''' compares 3 different agents

    Args:
        param1 (args) : command line argumente
    '''
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64", no_graphics=True)
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


    agent = Duelling_DDQNAgent(args, state_size=state_size, action_size=action_size)
    mem = ReplayMemory(args, args.evaluation_size)
    scores = dqn(agent, env, brain_name, mem, args, n_episodes=args.n_episodes, eps_decay=args.eps_decay)
    save_and_plot(scores, args, 1)

    mem = ReplayMemory(args, args.evaluation_size)
    agent = Double_DQNAgent(args, state_size=state_size, action_size=action_size)
    scores2 = dqn(agent, env, brain_name, mem, args, n_episodes=args.n_episodes, eps_decay=args.eps_decay)
    save_and_plot(scores2, args, 2)

    args.priority_exponent = 0.8
    args.multi_step = 7
    args.update_every = 4
    args.noise = True
    mem = ReplayMemory(args, args.evaluation_size)
    agent = Agent(args, state_size=state_size, action_size=action_size)
    scores3 = dqn(agent, env, brain_name, mem, args, n_episodes=args.n_episodes, eps_decay=args.eps_decay)
    save_and_plot(scores3, args, 3)


    # plot the scores
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label="Duelling Double DQN")
    plt.plot(np.arange(len(scores2)), scores2, label="Double DQN")
    plt.plot(np.arange(len(scores3)), scores3, label="Rainblow")
    plt.legend()
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', default=500)
    parser.add_argument('--max_t', default=2000)
    parser.add_argument('--eps_start', default=1.0)
    parser.add_argument('--eps_end', default=0.01)
    parser.add_argument('--eps_decay', default=0.990)
    parser.add_argument('--buffer-size', default=1e5, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--noise', default=False)
    parser.add_argument('--tau', default=1e-3)
    parser.add_argument('--lr', default=5e-4)
    parser.add_argument('--update-every', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--hidden_size_1', default=128)
    parser.add_argument('--hidden_size_2', default=64, type=int)
    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--replay-frequency', type=int, default=10, metavar='k', help='Frequency of sampling from memory')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='sigma', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--learn-start', type=int, default=int(800), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--adam-eps', type=float, default=1e-8, metavar='eps', help='Adam epsilon')
    parser.add_argument('--priority-exponent', type=float, default=0.0, metavar='omega', help='Prioritised experience replay exponent (originally denoted alpha)')
    parser.add_argument('--priority-weight', type=float, default=0.8, metavar='beata', help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--evaluation-size', type=int, default=50000, metavar='N', help='Number of transitions to use for validating Q')
    parser.add_argument('--history-length', type=int, default=1, metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--discount', type=float, default=0.99, metavar='gamma', help='Discount factor')
    parser.add_argument('--multi-step', type=int, default=1, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--target-update', type=int, default=int(4), metavar='tau', help='Number of steps after which to update target network')
    parser.add_argument('--reward-clip', type=int, default=0, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--model_num', default=1)
    arg = parser.parse_args()
    print(arg)
    main(arg)
