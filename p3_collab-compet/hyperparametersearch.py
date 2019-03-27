""" search for the optimal hyperparameter values  to run several in parallel use
    different """
import os
import argparse


def main(args):
    """ try different values for epsilon and the decay
        and the two parameter from per alpha and beta
    Args:
        param1 (args) :
    """
    counter = args.env_num
    for nodes_1 in [64, 128, 256, 512]:
        for nodes_2 in [64, 128, 256, 512]:
            for batch in [16, 32, 64, 128]:
                for lr_actor in  [1e-1, 1e-2, 1e-3, 5e-3, 1e-4, 3e-4, 5e-4]:
                    for lr_critic in  [1e-1, 1e-2, 1e-3, 5e-3, 1e-4, 3e-4, 5e-4]:
                        for step in [2, 3, 4, 5, 7, 8]:
                            if os.path.isfile('results/model-{counter}/scores.png') and counter == 0:
                                print('model-{counter} is already done')
                                counter += 1
                                continue
                            if counter % (args.env_num + args.num_parallel) == 0:
                                print(counter)
                                os.system(f'python3 ./training.py \
                                        --hidden_units_1 {nodes_1} \
                                        --hidden_units_2 {nodes_2}  \
                                        --batch-size {batch} \
                                        --actor-learning-rate {lr_actor} \
                                        --critic-learning-rate {lr_critic} \
                                        --update_every {step} \
                                        --env_num {args.env_num} \
                                        --model_num {counter}')
                            counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDPG-Multiagent')
    parser.add_argument('--env_num', type=int, default=1, help='enviroment')
    parser.add_argument('--num_parallel', type=int, default=0, help='enviroment')
    main(parser.parse_args())
