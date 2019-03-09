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
    for eps in [0.990, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998]:
        for a in [0.0, 0.4, 0.8]:
            for b in  [0.0, 0.4, 0.8]:
                for step in [2, 3, 4, 5, 7, 8]:
                    if os.path.isfile('results/model-{counter}/scores.png') and counter == 0:
                        print('model-{counter} is already done')
                        counter += 1
                        continue
                    if counter % (args.env_num + args.num_parallel) == 0:
                        print(counter)
                        os.system('python3 ./training.py \
                                --priority-exponent {a} \
                                --priority-weight {b}  \
                                --eps_decay {eps} \
                                --multi-step {step} \
                                --env_num {args.env_num} \
                                --model_num {counter}')
                    counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--env_num', type=int, default=1, help='enviroment')
    parser.add_argument('--num_parallel', type=int, default=0, help='enviroment')
    main(parser.parse_args())
