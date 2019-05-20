import argparse
from args import init_parser, post_processing
import numpy as np
import torch
from train import train_policy
from evaluate import evaluate_policy
from utils import setup_dirs
import random
from envs import make_env

parser = argparse.ArgumentParser(description='SPC')
init_parser(parser)  # See `args.py` for default arguments
args = parser.parse_args()
args = post_processing(args)

if __name__ == '__main__':
    setup_dirs(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if 'carla' in args.env:
        from carla.client import make_carla_client
        from envs.CARLA.carla_env import CarlaEnv

        with make_carla_client('localhost', 2019) as client:
            env = CarlaEnv(client)
            if args.eval:
                evaluate_policy(args, env)
            else:
                train_policy(args, env, max_steps=40000000)
    else:
        with make_env(args) as env:
            if args.eval:
                evaluate_policy(args, env)
            else:
                train_policy(args, env, max_steps=40000000)
