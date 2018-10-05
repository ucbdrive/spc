import pdb
import os
import argparse
from args import init_parser
import numpy as np
import torch
from train_cont_policy import train_policy
from utils import init_dirs
import sys
import random
from pong_wrapper import pong_wrapper

parser = argparse.ArgumentParser(description='Train-pong')
init_parser(parser)  # See `args.py` for default arguments
args = parser.parse_args()

if __name__ == '__main__':
    init_dirs([args.save_path,
               os.path.join(args.save_path, 'model'),
               os.path.join(args.save_path, 'optimizer'),
               os.path.join(args.save_path, 'cnfmat')])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    env = pong_wrapper()
    train_policy(args, env, num_steps=400000000)
