import pdb
import os
import argparse
from args import init_parser
import numpy as np
# from envs import create_env
from train_cont_policy import train_policy
from utils import init_dirs
import sys
sys.path.append('/media/xinleipan/data/git/pyTORCS/py_TORCS')
sys.path.append('/media/xinleipan/data/git/pyTORCS/')
from py_TORCS import torcs_envs
import multiprocessing as mp
# from mpc_agent import *

parser = argparse.ArgumentParser(description = 'Train-torcs')
init_parser(parser) # See `args.py` for default arguments
args = parser.parse_args()

if __name__ == '__main__':
    init_dirs([args.save_path,
               os.path.join(args.save_path, 'model'),
               os.path.join(args.save_path,'optimizer'),
               os.path.join(args.save_path, 'cnfmat')])
    if os.path.exists('/home/cxy/pyTORCS/game_config/michigan.xml'):
        args.game_config = '/home/cxy/pyTORCS/game_config/michigan.xml'
        args.xvfb = False
    
    envs = torcs_envs(num = 1, game_config = args.game_config,
                      isServer = int(args.xvfb), continuous = args.continuous, resize = True)
    env = envs.get_envs()[0]
    obs1 = env.reset()
    print(obs1.shape)
    obs, reward, done, info = env.step(np.array([1.0, 0.0]) if args.continuous else 1) # Action space is (-1,1)^2
    print(obs.shape, reward, done, info)

    train_policy(args, env, num_steps = 40000000, save_path = 'model')
    env.close()
