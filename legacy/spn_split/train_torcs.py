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
import gym, logging
import py_TORCS

parser = argparse.ArgumentParser(description = 'Train-torcs')
init_parser(parser) # See `args.py` for default arguments
args = parser.parse_args()

if __name__ == '__main__':
    init_dirs([args.save_path,
               os.path.join(args.save_path, 'model'),
               os.path.join(args.save_path,'optimizer'),
               os.path.join(args.save_path, 'cnfmat')])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # For tarasque
    if os.path.exists('/home/qizhicai/multitask/spn_new2/semantic-predictive-learning/game_config/michigan.xml'):
        args.game_config = '/home/qizhicai/multitask/spn_new2/semantic-predictive-learning/game_config/michigan.xml'

    # for 20500
    if os.path.exists('/home/cxy/pyTORCS/py_TORCS/py_TORCS/game_config/michigan.xml'):
        args.game_config = '/home/cxy/pyTORCS/py_TORCS/py_TORCS/game_config/michigan.xml'
        args.xvfb = False

    # for 20800
    if os.path.exists('/home/xiangyu/pyTORCS/game_config/michigan.xml'):
        args.game_config = '/home/xiangyu/pyTORCS/game_config/michigan.xml'
        args.xvfb = False
    try:
        env = gym.make('TORCS-v0')
        env.init(isServer=0, continuous=True, resize=True, ID=6)
    except:
        from py_TORCS import torcs_envs
        envs = torcs_envs(num = 1, game_config = args.game_config, mkey_start = 817 + args.id, screen_id = 160 + args.id,
                          isServer = int(args.xvfb), continuous = args.continuous, resize = True)
        env = envs.get_envs()[0]
        
    obs1 = env.reset()
    print(obs1.shape)
    obs, reward, done, info = env.step(np.array([1.0, 0.0]) if args.continuous else 1) # Action space is (-1,1)^2
    print(obs.shape, reward, done, info)

    train_policy(args, env, num_steps = 40000000)
    env.close()
