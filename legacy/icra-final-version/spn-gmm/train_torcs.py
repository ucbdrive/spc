import pdb
import os
import argparse
from args import init_parser
import numpy as np
import torch
from train_cont_policy import train_policy
from evaluate import evaluate_policy
from utils import init_dirs
import sys
import random

parser = argparse.ArgumentParser(description='Train-torcs')
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

    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
    if 'torcs' in args.env:
        # For tarasque
        if os.path.exists('/data2/qizhicai/pyTORCS/py_TORCS/py_TORCS/game_config/michigan.xml'):
            args.game_config = '/data2/qizhicai/pyTORCS/py_TORCS/py_TORCS/game_config/michigan.xml'

        # for 20500
        if os.path.exists('/home/cxy/pyTORCS/py_TORCS/py_TORCS/game_config/michigan.xml'):
            args.game_config = '/home/cxy/pyTORCS/py_TORCS/py_TORCS/game_config/michigan.xml'
            args.xvfb = False

        # for 20800
        if os.path.exists('/home/xiangyu/pyTORCS/game_config/michigan.xml'):
            args.game_config = '/home/xiangyu/pyTORCS/game_config/michigan.xml'
            args.xvfb = False

        try:
            import gym
            import py_TORCS
            env = gym.make('TORCS-v0')
            env.init(isServer=0, continuous=True, resize=not args.eval, ID=6)
        except:
            from py_TORCS import torcs_envs
            envs = torcs_envs(num=1, game_config=args.game_config, mkey_start=817 + args.id, screen_id=160 + args.id,
                              isServer=int(args.xvfb), continuous=args.continuous, resize=not args.eval)
            env = envs.get_envs()[0]

        obs1 = env.reset()
        print(obs1.shape)
        obs, reward, done, info = env.step(np.array([1.0, 0.0]) if args.continuous else 1)  # Action space is (-1,1)^2
        print(obs.shape, reward, done, info)

        if args.eval:
            evaluate_policy(args, env)
        else:
            train_policy(args, env, num_steps=40000000)
        env.close()

    elif 'carla' in args.env:
        if args.simple_seg:
            args.classes = 4
        from carla.client import make_carla_client
        from carla_env import carla_env
        with make_carla_client('localhost', 2000) as client:
            print('\033[1;32mCarla client connected\033[0m')
            env = carla_env(client, args.simple_seg)
            env.reset()
            train_policy(args, env, num_steps=40000000)

    elif 'gta' in args.env:
        from gta_env import *
        if args.simple_seg:
            args.classes = 4
        from gta_wrapper import *
        from deepgtav.utils import parseroadinfo, parsedirectioninfo
        env = GtaEnv(autodrive=None)
        env = GTAWrapper(env)
        _ = env.reset()
        train_policy(args, env, num_steps=400000000)
