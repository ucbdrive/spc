import pdb
import os
import argparse
import numpy as np
# from envs import create_env
from train_cont_policy import train_policy
from utils import init_dirs
import sys
sys.path.append('/home/xinleipan/pyTORCS/py_TORCS')
sys.path.append('/home/xinleipan/pyTORCS/')
from py_TORCS import torcs_envs
import multiprocessing as mp
# from mpc_agent import *

parser = argparse.ArgumentParser(description = 'Train-torcs')
parser.add_argument('--lr', type = float, default = 0.001, metavar = 'LR', help = 'learning rate')
parser.add_argument('--env', type = str, default = 'torcs-v0', metavar = 'ENV', help = 'environment')
parser.add_argument('--save-path', type=str, default = 'mpc_12_step')
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--pred-step', type=int, default=15)
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--buffer-size', type=int, default=50000)
parser.add_argument('--save-freq', type=int, default=10)
parser.add_argument('--frame-history-len', type=int, default=3)
parser.add_argument('--num-total-act', type=int, default=6)
parser.add_argument('--epsilon-frames', type=int, default=50000)
parser.add_argument('--learning-starts', type=int, default=100)
parser.add_argument('--learning-freq', type=int, default=10)
parser.add_argument('--target-update-freq', type=int, default=100)
parser.add_argument('--batch-step', type=int, default=400)
parser.add_argument('--with-dla', action='store_true')
parser.add_argument('--resume', type = bool, default = False)

if __name__ == '__main__':
    args = parser.parse_args()
    init_dirs([args.save_path,\
               args.save_path+'/model', \
               args.save_path+'/optimizer', \
               args.save_path+'/cnfmat'])
    
    envs = torcs_envs(num = 1, game_config = '/home/cxy/semantic-predictive-learning/mpc_final/game_config/michigan.xml', isServer = 0, continuous = True, resize = True)
    env = envs.get_envs()[0]
    obs1 = env.reset()
    print(obs1.shape)
    obs, reward, done, info = env.step(np.array([1.0, 0.0])) # Action space is (-1,1)^2
    print(obs.shape, reward, done, info)
    train_policy(args, env, 
                 num_steps=40000000,
                 batch_size=32,
                 pred_step=15,
                 normalize=True,
                 buffer_size=50000,
                 save_path='model',
                 save_freq=10,
                 frame_history_len=3,
                 num_total_act=2,
                 use_seg=True,
                 use_xyz=True,
                 use_dqn=True)
    env.close()
