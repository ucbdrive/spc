import pdb
import os
import argparse
from envs import create_env
from train_policy import train_policy
from utils import init_dirs
import sys
sys.path.append('/home/xinleipan/pyTORCS/py_TORCS')
sys.path.append('/home/xinleipan/pyTORCS/')
from py_TORCS import torcs_envs
import multiprocessing as mp
from mpc_agent import *

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

if __name__ == '__main__':
    args = parser.parse_args()
    init_dirs([args.save_path,\
               args.save_path+'/model', \
               args.save_path+'/optimizer', \
               args.save_path+'/cnfmat'])
    
    envs = torcs_envs(num = 1, game_config='/home/xinleipan/pyTORCS/game_config/michigan.xml', isServer = 1, screen_id = 162)
    env = envs.get_envs()[0]
    obs1 = env.reset()
    print(obs1.shape)
    obs, reward, done, info = env.step(np.array([1.0, 0.0, 0.0])) # Action space is (-1,1)^3
    print(obs.shape, reward, done, info)
    train_policy(args, env, 4000000,  
                 batch_size = args.batch_size,
                 pred_step = args.pred_step,
                 normalize = args.normalize,
                 start_step = 100,
                 buffer_size = args.buffer_size,
                 save_path = args.save_path,
                 save_freq = args.save_freq, # model saving frequency
                 frame_history_len = args.frame_history_len,
                 num_total_act = args.num_total_act)
    env.close()
