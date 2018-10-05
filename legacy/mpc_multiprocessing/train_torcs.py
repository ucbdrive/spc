import pdb
import os
import argparse
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Lock, Value, Array
from memory_pool import memory_pool
from collect_data import collect_data
from train_agent import train_agent
from utils import init_dirs
from kill_torcses import kill_torcses

parser = argparse.ArgumentParser(description = 'Train-torcs')
parser.add_argument('--lr', type = float, default = 0.001, metavar = 'LR', help = 'learning rate')
parser.add_argument('--env', type = str, default = 'torcs-v0', metavar = 'ENV', help = 'environment')
parser.add_argument('--observation-height', type = int, default = 256)
parser.add_argument('--observation-width', type = int, default = 256)
parser.add_argument('--observation-channels', type = int, default = 3)
parser.add_argument('--max-episode-length', type = int, default = 1000)
parser.add_argument('--num-steps', type = int, default = 4000000) # number of training steps 
parser.add_argument('--save-path', type = str, default = 'mpc_12_step')
parser.add_argument('--pred-step', type = int, default = 15) # number of prediction steps
parser.add_argument('--normalize', type = bool, default = True) # whether to normalize images or not
parser.add_argument('--buffer-size', type = int, default = 50000)
parser.add_argument('--batch-size', type = int, default = 5)
parser.add_argument('--use-pos-class', type = bool, default = True)
parser.add_argument('--save-freq', type = int, default = 10) # model saving frequency
parser.add_argument('--with-speed', type = bool, default = True)
parser.add_argument('--with-posinfo', type = bool, default = True)
parser.add_argument('--with-pos', type = bool, default = True)
parser.add_argument('--with-dla', type = bool, default = True)
parser.add_argument('--frame-history-len', type = int, default = 3)
parser.add_argument('--num-total-act', type = int, default = 6)
parser.add_argument('--epsilon-frames', type = int, default = 50000)
parser.add_argument('--learning-starts', type = int, default = 100)
parser.add_argument('--learning-freq', type = int, default = 10)
parser.add_argument('--target-update-freq', type = int, default = 100) # model loading frequency
parser.add_argument('--batch-step', type = int, default = 500)

if __name__ == '__main__':
    args = parser.parse_args()
    d_args = vars(args)
    print('====================== Args ======================')
    for k in d_args.keys():
        print('  %s: %s' % (k, d_args[k]))
    print('')

    init_dirs([args.save_path,
               args.save_path + '/model',
               args.save_path + '/optimizer',
               args.save_path + '/cnfmat',
               args.save_path + '/dataset'])
    
    mem_pool = memory_pool(args)
    p1 = Process(target = collect_data, args = (args, mem_pool))
    p2 = Process(target = train_agent, args = (args, mem_pool, 0)) # cuda share memory needs Python3
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    
    kill_torcses()