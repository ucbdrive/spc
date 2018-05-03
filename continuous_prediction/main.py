from __future__ import division, print_function
import numpy as np
np.set_printoptions(formatter = {'float_kind': lambda x: "%.6f" % x})
import torch
import torch.nn as nn
from py_TORCS import torcs_envs

from model import hybrid_net, DQN
from train import train
from test import test
from utils import init_dirs, load

import os
import argparse
from args import init_parser

from scipy.misc import imsave
from scipy.misc.pilutil import imshow
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = 'Train-TORCS')
init_parser(parser)

latest_epoch = 140

if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    init_dirs(args)
    
    model = hybrid_net(args)
    dqn = DQN()
    env = torcs_envs(num = 1, isServer = 0, mkey_start = 900,
                     resize = True, continuous = args.continuous).get_envs()[0]

    if not args.train or args.load:
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model_epoch%d.dat' % latest_epoch)))
        dqn.load_state_dict(torch.load(os.path.join(args.model_dir, 'dqn_epoch%d.dat' % latest_epoch)))

    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda())
        dqn = nn.DataParallel(dqn.cuda())

    if args.train:
        train(args, model, dqn, env)
    else:
        test(args, model, dqn, env)
