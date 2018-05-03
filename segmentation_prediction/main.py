from __future__ import division, print_function
import numpy as np
np.set_printoptions(formatter = {'float_kind': lambda x: "%.6f" % x})
import torch
import torch.nn as nn
from py_TORCS import torcs_envs

from model import DRNSeg, UP_Samper, PRED, FURTHER
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
    
    model = DRNSeg('drn_d_22', args.semantic_classes)
    up = UP_Samper()
    predictor = PRED(args.semantic_classes, args.num_actions)
    further = FURTHER()
    env = torcs_envs(num = 1, isServer = 0, mkey_start = 800, resize = True).get_envs()[0]

    model.load_state_dict(load(os.path.join('pretrained_models', '005', 'model.dat')))
    up.load_state_dict(load(os.path.join('pretrained_models', '005', 'up.dat')))
    predictor.load_state_dict(load(os.path.join('pretrained_models', '005', 'predictor.dat')))
    further.load_state_dict(load(os.path.join('pretrained_models', '005', 'further.dat')))

    # if not args.train or args.load:
    #     model.load_state_dict(torch.load(os.path.join(args.model_dir, 'model_epoch%d.dat' % latest_epoch)))
    #     up.load_state_dict(torch.load(os.path.join(args.model_dir, 'up_epoch%d.dat' % latest_epoch)))
    #     predictor.load_state_dict(torch.load(os.path.join(args.model_dir, 'predictor_epoch%d.dat' % latest_epoch)))
    #     further.load_state_dict(torch.load(os.path.join(args.model_dir, 'further_epoch%d.dat' % latest_epoch)))

    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda())
        up = nn.DataParallel(up.cuda())
        predictor = nn.DataParallel(predictor.cuda())
        further = nn.DataParallel(further.cuda())

    if args.train:
        train(args, model, up, predictor, further, env)
    else:
        test(args, model, up, predictor, further, env)
