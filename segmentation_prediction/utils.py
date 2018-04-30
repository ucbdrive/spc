import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import math

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def up_sampler(classes, use_torch_up=False):
    if use_torch_up:
        up = nn.UpsamplingBilinear2d(scale_factor=8)
    else:
        up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                output_padding=0, groups=classes, bias=False)
        fill_up_weights(up)
    return up

def draw_from_pred(pred):
    pred = pred.data
    if torch.cuda.is_available():
        pred = pred.cpu()
    pred = pred.numpy()
    illustration = np.zeros((256, 256, 3)).astype(np.uint8)
    illustration[:, :, 0] = 255
    illustration[pred == 1] = np.array([0, 255, 0])
    illustration[pred == 2] = np.array([0, 0, 0])
    illustration[pred == 3] = np.array([0, 0, 255])
    return illustration

def reset_env(args, env):
    obs = env.reset()
    for i in range(np.random.randint(150) + 1):
        obs, reward, done, info = env.step(np.random.randint(args.num_actions))

    true_obs = np.zeros((3 * args.frame_len, 256, 256))
    for i in range(args.frame_len):
        obs, reward, done, info = env.step(np.random.randint(args.num_actions))
        true_obs[i * 3: (i + 1) * 3] = (obs.transpose(2, 0, 1) - args.obs_avg) / args.obs_std

    return true_obs

def naive_driver(info):
    if info['angle'] > 0.5 or (info['trackPos'] < -1 and info['angle'] > 0):
        return 0
    elif info['angle'] < -0.5 or (info['trackPos'] > 3 and info['angle'] < 0):
        return 2
    return 1

def init_dirs(args):
    if os.path.exists(args.log):
        os.system('rm %s' % args.log)
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.isdir(args.visualize_dir):
        os.mkdir(args.visualize_dir)
    elif os.listdir(args.model_dir) and args.train and not args.load:
        os.system('rm %s/*.dat' % args.model_dir)

def clear_visualization_dir(args):
    if os.listdir(args.visualize_dir):
        os.system('rm %s/*' % args.visualize_dir)

def init_variables(args):
    inputs = Variable(torch.ones(args.num_steps + 1, args.batch_size, 9, 256, 256), requires_grad = False)
    outputs = Variable(torch.ones(args.num_steps + 1, args.batch_size, 4, 256, 256), requires_grad = False)
    prediction = Variable(torch.ones(args.num_steps + 1, args.batch_size, 4, 256, 256), requires_grad = False)
    feature_map = Variable(torch.ones(args.num_steps + 1, args.batch_size, 4, 32, 32), requires_grad = False)
    output_pos = Variable(torch.zeros(args.num_steps + 1, args.batch_size), requires_grad = False)
    output_angle = Variable(torch.zeros(args.num_steps + 1, args.batch_size), requires_grad = False)
    targets = Variable(torch.ones(args.num_steps + 1, args.batch_size, 256, 256), requires_grad = False).type(torch.LongTensor)
    target_pos = Variable(torch.zeros(args.num_steps + 1, args.batch_size), requires_grad = False)
    target_angle = Variable(torch.zeros(args.num_steps + 1, args.batch_size), requires_grad = False)
    actions  = Variable(torch.zeros(args.num_steps, args.batch_size), requires_grad = False).type(torch.LongTensor)

    if torch.cuda.is_available():
        inputs = inputs.cuda()
        outputs = outputs.cuda()
        prediction = prediction.cuda()
        feature_map = feature_map.cuda()
        output_pos = output_pos.cuda()
        output_angle = output_angle.cuda()
        targets = targets.cuda()
        target_pos = target_pos.cuda()
        target_angle = target_angle.cuda()
        actions = actions.cuda()

    return inputs, outputs, prediction, feature_map, output_pos, output_angle, targets, target_pos, target_angle, actions

def init_criteria():
    NLL = nn.NLLLoss()
    CE = nn.CrossEntropyLoss()
    L1 = nn.SmoothL1Loss()
    L2 = nn.MSELoss()

    if torch.cuda.is_available():
        NLL = NLL.cuda()
        CE = CE.cuda()
        L1 = L1.cuda()
        L2 = L2.cuda()

    return {'NLL': NLL, 'CE': CE, 'L1': L1, 'L2': L2}