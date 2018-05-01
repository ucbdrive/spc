from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import math
import time

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

def naive_driver(args, info):
    if info['angle'] > args.safe_angle or (info['trackPos'] < -args.safe_pos and info['angle'] > 0):
        return 0
    elif info['angle'] < -args.safe_angle or (info['trackPos'] > args.safe_pos and info['angle'] < 0):
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

def from_variable_to_numpy(x):
    x = x.data
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.numpy()
    return x

def generate_action_sample(args, prob, batch_size, length, LAST_ACTION = 1):
    all_actions = torch.zeros(length, batch_size).type(torch.LongTensor)
    all_actions[0] = prob[LAST_ACTION].multinomial(num_samples = batch_size, replacement=True).data
    # if torch.cuda.is_available():
    #     all_actions = all_actions.cuda()

    for step in range(1, length):
        indices = [torch.nonzero(all_actions[step - 1] == x).squeeze() for x in range(args.num_actions)]
        for action in range(args.num_actions):
            if indices[action].numel() > 0:
                all_actions[step, indices[action]] = prob[action].multinomial(num_samples = indices[action].numel(), replacement=True).data

    return all_actions

def generate_probs(args, all_actions, last_action = 1):
    all_actions = from_variable_to_numpy(all_actions)
    prob_map = np.concatenate((np.expand_dims(last_action * args.num_actions + all_actions[:, 0], axis = 1), all_actions[:, :-1] * args.num_actions + all_actions[:, 1:]), axis = 1)
    prob = torch.histc(torch.from_numpy(prob_map).type(torch.Tensor), bins = args.num_actions * args.num_actions).view(args.num_actions, args.num_actions)

    # if torch.cuda.is_available():
    #     prob = prob.cuda()

    prob[prob.sum(dim = 1) == 0, :] = 1
    prob /= prob.sum(dim = 1).unsqueeze(1)

    return prob

def sample_action(args, true_obs, model, predictor, further, prev_action = 1):
    # start_time = time.time()
    obs = np.repeat(np.expand_dims(true_obs, axis = 0), args.batch_size, axis = 0)
    obs = Variable(torch.from_numpy(obs), requires_grad = False).type(torch.Tensor)
    prob = torch.ones(args.num_actions, args.num_actions) / float(args.num_actions)
    # if torch.cuda.is_available():
    #     obs = obs.cuda()
    #     prob = prob.cuda()

    with torch.no_grad():
        for i in range(6):
            all_actions = generate_action_sample(args, prob, 6 * args.batch_size, args.num_steps, prev_action)
            all_losses = np.zeros(6 * args.batch_size)

            for ii in range(6):
                actions = Variable(all_actions[:, ii * args.batch_size: (ii + 1) * args.batch_size], requires_grad = False)
                pos_loss, angle_loss = get_action_loss(args, model, predictor, further, obs, actions)
                all_losses[ii * args.batch_size: (ii + 1) * args.batch_size] = pos_loss + angle_loss

            if i < 5:
                indices = np.argsort(all_losses)[:args.batch_size]
                prob = generate_probs(args, all_actions[:, indices], prev_action)
            else:
                idx = np.argmin(all_losses)
                which_action = int(from_variable_to_numpy(all_actions)[0, idx])

    # print(time.time() - start_time, which_action)
    return which_action

def get_action_loss(args, model, predictor, further, obs, actions):
    output_pos = Variable(torch.zeros(args.num_steps + 1, args.batch_size), requires_grad = False)
    output_angle = Variable(torch.zeros(args.num_steps + 1, args.batch_size), requires_grad = False)
    target_pos = Variable(torch.zeros(args.batch_size), requires_grad = False)
    target_angle = Variable(torch.zeros(args.batch_size), requires_grad = False)
    if torch.cuda.is_available():
        output_pos = output_pos.cuda()
        output_angle = output_angle.cuda()
        target_pos = target_pos.cuda()
        target_angle = target_angle.cuda()

    weight = 1
    L2 = nn.MSELoss(reduce = False)

    feature_map = model(obs)
    output_pos, output_angle = further(feature_map)
    pos_loss = L2(output_pos, target_pos)
    angle_loss = L2(output_angle, target_angle)

    for i in range(1, args.num_steps + 1):
        weight *= args.loss_decay
        feature_map = predictor(feature_map, actions[i - 1])
        output_pos, output_angle = further(feature_map)
        pos_loss = L2(output_pos, target_pos)
        angle_loss = L2(output_angle, target_angle)

    return from_variable_to_numpy(pos_loss), from_variable_to_numpy(angle_loss)
