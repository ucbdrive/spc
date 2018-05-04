from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import math
import time
from collections import OrderedDict

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

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
    for i in range(np.random.randint(300)):
        obs, reward, done, info = env.step(naive_driver(args, env.get_info()))
    true_obs = (obs.transpose(2, 0, 1) - args.obs_avg) / args.obs_std

    if args.frame_len > 1:
        true_obs = np.zeros((3 * args.frame_len, 256, 256))
        for i in range(args.frame_len):
            obs, reward, done, info = env.step(np.random.rand(2) * 2 - 1 if args.continuous else np.random.randint(args.num_actions))
            true_obs[i * 3: (i + 1) * 3] = (obs.transpose(2, 0, 1) - args.obs_avg) / args.obs_std

    return true_obs, np.array(info['pos'])

def naive_driver(args, info):
    if info['angle'] > args.safe_angle or (info['trackPos'] < -args.safe_pos and info['angle'] > 0):
        return np.array([1, 0, -1]) if args.continuous else 0
    elif info['angle'] < -args.safe_angle or (info['trackPos'] > args.safe_pos and info['angle'] < 0):
        return np.array([1, 0, 1]) if args.continuous else 2
    return np.array([1, 0, 0]) if args.continuous else 1 # np.random.randint(args.num_actions)

def init_dirs(args):
    if os.path.exists(args.log):
        os.system('rm %s' % args.log)
    if os.path.exists('env_log.txt'):
        os.system('rm env_log.txt')
    if not os.path.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.isdir(args.visualize_dir):
        os.mkdir(args.visualize_dir)
    elif os.listdir(args.model_dir) and args.train and not args.load:
        os.system('rm %s/*.dat' % args.model_dir)

def clear_visualization_dir(args):
    if os.listdir(args.visualize_dir):
        os.system('rm %s/*' % args.visualize_dir)

def init_variable(*kwargs):
    result = Variable(torch.zeros(*kwargs), requires_grad = False)
    if torch.cuda.is_available():
        result = result.cuda()
    return result

def init_training_variables(args):
    data_dict = dict()
    data_dict['obs'] = init_variable(args.batch_size, 3 * args.frame_len, 256, 256)

    if args.continuous:
        data_dict['action'] = init_variable(args.num_steps, args.batch_size, args.action_dim)
    else:
        data_dict['action'] = init_variable(args.num_steps, args.batch_size).type(torch.LongTensor)

    if args.use_dqn:
        data_dict['dqn_action'] = init_variable(args.num_steps, args.batch_size, 1).type(torch.LongTensor)
        data_dict['reward'] = init_variable(args.num_steps, args.batch_size)

    if args.use_seg:
        data_dict['seg'] = init_variable(args.num_steps, args.batch_size, args.semantic_classes, 256, 256)

    data_dict['target_coll'] = init_variable(args.num_steps, args.batch_size)
    data_dict['target_off'] = init_variable(args.num_steps, args.batch_size)
    data_dict['target_dist'] = init_variable(args.num_steps, args.batch_size)

    if args.use_xyz:
        data_dict['target_xyz'] = init_variable(args.num_steps, args.batch_size, 3)

    return data_dict

def init_criteria():
    NLL = nn.NLLLoss(reduce = False)
    CE = nn.CrossEntropyLoss(reduce = False)
    L1 = nn.SmoothL1Loss(reduce = False)
    L2 = nn.MSELoss(reduce = False)
    BCE = nn.BCELoss(reduce = False)

    if torch.cuda.is_available():
        NLL = NLL.cuda()
        CE = CE.cuda()
        L1 = L1.cuda()
        L2 = L2.cuda()
        BCE = BCE.cuda()

    return {'NLL': NLL, 'CE': CE, 'L1': L1, 'L2': L2, 'BCE': BCE}

def from_variable_to_numpy(x):
    x = x.data
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.numpy()
    return x

def generate_action_sample(args, prob, batch_size, length, LAST_ACTION = 1):
    # start_time = time.time()
    all_actions = torch.zeros(length, batch_size).type(torch.LongTensor)
    all_actions[0] = prob[LAST_ACTION].multinomial(num_samples = batch_size, replacement=True).data
    # if torch.cuda.is_available():
    #     all_actions = all_actions.cuda()

    for step in range(1, length):
        indices = [torch.nonzero(all_actions[step - 1] == x).squeeze() for x in range(args.num_actions)]
        for action in range(args.num_actions):
            if indices[action].numel() > 0:
                all_actions[step, indices[action]] = prob[action].multinomial(num_samples = indices[action].numel(), replacement=True).data

    # print('Gerenating action samples takes %0.2f seconds.' % (time.time() - start_time))
    return all_actions

def generate_probs(args, all_actions, last_action = 1):
    # start_time = time.time()
    all_actions = from_variable_to_numpy(all_actions)
    prob_map = np.concatenate((np.expand_dims(last_action * args.num_actions + all_actions[:, 0], axis = 1), all_actions[:, :-1] * args.num_actions + all_actions[:, 1:]), axis = 1)
    prob = torch.histc(torch.from_numpy(prob_map).type(torch.Tensor), bins = args.num_actions * args.num_actions).view(args.num_actions, args.num_actions)

    # if torch.cuda.is_available():
    #     prob = prob.cuda()

    prob[prob.sum(dim = 1) == 0, :] = 1
    prob /= prob.sum(dim = 1).unsqueeze(1)

    # print('Gerenating probability distribution takes %0.2f seconds.' % (time.time() - start_time))

    return prob

def sample_action(args, true_obs, model, prev_action = 1):
    start_time = time.time()
    obs = np.repeat(np.expand_dims(true_obs, axis = 0), args.action_batch_size, axis = 0)
    obs = Variable(torch.from_numpy(obs), requires_grad = False).type(torch.Tensor)
    prob = torch.ones(args.num_actions, args.num_actions) / float(args.num_actions)
    # if torch.cuda.is_available():
    #     obs = obs.cuda()
    #     prob = prob.cuda()

    with torch.no_grad():
        for i in range(6):
            all_actions = generate_action_sample(args, prob, 6 * args.action_batch_size, args.num_steps, prev_action)
            all_losses = np.zeros(6 * args.action_batch_size)

            for ii in range(6):
                actions = Variable(all_actions[:, ii * args.action_batch_size: (ii + 1) * args.action_batch_size], requires_grad = False)
                loss_dict = get_action_loss(args, model, obs, actions, args.action_batch_size) # needs updating
                all_losses[ii * args.action_batch_size: (ii + 1) * args.action_batch_size] = from_variable_to_numpy(loss_dict['coll_loss']) + from_variable_to_numpy(loss_dict['off_loss']) - 0.1 * from_variable_to_numpy(loss_dict['dist_loss'])

            if i < 5:
                indices = np.argsort(all_losses)[:args.action_batch_size]
                prob = generate_probs(args, all_actions[:, indices], prev_action)
            else:
                idx = np.argmin(all_losses)
                which_action = int(from_variable_to_numpy(all_actions)[0, idx])

    print('Sampling takes %0.2f seconds, selected action: %d.' % (time.time() - start_time, which_action))
    return which_action

def get_action_loss(args, model, obs, actions, batch_size):
    output = model(obs, actions)
    target_coll = Variable(torch.zeros(args.num_steps, batch_size), requires_grad = False)
    target_off = Variable(torch.zeros(args.num_steps, batch_size), requires_grad = False)
    target_dist = Variable(torch.zeros(args.num_steps, batch_size), requires_grad = False)
    weight = Variable(args.loss_decay ** torch.arange(args.num_steps), requires_grad = False).view(args.num_steps, 1)
    if torch.cuda.is_available():
        target_coll = target_coll.cuda()
        target_off = target_off.cuda()
        weight = weight.cuda()

    L2 = nn.MSELoss(reduce = False)
    BCE = nn.BCELoss(reduce = False)

    coll_loss = torch.sum(BCE(output['output_coll'], target_coll) * weight)
    off_loss = torch.sum(BCE(output['output_off'], target_off) * weight)
    dist_loss = torch.sum(BCE(output['output_dist'], target_dist) * weight)

    return {'coll_loss': coll_loss, 'off_loss': off_loss, 'dist_loss': dist_loss}

def load(path):
    state_dict = torch.load(path)
    return convert_state_dict(state_dict)

def convert_state_dict(state_dict):
    target_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[0:7] == 'module.':
            target_state_dict[k[7:]] = v
        else:
            target_state_dict[k] = v
    return target_state_dict

def sample_dqn_action(args, obs, dqn):
    if np.random.random() < args.exploration_decay ** args.epoch:
        action = np.random.randint(9) 
    else:
        action = dqn(Variable(torch.from_numpy(obs), requires_grad = False)).data.max(1)[1]
        if torch.cuda.is_available():
            action = action.cpu()
        action = int(action.numpy())
    return action

def sample_continuous_action(args, obs, model):
    if np.random.random() < args.exploration_decay ** args.epoch:
        action = np.random.rand(2) * 2 - 1
    else:
        action = sample_cont_action(args, obs, model)
        action = np.clip(action, -1.0, 1.0)

    return action

def sample_cont_action(args, obs, model):
    start_time = time.time()
    obs = np.repeat(np.expand_dims(true_obs, axis = 0), args.action_batch_size, axis = 0)
    obs = Variable(torch.from_numpy(obs), requires_grad = False).type(torch.Tensor)

    this_action = torch.from_numpy(np.random.rand(2 * args.num_steps) * 2 - 1)
    this_action = Variable(this_action, requires_grad = True).view(args.num_steps, 1, 2)
    if torch.cuda.is_available():
        this_action = this_action.cuda()
    prev_loss = 1000
    cnt = 0
    while sign:
        this_action.zero_grad()
        loss = get_action_loss(args, model, obs, this_action, 1)
        loss = loss_dict['coll_loss'] + loss_dict['off_loss'] - 0.1 * loss_dict['dist_loss']
        loss.backward()
        this_loss = float(from_variable_to_numpy(loss))
        if cnt >= 10 and (np.abs(prev_loss - this_loss) <= 0.0005 * prev_loss or this_loss > prev_loss):
            which_action = from_variable_to_numpy(this_action)[0, 0, :]
            break
        cnt += 1 
        this_action.data -= 0.01 * this_action.grad.data
        this_action.data.clamp(-1, 1)# = torch.clamp(this_action, -1, 1)
        prev_loss = this_loss
    model.zero_grad()
    print('Sampling takes %0.2f seconds, selected action: %d.' % (time.time() - start_time, which_action))
    return which_action

def one_hot(args, action):
    one_hot_vector = Variable(torch.zeros(args.batch_size, self.num_actions), requires_grad = False)
    one_hot_vector[torch.arange(args.batch_size).type(torch.LongTensor), action] = 1
    if torch.cuda.is_available():
        one_hot_vector = one_hot_vector.cuda()
    return one_hot_vector