from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import os
import time
import math
import PIL.Image as Image
import random
from sklearn.metrics import confusion_matrix
import pdb
from scipy.misc import imsave
import copy
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from eval_segm import mean_IU, mean_accuracy, pixel_accuracy, frequency_weighted_IU
from scipy.special import comb
from itertools import combinations, chain


def get_from_dict(info, key):
    return info[key] if key in info.keys() else None


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1: 4])
        fan_out = np.prod(weight_shape[2: 4]) * weight_shape[0]
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
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)


def train_model(args, train_net, mpc_buffer, epoch, avg_img_t, std_img_t):
    if epoch % 20 == 0:
        target, idxes = mpc_buffer.sample(args.batch_size, sample_early=False)
    else:
        target, idxes = mpc_buffer.sample(args.batch_size, sample_early=False)

    for key in target.keys():
        target[key] = Variable(torch.from_numpy(target[key]).float(), requires_grad=False)

    if args.use_seg:
        target['seg_batch'] = target['seg_batch'].long()
    else:
        with torch.no_grad():
            nximg_enc = train_net(target['nx_obs_batch'], get_feature=True).detach()

    if args.normalize:
        target['obs_batch'] = (target['obs_batch'] - avg_img_t) / std_img_t
        target['nx_obs_batch'] = (target['nx_obs_batch'] - avg_img_t) / std_img_t
    else:
        target['obs_batch'] = target['obs_batch'] / 255.0
        target['nx_obs_batch'] = target['nx_obs_batch'] / 255.0

    if torch.cuda.is_available():
        for key in target.keys():
            target[key] = target[key].to(torch.device("cuda:0"))

    output = train_net(target['obs_batch'], target['act_batch'], action_var=target['prev_action'])

    loss = 0

    weight = (args.gamma ** np.arange(args.pred_step)).reshape((1, args.pred_step, 1))
    weight = Variable(torch.from_numpy(weight).float().cuda(), requires_grad=False).repeat(args.batch_size, 1, 1)

    coll_ls = nn.CrossEntropyLoss()(output['coll_prob'].view(-1, 2), target['coll_batch'].view(-1).long())
    loss += coll_ls
    print('coll ls', coll_ls.data.cpu().numpy())

    reward_ls = torch.sqrt(nn.MSELoss()(output['reward'].view(-1, args.pred_step), target['reward_batch'][:, 1:].view(-1, args.pred_step)))
    loss += reward_ls
    print('reward ls', reward_ls.data.cpu().numpy())

    value_ls = torch.sqrt(nn.MSELoss()(output['value'].view(-1, args.pred_step), target['value_batch'].view(-1, args.pred_step)))
    loss += value_ls
    print('value ls', value_ls.data.cpu().numpy())

    if args.use_seg:
        output['seg_pred'] = output['seg_pred'].view(args.batch_size * (args.pred_step + 1), args.classes, args.frame_height, args.frame_width)
        target['seg_batch'] = target['seg_batch'].view(args.batch_size * (args.pred_step + 1), args.frame_height, args.frame_width)
        pred_ls = nn.NLLLoss()(output['seg_pred'], target['seg_batch'])
    else:
        pred_ls = torch.sqrt(nn.MSELoss()(output['seg_pred'], nximg_enc))
    print('pred ls', pred_ls.data.cpu().numpy()) # nan here!
    loss += pred_ls

    loss_value = float(loss.data.cpu().numpy())
    if np.isnan(loss_value):
        pdb.set_trace()
    visualize(args, target, output)
    return loss

def draw_from_pred(array):
    classes = {
        0: [26, 26, 26],
        1: [149, 216, 222],
        2: [0, 255, 0],
        3: [0, 56, 252]
    }

    result = np.zeros((array.shape[0], array.shape[1], 3))
    for key, value in classes.items():
        result[np.where(array == key)] = value
    return result

def visualize(args, target, output):
    if not os.path.isdir('visualize'):
        os.mkdir('visualize')
    batch_id = np.random.randint(args.batch_size)

    if args.use_seg:
        # observation = (from_variable_to_numpy(target['obs_batch'][batch_id, :, -3:, :, :]) * 56.1524832523 + 112.62289744791671).astype(np.uint8).transpose(0, 2, 3, 1)
        observation = (from_variable_to_numpy(target['obs_batch'][batch_id, :, -3:, :, :]) * 255.0).astype(np.uint8).transpose(0, 2, 3, 1)
        target['seg_batch'] = target['seg_batch'].view(args.batch_size, args.pred_step + 1, args.frame_height, args.frame_width)
        segmentation = from_variable_to_numpy(target['seg_batch'][batch_id])
        output['seg_pred'] = output['seg_pred'].view(args.batch_size, args.pred_step + 1, args.classes, args.frame_height, args.frame_width)
        _, prediction = torch.max(output['seg_pred'][batch_id], 1)
        prediction = from_variable_to_numpy(prediction)
        for i in range(args.pred_step):
            imsave('visualize/%d.png' % i, np.concatenate([observation[i], draw_from_pred(segmentation[i]), draw_from_pred(prediction[i])], axis=1))

    with open(os.path.join(args.save_path, 'report.txt'), 'a') as f:
        f.write('target reward:\n')
        f.write(str(from_variable_to_numpy(target['reward_batch'][batch_id])) + '\n')
        f.write('output reward:\n')
        f.write(str(from_variable_to_numpy(output['reward'][batch_id])) + '\n')

class DoneCondition:
    def __init__(self, size):
        self.size = size
        self.off_cnt = 0
        self.pos = []

    def isdone(self, pos, dist, posxyz, angle):
        if pos <= -6.2 and dist < 0:
            self.off_cnt += 1
        elif pos > -6.2 or dist > 0:
            self.off_cnt = 0
        if self.off_cnt > self.size:
            self.off_cnt = 0
            self.pos = []
            return True
        if abs(pos) >= 21.0:
            self.off_cnt = 0
            self.pos = []
            return True
        self.pos.append(list(posxyz))
        real_pos = np.concatenate(self.pos[-100:])
        real_pos = real_pos.reshape(-1,3)
        std = np.sum(np.std(real_pos, 0))
        if std < 2.0 and len(self.pos) > 100:
            self.pos = []
            self.off_cnt = 0
            return True
        return False


class ObsBuffer:
    def __init__(self, frame_history_len=3):
        self.frame_history_len = frame_history_len
        self.last_obs_all = []

    def store_frame(self, frame):
        obs_np = frame.transpose(2, 0, 1)
        if len(self.last_obs_all) < self.frame_history_len:
            self.last_obs_all = []
            for ii in range(self.frame_history_len):
                self.last_obs_all.append(obs_np)
        else:
            self.last_obs_all = self.last_obs_all[1:] + [obs_np]
        return np.concatenate(self.last_obs_all, 0)

    def clear(self):
        self.last_obs_all = []
        return


class ActionBuffer:
    def __init__(self, frame_history_len=3):
        self.frame_history_len = frame_history_len
        self.last_action_all = []

    def store_frame(self, action):
        action = action.reshape(1, -1)
        if len(self.last_action_all) < self.frame_history_len:
            self.last_action_all = []
            for ii in range(self.frame_history_len):
                self.last_action_all.append(action)
        else:
            self.last_action_all = self.last_action_all[1:] + [action]
        return np.concatenate(self.last_action_all, 0)[np.newaxis,]

    def clear(self):
        self.last_action_all = []
        return


def init_dirs(dir_list):
    for path in dir_list:
        make_dir(path)


def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def load_model(path, net, data_parallel=True, optimizer=None, resume=True):
    if resume:
        file_list = sorted(os.listdir(path+'/model'))
        if len(file_list) == 0:
            print('no model to resume!')
            epoch = 0
        else:
            model_path = file_list[-1]
            epoch = pkl.load(open(path+'/epoch.pkl', 'rb'))
            state_dict = torch.load(os.path.join(path, 'model', model_path))
            net.load_state_dict(state_dict)
            print('load success')
    else:
        epoch = 0

    # if optimizer is not None and epoch > 0:
    #     optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer', 'optimizer.pt')))

    # if optimizer is None:
    return net, epoch
    # else:
    #     return net, epoch, optimizer


def get_info_np(info, use_pos_class = False):
    speed_np = np.array([[info['speed'], info['angle']]])
    if use_pos_class:
        pos = int(round(np.clip(info['trackPos'], -9, 9)) + 9)
        pos_np = np.zeros((1, 19))
        pos_np[0, pos] = 1
    else:
        pos_np = np.array([[info['trackPos']]])
    posxyz_np = np.array([list(info['pos'])])
    return speed_np, pos_np, posxyz_np


def get_info_ls(info):
    speed = [info['speed'], info['angle']]
    pos = [info['trackPos']] + list(info['pos'])
    return speed, pos


def Focal_Loss(probs, target, reduce=True):
    # probs : batch * num_class
    # target : batch,
    loss = -1.0 * (1-probs).pow(3) * torch.log(probs)
    batch_size = int(probs.size()[0])
    loss = loss[torch.arange(batch_size).long().cuda(), target.long()]
    if reduce:
        loss = loss.sum() / (batch_size * 1.0)
    return loss


def sample_cont_action(args, net, imgs, info=None, prev_action=None, testing=False, avg_img=0, std_img=1.0, tt=0, action_var=None):
    imgs = copy.deepcopy(imgs)
    if args.normalize:
        imgs = (imgs.contiguous() - avg_img) / (std_img)
    else:
        imgs = imgs / 255.0
    batch_size, c, w, h = int(imgs.size()[0]), int(imgs.size()[-3]), int(imgs.size()[-2]), int(imgs.size()[-1])
    imgs = imgs.view(batch_size, 1, c, w, h)
    prev_action = prev_action.reshape((1, 1, args.num_total_act))
    prev_action = np.repeat(prev_action, args.pred_step, axis=1)
    this_action = torch.from_numpy(prev_action).float()
    this_action = Variable(this_action, requires_grad=True)
    this_action.data.clamp(-1, 1)
    prev_loss = 1000
    sign = True
    cnt = 0

    if args.sample_based_planning:
        num_candidates = sum([comb(args.pred_step, i, exact=True) for i in range(3)])
        imgs = imgs.repeat(num_candidates, 1, 1, 1, 1)
        action_var = action_var.repeat(num_candidates, 1, 1)
        this_action0 = torch.zeros(num_candidates, args.pred_step, 1)
        selections = list(chain(*[combinations(range(args.pred_step), i) for i in range(3)]))
        for i in range(num_candidates):
            for j in selections[i]:
                this_action0[i, j, 0] = 1
        this_action = Variable(this_action0.cuda().float(), requires_grad=False)
        # loss = np.zeros(num_candidates)
        start_time = time.time()
        loss = get_action_loss(args, net, imgs, this_action, action_var, None, None, None).data.cpu().numpy()
        # for i in range(4):
        #     with torch.no_grad():
        #         loss[i*(2 ** (args.pred_step-2)): (i+1)*(2 ** (args.pred_step-2))] = get_action_loss(args, net, imgs[i*(2 ** (args.pred_step-2)): (i+1)*(2 ** (args.pred_step-2))], this_action[i*(2 ** (args.pred_step-2)): (i+1)*(2 ** (args.pred_step-2))], action_var, None, None, None).data.cpu().numpy()
        print('Sampling takes %0.2f seconds.' % (time.time() - start_time))

        if False:
            plt.figure()
            plt.title('pos [%0.2f, %0.2f], trackPos %0.2f, angle %0.2f' % (info['pos'][0], info['pos'][1], info['trackPos'], info['angle']))
            loss2 = loss.reshape((20, 20))
            plt.imshow(loss2)
            plt.colorbar()
            plt.xlabel('acceleration')
            plt.ylabel('steering angle')
            # plt.plot(np.arange(100) / 50.0 - 1, loss)
            plt.tight_layout()
            if not os.path.isdir(os.path.join(args.save_path, 'actions')):
                os.mkdir(os.path.join(args.save_path, 'actions'))
            plt.savefig(os.path.join(args.save_path, 'actions/step_%04d.png' % (tt % 10000)), dpi=100)
            plt.close()
        idx = np.argmin(loss)
        res = this_action0[idx, 0, 0]
        # return np.array([0.5, idx / 50 - 1])
        return res

    else:
        optimizer = optim.Adam([this_action], lr=0.01)
        while sign:
            if testing:
                loss = get_action_loss_test(args, net, imgs, this_action, None, None, None)
            else:
                loss = get_action_loss(args, net, imgs, this_action, None, None, None)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            this_loss = float(loss.data.cpu().numpy())
            if cnt >= 20 and (np.abs(prev_loss-this_loss)/prev_loss <= 0.0005 or this_loss > prev_loss):
                sign = False
                return this_action.data.cpu().numpy()[0, 0, :].reshape(-1)
            if cnt >= 25:
                sign = False
            cnt += 1
            this_action.data.clamp(-1, 1)  # = torch.clamp(this_action, -1, 1)
            prev_loss = this_loss
        return this_action.data.cpu().numpy()[0, 0, :].reshape(-1)


def from_variable_to_numpy(x):
    x = x.data
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.numpy()
    return x


def generate_action_sample(args, prob, batch_size, length, LAST_ACTION=1):
    all_actions = torch.zeros(batch_size, length).type(torch.LongTensor)
    all_actions[:, 0] = prob[LAST_ACTION].multinomial(num_samples=batch_size, replacement=True).data

    for step in range(1, length):
        indices = [torch.nonzero(all_actions[:, step - 1] == x).squeeze() for x in range(args.num_total_act)]
        for action in range(args.num_total_act):
            if indices[action].numel() > 0:
                all_actions[indices[action], step] = prob[action].multinomial(num_samples = indices[action].numel(), replacement=True).data

    return all_actions


def generate_one_hot_actions(actions, num_actions):
    batch_size, length = actions.size()
    result = torch.zeros(batch_size * length, num_actions)
    actions = actions.view(batch_size * length)
    result[torch.arange(batch_size * length).type(torch.LongTensor), actions] = 1
    result = result.view(batch_size, length, num_actions)
    return result


def generate_probs(args, all_actions, last_action=1):
    all_actions = from_variable_to_numpy(all_actions)
    prob_map = np.concatenate((np.expand_dims(last_action * args.num_total_act + all_actions[:, 0], axis = 1), all_actions[:, :-1] * args.num_total_act + all_actions[:, 1:]), axis = 1)
    prob = torch.histc(torch.from_numpy(prob_map).type(torch.Tensor), bins=args.num_total_act * args.num_total_act).view(args.num_total_act, args.num_total_act)

    prob[prob.sum(dim=1) == 0, :] = 1
    prob /= prob.sum(dim=1).unsqueeze(1)

    return prob


def sample_discrete_action(args, net, obs_var, prev_action=None):
    start_time = time.time()
    obs = np.repeat(np.expand_dims(obs_var, axis=0), args.batch_size, axis=0)
    obs = Variable(torch.from_numpy(obs), requires_grad=False).type(torch.Tensor)
    if torch.cuda.is_available():
        obs = obs.cuda()
    prob = torch.ones(args.num_total_act, args.num_total_act) / float(args.num_total_act)
    with torch.no_grad():
        for i in range(6):
            all_actions = generate_action_sample(args, prob, 6 * args.batch_size, args.pred_step, prev_action)
            one_hot_actions = generate_one_hot_actions(all_actions, args.num_total_act)
            if torch.cuda.is_available():
                one_hot_actions = one_hot_actions.cuda()

            actions = Variable(one_hot_actions, requires_grad=False)
            loss = get_action_loss(args, net, obs, actions)  # needs updating
            all_losses = from_variable_to_numpy(loss)

            if i < 5:
                indices = np.argsort(all_losses)[:args.batch_size]
                prob = generate_probs(args, all_actions[indices], prev_action)
            else:
                idx = np.argmin(all_losses)
                which_action = int(from_variable_to_numpy(all_actions)[idx, 0])

    print('Sampling takes %0.2f seconds, selected action: %d.' % (time.time() - start_time, which_action))
    return which_action


def tile_single(x, action):
    batch_size, c, w, h = x.size()
    assert action.size(0) == batch_size
    action = action.view(action.size(0), -1, 1, 1).repeat(1, 1, w, h)
    return torch.cat([x, action], dim=1)

def tile(x, action):
    return list(map(lambda t: tile_single(t, action), x))


def tile_first(x, action):
    for i in range(len(x) - 1):
        x[i] = tile(x[i], action[:, i, :].float())
    return x


def get_action_loss(args, net, imgs, actions, action_var=None, target=None, hidden=None, cell=None, gpu=0):
    batch_size = int(imgs.size()[0])
    if args.continuous and not args.sample_based_planning:
        batch_size = 1
    if target is None:
        target = dict()
        # if args.sample_with_collision:
        #     target['coll_batch'] = Variable(torch.zeros(batch_size * args.pred_step), requires_grad = False).type(torch.LongTensor)
        # if args.sample_with_offroad:
        #     target['off_batch'] = Variable(torch.zeros(batch_size * args.pred_step), requires_grad = False).type(torch.LongTensor)
        if args.sample_with_pos:
            target['pos_batch'] = Variable(torch.ones(batch_size, args.pred_step, 1) * args.target_pos, requires_grad = False)
        if args.sample_with_angle:
            target['angle_batch'] = Variable(torch.zeros(batch_size, args.pred_step, 1), requires_grad = False)
        if args.target_speed > 0:
            target['speed_batch'] = Variable(torch.ones(batch_size, args.pred_step, 1) * args.target_speed, requires_grad = False)
        if args.target_dist > 0:
            target['dist_batch'] = Variable(torch.ones(batch_size, args.pred_step, 1) * args.target_dist, requires_grad = False)

        if torch.cuda.is_available():
            # if args.sample_with_collision:
            #     target['coll_batch'] = target['coll_batch'].cuda()
            # if args.sample_with_offroad:
            #     target['off_batch'] = target['off_batch'].cuda()
            if args.sample_with_pos:
                target['pos_batch'] = target['pos_batch'].cuda()
            if args.sample_with_angle:
                target['angle_batch'] = target['angle_batch'].cuda()
            if args.target_speed > 0:
                target['speed_batch'] = target['speed_batch'].cuda()
            if args.target_dist > 0:
                target['dist_batch'] = target['dist_batch'].cuda()

    weight = (args.gamma ** np.arange(args.pred_step)).reshape((1, args.pred_step, 1))
    weight = Variable(torch.from_numpy(weight).float().cuda()).repeat(batch_size, 1, 1)
    with torch.no_grad():
        output = net(imgs, actions, hidden=hidden, cell=cell, training=False, action_var=action_var)

    output['coll_prob'] = F.softmax(output['coll_prob'], -1)
    loss = -torch.round(output['coll_prob'][:, :, 0]) * output['value'].view(-1, args.pred_step) + torch.round(output['coll_prob'][:, :, 1]) * 2
    # loss = -(output['reward'].view(-1, args.pred_step, 1) * weight).sum(-1).sum(-1) - output['value'][:, -1, 0].contiguous().view(-1) * args.gamma ** args.pred_step
    loss = (loss.view(-1, args.pred_step, 1) * weight).sum(-1).sum(-1)
    return loss


def show_accuracy(output, target, label):
    tn, fp, fn, tp = confusion_matrix(output, target, labels=[0, 1]).ravel()
    score = (tn + tp) / (tn + fp + fn + tp) * 100.0
    print('%s accuracy: %0.2f%%' % (label, score))
    return score


if __name__ == '__main__':
    class dummy(object):
        def __init__(self):
            self.num_total_act = 3
    prob = torch.ones(3, 3) / 3
    all_actions = generate_action_sample(dummy(), prob, 6, 15, 1)
    one_hot_actions = generate_one_hot_actions(all_actions, 3)
    print(all_actions)
    print(all_actions.size())
    print(one_hot_actions)
    print(one_hot_actions.size())
