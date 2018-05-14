from __future__ import division, print_function
import torch
import torch.nn as nn
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

def train_model_imitation(train_net, mpc_buffer, batch_size, epoch, avg_img_t, std_img_t, pred_step = 15):
    if epoch % 20 == 0:
        x, idxes = mpc_buffer.sample(batch_size, sample_early = True)
    else:
        x, idxes = mpc_buffer.sample(batch_size, sample_early = False)
    x = list(x)
    for ii in range(len(x)):
        if x[ii] is not None:
            x[ii] = torch.from_numpy(x[ii]).float().cuda()
    act_batch = Variable(x[0], requires_grad = False)
    coll_batch = Variable(x[1], requires_grad = False)
    offroad_batch = Variable(x[3], requires_grad = False)
    dist_batch = Variable(x[4])
    img_batch = Variable(((x[5].float() - avg_img_t) / (std_img_t + 0.0001)), requires_grad = False)
    nximg_batch = Variable(((x[6].float() - avg_img_t) / (std_img_t + 0.0001)), requires_grad = False)
    with torch.no_grad():
        nximg_enc = train_net(nximg_batch, get_feature=True)
        nximg_enc = nximg_enc.detach()
    actions = torch.rand(*act_batch.size()).float()
    actions = Variable(actions.cuda(), requires_grad=True)
    actions.data.clamp(-1,1)
    train_net.zero_grad()
    for i in range(10):
        pred_coll, pred_enc, pred_off, pred_dist, _, _ = train_net(img_batch,actions, pred_step)
        coll_ls = Focal_Loss(pred_coll.view(-1, 2), (torch.max(coll_batch.view(-1, 2), -1)[1]).view(-1), reduce = True)
        offroad_ls = Focal_Loss(pred_off.view(-1, 2), (torch.max(offroad_batch.view(-1, 2), -1)[1]).view(-1), reduce = True)
        dist_ls = torch.sqrt(nn.MSELoss()(pred_dist.view(-1, pred_step), dist_batch[:, 1:].view(-1, pred_step)))
        pred_ls = nn.L1Loss()(pred_enc, nximg_enc).sum()
        loss = coll_ls + offroad_ls + dist_ls
        loss.backward(retain_graph=True)
        actions.data -= 0.001 * actions.grad.data
        actions.data.clamp(-1,1)
    action_loss = torch.sqrt(nn.MSELoss()(actions, act_batch))
    return action_loss 
    
def train_model(args, train_net, mpc_buffer, epoch, avg_img_t, std_img_t):
    if epoch % 20 == 0:
        target, idxes = mpc_buffer.sample(args.batch_size, sample_early = True)
    else:
        target, idxes = mpc_buffer.sample(args.batch_size, sample_early = False)
    for key in target.keys():
        target[key] = Variable(torch.from_numpy(target[key]).float(), requires_grad = False)
        if torch.cuda.is_available():
            target[key] = target[key].cuda()

    #img_batch = Variable(((x[5].float()-avg_img_t)/(std_img_t+0.0001)), requires_grad=False)
    #nximg_batch = Variable(((x[6].float()-avg_img_t)/(std_img_t+0.0001)), requires_grad=False)
    target['obs_batch'] = (target['obs_batch'] - 112.62289744791671) / 56.1524832523
    target['nx_obs_batch'] = (target['nx_obs_batch'] - 112.62289744791671) / 56.1524832523
    if args.use_seg:
        target['seg_batch'] = target['seg_batch'].long()
    else:
        with torch.no_grad():
            nximg_enc = train_net(target['nx_obs_batch'], get_feature = True).detach()

    output = train_net(target['obs_batch'], target['act_batch'])

    if args.use_collision:
        show_accuracy(target['coll_batch'].view(-1), torch.max(output['coll_prob'].view(-1, 2), -1)[1], 'collision')
        weight = torch.zeros(2)
        weight[0] = target['coll_batch'].sum() / args.batch_size / args.pred_step
        weight[1] = 1 - weight[0]
        if torch.cuda.is_available():
            weight = weight.cuda()
        coll_ls = nn.CrossEntropyLoss(weight = weight)(output['coll_prob'].view(args.batch_size * args.pred_step, 2), target['coll_batch'].view(args.batch_size * args.pred_step).long())
        # coll_ls = Focal_Loss(output['coll_prob'].view(-1, 2), target['coll_batch'].view(-1), reduce = True)
        print('coll ls', coll_ls.data.cpu().numpy())

    if args.use_offroad:
        show_accuracy(target['off_batch'].view(-1), torch.max(output['offroad_prob'].view(-1, 2), -1)[1], 'offroad')
        weight = torch.zeros(2)
        weight[0] = target['off_batch'].sum() / args.batch_size / args.pred_step
        weight[1] = 1 - weight[0]
        if torch.cuda.is_available():
            weight = weight.cuda()
        offroad_ls = nn.CrossEntropyLoss(weight = weight)(output['offroad_prob'].view(args.batch_size * args.pred_step, 2), target['off_batch'].view(args.batch_size * args.pred_step).long())
        # offroad_ls = Focal_Loss(output['offroad_prob'].view(-1, 2), target['off_batch'].view(-1), reduce = True)
        print('offroad ls', offroad_ls.data.cpu().numpy())

    if args.use_distance:
        dist_ls = torch.sqrt(nn.MSELoss()(output['dist'].view(-1, args.pred_step), target['dist_batch'][:, 1:].view(-1, args.pred_step))) / 40
        print('dist ls', dist_ls.data.cpu().numpy())
    
    if args.use_seg:
        output['seg_pred'] = output['seg_pred'].view(args.batch_size * (args.pred_step + 1), args.classes, 256, 256)
        target['seg_batch'] = target['seg_batch'].view(args.batch_size * (args.pred_step + 1), 256, 256)
        pred_ls = nn.CrossEntropyLoss()(output['seg_pred'], target['seg_batch'])
    else:
        pred_ls = nn.L1Loss()(output['seg_pred'], nximg_enc).sum()
    print('pred ls', pred_ls.data.cpu().numpy()) # nan here!
    loss = pred_ls + coll_ls + offroad_ls + dist_ls

    if args.use_pos:
        pos_loss = torch.sqrt(nn.MSELoss()(output['pos'], target['pos_batch'][:, :-1, :]))
        loss += pos_loss
        print('pos ls', pos_loss.data.cpu().numpy())
    if args.use_angle:
        angle_loss = torch.sqrt(nn.MSELoss()(output['angle'], target['angle_batch'][:, :-1, :]))
        loss += angle_loss
        print('angle ls', angle_loss.data.cpu().numpy())
    if args.use_speed:
        speed_loss = torch.sqrt(nn.MSELoss()(output['speed'], target['sp_batch'][:, :-1, :]))
        loss += speed_loss
        print('speed ls', speed_loss.data.cpu().numpy())
    if args.use_xyz:
        xyz_loss = torch.sqrt(nn.MSELoss()(output['xyz'], target['xyz_batch'])) / 100.0
        loss += xyz_loss
        print('xyz ls', xyz_loss.data.cpu().numpy())

    loss_value = float(loss.data.cpu().numpy())
    if np.isnan(loss_value):
        pdb.set_trace()

    #if use_seg:
    #    seg_loss = sum([nn.CrossEntropyLoss()(seg_out[:, i], seg_batch[:, i]) for i in range(pred_step)])
    #    loss += seg_loss
    #else:
    #    seg_loss = Variable(torch.zeros(1))

    #coll_acc, off_acc, total_dist_ls = log_info(pred_coll, coll_batch, pred_off, offroad_batch, \
    #    float(coll_ls.data.cpu().numpy()), float(offroad_ls.data.cpu().numpy()),\
    #    float(pred_ls.data.cpu().numpy()), float(dist_ls.data.cpu().numpy()), \
    #    float(xyz_loss.data.cpu().numpy()), float(seg_loss.data.cpu().numpy()), float(loss.data.cpu().numpy()), epoch, 1)
    print('1 step training')
    visualize(args, target, output)
    return loss

def draw_from_pred(pred):
    illustration = np.zeros((256, 256, 3)).astype(np.uint8)
    illustration[:, :, 0] = 255
    illustration[pred == 1] = np.array([0, 255, 0])
    illustration[pred == 2] = np.array([0, 0, 0])
    illustration[pred == 3] = np.array([0, 0, 255])
    return illustration

def visualize(args, target, output):
    batch_id = np.random.randint(args.batch_size)
    if args.use_seg:
        observation = (from_variable_to_numpy(target['obs_batch'][batch_id, :, -3:, :, :]) * 56.1524832523 + 112.62289744791671).astype(np.uint8).transpose(0, 2, 3, 1)
        target['seg_batch'] = target['seg_batch'].view(args.batch_size, args.pred_step + 1, 256, 256)
        segmentation = from_variable_to_numpy(target['seg_batch'][batch_id])
        output['seg_pred'] = output['seg_pred'].view(args.batch_size, args.pred_step + 1, args.classes, 256, 256)
        _, prediction = torch.max(output['seg_pred'][batch_id], 1)
        prediction = from_variable_to_numpy(prediction)
        for i in range(args.pred_step):
            imsave('visualize/%d.png' % i, np.concatenate([observation[i], draw_from_pred(segmentation[i]), draw_from_pred(prediction[i])], 1))
    with open(args.save_path+'/report.txt', 'a') as f:
        f.write('target collision:\n')
        f.write(str(from_variable_to_numpy(target['coll_batch'][batch_id])) + '\n')
        f.write('output collision:\n')
        f.write(str(from_variable_to_numpy(output['coll_prob'][batch_id])) + '\n')

        f.write('target offroad:\n')
        f.write(str(from_variable_to_numpy(target['off_batch'][batch_id])) + '\n')
        f.write('output offroad:\n')
        f.write(str(from_variable_to_numpy(output['offroad_prob'][batch_id])) + '\n')

        if args.use_pos:
            f.write('target pos:\n')
            f.write(str(from_variable_to_numpy(target['pos_batch'][batch_id, :-1])) + '\n')
            f.write('output pos:\n')
            f.write(str(from_variable_to_numpy(output['pos'][batch_id])) + '\n')

        if args.use_angle:
            f.write('target angle:\n')
            f.write(str(from_variable_to_numpy(target['angle_batch'][batch_id, :-1])) + '\n')
            f.write('output angle:\n')
            f.write(str(from_variable_to_numpy(output['angle'][batch_id])) + '\n')

        if args.use_speed:
            f.write('target speed:\n')
            f.write(str(from_variable_to_numpy(target['sp_batch'][batch_id, :-1])) + '\n')
            f.write('output speed:\n')
            f.write(str(from_variable_to_numpy(output['speed'][batch_id])) + '\n')

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
            return True
        if abs(pos) >= 21.0:
            return True
        self.pos.append(list(posxyz))
        real_pos = np.concatenate(self.pos[-100:])
        real_pos = real_pos.reshape(-1,3)
        std = np.sum(np.std(real_pos, 0))
        if std < 2.0 and len(self.pos) > 100:
            self.pos = []
            return True
        return False 

class ObsBuffer:
    def __init__(self, frame_history_len=3):
        self.frame_history_len = frame_history_len
        self.last_obs_all = []

    def store_frame(self, frame, avg_img, std_img):
        obs_np = (frame-avg_img)/(std_img+0.0001)
        obs_np = obs_np.transpose(2,0,1)
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

def log_info(pred_coll, coll_batch, pred_off, offroad_batch, 
            total_coll_ls, total_off_ls, total_pred_ls, 
            total_dist_ls, xyz_loss, seg_loss, total_loss, 
            epoch, num_batch):
    coll_label, coll_pred, off_label, off_pred = [], [], [], []
    pred_coll_np = pred_coll.view(-1,2).data.cpu().numpy()
    coll_np = coll_batch.view(-1,2).data.cpu().numpy()
    pred_coll_np = np.argmax(pred_coll_np, 1)
    coll_np = np.argmax(coll_np, 1)
    coll_label.append(coll_np)
    coll_pred.append(pred_coll_np)

    pred_off_np = pred_off.view(-1,2).data.cpu().numpy()
    off_np = offroad_batch.view(-1,2).data.cpu().numpy()
    pred_off_np = np.argmax(pred_off_np, 1)
    off_np = np.argmax(off_np, 1)
    off_label.append(off_np)
    off_pred.append(pred_off_np)
    
    coll_label = np.concatenate(coll_label)
    coll_pred = np.concatenate(coll_pred)
    off_label = np.concatenate(off_label)
    off_pred = np.concatenate(off_pred)
    cnf_matrix = confusion_matrix(coll_label, coll_pred)
    cnf_matrix_off = confusion_matrix(off_label, off_pred)
    coll_acc, off_accuracy = 0, 0
    try:
        coll_acc = (cnf_matrix[0,0] + cnf_matrix[1,1]) / (cnf_matrix.sum() * 1.0)
        off_accuracy = (cnf_matrix_off[0,0] + cnf_matrix_off[1,1]) / (cnf_matrix_off.sum() * 1.0)
        if epoch % 20 == 0:
            print('sample early collacc', "{0:.3f}".format(coll_acc), \
                "{0:.3f}".format(total_coll_ls / num_batch), \
                'offacc', "{0:.3f}".format(off_accuracy), \
                "{0:.3f}".format(total_off_ls / num_batch), \
                'xyzls', "{0:.3f}".format(xyz_loss / num_batch), \
                'segls', "{0:.3f}".format(seg_loss / num_batch), \
                'ttls', "{0:.3f}".format(total_loss / num_batch), \
                'predls', "{0:.3f}".format(total_pred_ls / num_batch), \
                'distls', "{0:.3f}".format(total_dist_ls / num_batch))
        else:
            print('collacc', "{0:.3f}".format(coll_acc), \
                "{0:.3f}".format(total_coll_ls / num_batch), \
                'offacc', "{0:.3f}".format(off_accuracy), \
                "{0:.3f}".format(total_off_ls / num_batch), \
                'xyzls', "{0:.3f}".format(xyz_loss / num_batch), \
                'segls', "{0:.3f}".format(seg_loss / num_batch), \
                'ttls', "{0:.3f}".format(total_loss / num_batch), \
                'predls', "{0:.3f}".format(total_pred_ls / num_batch), \
                'distls', "{0:.3f}".format(total_dist_ls / num_batch))
    except:
        print('dist ls', total_dist_ls / num_batch)
    return coll_acc, off_accuracy, total_dist_ls / num_batch

def init_dirs(dir_list):
    for path in dir_list:
        make_dir(path)

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def load_model(path, net, data_parallel = True, optimizer = None, resume=True):
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
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net) if data_parallel else net
        net = net.cuda()

    if optimizer is not None and epoch > 0:
        optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer', 'optimizer.pt')))

    if optimizer is None:
        return net, epoch
    else:
        return net, epoch, optimizer
        
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
    loss = -1.0 * (1-probs).pow(1) * torch.log(probs)
    batch_size = int(probs.size()[0])
    loss = loss[torch.arange(batch_size).long().cuda(), target.long()]
    if reduce == True:
        loss = loss.sum()/(batch_size*1.0)
    return loss

def sample_cont_action(args, net, imgs, prev_action = None):
    imgs = imgs.contiguous()
    batch_size, c, w, h = int(imgs.size()[0]), int(imgs.size()[-3]), int(imgs.size()[-2]), int(imgs.size()[-1])
    imgs = imgs.view(batch_size, 1, c, w, h)
    prev_action = prev_action.reshape((1, 1, 2))
    prev_action = np.repeat(prev_action, args.pred_step, axis=1) 
    this_action = torch.from_numpy(prev_action).float()
    this_action = Variable(this_action.cuda(), requires_grad=True)
    this_action.data.clamp(-1,1)
    prev_loss = 1000
    sign = True
    cnt = 0
    optimizer = optim.Adam([this_action], lr = 0.01)
    while sign:
        loss = get_action_loss(args, net, imgs, this_action, None, None, None)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        this_loss = float(loss.data.cpu().numpy())
        if cnt >= 10 and (np.abs(prev_loss-this_loss)/prev_loss <= 0.0005 or this_loss > prev_loss):
            sign = False
            return this_action.data.cpu().numpy()[0,0,:].reshape(-1)
        cnt += 1 
        this_action.data.clamp(-1, 1)# = torch.clamp(this_action, -1, 1)
        prev_loss = this_loss
    return this_action.data.cpu().numpy()[0,0,:].reshape(-1) 

def from_variable_to_numpy(x):
    x = x.data
    if torch.cuda.is_available():
        x = x.cpu()
    x = x.numpy()
    return x

def generate_action_sample(args, prob, batch_size, length, LAST_ACTION = 1):
    all_actions = torch.zeros(batch_size, length).type(torch.LongTensor)
    all_actions[: ,0] = prob[LAST_ACTION].multinomial(num_samples = batch_size, replacement = True).data

    for step in range(1, length):
        indices = [torch.nonzero(all_actions[: ,step - 1] == x).squeeze() for x in range(args.num_total_act)]
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

def generate_probs(args, all_actions, last_action = 1):
    all_actions = from_variable_to_numpy(all_actions)
    prob_map = np.concatenate((np.expand_dims(last_action * args.num_total_act + all_actions[:, 0], axis = 1), all_actions[:, :-1] * args.num_total_act + all_actions[:, 1:]), axis = 1)
    prob = torch.histc(torch.from_numpy(prob_map).type(torch.Tensor), bins = args.num_total_act * args.num_total_act).view(args.num_total_act, args.num_total_act)


    prob[prob.sum(dim = 1) == 0, :] = 1
    prob /= prob.sum(dim = 1).unsqueeze(1)

    return prob

def sample_discrete_action(args, net, obs_var, prev_action = None):
    start_time = time.time()
    obs = np.repeat(np.expand_dims(obs_var, axis = 0), args.batch_size, axis = 0)
    obs = Variable(torch.from_numpy(obs), requires_grad = False).type(torch.Tensor)
    if torch.cuda.is_available():
        obs = obs.cuda()
    prob = torch.ones(args.num_total_act, args.num_total_act) / float(args.num_total_act)
    with torch.no_grad():
        for i in range(6):
            all_actions = generate_action_sample(args, prob, 6 * args.batch_size, args.pred_step, prev_action)
            one_hot_actions = generate_one_hot_actions(all_actions, args.num_total_act)
            if torch.cuda.is_available():
                one_hot_actions = one_hot_actions.cuda()
            all_losses = np.zeros(6 * args.batch_size)

            for ii in range(6):
                actions = Variable(one_hot_actions[ii * args.batch_size: (ii + 1) * args.batch_size], requires_grad = False)
                loss = get_action_loss(args, net, obs, actions) # needs updating
                all_losses[ii * args.batch_size: (ii + 1) * args.batch_size] = from_variable_to_numpy(loss)

            if i < 5:
                indices = np.argsort(all_losses)[:args.batch_size]
                prob = generate_probs(args, all_actions[indices], prev_action)
            else:
                idx = np.argmin(all_losses)
                which_action = int(from_variable_to_numpy(all_actions)[idx, 0])

    print('Sampling takes %0.2f seconds, selected action: %d.' % (time.time() - start_time, which_action))
    return which_action

def get_action_loss(args, net, imgs, actions, target = None, hidden = None, cell = None, gpu = 0):
    batch_size = int(imgs.size()[0])
    if args.continuous:
        batch_size = 1
    if target is None:
        target = dict()
        target['coll_batch'] = Variable(torch.zeros(batch_size * args.pred_step), requires_grad = False).type(torch.LongTensor)
        target['off_batch'] = Variable(torch.zeros(batch_size * args.pred_step), requires_grad = False).type(torch.LongTensor)
        if args.use_pos and args.use_angle:
            target['pos_batch'] = Variable(torch.zeros(batch_size, args.pred_step, 1), requires_grad = False)

        if torch.cuda.is_available():
            target['coll_batch'] = target['coll_batch'].cuda()
            target['off_batch'] = target['off_batch'].cuda()
            if args.use_pos and args.use_angle:
                target['pos_batch'] = target['pos_batch'].cuda()

    weight = (0.97 ** np.arange(args.pred_step)).reshape((1, args.pred_step, 1))
    weight = Variable(torch.from_numpy(weight).float().cuda()).repeat(batch_size, 1, 1)
    output = net(imgs, actions, hidden = hidden, cell = cell)

    loss = 0
    if args.sample_with_collision:
        coll_ls = nn.CrossEntropyLoss(reduce = False)(output['coll_prob'].view(batch_size * args.pred_step, 2), target['coll_batch'])
        coll_ls = (coll_ls.view(-1, args.pred_step, 1) * weight).sum()
        loss += coll_ls
    if args.sample_with_offroad:
        off_ls = nn.CrossEntropyLoss(reduce = False)(output['offroad_prob'].view(batch_size * args.pred_step, 2), target['off_batch'])
        off_ls = (off_ls.view(-1, args.pred_step, 1) * weight).sum()
        loss += off_ls
    if args.sample_with_distance:
        dist_ls = (output['dist'].view(-1, args.pred_step, 1) * weight).sum()
        loss -= 0.1 * dist_ls

    if 'pos_batch' in target.keys() and 'angle_batch' in target.keys():
        pos_loss = torch.sqrt(nn.MSELoss()(output['pos'] + torch.sin(output['angle']), target['pos_batch']))
        loss += pos_loss
    
    if 'sp_batch' in target.keys():
        speed_loss = torch.sqrt(nn.MSELoss()(output['speed'], target['sp_batch']))
        loss += speed_loss
    if 'xyz_batch' in target.keys():
        xyz_loss = torch.sqrt(nn.MSELoss()(output['xyz'], target['xyz_batch'])) / 100.0
        loss += xyz_loss

    return loss

def show_accuracy(output, target, label):
    tn, fp, fn, tp = confusion_matrix(output, target, labels = [0, 1]).ravel()
    print('%s accuracy: %0.2f%%' % (label, (tn + tp) / (tn + fp + fn + tp) * 100.0))

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
