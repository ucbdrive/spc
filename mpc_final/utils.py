from __future__ import division, print_function
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import os
import PIL.Image as Image
import random
from sklearn.metrics import confusion_matrix
import pdb
from model import *

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

def train_model_imitation(train_net, mpc_buffer, batch_size, epoch, avg_img_t, std_img_t, pred_step=15):
    if epoch % 20 == 0:
        x, idxes = mpc_buffer.sample(batch_size, sample_early = True)
    else:
        x, idxes = mpc_buffer.sample(batch_size, sample_early = False)
    x = list(x)
    for ii in range(len(x)):
        if x[ii] is not None:
            x[ii] = torch.from_numpy(x[ii]).float().cuda()
    act_batch = Variable(x[0], requires_grad = False)
    coll_batch = Variable(x[1], requires_grad=False)
    offroad_batch = Variable(x[3], requires_grad=False)
    dist_batch = Variable(x[4])
    img_batch = Variable(((x[5].float()-avg_img_t)/(std_img_t+0.0001)), requires_grad=False)
    nximg_batch = Variable(((x[6].float()-avg_img_t)/(std_img_t+0.0001)), requires_grad=False)
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
    target['obs_batch'] /= 255.0
    target['nx_obs_batch'] /= 255.0
    if args.use_seg:
        target['seg_batch'] = target['seg_batch'].long()
    else:
        with torch.no_grad():
            nximg_enc = train_net(target['nx_obs_batch'], get_feature = True).detach()

    output = train_net(target['obs_batch'], target['act_batch'])

    coll_ls = Focal_Loss(output['coll_prob'].view(-1, 2), (torch.max(target['coll_batch'].view(-1, 2), -1)[1]).view(-1), reduce = True)
    offroad_ls = Focal_Loss(output['offroad_prob'].view(-1, 2), (torch.max(target['off_batch'].view(-1, 2), -1)[1]).view(-1), reduce = True)
    dist_ls = torch.sqrt(nn.MSELoss()(output['dist'].view(-1, args.pred_step), target['dist_batch'][:,1:].view(-1, args.pred_step)))
    if args.use_seg == False:
        pred_ls = nn.L1Loss()(output['seg_pred'], nximg_enc).sum()
    else:
        output['seg_pred'] = output['seg_pred'].permute(0, 1, 3, 4, 2).contiguous()#.view(-1, 4)
        target['seg_batch'] = target['seg_batch'].permute(0, 1, 3, 4, 2)#.view(-1, 1)
        pred_ls = nn.CrossEntropyLoss()(output['seg_pred'].view(-1, 4), target['seg_batch'].view(-1))
    loss = pred_ls + coll_ls + offroad_ls + 10 * dist_ls

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

    print('pred ls', pred_ls.data.cpu().numpy()) # nan here!
    print('coll ls', coll_ls.data.cpu().numpy())
    print('offroad ls', offroad_ls.data.cpu().numpy())
    print('dist ls', dist_ls.data.cpu().numpy())
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
    return loss
 
class DoneCondition:
    def __init__(self, size):
        self.size = size
        self.off_cnt = 0
        self.pos = []

    def isdone(self, pos, dist, posxyz):
        if pos <=-6.2 and dist < 0:
            self.off_cnt += 1
        elif pos > -6.2 or dist > 0:
            self.off_cnt = 0
        if self.off_cnt > self.size:
            return True
        if abs(pos) >= 15.0:
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
        #obs_np = (frame-avg_img)/(std_img+0.0001)
        obs_np = frame/255.0
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
        optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer/optim_' + model_path.split('_')[-1])))

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
    optimizer = optim.Adam([this_action], lr = 0.06)
    while sign:
        loss = get_action_loss(args, net, imgs, this_action, None, None)
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

def get_action_loss(args, net, imgs, actions, target = None, hidden = None, cell = None, gpu = 0):
    batch_size = int(imgs.size()[0])
    if target is None:
        target_coll_np = np.zeros((args.pred_step, 2))
        target_coll_np[:, 0] = 1.0
        target['coll'] = Variable(torch.from_numpy(target_coll_np).float(), requires_grad = False).cuda()
        target['off'] = Variable(torch.from_numpy(target_coll_np).float(), requires_grad = False).cuda()

    weight = (0.97 ** np.arange(args.pred_step)).reshape((1, args.pred_step, 1))
    weight = Variable(torch.from_numpy(weight).float().cuda()).repeat(batch_size, 1, 1)
    output = net(imgs, actions, hidden = hidden, cell = cell)

    coll_ls = nn.CrossEntropyLoss(reduce = False)(output['coll_prob'].view(-1, 2), torch.max(target['coll_batch'].view(-1, 2), -1)[1])
    off_ls = nn.CrossEntropyLoss(reduce = False)(output['coll_prob'].view(-1, 2), torch.max(target['off_batch'].view(-1, 2), -1)[1])
    coll_ls = (coll_ls.view(-1, args.pred_step, 1) * weight).view(-1, args.pred_step).sum(-1)
    off_ls = (off_ls.view(-1, args.pred_step, 1) * weight).view(-1, args.pred_step).sum(-1)
    dist_ls = (output['dist'].view(-1, args.pred_step, 1) * weight).view(-1, args.pred_step).sum(-1)
    loss = off_ls + coll_ls - 0.1 * dist_ls

    if args.use_pos:
        pos_loss = torch.sqrt(nn.MSELoss()(output['pos'], target['pos_batch']))
        loss += pos_loss
    if args.use_angle:
        angle_loss = torch.sqrt(nn.MSELoss()(output['angle'], target['angle_batch']))
        loss += angle_loss
    if args.use_speed:
        speed_loss = torch.sqrt(nn.MSELoss()(output['speed'], target['sp_batch']))
        loss += speed_loss
    if args.use_xyz:
        xyz_loss = torch.sqrt(nn.MSELoss()(output['xyz'], target['xyz_batch'])) / 100.0
        loss += xyz_loss

    return loss
