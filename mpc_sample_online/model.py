import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import dla
import pdb
import numpy as np
from generate_action_samples import *
import random
import pickle as pkl

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.fc1 = nn.Linear(input_dim + hidden_dim, hidden_dim, bias = self.bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias = self.bias)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias = self.bias)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim, bias = self.bias)
        self.W = nn.Linear(hidden_dim, 4 * hidden_dim, bias = self.bias)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim = 1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        combined_conv = F.relu(self.W(x))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim = 1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next # h_next is the output

    def init_hidden(self, batch_size):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            return (Variable(torch.zeros(batch_size, self.hidden_dim)).cuda(),
                    Variable(torch.zeros(batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(batch_size, self.hidden_dim)),
                    Variable(torch.zeros(batch_size, self.hidden_dim)))

class ConvLSTMNet(nn.Module):
    def __init__(self, 
                num_actions=9,
                pretrain=True, # use pretrained dla model 
                frame_history_len=4,
                freeze_dla=False,
                hidden_dim=512):
        super(ConvLSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.frame_history_len = frame_history_len
        self.dla = dla.dla46x_c(pretrained = pretrain)
        if freeze_dla:
            for param in self.dla.parameters():
                param.requires_grad = False
        
        self.feature_encode = nn.Linear(256 * frame_history_len, self.hidden_dim)
        self.outfeature_encode = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.act_encode = nn.Linear(num_actions, 16) # action encoding
        self.info_dim = 16
        self.info_encode = nn.Linear(self.info_dim + self.hidden_dim, self.hidden_dim)
        self.lstm = ConvLSTMCell(self.hidden_dim, self.hidden_dim, True)
        self.fc_coll_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
        self.fc_coll_2 = nn.Linear(128+16, 32)
        self.fc_coll_3 = nn.Linear(32+16, 2)
        self.fc_off_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
        self.fc_off_2 = nn.Linear(128+16, 32)
        self.fc_off_3 = nn.Linear(32+16, 2)
        self.fc_dist_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
        self.fc_dist_2 = nn.Linear(128+16, 32)
        self.fc_dist_3 = nn.Linear(32+16, 1)
        self.fc_dist_tanh = nn.Tanh()
 
    def get_feature(self, x):
        res = []
        for i in range(self.frame_history_len):
            out = self.dla(x[:, i * 3 : (i + 1) * 3, :, :])
            out = out.squeeze().view(out.size(0), -1)
            res.append(out)
        res = torch.cat(res, dim = 1)
        res = self.feature_encode(res)
        return res # batch * 128

    def forward(self, x, action, with_encode=False, hidden=None, cell=None):
        if with_encode == False:
            x = self.get_feature(x) # batch * 128
        if hidden is None or cell is None:
            hidden, cell = x, x # batch * 128
        action_enc = F.relu(self.act_encode(action))
        info_enc = action_enc
        encode = torch.cat([x, info_enc], dim=1) # batch * 256
        encode = F.relu(self.info_encode(encode))
        hidden, cell = self.lstm(encode, [hidden, cell])
        pred_encode_nx = hidden.view(-1, self.hidden_dim)
        nx_feature_enc = self.outfeature_encode(F.relu(pred_encode_nx))
        hidden_enc = torch.cat([pred_encode_nx, info_enc], dim = 1)

        # outputs
        coll_prob = F.relu(self.fc_coll_1(hidden_enc))
        coll_prob = torch.cat([coll_prob, action_enc], dim = 1)
        coll_prob = F.relu(self.fc_coll_2(coll_prob))
        coll_prob = torch.cat([coll_prob, action_enc], dim = 1)
        coll_prob = nn.Softmax(dim=-1)(F.relu(self.fc_coll_3(coll_prob)))

        offroad_prob = F.relu(self.fc_off_1(hidden_enc))
        offroad_prob = torch.cat([offroad_prob, action_enc], dim=1)
        offroad_prob = F.relu(self.fc_off_2(offroad_prob))
        offroad_prob = torch.cat([offroad_prob, action_enc], dim=1)
        offroad_prob = nn.Softmax(dim=-1)(F.relu(self.fc_off_3(offroad_prob)))

        dist = F.relu(self.fc_dist_1(hidden_enc))
        dist = torch.cat([dist, action_enc], dim=1)
        dist = F.relu(self.fc_dist_2(dist))
        dist = torch.cat([dist, action_enc], dim=1)
        dist = self.fc_dist_tanh(F.relu(self.fc_dist_3(dist)))*100
        return coll_prob, nx_feature_enc, offroad_prob, dist, hidden, cell

class ConvLSTMMulti(nn.Module):
    def __init__(self, na, pretrain = True, frame_history_len = 4, freeze_dla=False):
        super(ConvLSTMMulti, self).__init__()
        self.conv_lstm = ConvLSTMNet(na, pretrain = pretrain, frame_history_len = frame_history_len, freeze_dla=freeze_dla)
        self.frame_history_len = frame_history_len
        self.num_actions = na

    def get_feature(self, x):
        feat = []
        x = x.contiguous()
        _, num_time, _, _, _ = int(x.size()[0]), int(x.size()[1]), int(x.size()[2]), int(x.size()[3]), int(x.size()[4])
        for i in range(num_time):
            feat.append(self.conv_lstm.get_feature(x[:, i, :, :, :].squeeze(1)))
        return torch.stack(feat, dim = 1)

    def forward(self, imgs, actions=None, num_time=None, hidden=None, cell=None, get_feature=False):
        if get_feature:
            res = self.get_feature(imgs)
            return res
        batch_size, num_step, c, w, h = int(imgs.size()[0]), int(imgs.size()[1]), int(imgs.size()[-3]), int(imgs.size()[-2]), int(imgs.size()[-1])
        coll, pred, offroad, dist, hidden, cell = self.conv_lstm(imgs[:,0,:,:,:].squeeze(1), actions[:,0,:].squeeze(1), hidden=hidden, cell=cell)
        num_act = self.num_actions
        coll_list = [coll]
        pred_list = [pred]
        offroad_list = [offroad]
        dist_list = [dist]
        for i in range(1, num_time):
            coll, pred, offroad, dist, hidden, cell = self.conv_lstm(pred, actions[:,i,:].squeeze(1), with_encode=True, hidden=hidden, cell=cell)
            coll_list.append(coll)
            pred_list.append(pred)
            offroad_list.append(offroad)
            dist_list.append(dist)
        return torch.stack(coll_list, dim=1), torch.stack(pred_list, dim=1), torch.stack(offroad_list,dim=1), \
            torch.stack(dist_list, dim=1), hidden, cell

def get_action_loss(net, imgs, actions, num_time = 3, hidden = None, cell = None, gpu=0):
    batch_size = int(imgs.size()[0])
    target_coll_np = np.zeros((batch_size, num_time, 2))
    target_coll_np[:,:,0] = 1.0
    target_coll = Variable(torch.from_numpy(target_coll_np).float()).cuda()
    target_off = Variable(torch.from_numpy(target_coll_np).float()).cuda()
    weight = []
    for i in range(num_time):
        weight.append(0.97**i)
    weight = Variable(torch.from_numpy(np.array(weight).reshape((1, num_time, 1))).float().cuda()).repeat(batch_size, 1, 1)
    outs = net.forward(imgs, actions, num_time=num_time, hidden=hidden, cell=cell)
    coll_ls = nn.CrossEntropyLoss(reduce=False)(outs[0].view(-1,2), torch.max(target_coll.view(-1,2),-1)[1])
    off_ls = nn.CrossEntropyLoss(reduce=False)(outs[2].view(-1,2), torch.max(target_off.view(-1,2),-1)[1])
    coll_ls = (coll_ls.view(-1,num_time,1)*weight).view(-1,num_time).sum(-1)
    off_ls = (off_ls.view(-1,num_time,1)*weight).view(-1,num_time).sum(-1)
    dist_ls = (outs[3].view(-1,num_time,1)*weight).view(-1,num_time).sum(-1)
    loss = off_ls + coll_ls - 0.1 * dist_ls
    return coll_ls.data.cpu().numpy().reshape((-1)), off_ls.data.cpu().numpy().reshape((-1)), dist_ls.data.cpu().numpy().reshape((-1)),\
        outs[0][:,:,0].data.cpu().numpy(), outs[2][:,:,0].data.cpu().numpy(), outs[3][:,:,0].data.cpu().numpy(), loss        
     
def sample_action(net, imgs, num_time=3, hidden=None, cell=None, num_actions = 6, calculate_loss=False, batch_step=200, gpu=2, same_step=False, use_optimize=False):
    imgs = imgs.contiguous()
    batch_size, c, w, h = int(imgs.size()[0]), int(imgs.size()[-3]), int(imgs.size()[-2]), int(imgs.size()[-1])
    imgs = imgs.view(batch_size, 1, c, w, h)

    one_step = torch.zeros(6, 1, 6)
    for i in range(6):
        one_step[i, :, i] = 1
    if torch.cuda.is_available():
        one_step = one_step.cuda()

    this_action = Variable(one_step, requires_grad = False)
    this_imgs = imgs.repeat(6, 1, 1, 1, 1)
    coll_ls, off_ls, dist_ls, coll_prob, off_prob, distance, _ = get_action_loss(net, this_imgs, this_action, 1, hidden, cell, gpu=gpu)
    batch_ls = coll_ls + off_ls - 0.1 * dist_ls
    prob = F.softmax(-Variable(torch.from_numpy(batch_ls), requires_grad = False), dim = 0)

    action = prob.multinomial(num_samples = 1).data
    if torch.cuda.is_available():
        action = action.cpu()
    action = action.numpy()[0]

    return action

    num_choices = 60
    if num_choices < batch_step:
        batch_step = num_choices

    all_actions = generate_action_sample(prob, num_choices, num_time)
    this_imgs = imgs.repeat(batch_step, 1, 1, 1, 1)

    min_ls = 100000000
    which_action = -1
    for ii in range(int(num_choices / batch_step)):
        this_action = Variable(all_actions[ii * batch_step:min((ii + 1) * batch_step, num_choices), :, :])
        if (ii + 1) * batch_step > num_choices:
            this_imgs = imgs.repeat(int(this_action.size()[0]), 1, 1, 1, 1)
        coll_ls, off_ls, dist_ls, coll_prob, off_prob, distance, _ = get_action_loss(net, this_imgs, this_action, num_time, hidden, cell, gpu=gpu)
        # print(coll_ls.shape, off_ls.shape, dist_ls.shape, coll_prob.shape, off_prob.shape, distance.shape) # ((20,), (20,), (20,), (20, 15), (20, 15), (20, 15))
        batch_ls = coll_ls + off_ls - 0.1 * dist_ls
        idx = np.argmin(batch_ls)
        this_loss = batch_ls[idx]
        if this_loss < min_ls or ii == 0:
            which_action = np.argmax(this_action.data.cpu().numpy()[idx,0,:].squeeze())
            min_ls = this_loss
    return which_action

def quantize_action(action, batch_size, num_time, num_actions, requires_grad=False, prev_action=None):
    act = torch.max(action, -1)[1]
    act_np = np.zeros((batch_size, num_time, num_actions))
    if prev_action is None:
        for j in range(batch_size):
            act_np[j, np.arange(num_time), act.cpu().data.numpy().astype(np.uint8)[j,:]] = 1
    elif prev_action == -1:
        pass
    else:
        for j in range(batch_size):
            act_np[j, np.arange(num_time), (np.arange(num_time)*0+prev_action).astype(np.uint8)] = 1
    act_np = act_np.reshape((batch_size, num_time, num_actions))
    action_v = Variable(torch.from_numpy(act_np).float().cuda(), requires_grad=requires_grad)
    return action_v

def generate_action_sample(prob, batch_size, length):
    num_actions = prob.size(0)
    all_actions = torch.zeros(batch_size, length, num_actions)

    for i in range(batch_size):
        for j in range(length):
            action = prob.multinomial(num_samples = 1).data
            if torch.cuda.is_available():
                action = action.cpu()
            action = action.numpy()[0]
            all_actions[i, j, action] = 1

    if torch.cuda.is_available():
        all_actions = all_actions.cuda()

    return all_actions