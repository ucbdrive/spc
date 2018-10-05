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

class atari_model(nn.Module):
    def __init__(self, in_channels = 12, num_actions = 18):
        super(atari_model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        self.num_actions = num_actions

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        res = self.fc5(x)
        return res

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
                in_channel=3, 
                out_channel=3, 
                num_actions=9,
                pretrain=True, # use pretrained dla model 
                with_lstm=True, # with lstm
                multi_info=False, # multi features with input image feature
                with_posinfo=False, # with xyz position information
                use_pos_class=False, # convert regression problem into classification problem
                with_speed=False,
                with_pos=False,
                frame_history_len=4,
                freeze_dla=False,
                hidden_dim=512,
                batch_norm=False,
                with_dla=True): # using speed as input
        super(ConvLSTMNet, self).__init__()
        self.with_lstm = with_lstm
        self.multi_info = multi_info
        self.use_pos_class = use_pos_class
        self.with_speed = with_speed
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.with_posinfo = with_posinfo
        self.with_pos = with_pos
        self.frame_history_len = frame_history_len
        self.batch_norm = batch_norm
        self.with_dla = with_dla
        if self.with_dla:
            self.dla = dla.dla46x_c(pretrained = pretrain)
        else:
            self.dla = nn.Sequential(nn.Conv2d(in_channel, 16, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, 2, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )

        if torch.cuda.is_available():
            self.dla = self.dla.cuda()
        if freeze_dla:
            for param in self.dla.parameters():
                param.requires_grad = False
        self.feature_encode = nn.Linear(256 * frame_history_len, self.hidden_dim)
        self.outfeature_encode = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.info_dim = 0
        self.act_encode = nn.Linear(num_actions, 16) # action encoding
        self.act_bn = nn.BatchNorm1d(16)
        self.info_dim += 16
        if with_pos:
            if use_pos_class:
                self.pos_encode = nn.Linear(19, 32)
                if self.batch_norm:
                    self.pos_bn = nn.BatchNorm1d(32)
                self.info_dim += 32
            else:
                self.pos_encode = nn.Linear(1, 16)
                if self.batch_norm:
                    self.pos_bn = nn.BatchNorm1d(16)
                self.info_dim += 16
        if with_posinfo:
            self.posxyz_encode = nn.Linear(3, 32)
            if self.batch_norm:
                self.posxyz_bn = nn.BatchNorm1d(32)
            self.info_dim += 32
        if with_speed:
            self.speed_encode = nn.Linear(2, 16)
            if self.batch_norm:
                self.speed_bn = nn.BatchNorm1d(16)
            self.info_dim += 16
        if multi_info:
            self.info_encode = nn.Linear(self.info_dim, self.hidden_dim)
        else:
            self.info_encode = nn.Linear(self.info_dim + self.hidden_dim, self.hidden_dim)
        if self.batch_norm:
            self.info_bn = nn.BatchNorm1d(self.hidden_dim)
        if with_lstm:
            self.lstm = ConvLSTMCell(self.hidden_dim, self.hidden_dim, True)
        self.fc_coll_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
        if self.batch_norm:
            self.fc_coll_bn1 = nn.BatchNorm1d(128)
        self.fc_coll_2 = nn.Linear(128, 32)
        if self.batch_norm:
            self.fc_coll_bn2 = nn.BatchNorm1d(32)
        self.fc_coll_3 = nn.Linear(32, 2)
        self.fc_off_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
        if self.batch_norm:
            self.fc_off_bn1 = nn.BatchNorm1d(128)
        self.fc_off_2 = nn.Linear(128, 32)
        if self.batch_norm:
            self.fc_off_bn2 = nn.BatchNorm1d(32)
        self.fc_off_3 = nn.Linear(32, 2)
        if with_pos:
            self.fc_pos_1 = nn.Linear(self.hidden_dim + self.info_dim, 32)
            if self.batch_norm:
                self.fc_pos_bn = nn.BatchNorm1d(32)
            self.fc_pos_tanh = nn.Tanh()
            if use_pos_class:
                self.fc_pos_2 = nn.Linear(32, 19)
            else:
                self.fc_pos_2 = nn.Linear(32, 1)
        self.fc_dist_1 = nn.Linear(self.hidden_dim + self.info_dim, 128)
        if self.batch_norm:
            self.fc_dist_bn1 = nn.BatchNorm1d(128)
        self.fc_dist_2 = nn.Linear(128, 32)
        if self.batch_norm:
            self.fc_dist_bn2 = nn.BatchNorm1d(32)
        self.fc_dist_3 = nn.Linear(32, 1)
        self.fc_dist_tanh = nn.Tanh()
        if with_speed:
            self.fc_speed_1 = nn.Linear(self.hidden_dim + self.info_dim, 32)
            if self.batch_norm:
                self.fc_speed_bn = nn.BatchNorm1d(32)
            self.fc_speed_2 = nn.Linear(self.hidden_dim + self.info_dim, 2)
        if with_posinfo:
            self.fc_posxyz_1 = nn.Linear(self.hidden_dim + self.info_dim, 32)
            if self.batch_norm:
                self.fc_posxyz_bn = nn.BatchNorm1d(32)
            self.fc_posxyz_2 = nn.Linear(32, 3)
 
    def get_feature(self, x):
        res = []
        for i in range(self.frame_history_len):
            out = self.dla(x[:, i * 3 : (i + 1) * 3, :, :])
            out = out.squeeze().view(out.size(0), -1)
            res.append(out)
        res = torch.cat(res, dim = 1)
        res = self.feature_encode(res)
        return res # batch * 128

    def forward(self, x, action, speed=None, pos=None, posxyz=None, with_encode=False, hidden=None, cell=None):
        if with_encode == False:
            x = self.get_feature(x) # batch * 128
        if self.with_lstm:
            if hidden is None or cell is None:
                hidden, cell = x, x # batch * 128
        if self.batch_norm:
            action_enc = F.relu(self.act_bn(self.act_encode(action))) # batch * 64
        else:
            action_enc = F.relu(self.act_encode(action))
        if self.with_pos:
            if self.batch_norm:
                pos_enc = F.relu(self.pos_bn(self.pos_encode(pos)))
            else:
                pos_enc = F.relu(self.pos_encode(pos))
            info_enc = torch.cat([action_enc, pos_enc], dim=1)
        else:
            info_enc = action_enc
        if self.with_speed:
            if self.batch_norm:
                speed_enc = F.relu(self.speed_bn(self.speed_encode(speed))) # batch * 16
            else:
                speed_enc = F.relu(self.speed_encode(speed))
            info_enc = torch.cat([info_enc, speed_enc], dim=1)
        if self.with_posinfo:
            if self.batch_norm:
                posxyz_enc = F.relu(self.posxyz_bn(self.posxyz_encode(posxyz)))
            else:
                posxyz_enc = F.relu(self.posxyz_encode(posxyz))
            info_enc = torch.cat([info_enc, posxyz_enc], dim=1)
        if self.multi_info == True:
            if self.batch_norm:
                encode = F.relu(self.info_bn(self.info_encode(info_enc)))
            else:
                encode = F.relu(self.info_encode(info_enc))
            encode = F.relu(encode * x)  # batch * 256
        else:
            encode = torch.cat([x, info_enc], dim=1) # batch * 256
            if self.batch_norm:
                encode = F.relu(self.info_bn(self.info_encode(encode)))
            else:
                encode = F.relu(self.info_encode(encode))
        if self.with_lstm:
            hidden, cell = self.lstm(encode, [hidden, cell])
            pred_encode_nx = hidden.view(-1, self.hidden_dim)
            nx_feature_enc = self.outfeature_encode(F.relu(pred_encode_nx))
        else:
            pred_encode_nx = encode # batch * 256
            nx_feature_enc = self.outfeature_encode(F.relu(pred_encode_nx))
        hidden_enc = torch.cat([pred_encode_nx, info_enc], dim = 1)

        # outputs
        if self.batch_norm:
            coll_prob = nn.Softmax(dim = -1)(self.fc_coll_3(F.relu(self.fc_coll_bn2(self.fc_coll_2(F.relu(self.fc_coll_bn1(self.fc_coll_1(hidden_enc))))))))
        else:
            coll_prob = nn.Softmax(dim=-1)(self.fc_coll_3(F.relu(self.fc_coll_2(F.relu(self.fc_coll_1(hidden_enc))))))
        if self.with_pos:
            if self.batch_norm:
                pos_pred = 20 * self.fc_pos_tanh(self.fc_pos_2(F.relu(self.fc_pos_bn(self.fc_pos_1(hidden_enc)))))
            else:
                pos_pred = 20 * self.fc_pos_tanh(self.fc_pos_2(F.relu(self.fc_pos_1(hidden_enc))))
        if self.use_pos_class and self.with_pos:
            pos_pred = nn.Softmax(dim=-1)(pos_pred)
        if self.with_pos == False:
            pos_pred = None
        if self.batch_norm:
            offroad_prob = nn.Softmax(dim=-1)(self.fc_off_3(F.relu(self.fc_off_bn2(self.fc_off_2(F.relu(self.fc_off_bn1(self.fc_off_1(hidden_enc))))))))
        else:
            offroad_prob = nn.Softmax(dim=-1)(self.fc_off_3(F.relu(self.fc_off_2(F.relu(self.fc_off_1(hidden_enc))))))
        if self.batch_norm:
            dist = self.fc_dist_tanh(self.fc_dist_3(F.relu(self.fc_dist_bn2(self.fc_dist_2(F.relu(self.fc_dist_bn1(self.fc_dist_1(hidden_enc))))))))*100
        else:
            dist = self.fc_dist_tanh(self.fc_dist_3(F.relu(self.fc_dist_2(F.relu(self.fc_dist_1(hidden_enc))))))*100
        if self.with_speed:
            if self.batch_norm:
                speed_pred = self.fc_speed_2(F.relu(self.fc_speed_bn(self.fc_speed_1(hidden_enc))))
            else:
                speed_pred = self.fc_speed_2(F.relu(self.fc_speed_1(hidden_enc)))
        else:
            speed_pred = None
        if self.with_posinfo:
            if self.batch_norm:
                posxyz_pred = self.fc_posxyz_2(F.relu(self.fc_posxyz_bn(self.fc_posxyz_1(hidden_enc))))
            else:
                posxyz_pred = self.fc_posxyz_2(F.relu(self.fc_posxyz_1(hidden_enc)))
        else:
            posxyz_pred = None
        return coll_prob, nx_feature_enc, offroad_prob, speed_pred, dist, pos_pred, posxyz_pred, hidden, cell

class ConvLSTMMulti(nn.Module):
    def __init__(self, inc, outc, na, pretrain = True, with_lstm = True, multi_info = False, with_posinfo = False, use_pos_class = False, with_speed = False, with_pos = False, frame_history_len = 4, with_dla=False):
        super(ConvLSTMMulti, self).__init__()
        self.conv_lstm = ConvLSTMNet(inc, outc, na, pretrain = pretrain, with_lstm = with_lstm, multi_info = multi_info, with_posinfo = with_posinfo, use_pos_class = use_pos_class, with_speed = with_speed, with_pos = with_pos, frame_history_len = frame_history_len, with_dla=with_dla)
        self.with_posinfo = with_posinfo
        self.use_pos_class = use_pos_class
        self.with_speed = with_speed
        self.with_pos = with_pos
        self.frame_history_len = frame_history_len
        self.num_actions = na
        self.actions = []
        self.cnt = 0

    def get_feature(self, x):
        feat = []
        x = x.contiguous()
        _, num_time, _, _, _ = int(x.size()[0]), int(x.size()[1]), int(x.size()[2]), int(x.size()[3]), int(x.size()[4])
        for i in range(num_time):
            feat.append(self.conv_lstm.get_feature(x[:, i, :, :, :].squeeze(1)))
        return torch.stack(feat, dim = 1)

    def get_action_loss(self, imgs, actions, speed = None, pos = None, num_time = 3, hidden = None, cell = None, posxyz = None, gpu=0):
        batch_size = int(imgs.size()[0])
        target_coll_np = np.zeros((batch_size, num_time, 2))
        target_coll_np[:,:,0] = 1.0
        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        target_coll = Variable(torch.from_numpy(target_coll_np).float()).cuda(gpu)
        target_off = Variable(torch.from_numpy(target_coll_np).float()).cuda(gpu)
        weight = []
        for i in range(num_time):
            weight.append(0.97**i)
        weight = Variable(torch.from_numpy(np.array(weight).reshape((1, num_time, 1))).float().cuda(gpu)).repeat(batch_size, 1, 1)
        outs = self.forward(imgs, actions, speed, pos, num_time=num_time, hidden=hidden, cell=cell, posxyz=posxyz)
        coll_ls = nn.CrossEntropyLoss(reduce=False)(outs[0].view(-1,2), torch.max(target_coll.view(-1,2),-1)[1])
        off_ls = nn.CrossEntropyLoss(reduce=False)(outs[2].view(-1,2), torch.max(target_off.view(-1,2),-1)[1])
        coll_ls = (coll_ls.view(-1,num_time,1)*weight).view(-1,num_time).sum(-1)
        off_ls = (off_ls.view(-1,num_time,1)*weight).view(-1,num_time).sum(-1)
        dist_ls = (outs[4].view(-1,num_time,1)*weight).view(-1,num_time).sum(-1)
        return coll_ls.data.cpu().numpy().reshape((-1)), off_ls.data.cpu().numpy().reshape((-1)), dist_ls.data.cpu().numpy().reshape((-1)),\
            outs[0][:,:,0].data.cpu().numpy(), outs[2][:,:,0].data.cpu().numpy(), outs[4][:,:,0].data.cpu().numpy()           
     
    def sample_action(self, imgs, prev_action, speed=None, pos=None, num_time=3, num_actions=9, hidden=None, cell=None, calculate_loss=False, posxyz=None, batch_step=200, hand=True, gpu=2):
        use_cuda = torch.cuda.is_available()
        imgs = imgs.contiguous()
        if self.with_speed:
            speed = speed.contiguous()
            speed = speed.view(-1, 1, 2) 
        batch_size, c, w, h = int(imgs.size()[0]), int(imgs.size()[-3]), int(imgs.size()[-2]), int(imgs.size()[-1])
        imgs = imgs.view(batch_size, 1, c, w, h)
        pos = pos.view(batch_size, 1, -1)
        if self.with_posinfo:
            posxyz = posxyz.view(batch_size, 1, 3)
        if use_cuda:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        if calculate_loss:
            this_action = Variable(torch.randn(1, num_time, self.num_actions), requires_grad=False)
            this_action = self.quantize_action(this_action, batch_size, num_time, self.num_actions, requires_grad=False, prev_action=prev_action)
            coll_ls, off_ls, dist_ls, coll_prob, off_prob, distance = self.get_action_loss(imgs, this_action, speed, pos, num_time, hidden, cell, posxyz)
            return coll_prob, off_prob, distance, coll_ls, off_ls, dist_ls
        elif hand == True:
            all_colls = []
            all_offs = []
            all_off_probs = []
            all_coll_probs = []
            all_dists = []
            for prev_act in range(self.num_actions):
                this_action = Variable(torch.randn(1, num_time, self.num_actions), requires_grad=False)
                this_action = self.quantize_action(this_action, batch_size, num_time, self.num_actions, requires_grad=False, prev_action=prev_act)
                coll_ls, off_ls, dist_ls, coll_prob, off_prob, distance = self.get_action_loss(imgs, this_action, speed, pos, num_time, hidden, cell, posxyz)
                all_colls.append(np.mean(coll_ls))
                all_offs.append(np.mean(off_ls))
                all_off_probs.append(np.mean(off_prob))
                all_dists.append(np.mean(dist_ls))
                all_coll_probs.append(np.mean(coll_prob))
            off_rank = np.array(all_offs).argsort().argsort()
            coll_rank = np.array(all_colls).argsort().argsort()
            dist_rank = self.num_actions-1-np.array(all_dists).argsort().argsort()
            all_rank = (off_rank + dist_rank + coll_rank).argsort().argsort()
            if np.max(all_off_probs) < 0.7 and np.max(all_coll_probs) >= 0.7:
                action = np.argmin(off_rank)
            elif np.max(all_coll_probs) < 0.7 and np.max(all_off_probs) >= 0.7:
                action = np.argmin(coll_rank)
            elif np.max(all_off_probs) < 0.7 and np.max(all_coll_probs) < 0.7:
                action = np.argmin(np.array(all_colls)+np.array(all_offs))
            else:
                action = np.argmin(dist_rank)
            if prev_action == 5 or prev_action == 2:
                action = 4
            elif prev_action == 0 or prev_action == 3:
                action = 4
            if np.max(all_off_probs) <=0.5:
                curr_pos = float(pos[0,0,0].data.cpu().numpy()) 
                curr_sp = float(speed[0,0].data.cpu().numpy())
                if curr_pos <= -4:
                    if curr_sp <= 3.0:
                        action = 0
                    else:
                        action = 3
                elif curr_pos <= -1 and curr_pos > -4:
                    if prev_action != 3 and prev_action!=0:
                        if curr_sp <= 3.0:
                            action = 0
                        else:
                            action = 3
                    else:
                        action = 4
                elif curr_pos >=3 and curr_pos < 6:
                    if prev_action != 2 and prev_action!=5:
                        if curr_sp <= 3.0:
                            action = 2
                        else:
                            action = 5
                    else:
                        action = 4
                elif curr_pos >= 6:
                    if curr_sp <= 3.0:
                        action = 2
                    else:
                        action = 5
                else:
                    if curr_sp <= 10:
                        action = 1
                    else:
                        action = 4
                print('hand select')
            return action, None, None
        else: # sample action
            if num_time == 15:
                all_actions,_ = get_act_samps(num_time, self.num_actions, prev_action, 1000)
            else:
                all_actions = get_action_sample(num_time, 3, self.num_actions)
            num_choice = all_actions.shape[0]
            total_ls = 100000000
            which_action = -1
            all_off_loss, all_coll_loss, all_dist_loss = [], [], []
            off_ls_act = np.ones(self.num_actions)*100000
            coll_ls_act = np.ones(self.num_actions)*100000
            dist_ls_act = np.ones(self.num_actions)*-10000
            off_probs = np.zeros(self.num_actions)
            all_actions = torch.from_numpy(all_actions).float().cuda(gpu)
            for ii in range(int(num_choice/batch_step)):
                this_action = Variable(all_actions[ii*batch_step:min((ii+1)*batch_step, num_choice),:,:])
                this_imgs = imgs.repeat(int(this_action.size()[0]), 1,1,1,1)
                if self.with_speed:
                    this_sp = speed.repeat(int(this_action.size()[0]), 1, 1)
                else:
                    this_sp = None
                this_pos = pos.repeat(int(this_action.size()[0]), 1, 1)
                coll_ls, off_ls, dist_ls, coll_prob, off_prob, distance = self.get_action_loss(this_imgs, this_action, this_sp, this_pos, num_time, hidden, cell, posxyz)
                batch_ls = coll_ls + off_ls -0.1*dist_ls
                idx = np.argmin(batch_ls)
                this_act_simp = torch.max(this_action,-1)[1].data.cpu().numpy()[:,0]
                this_loss = batch_ls[idx]
                if this_loss < total_ls or ii == 0:
                    poss_action = np.argmax(this_action.data.cpu().numpy()[idx,0,:].squeeze())
                    if (float(coll_prob[idx, 0]) > 0.4 and float(off_prob[idx, 0]) > 0.4) or ii==0: 
                        total_ls = this_loss
                        which_action = poss_action
            return which_action, None, None

    def quantize_action(self, action, batch_size, num_time, num_actions, requires_grad=False, prev_action=None):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
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
        action_v = Variable(torch.from_numpy(act_np).type(dtype), requires_grad=requires_grad)
        return action_v

    def forward(self, imgs, actions=None, speed=None, pos=None, num_time=None, hidden=None, cell=None, posxyz=None, get_feature=False):
        if get_feature:
            res = self.get_feature(imgs)
            return res
        batch_size, num_step, c, w, h = int(imgs.size()[0]), int(imgs.size()[1]), int(imgs.size()[-3]), int(imgs.size()[-2]), int(imgs.size()[-1])
        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        if self.use_pos_class:
            this_pos = pos[:,0,:].squeeze(1).view(-1,19)
        else:
            this_pos = pos[:,0,:].squeeze(1).view(-1,1)
        if self.with_speed:
            this_speed = speed[:,0,:].squeeze(1)
        else:
            this_speed = None
        if self.with_posinfo:
            this_posxyz = posxyz[:,0,:].squeeze(1)
        else:
            this_posxyz = None
        coll, pred, offroad, speed_pred, dist, pos_pred, posxyz_pred, hidden, cell = self.conv_lstm(imgs[:,0,:,:,:].squeeze(1), actions[:,0,:].squeeze(1),this_speed, this_pos, this_posxyz, hidden=hidden, cell=cell) 
        num_act = self.num_actions 
        coll_list = [coll]
        pred_list = [pred]
        offroad_list = [offroad]
        speed_list = [speed_pred]
        dist_list = [dist]
        pos_list = [pos_pred]
        posxyz_list = [posxyz_pred]
        for i in range(1, num_time):
            if self.use_pos_class and self.with_pos:
                mask = Variable(torch.zeros(batch_size, 19), requires_grad=False).type(dtype)
                mask[torch.arange(batch_size).type(dtype).long(), torch.max(pos_pred,-1)[1].long()] = 1.0
                pos_pred2 = torch.abs(mask * pos_pred)
                pos_pred2[torch.arange(batch_size).type(dtype).long(), torch.max(pos_pred,-1)[1].long()] = 1.0
            else:
                pos_pred2 = pos_pred
            coll, pred, offroad, speed_pred, dist, pos_pred, posxyz_pred, hidden, cell = self.conv_lstm(pred, actions[:,i,:].squeeze(1), speed_pred, pos_pred2, posxyz_pred, with_encode=True, hidden=hidden, cell=cell)
            coll_list.append(coll)
            pred_list.append(pred)
            offroad_list.append(offroad)
            speed_list.append(speed_pred)
            dist_list.append(dist)
            pos_list.append(pos_pred)
            posxyz_list.append(posxyz_pred)
        if self.with_speed == False:
            speed_out = None
        else:
            speed_out = torch.stack(speed_list, dim=1)
        if self.with_posinfo == False:
            posxyz_out = None
        else:
            posxyz_out = torch.stack(posxyz_list, dim=1)
        if self.with_pos:
            pos_out = torch.stack(pos_list,dim=1)
        else:
            pos_out = None
        return torch.stack(coll_list, dim=1), torch.stack(pred_list, dim=1), torch.stack(offroad_list,dim=1), \
            speed_out, torch.stack(dist_list, dim=1), pos_out, posxyz_out, hidden, cell
