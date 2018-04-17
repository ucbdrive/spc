import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import dla
import pdb
import numpy as np
import random
import pickle as pkl
import torch.optim as optim

class ConvLSTMCell(nn.Module):
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.bias        = bias
        self.fc1        = nn.Linear(input_dim+hidden_dim, 2 * hidden_dim, bias=self.bias)
        self.fc2        = nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=self.bias)
        self.fc3        = nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=self.bias)
        self.fc4        = nn.Linear(2 * hidden_dim, 2 * hidden_dim, bias=self.bias)
        self.W            = nn.Linear(2 * hidden_dim, 4 * hidden_dim, bias=self.bias)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1) 
        combined_conv = self.W(F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(combined)))))))))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
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
    def __init__(self, in_channel=3, out_channel=3, num_actions=3,
                pretrain=True, # use pretrained dla model 
                with_lstm=True, # with lstm
                multi_info=False, # multi features with input image feature
                with_posinfo=False, # with xyz position information
                use_pos_class=False, # convert regression problem into classification problem
                with_speed=False,
                with_pos=False,
                frame_history_len=4,
                freeze_dla=False,
                hidden_dim=512): # using speed as input
        super(ConvLSTMNet, self).__init__()
        self.dla = dla.dla46x_c(pretrained=pretrain).cuda()
        if freeze_dla:
            for param in self.dla.parameters():
                param.requires_grad = False
        self.with_lstm = with_lstm
        self.multi_info = multi_info
        self.use_pos_class = use_pos_class
        self.with_speed = with_speed
        self.hidden_dim = hidden_dim
        self.feature_encode = nn.Linear(256*frame_history_len, self.hidden_dim)
        self.outfeature_encode = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.num_actions = num_actions
        self.with_posinfo = with_posinfo
        self.with_pos = with_pos
        self.info_dim = 0
        self.frame_history_len = frame_history_len
        self.act_encode = nn.Linear(num_actions, 16) # action encoding
        self.act_bn = nn.BatchNorm1d(16)
        self.info_dim += 16
        if use_pos_class and with_pos:
            self.pos_encode = nn.Linear(19, 32)
            self.pos_bn = nn.BatchNorm1d(32)
            self.info_dim += 32
        else:
            if with_pos:
                self.pos_encode = nn.Linear(1, 16)
                self.pos_bn = nn.BatchNorm1d(16)
                self.info_dim += 16
        if with_posinfo:
            self.posxyz_encode = nn.Linear(3, 32)
            self.posxyz_bn = nn.BatchNorm1d(32)
            self.info_dim += 32
        if with_speed:
            self.speed_encode = nn.Linear(2, 16)
            self.speed_bn = nn.BatchNorm1d(16)
            self.info_dim += 16
        if multi_info:
            self.info_encode = nn.Linear(self.info_dim, self.hidden_dim)
        else:
            self.info_encode = nn.Linear(self.info_dim+self.hidden_dim, self.hidden_dim)
        self.info_bn = nn.BatchNorm1d(self.hidden_dim)
        if with_lstm:
            self.lstm = ConvLSTMCell(self.hidden_dim, self.hidden_dim, True)
        self.fc_coll_1 = nn.Linear(self.hidden_dim+self.info_dim, 128)
        self.fc_coll_bn1 = nn.BatchNorm1d(128)
        self.fc_coll_2 = nn.Linear(128, 32)
        self.fc_coll_bn2 = nn.BatchNorm1d(32)
        self.fc_coll_3 = nn.Linear(32, 2)
        self.fc_off_1 = nn.Linear(self.hidden_dim+self.info_dim, 128)
        self.fc_off_bn1 = nn.BatchNorm1d(128)
        self.fc_off_2 = nn.Linear(128, 32)
        self.fc_off_bn2 = nn.BatchNorm1d(32)
        self.fc_off_3 = nn.Linear(32, 2)
        if with_pos:
            self.fc_pos_1 = nn.Linear(self.hidden_dim+self.info_dim, 32)
            self.fc_pos_bn = nn.BatchNorm1d(32)
            self.fc_pos_tanh = nn.Tanh()
        if use_pos_class and with_pos:
            self.fc_pos_2 = nn.Linear(32, 19)
        else:
            if with_pos:
                self.fc_pos_2 = nn.Linear(32, 1)
        self.fc_dist_1 = nn.Linear(self.hidden_dim+self.info_dim, 128)
        self.fc_dist_bn1 = nn.BatchNorm1d(128)
        self.fc_dist_2 = nn.Linear(128, 32)
        self.fc_dist_bn2 = nn.BatchNorm1d(32)
        self.fc_dist_3 = nn.Linear(32, 1)
        if with_speed:
            self.fc_speed_1 = nn.Linear(self.hidden_dim+self.info_dim, 32)
            self.fc_speed_bn = nn.BatchNorm1d(32)
            self.fc_speed_2 = nn.Linear(self.hidden_dim+self.info_dim, 2)
        if with_posinfo:
            self.fc_posxyz_1 = nn.Linear(self.hidden_dim+self.info_dim, 32)
            self.fc_posxyz_bn = nn.BatchNorm1d(32)
            self.fc_posxyz_2 = nn.Linear(32, 3)
 
    def get_feature(self, x):
        res = []
        for i in range(self.frame_history_len):
            res.append(self.dla(x[:,i*3:(i+1)*3,:,:]))
        res = torch.cat(res, dim=1)
        res = self.feature_encode(res)
        return res # batch * 128

    def forward(self, x, action, speed=None, pos=None, posxyz=None, with_encode=False, hidden=None, cell=None):
        if with_encode == False:
            x = self.get_feature(x) # batch * 128
        if self.with_lstm:
            if hidden is None or cell is None:
                hidden, cell = x, x # batch * 128
        action_enc = F.relu(self.act_encode(action)) # batch * 64
        if self.with_pos:
            pos_enc = F.relu(self.pos_bn(self.pos_encode(pos)))
            info_enc = torch.cat([action_enc, pos_enc], dim=1)
        else:
            info_enc = action_enc
        if self.with_speed:
            speed_enc = F.relu(self.speed_bn(self.speed_encode(speed))) # batch * 16
            info_enc = torch.cat([info_enc, speed_enc], dim=1)
        if self.with_posinfo:
            posxyz_enc = F.relu(self.posxyz_bn(self.posxyz_encode(posxyz)))
            info_enc = torch.cat([info_enc, posxyz_enc], dim=1)
        if self.multi_info == True:
            encode = F.relu(self.info_bn(self.info_encode(info_enc)))
            encode = F.relu(encode * x)  # batch * 256
        else:
            encode = torch.cat([x, info_enc], dim=1) # batch * 256
            encode = F.relu(self.info_bn(self.info_encode(encode)))
        if self.with_lstm:
            hidden, cell = self.lstm(encode, [hidden, cell])
            pred_encode_nx = hidden.view(-1, self.hidden_dim)
            nx_feature_enc = self.outfeature_encode(F.relu(pred_encode_nx))
        else:
            pred_encode_nx = encode # batch * 256
            nx_feature_enc = self.outfeature_encode(F.relu(pred_encode_nx))
        hidden_enc = torch.cat([pred_encode_nx, info_enc], dim=1)

        # outputs
        coll_prob = nn.Softmax(dim=-1)(self.fc_coll_3(F.relu(self.fc_coll_bn2(self.fc_coll_2(F.relu(self.fc_coll_bn1(self.fc_coll_1(hidden_enc))))))))
        if self.with_pos:
            pos_pred = 20 * self.fc_pos_tanh(self.fc_pos_2(F.relu(self.fc_pos_bn(self.fc_pos_1(hidden_enc)))))
        if self.use_pos_class and self.with_pos:
            pos_pred = nn.Softmax(dim=-1)(pos_pred)
        if self.with_pos == False:
            pos_pred = None
        offroad_prob = nn.Softmax(dim=-1)(self.fc_off_3(F.relu(self.fc_off_bn2(self.fc_off_2(F.relu(self.fc_off_bn1(self.fc_off_1(hidden_enc))))))))
        dist = self.fc_dist_3(F.relu(self.fc_dist_bn2(self.fc_dist_2(F.relu(self.fc_dist_bn1(self.fc_dist_1(hidden_enc)))))))
        if self.with_speed:
            speed_pred = self.fc_speed_2(F.relu(self.fc_speed_bn(self.fc_speed_1(hidden_enc))))
        else:
            speed_pred = None
        if self.with_posinfo:
            posxyz_pred = self.fc_posxyz_2(F.relu(self.fc_posxyz_bn(self.fc_posxyz_1(hidden_enc))))
        else:
            posxyz_pred = None
        return coll_prob, nx_feature_enc, offroad_prob, speed_pred, dist, pos_pred, posxyz_pred, hidden, cell

class ConvLSTMMulti(nn.Module):
    def __init__(self, inc, outc, na=3, pretrain=True, with_lstm=True, multi_info=False, with_posinfo=False, use_pos_class=False, with_speed=False, with_pos=False, frame_history_len=3):
        super(ConvLSTMMulti, self).__init__()
        self.conv_lstm = ConvLSTMNet(inc, outc, na, pretrain=pretrain, with_lstm=with_lstm, multi_info=multi_info, with_posinfo=with_posinfo, use_pos_class=use_pos_class, with_speed=with_speed, with_pos=with_pos, frame_history_len=frame_history_len)
        self.with_posinfo = with_posinfo
        self.use_pos_class = use_pos_class
        self.with_speed = with_speed
        self.with_pos = with_pos
        self.frame_history_len = frame_history_len
        self.num_actions = na

    def get_feature(self, x):
        feat = []
        x = x.contiguous()
        _,num_time,_,_,_ = int(x.size()[0]), int(x.size()[1]), int(x.size()[2]), int(x.size()[3]), int(x.size()[4])
        for i in range(num_time):
            feat.append(self.conv_lstm.get_feature(x[:,i,:,:,:].squeeze(1)))
        return torch.stack(feat, dim=1)          

    def forward(self, imgs, actions=None, speed=None, pos=None, num_time=None, hidden=None, cell=None, posxyz=None, get_feature=False):
        # batch * time * c * w * h
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
