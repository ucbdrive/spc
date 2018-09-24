''' Model Definition '''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import drn
import dla
import dla_up
import numpy as np
import math
import pdb
from end_layer import end_layer
from conv_lstm import convLSTM
from fcn import fcn
from utils import weights_init, tile, tile_first
from convLSTM2 import convLSTM2


class atari_model(nn.Module):
    def __init__(self, inc=12, num_actions=9, frame_history_len=4):
        super(atari_model, self).__init__()
        self.conv1 = nn.Conv2d(inc, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        self.frame_history_len = frame_history_len

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
        self.fc1 = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=self.bias)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=self.bias)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim, bias = self.bias)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim, bias = self.bias)
        self.W = nn.Linear(hidden_dim, 4 * hidden_dim, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        combined_conv = F.relu(self.W(x))
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next  # h_next is the output

    def init_hidden(self, batch_size):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            return (Variable(torch.zeros(batch_size, self.hidden_dim)).cuda(),
                    Variable(torch.zeros(batch_size, self.hidden_dim)).cuda())
        else:
            return (Variable(torch.zeros(batch_size, self.hidden_dim)),
                    Variable(torch.zeros(batch_size, self.hidden_dim)))


class DRNSeg(nn.Module):
    ''' Network For Feature Extraction for Segmentation Prediction '''
    def __init__(self, args):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(args.drn_model)(pretrained=args.pretrained, num_classes=1000)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.seg = nn.Conv2d(model.out_dim, args.classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.model(x)
        feature_map = self.seg(x)
        return feature_map  # size: batch_size x classes x 32 x 32


class ConvLSTMNet(nn.Module):
    def __init__(self, args):
        super(ConvLSTMNet, self).__init__()

        # feature extraction part
        if args.use_seg:
            if not args.one_hot:
                args.classes = 1

            if args.use_lstm:
                self.action_encode = nn.Linear(args.num_total_act, 1*32*32)
                self.action_up1 = nn.ConvTranspose2d(1, 2, 4, stride=4)
                self.action_up2 = nn.ConvTranspose2d(2, 3, 2, stride=2)
                self.feature_map_predictor = convLSTM(3, args.classes * args.frame_history_len, args.classes)

            elif args.lstm2:
                self.feature_map_predictor = convLSTM2(args.classes + args.num_total_act, args.frame_history_len, args.classes)

            else:
                self.actionEncoder = nn.Linear(args.num_total_act, 32)
                self.feature_map_predictor = fcn(args.classes * args.frame_history_len, args.classes)

        else:
            self.action_encode = nn.Linear(args.num_total_act, args.info_dim)
            self.info_encode = nn.Linear(args.info_dim + args.hidden_dim, args.hidden_dim)
            self.dla = dla.dla46x_c(pretrained = args.pretrained)
            self.feature_encode = nn.Linear(256 * args.frame_history_len, args.hidden_dim)
            self.lstm = ConvLSTMCell(args.hidden_dim, args.hidden_dim, True)
            self.outfeature_encode = nn.Linear(args.hidden_dim + args.info_dim, args.hidden_dim)
            self.hidden_encode = nn.Linear(args.hidden_dim + args.hidden_dim, args.hidden_dim)
            self.cell_encode = nn.Linear(args.hidden_dim + args.hidden_dim, args.hidden_dim)

        # output layers
        if args.one_hot:
            self.coll_layer = end_layer(args, args.classes, 2)
            self.off_layer = end_layer(args, args.classes, 2)  # if 'torcs' in args.env else 1)
            self.dist_layer = end_layer(args, args.classes * args.frame_history_len, 1)
        else:
            self.coll_layer = end_layer(args, 1, 2)
            self.off_layer = end_layer(args, 1, 2)
            self.dist_layer = end_layer(args, args.frame_history_len, 1)

        # optional layers
        if args.use_otherlane:
            self.otherlane_layer = end_layer(args, args.classes, 1)
        if args.use_pos:
            self.pos_layer = end_layer(args, args.classes, 1)
        if args.use_angle:
            self.angle_layer = end_layer(args, args.classes, 1)
        if args.use_speed:
            self.speed_layer = end_layer(args, args.classes * args.frame_history_len, 1)
        if args.use_xyz:
            self.xyz_layer = end_layer(args, args.classes, 3)

        self.args = args

    def encode_action(self, action):
        if self.args.use_seg:
            if self.args.use_lstm:
                action_enc = F.sigmoid(self.action_encode(action).view(-1, 1, 32, 32))
                action_enc = self.action_up2(F.sigmoid(self.action_up1(action_enc)))
            else:
                action_enc = F.sigmoid(self.actionEncoder(action))
        else:
            action_enc = F.relu(self.action_encode(action))
        return action_enc

    def forward(self, x=None, action=None, phase=1, with_encode=False, action_var=None, softmax=None, softmax_stack=None):
        if phase == 1:
            if not with_encode and torch.cuda.is_available():
                action_var = action_var.cuda()
            xa = x if with_encode else tile_first(x, action_var, self.args.frame_history_len, self.args.classes, self.args.num_total_act)
            xa = torch.cat([xa, tile(action, self.args.num_total_act)], dim=1)
            hx = self.feature_map_predictor(xa)
            nx_feature_enc = torch.cat([xa[:, (self.args.classes + self.args.num_total_act):, :, :], hx], dim=1)
            return hx, nx_feature_enc
        else:
            output_dict = dict()
            output_dict['coll_prob'] = self.coll_layer(softmax.detach())
            output_dict['offroad_prob'] = self.off_layer(softmax.detach())
            output_dict['dist'] = self.dist_layer(softmax_stack.detach())
            return output_dict

    def get_feature(self, x):
        return x


class ConvLSTMMulti(nn.Module):
    def __init__(self, args):
        super(ConvLSTMMulti, self).__init__()
        self.args = args
        self.conv_lstm = ConvLSTMNet(self.args)

    def forward(self, imgs=None, actions=None, phase=1, action_var=None, softmax=None, softmax_stack=None):
        if phase == 1:
            feature, pred = self.conv_lstm(
                x=imgs[:, 0, :, :, :].squeeze(1),
                action=actions[:, 0, :].squeeze(1),
                phase=1,
                action_var=action_var
            )
            features = [imgs[:, 0, :, :, :], feature]

            for i in range(1, self.args.pred_step):
                feature, pred = self.conv_lstm(
                    x=pred,
                    action=actions[:, i, :].squeeze(1),
                    phase=1,
                    with_encode=True
                )
                features.append(feature)
            return torch.cat(features, dim=1)

        elif phase == 2:
            output_dict = self.conv_lstm(
                phase=2,
                softmax=softmax,
                softmax_stack=softmax_stack
            )
            return output_dict

if __name__ == '__main__':
    net = ConvLSTMMulti(num_actions=2, pretrained=True, frame_history_len=4, use_seg=True,
                        use_xyz=True, model_name='drn_d_22', num_classes=4, hidden_dim=512, info_dim=16)
    action = Variable(torch.zeros(16, 2, 2), requires_grad=False)
    img = Variable(torch.zeros(16, 2, 12, 256, 256), requires_grad=False)
    result = net(img, action)
    for i in result:
        print(i.size())
