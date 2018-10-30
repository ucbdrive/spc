''' Model Definition '''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import drn
import dla
from dla_up import DLAUp, Identity, fill_up_weights, BatchNorm
import dla_up
import numpy as np
import math
import pdb
from end_layer import end_layer2
from conv_lstm import convLSTM
from fcn import fcn
from utils import weights_init, tile, tile_first
from convLSTM3 import convLSTM3


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


class DLASeg(nn.Module):
    def __init__(self, args, down_ratio=2):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.base = dla.__dict__[args.drn_model](pretrained=args.pretrained, return_levels=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)
        self.fc = nn.Sequential(
            nn.Conv2d(channels[self.first_level], args.classes, kernel_size=1,
                      stride=1, padding=0, bias=True)
        )
        up_factor = 2 ** self.first_level
        if up_factor == 4:
            up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(args.classes, args.classes, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(args.classes, args.classes, 5, padding=2)
            )
            up.apply(weights_init)
        elif up_factor > 1:
            up = nn.ConvTranspose2d(args.classes, args.classes, up_factor * 2,
                                    stride=up_factor, padding=up_factor // 2,
                                    output_padding=0, groups=args.classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
        else:
            up = Identity()
        self.up = up
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.base(x)[self.first_level:]
        xx = x
        x = self.dla_up(x)
        x = self.fc(x)
        x = self.up(x)
        y = self.logsoftmax(x)
        x = self.softmax(x)
        return xx, x, y

    def infer(self, x):
        x = self.dla_up(x)
        x = self.fc(x)
        x = self.up(x)
        y = self.logsoftmax(x)
        x = self.softmax(x)
        return x, y

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.dla_up.parameters():
            yield param
        for param in self.fc.parameters():
            yield param


class ConvLSTMNet(nn.Module):
    def __init__(self, args):
        super(ConvLSTMNet, self).__init__()

        # feature extraction part
        if args.use_seg:
            self.drnseg = DLASeg(args, down_ratio=4)
            self.feature_map_predictor = convLSTM3()

        self.coll_layer = end_layer2(args, args.classes, 2)
        self.off_layer = end_layer2(args, args.classes, 2)  # if 'torcs' in args.env else 1)
        self.dist_layer = end_layer2(args, args.classes * args.frame_history_len, 1)

        # optional layers
        if args.use_otherlane:
            self.otherlane_layer = end_layer2(args, args.classes, 1)
        if args.use_pos:
            self.pos_layer = end_layer2(args, args.classes, 1)
        if args.use_angle:
            self.angle_layer = end_layer2(args, args.classes, 1)
        if args.use_speed:
            self.speed_layer = end_layer2(args, args.classes * args.frame_history_len, 1)
        if args.use_xyz:
            self.xyz_layer = end_layer2(args, args.classes, 3)

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

    def forward(self, x, action, with_encode=False, hidden=None, cell=None, training=True, action_var=None):
        output_dict = dict()
        if not with_encode:
            x, hidden, output_dict['seg_current'] = self.get_feature(x)
            if torch.cuda.is_available():
                action_var = action_var.cuda()
            x = tile_first(x, action_var)

        x[-1] = tile(x[-1], action)
        hx = self.feature_map_predictor(x)
        rx, output_dict['seg_pred'] = self.drnseg.infer(hx)
        nx_feature_enc = x[1:] + [hx]
        hidden = torch.cat([hidden[:, self.args.classes:, :, :], rx], dim=1)

        output_dict['coll_prob'] = self.coll_layer(rx.detach())
        output_dict['offroad_prob'] = self.off_layer(rx.detach())
        output_dict['dist'] = self.dist_layer(hidden.detach())

        # optional outputs
        if self.args.use_otherlane:
            output_dict['otherlane'] = self.pos_layer(rx.detach())
        if self.args.use_pos:
            output_dict['pos'] = self.pos_layer(rx.detach())
        if self.args.use_angle:
            output_dict['angle'] = self.angle_layer(rx.detach())
        if self.args.use_speed:
            output_dict['speed'] = self.speed_layer(hidden.detach())
        if self.args.use_xyz:
            output_dict['xyz'] = self.xyz_layer(nx_feature_enc)
        return output_dict, nx_feature_enc, hidden, None

    def get_feature(self, x):
        batch_size, frame_history_len, height, width = x.size()
        frame_history_len = int(frame_history_len / 3)
        res = []
        hidden = []

        for i in range(frame_history_len):
            xx, rx, y = self.drnseg(x[:, i*3:(i+1)*3, :, :])
            res.append(xx)
            hidden.append(rx)
        hidden = torch.cat(hidden, dim=1)
        return res, hidden, y


class ConvLSTMMulti(nn.Module):
    def __init__(self, args):
        super(ConvLSTMMulti, self).__init__()
        self.args = args
        self.conv_lstm = ConvLSTMNet(self.args)

    def get_feature(self, x):
        x = x.contiguous()
        xx, x, y = self.conv_lstm.get_feature(x[:, 0, :, :, :])

        return xx, x, y

    def predict_seg(self, x):
        _, _, seg = self.conv_lstm.drnseg(x)
        return seg

    def predict_collision(self, x):
        return self.conv_lstm.coll_layer(x)

    def predict_offroad(self, x):
        return self.conv_lstm.off_layer(x)

    def predict_distance(self, x):
        return self.conv_lstm.dist_layer(x)

    def predict_feature(self, x, action):
        batch_size = x.size(0)
        pred_steps = action.size(1)
        hidden = x
        cell = Variable(torch.zeros(batch_size, self.args.classes, 256, 256), requires_grad=False)
        if torch.cuda.is_available():
            cell = cell.cuda()
        result = []
        for step in range(pred_steps):
            action_enc = self.conv_lstm.encode_action(action[:, step, :].view(batch_size, self.args.num_total_act))
            hx, cell, y = self.conv_lstm.feature_map_predictor(action_enc, (hidden, cell))
            result.append(y)
            hidden = torch.cat([hidden[:, self.args.classes:, :, :], hx], dim=1)
        return torch.cat(result, dim=0)

    def predict_fcn(self, x, action):
        batch_size = x.size(0)
        action_encoding = self.conv_lstm.actionEncoder(action)
        x, y = self.conv_lstm.feature_map_predictor(x, action_encoding)
        return x, y

    def forward(self, imgs, actions=None, hidden=None, cell=None, get_feature=False, training=True, function='', action_var=None):
        if function == 'predict_seg':
            return self.predict_seg(imgs)
        elif function == 'predict_collision':
            return self.predict_collision(imgs)
        elif function == 'predict_offroad':
            return self.predict_offroad(imgs)
        elif function == 'predict_distance':
            return self.predict_distance(imgs)
        elif function == 'predict_feature':
            return self.predict_feature(imgs, actions)
        elif function == 'predict_fcn':
            return self.predict_fcn(imgs, actions)

        if get_feature:
            return self.get_feature(imgs)
        batch_size, num_step, c, w, h = int(imgs.size()[0]), int(imgs.size()[1]), int(imgs.size()[-3]), int(imgs.size()[-2]), int(imgs.size()[-1])
        output_dict, pred, hidden, cell = self.conv_lstm(imgs[:, 0, :, :, :].squeeze(1), actions[:, 0, :].squeeze(1), hidden=hidden, cell=cell, training=training, action_var=action_var)

        # create dictionary to store outputs
        final_dict = dict()
        for key in output_dict.keys():
            final_dict[key] = [output_dict[key]]
        if self.args.use_seg:
            final_dict['seg_pred'] = [output_dict['seg_current'], output_dict['seg_pred']]

        for i in range(1, self.args.pred_step):
            output_dict, pred, hidden, cell = self.conv_lstm(pred, actions[:, i, :], with_encode=True, hidden=hidden, cell=cell, training=training, action_var=None)
            for key in output_dict.keys():
                final_dict[key].append(output_dict[key])

        for key in final_dict.keys():
            final_dict[key] = torch.stack(final_dict[key], dim=1)

        return final_dict


if __name__ == '__main__':
    net = ConvLSTMMulti(num_actions=2, pretrained=True, frame_history_len=4, use_seg=True,
                        use_xyz=True, model_name='drn_d_22', num_classes=4, hidden_dim=512, info_dim=16)
    action = Variable(torch.zeros(16, 2, 2), requires_grad=False)
    img = Variable(torch.zeros(16, 2, 12, 256, 256), requires_grad=False)
    result = net(img, action)
    for i in result:
        print(i.size())
