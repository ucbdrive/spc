import torch
import torch.nn as nn
from torch import optim
import numpy as np
import os
from models.DLASeg import DLASeg
from models.convLSTM import convLSTM
from models.end_layer import end_layer
from utils import PiecewiseSchedule, tile, tile_first, load_model


class ConvLSTMNet(nn.Module):
    def __init__(self, args):
        super(ConvLSTMNet, self).__init__()
        self.args = args

        # Feature extraction and prediction
        self.dlaseg = DLASeg(args, down_ratio=4)
        self.feature_map_predictor = convLSTM()

        # Information prediction
        self.guide_layer = end_layer(args, args.classes, int(np.prod(args.bin_divide)))
        self.coll_layer = end_layer(args, args.classes, 2)
        self.off_layer = end_layer(args, args.classes, 2)
        self.speed_layer = end_layer(args, args.classes*args.frame_history_len, 1)

    def forward(self, x, action, with_encode=False, hidden=None, cell=None, training=True, action_var=None):
        output_dict = dict()
        if not with_encode:
            x, hidden, output_dict['seg_current'] = self.get_feature(x)
            if torch.cuda.is_available():
                action_var = action_var.cuda()
            x = tile_first(x, action_var)

        x[-1] = tile(x[-1], action)
        hx = self.feature_map_predictor(x)
        rx, output_dict['seg_pred'] = self.dlaseg.infer(hx)
        nx_feature_enc = x[1:] + [hx]
        hidden = torch.cat([hidden[:, self.args.classes:, :, :], rx], dim=1)

        output_dict['coll_prob'] = self.coll_layer(rx.detach())
        output_dict['offroad_prob'] = self.off_layer(rx.detach())
        output_dict['speed'] = self.speed_layer(hidden.detach())

        return output_dict, nx_feature_enc, hidden, None

    def get_feature(self, x, train=True):
        batch_size, frame_history_len, height, width = x.size()
        frame_history_len = int(frame_history_len / 3)
        res = []
        hidden = []

        if train:
            for i in range(frame_history_len):
                xx, rx, y = self.dlaseg(x[:, i*3:(i+1)*3, :, :])
                res.append(xx)
                hidden.append(rx)
            hidden = torch.cat(hidden, dim=1)
            return res, hidden, y
        else:
            for i in range(frame_history_len):
                xx = self.dlaseg(x[:, i*3:(i+1)*3, :, :], train=train)
                res.append(xx)
            return res


class ConvLSTMMulti(nn.Module):
    def __init__(self, args):
        super(ConvLSTMMulti, self).__init__()
        self.args = args
        self.conv_lstm = ConvLSTMNet(self.args)

    def get_feature(self, x, next_obs=False):
        x = x.contiguous()
        if next_obs:
            batch_size, pred_step, frame_history_len, frame_height, frame_width = x.size()
            frame_history_len = int(frame_history_len / 3)
            x = x.view(batch_size * pred_step, 3 * frame_history_len, frame_height, frame_width)
            x = self.conv_lstm.get_feature(x)[1]
            x = x.contiguous().view(batch_size, pred_step, self.args.classes * frame_history_len, frame_height, frame_width)
            return x
        else:
            xx, x, y = self.conv_lstm.get_feature(x[:, 0, :, :, :])
            return xx, x, y

    def guide_action(self, x):
        _, enc, seg = self.conv_lstm.dlaseg(x)
        logit = self.conv_lstm.guide_layer(enc.detach())
        return logit

    def extract_feature(self, x):
        res, hidden, _ = self.conv_lstm.get_feature(x)
        return res, hidden

    def get_p(self, x):
        x, _ = self.conv_lstm.dlaseg.infer(x)
        return self.conv_lstm.guide_layer(x)

    def one_step(self, x, action, hidden):
        output_dict, nx_feature_enc, hidden, _ = self.conv_lstm(x, action, with_encode=True, hidden=hidden)
        return output_dict, nx_feature_enc, hidden

    def forward(self, imgs, actions=None, hidden=None, cell=None, get_feature=False, training=True, function='', action_var=None, next_obs=False):
        if function == 'guide_action':
            return self.guide_action(imgs)
        elif function == 'extract_feature':
            return self.extract_feature(imgs)
        elif function == 'get_p':
            return self.get_p(imgs)
        elif function == 'one_step':
            return self.one_step(imgs, actions, hidden)

        if get_feature:
            return self.get_feature(imgs, next_obs=next_obs)
        batch_size, num_step, c, w, h = int(imgs.size()[0]), int(imgs.size()[1]), int(imgs.size()[-3]), int(imgs.size()[-2]), int(imgs.size()[-1])
        output_dict, pred, hidden, cell = self.conv_lstm(imgs[:, 0, :, :, :].squeeze(1), actions[:, 0, :].squeeze(1), hidden=hidden, cell=cell, training=training, action_var=action_var)

        # create dictionary to store outputs
        final_dict = dict()
        for key in output_dict.keys():
            final_dict[key] = [output_dict[key]]
        final_dict['seg_pred'] = [output_dict['seg_current'], output_dict['seg_pred']]

        for i in range(1, self.args.pred_step):
            output_dict, pred, hidden, cell = self.conv_lstm(pred, actions[:, i, :], with_encode=True, hidden=hidden, cell=cell, training=training, action_var=None)
            for key in output_dict.keys():
                final_dict[key].append(output_dict[key])

        for key in final_dict.keys():
            final_dict[key] = torch.stack(final_dict[key], dim=1)

        return final_dict


def init_models(args):
    train_net = ConvLSTMMulti(args)
    for param in train_net.parameters():
        param.requires_grad = True
    train_net.train()

    net = ConvLSTMMulti(args)
    for param in net.parameters():
        param.requires_grad = False
    net.eval()

    train_net, epoch = load_model(args.save_path, train_net, resume=args.resume)
    net.load_state_dict(train_net.state_dict())

    if torch.cuda.is_available():
        train_net = train_net.cuda()
        net = net.cuda()
        if args.data_parallel:
            train_net = torch.nn.DataParallel(train_net)
            net = torch.nn.DataParallel(net)
    optimizer = optim.Adam(train_net.parameters(), lr=args.lr, amsgrad=True)

    exploration = PiecewiseSchedule([
            (0, 1.0),
            (args.epsilon_frames, 0.02),
        ], outside_value=0.02
    )

    if args.resume:
        try:
            num_imgs_start = max(int(open(os.path.join(args.save_path, 'log_train_torcs.txt')).readlines()[-1].split(' ')[1]) - 1000, 0)
        except:
            num_imgs_start = 0
    else:
        num_imgs_start = 0

    return train_net, net, optimizer, epoch, exploration, num_imgs_start
