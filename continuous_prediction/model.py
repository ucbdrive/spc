import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import from_variable_to_numpy

from models.DRNSeg import DRNSeg
from models.UP_Samper import UP_Samper
from models.PRED import PRED
from models.FURTHER import FURTHER
from models.RNN import RNN
from models.FURTHER_continuous import FURTHER_continuous
from models.DQN import DQN
import models.dla as dla

class hybrid_net(nn.Module):
    def __init__(self, args):
        super(hybrid_net, self).__init__()
        self.args = args

        if self.args.continuous:
            self.action_encoder = nn.Linear(2, args.info_dim)
            if args.use_dqn:
                self.dqn = DQN(args)

        if args.use_seg:
            self.feature_encoder = DRNSeg('drn_d_22', args.semantic_classes)
            self.up_sampler = UP_Samper()
            self.predictor = PRED(args.semantic_classes, args.num_actions)
            self.further = FURTHER()
        else:
            self.dla = dla.dla46x_c(pretrained = True)
            self.feature_encoder = nn.Linear(256 * args.frame_len, args.hidden_dim)
            self.predictor = RNN(args)
            self.further = FURTHER_continuous(args)


    def forward(self, obs, action, hx = None, cx = None):
        if self.args.continuous:
            action = self.action_encoder(action)
        else:
            action = one_hot(self.args, action)

        coll_list, off_list, dist_list = []
        if self.args.use_xyz:
            xyz_list = []

        if self.args.use_seg:
            feature_map_list, segmentation_list = [], []
            feature_map = self.feature_encoder(obs)
            for i in range(self.args.num_steps):
                feature_map = self.predictor(feature_map, action[i])
                segmentation = self.up_sampler(feature_map)
                output_dict = self.further(feature_map)
                feature_map_list.append(feature_map)
                segmentation_list.append(segmentation)
                coll_list.append(output_dict['collison'])
                off_list.append(output_dict['offroad'])
                dist_list.append(output_dict['distance'])
                if self.args.use_xyz:
                    xyz_list.append(output_dict['xyz'])
            return feature_map, segmentation, prediction, self.further(feature_map)
        else:
            hx, cx = Variable(torch.zeros(self.args.batch_size, self.args.hidden_dim)), Variable(torch.zeros(self.args.batch_size, self.args.hidden_dim))

            for i in range(self.args.num_steps):
                feature = self.feature_encoder(self.dla(obs)) if i == 0 else hx
                hx, cx = self.predictor(feature, action, hx, cx)
                output_dict = self.further(hx, action)
                coll_list.append(output_dict['collison'])
                off_list.append(output_dict['offroad'])
                dist_list.append(output_dict['distance'])
                if self.args.use_xyz:
                    xyz_list.append(output_dict['xyz'])

        output_dict = {'output_coll': torch.stack(coll_list, dim = 0),
                       'output_off': torch.stack(off_list, dim = 0),
                       'output_dist': torch.stack(dist_list, dim = 0)}
        if self.args.use_xyz:
            output_dict['output_xyz'] = torch.stack(xyz_list, dim = 0)

        return output_dist

if __name__ == '__main__':
    dqn = DQN().cuda()
    obs = Variable(torch.zeros(16, 3, 256, 256)).cuda()
    act = Variable(torch.zeros(16, 1)).type(torch.LongTensor).cuda()
    x = dqn(obs).detach().max(1)[0]
    print(x.size())
    # dla = dla.dla46x_c(pretrained = True).cuda()
    # obs = Variable(torch.zeros(16, 3, 256, 256)).cuda()
    # res = dla(obs)
    # print(res.size())