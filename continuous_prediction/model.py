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

        if self.args.use_seg:
            feature_map = self.feature_encoder(obs)
            segmentation = self.up_sampler(feature_map)
            prediction = self.predictor(feature_map, action)
            return feature_map, segmentation, prediction, self.further(feature_map)
        else:
            feature = self.feature_encoder(self.dla(obs))
            hx, cx = self.predictor(feature, action, hx, cx)
            return hx, cx, self.further(hx, action)

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