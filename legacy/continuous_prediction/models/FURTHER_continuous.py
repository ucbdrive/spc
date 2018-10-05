import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import weights_init

class FURTHER_continuous(nn.Module):
    def __init__(self, args):
        super(FURTHER_continuous, self).__init__()
        self.fc_coll_1 = nn.Linear(args.hidden_dim + args.info_dim, 128)
        self.fc_coll_2 = nn.Linear(128 + args.info_dim, 32)
        self.fc_coll_3 = nn.Linear(32 + args.info_dim, 1)

        self.fc_off_1 = nn.Linear(args.hidden_dim + args.info_dim, 128)
        self.fc_off_2 = nn.Linear(128 + args.info_dim, 32)
        self.fc_off_3 = nn.Linear(32 + args.info_dim, 1)

        self.fc_dist_1 = nn.Linear(args.hidden_dim + args.info_dim, 128)
        self.fc_dist_2 = nn.Linear(128 + args.info_dim, 32)
        self.fc_dist_3 = nn.Linear(32 + args.info_dim, 1)

        if args.use_xyz:
            self.fc_xyz_1 = nn.Linear(args.hidden_dim + args.info_dim, 128)
            self.fc_xyz_2 = nn.Linear(128 + args.info_dim, 32)
            self.fc_xyz_3 = nn.Linear(32 + args.info_dim, 3)   

        self.apply(weights_init)

    def forward(self, feature, action):
        x = torch.cat([feature, action], dim = 1)

        coll_prob = F.relu(self.fc_coll_1(x))
        coll_prob = F.relu(self.fc_coll_2(torch.cat([coll_prob, action], dim = 1)))
        coll_prob = F.sigmoid(self.fc_coll_3(torch.cat([coll_prob, action], dim = 1)))

        offroad_prob = F.relu(self.fc_off_1(x))
        offroad_prob = F.relu(self.fc_off_2(torch.cat([offroad_prob, action], dim = 1)))
        offroad_prob = F.sigmoid(self.fc_off_3(torch.cat([offroad_prob, action], dim = 1)))

        dist = F.relu(self.fc_dist_1(x))
        dist = F.relu(self.fc_dist_2(torch.cat([dist, action], dim = 1)))
        dist = self.fc_dist_3(torch.cat([dist, action], dim = 1))

        result_dict = {'collison': coll_prob, 'offroad': offroad_prob, 'distance': dist}

        if args.use_xyz:
            xyz = F.relu(self.fc_xyz_1(x))
            xyz = F.relu(self.fc_xyz_2(torch.cat([xyz, action], dim = 1)))
            xyz = self.fc_xyz_3(torch.cat([xyz, action], dim = 1))
            result_dict['xyz'] = xyz
            
        return result_dict