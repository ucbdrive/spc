import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import weights_init, from_variable_to_numpy, convert_state_dict
import cv2

class _DQN(nn.Module):
    def __init__(self, args, num_actions=11):
        super(_DQN, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(3 * args.frame_len, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        self.apply(weights_init)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        res = self.fc5(x)
        return res

class DQN(nn.Module):
    def __init__(self, args, num_actions=11):
        super(DQN, self).__init__()
        self.args = args
        self.dqn = _DQN(args, num_actions)

    def forward(self, x):
        return self.dqn(x)
