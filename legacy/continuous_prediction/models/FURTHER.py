import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from utils import weights_init

class FURTHER(nn.Module):
    def __init__(self):
        super(FURTHER, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 256)

        self.fc3_1 = nn.Linear(256, 64)
        self.fc3_2 = nn.Linear(256, 64)
        # self.fc3_3 = nn.Linear(256, 64)

        self.fc4_1 = nn.Linear(64, 16)
        self.fc4_2 = nn.Linear(64, 16)
        # self.fc4_3 = nn.Linear(64, 16)

        self.fc5_1 = nn.Linear(16, 1)
        self.fc5_2 = nn.Linear(16, 1)
        # self.fc5_3 = nn.Linear(16, 1)
        
        self.apply(weights_init)


    def forward(self, x):
        x = F.tanh(F.max_pool2d(self.conv1(x), kernel_size = 2, stride = 2))
        x = F.tanh(F.max_pool2d(self.conv2(x), kernel_size = 2, stride = 2))
        x = F.tanh(F.max_pool2d(self.conv3(x), kernel_size = 2, stride = 2)) # 1x64x4x4
        x = x.view(x.size(0), -1) # 1024
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pos = self.fc5_1(F.tanh(self.fc4_1(F.tanh(self.fc3_1(x))))).view(-1)
        angle = self.fc5_2(F.tanh(self.fc4_2(F.tanh(self.fc3_2(x))))).view(-1)
        # speed = self.fc5_3(F.relu(self.fc4_3(F.relu(self.fc3_3(x))))).view(1)
        return pos, angle