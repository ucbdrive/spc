import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from utils import weights_init

class end_layer(nn.Module):
    def __init__(self, hidden_dim, info_dim, out_dim, activate = None):
        super(end_layer, self).__init__()
        self.activate = activate

        self.fc1 = nn.Linear(hidden_dim + info_dim, 512)
        self.fc2 = nn.Linear(512 + info_dim, 128)
        self.fc3 = nn.Linear(128 + info_dim, 32)
        self.fc4 = nn.Linear(32 + info_dim, out_dim)

        self.apply(weights_init)


    def forward(self, hidden, info):
        x = torch.cat([hidden, info], dim = 1)
        x = F.relu(self.fc1(hidden))
        x = torch.cat([x, info], dim = 1)
        x = F.relu(self.fc2(x))
        x = torch.cat([x, info], dim = 1)
        x = F.relu(self.fc3(x))
        x = torch.cat([x, info], dim = 1)
        x = self.fc4(x)
        if self.activate is not None:
            x = self.activate(x)
        return x
