import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import weights_init


class end_layer(nn.Module):
    def __init__(self, args, in_channels, out_dim, activate=None):
        super(end_layer, self).__init__()
        self.args, self.activate = args, activate
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, out_dim)
        self.apply(weights_init)

    def forward(self, x):
        x = F.interpolate(x, (80, 80), mode='bilinear')
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


if __name__ == '__main__':
    class dummy(object):
        def __init__(self):
            self.use_seg = True
            self.hidden_dim = 256
            self.info_dim = 16
            self.classes = 4
            self.frame_history_len = 3
    net = end_layer(dummy(), 4, 2)
    # if dummy().use_seg:
    #     hidden = Variable(torch.zeros(1, 12, 32, 32))
    #     info = Variable(torch.zeros(1, 1, 32, 32))
    # else:
    #     hidden = Variable(torch.zeros(1, 256))
    #     info = Variable(torch.zeros(1, 16))
    # y = net(hidden, info)
    x = Variable(torch.zeros(1, 4, 288, 512))
    y = net(x)
    print(y.size())
