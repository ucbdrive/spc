import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
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
        x = F.interpolate(x, (80, 80), mode='bilinear', align_corners=True)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        if self.activate is not None:
            x = self.activate(x)
        return x

class end_layer2(nn.Module):
    def __init__(self, args, in_channels, out_dim, activate = None):
        super(end_layer2, self).__init__()
        self.args, self.activate = args, activate

        if args.use_seg:
            self.conv1 = nn.Conv2d(in_channels, 16, 5, stride=2, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(64, 32, 1, stride=1, padding=0)
            self.fc1 = nn.Linear(128, 128)
            self.fc2 = nn.Linear(128, 32)
            self.fc3 = nn.Linear(32, out_dim)
        else:
            self.fc1 = nn.Linear(args.hidden_dim + args.info_dim, 512)
            #self.fc2 = nn.Linear(512 + args.info_dim, 128)
            #self.fc3 = nn.Linear(128 + args.info_dim, 32)
            self.fc4 = nn.Linear(512 + args.info_dim, out_dim)

        self.apply(weights_init)


    def forward(self, x):
        if self.args.use_seg:
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
            x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
            x = F.relu(F.max_pool2d(self.conv4(x), kernel_size=2, stride=2))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x.view(x.size(0), -1)))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = torch.cat([hidden, info], dim = 1)
            x = F.sigmoid(self.fc1(x))
            # x = torch.cat([x, info], dim = 1)
            # x = F.sigmoid(self.fc2(x))
            # x = torch.cat([x, info], dim = 1)
            # x = F.sigmoid(self.fc3(x))
            x = torch.cat([x, info], dim = 1)
            x = self.fc4(x)
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
    x = Variable(torch.zeros(1, 4, 256, 256))
    y = net(x)
    print(y.size())
