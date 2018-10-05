import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from utils import weights_init

class end_layer(nn.Module):
    def __init__(self, args, out_dim, activate = None):
        super(end_layer, self).__init__()
        self.args, self.activate = args, activate

        self.conv1 = nn.Conv2d(args.classes * args.frame_history_len + 1, 16, 5, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(64, 32, 1, stride = 1, padding = 0)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, out_dim)

        self.apply(weights_init)


    def forward(self, hidden, info):
        x = torch.cat([hidden, info], dim = 1)
        x = F.tanh(F.max_pool2d(self.conv1(x), kernel_size = 2, stride = 2))
        x = F.tanh(F.max_pool2d(self.conv2(x), kernel_size = 2, stride = 2))
        x = F.tanh(F.max_pool2d(self.conv3(x), kernel_size = 2, stride = 2)) # 1x64x4x4
        x = F.tanh(self.conv4(x))
        x = x.view(x.size(0), -1) # 1024

        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
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
    net = end_layer(dummy(), 3)
    if dummy().use_seg:
        hidden = Variable(torch.zeros(1, 12, 32, 32))
        info = Variable(torch.zeros(1, 1, 32, 32))
    else:
        hidden = Variable(torch.zeros(1, 256))
        info = Variable(torch.zeros(1, 16))
    y = net(hidden, info)
    print(y.size())
