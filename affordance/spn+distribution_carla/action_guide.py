import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init

class guide(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(guide, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, out_dim)
        self.apply(weights_init)

    def forward(self, x):
        # x = F.interpolate(x, (80, 80), mode='bilinear', align_corners=True)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    net = guide(3, 6)
    x = torch.rand(2, 3, 256, 256)
    y = net(x)
    print(y)
    print(y.shape)
