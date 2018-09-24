import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import weights_init

class convLSTMCell2(nn.Module):
    def __init__(self, in_channels, feature_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True):
        super(convLSTMCell2, self).__init__()
        self.feature_channels = feature_channels
        self.conv = nn.Conv2d(in_channels + feature_channels, 4 * feature_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.apply(weights_init)
    
    def forward(self, x, hidden_states):
        hx, cx = hidden_states
        combined = torch.cat([x, hx], dim=1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.feature_channels, dim=1)#it should return 4 tensors
        i = F.sigmoid(ai)
        f = F.sigmoid(af)
        o = F.sigmoid(ao)
        g = F.tanh(ag)

        next_c = f * cx + i * g
        next_h = o * F.tanh(next_c)
        return next_h, next_c

class convLSTM2(nn.Module):
    def __init__(self, in_channels, frame_history_length, feature_channels):
        super(convLSTM2, self).__init__()
        self.in_channels = in_channels
        self.frame_history_length = frame_history_length
        self.feature_channels = feature_channels

        self.downsample1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.downsample3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

        self.lstm0 = convLSTMCell2(in_channels, feature_channels, kernel_size=5, padding=2)
        self.lstm1 = convLSTMCell2(in_channels, feature_channels, kernel_size=3, padding=1)
        self.lstm2 = convLSTMCell2(in_channels, feature_channels, kernel_size=3, padding=1)
        self.lstm3 = convLSTMCell2(in_channels, feature_channels, kernel_size=1, padding=0)

        self.upsample1 = nn.ConvTranspose2d(feature_channels, feature_channels, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(feature_channels, feature_channels, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(feature_channels, feature_channels, kernel_size=2, stride=2)

    def forward(self, x0):
        batch_size, channels, height, width = x0.size()

        hx0 = Variable(torch.zeros(batch_size, self.feature_channels, height, width))
        cx0 = Variable(torch.zeros(batch_size, self.feature_channels, height, width))
        hx1 = Variable(torch.zeros(batch_size, self.feature_channels, int(height/2), int(width/2)))
        cx1 = Variable(torch.zeros(batch_size, self.feature_channels, int(height/2), int(width/2)))
        hx2 = Variable(torch.zeros(batch_size, self.feature_channels, int(height/4), int(width/4)))
        cx2 = Variable(torch.zeros(batch_size, self.feature_channels, int(height/4), int(width/4)))
        hx3 = Variable(torch.zeros(batch_size, self.feature_channels, int(height/8), int(width/8)))
        cx3 = Variable(torch.zeros(batch_size, self.feature_channels, int(height/8), int(width/8)))

        if torch.cuda.is_available():
            hx0 = hx0.cuda()
            cx0 = cx0.cuda()
            hx1 = hx1.cuda()
            cx1 = cx1.cuda()
            hx2 = hx2.cuda()
            cx2 = cx2.cuda()
            hx3 = hx3.cuda()
            cx3 = cx3.cuda()

        x1 = self.downsample1(x0.view(batch_size*self.frame_history_length, self.in_channels, height, width))
        x2 = self.downsample2(x1)
        x3 = self.downsample3(x2)

        x1 = x1.view(batch_size, channels, int(height/2), int(width/2))
        x2 = x2.view(batch_size, channels, int(height/4), int(width/4))
        x3 = x3.view(batch_size, channels, int(height/8), int(width/8))

        for step in range(self.frame_history_length):
            hx0, cx0 = self.lstm0(x0[:, step*self.in_channels: (step+1)*self.in_channels, :, :], (hx0, cx0))
            hx1, cx1 = self.lstm1(x1[:, step*self.in_channels: (step+1)*self.in_channels, :, :], (hx1, cx1))
            hx2, cx2 = self.lstm2(x2[:, step*self.in_channels: (step+1)*self.in_channels, :, :], (hx2, cx2))
            hx3, cx3 = self.lstm3(x3[:, step*self.in_channels: (step+1)*self.in_channels, :, :], (hx3, cx3))

        hx2 = hx2 + self.upsample3(hx3)
        hx1 = hx1 + self.upsample2(hx2)
        return hx0 + self.upsample1(hx1)

if __name__ == '__main__':
    net = convLSTM2(5, 6, 4)
    x = Variable(torch.rand(2, 30, 256, 256))
    y = net(x)
    print(y.size())