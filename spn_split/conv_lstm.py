import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import weights_init

class convLSTMCell(nn.Module):
    def __init__(self, in_channels, in_feature_channels, out_feature_channels, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True):
        super(convLSTMCell, self).__init__()
        self.out_feature_channels = out_feature_channels
        self.conv = nn.Conv2d(in_channels + in_feature_channels, 4 * out_feature_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.apply(weights_init)
    
    def forward(self, x, hidden_states):
        hx, cx = hidden_states
        combined = torch.cat([x, hx], dim=1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.out_feature_channels, dim = 1)#it should return 4 tensors
        i = F.sigmoid(ai)
        f = F.sigmoid(af)
        o = F.sigmoid(ao)
        g = F.tanh(ag)
        
        next_c = f * cx + i * g
        next_h = o * F.tanh(next_c)
        return next_h, next_c


class convLSTM(nn.Module):
    def __init__(self, in_channels, in_feature_channels, out_feature_channels):
        super(convLSTM, self).__init__()
        self.cell_1 = convLSTMCell(in_channels, in_feature_channels, out_feature_channels, 3, stride = 1, padding = 2, dilation = 2)
        self.cell_2 = convLSTMCell(in_channels, out_feature_channels, out_feature_channels, 5, stride = 1, padding = 2)
        self.cell_3 = convLSTMCell(in_channels, out_feature_channels, out_feature_channels, 3, stride = 1, padding = 1)
        # self.cell_4 = convLSTMCell(in_channels, feature_channels, 3, stride = 1, padding = 1)
        # self.cell_5 = convLSTMCell(in_channels, feature_channels, 5, stride = 1, padding = 4, dilation = 2)
        # self.cell_6 = convLSTMCell(in_channels, feature_channels, 5, stride = 1, padding = 2)
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, x, hidden_states):
        hx, cx = hidden_states
        hx, cx = self.cell_1(x, (hx, cx))
        hx, cx = self.cell_2(x, (hx, cx))
        hx, cx = self.cell_3(x, (hx, cx))
        x = self.softmax(hx)
        y = self.logsoftmax(hx)

        # _x = F.avg_pool2d(x, kernel_size = 2, stride = 2)
        # hx = F.avg_pool2d(hx, kernel_size = 2, stride = 2)
        # cx = F.avg_pool2d(cx, kernel_size = 2, stride = 2)

        # hx, cx = self.cell_3(_x, (hx, cx))
        # hx, cx = self.cell_4(_x, (hx, cx))

        # hx = F.upsample(hx, scale_factor = 2, mode = 'bilinear', align_corners = True)
        # cx = F.upsample(cx, scale_factor = 2, mode = 'bilinear', align_corners = True)

        # hx, cx = self.cell_5(x, (hx, cx))
        # hx, cx = self.cell_6(x, (hx, cx))
        return x, cx, y

if __name__ == '__main__':
    net = convLSTM(3, 1)
    x = Variable(torch.zeros(1, 3, 256, 256))
    hx = Variable(torch.zeros(1, 1, 256, 256))
    cx = Variable(torch.zeros(1, 1, 256, 256))
    hx, cx = net(x, (hx, cx))
    print(hx.size(), cx.size())