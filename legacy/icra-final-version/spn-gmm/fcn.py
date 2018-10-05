import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
from utils import weights_init

class fcn(nn.Module):
    def __init__(self, classes_in, classes_out, num_actions=32, deep=False):
        super(fcn, self).__init__()
        self.classes_in = classes_in
        self.classes_out = classes_out
        self.num_actions = num_actions
        self.deep = deep

        # self.bn0 = nn.BatchNorm2d(classes_in)
        # self.bn0.weight.data.fill_(1)
        # self.bn0.bias.data.zero_()

        self.action_encoder1 = nn.Linear(num_actions, 16*classes_in*5*5)
        # self.action_encoder1.weight.data.normal_(0, math.sqrt(2. / (16*5*5)))

        self.action_encoder2 = nn.Linear(num_actions, 32*16*5*5)
        # self.action_encoder2.weight.data.normal_(0, math.sqrt(2. / (16*5*5)))

        self.bn1 = nn.BatchNorm2d(16)
        # self.bn1.weight.data.fill_(1)
        # self.bn1.bias.data.zero_()

        self.bn2 = nn.BatchNorm2d(32)
        # self.bn2.weight.data.fill_(1)
        # self.bn2.bias.data.zero_()

        if deep:
            self.action_encoder3 = nn.Linear(num_actions, 64*32*5*5)
            # self.action_encoder3.weight.data.normal_(0, math.sqrt(2. / (32*5*5)))

            self.action_encoder4 = nn.Linear(num_actions, 32*64*5*5)
            # self.action_encoder4.weight.data.normal_(0, math.sqrt(2. / (32*5*5)))

            self.bn3 = nn.BatchNorm2d(64)
            # self.bn3.weight.data.fill_(1)
            # self.bn3.bias.data.zero_()

            self.bn4 = nn.BatchNorm2d(32)
            # self.bn4.weight.data.fill_(1)
            # self.bn4.bias.data.zero_()

            self.action_encoder5 = nn.Linear(num_actions, 64*32*5*5)
            # self.action_encoder5.weight.data.normal_(0, math.sqrt(2. / (64*5*5)))

            self.action_encoder6 = nn.Linear(num_actions, 32*64*5*5)
            # self.action_encoder6.weight.data.normal_(0, math.sqrt(2. / (64*5*5)))

            self.bn5 = nn.BatchNorm2d(64)
            # self.bn5.weight.data.fill_(1)
            # self.bn5.bias.data.zero_()

            self.bn6 = nn.BatchNorm2d(32)
            # self.bn6.weight.data.fill_(1)
            # self.bn6.bias.data.zero_()

            self.up = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            # self.up = nn.UpsamplingBilinear2d(scale_factor = 2)

        self.action_encoder7 = nn.Linear(num_actions, classes_out*32*1*1)
        # self.action_encoder7.weight.data.normal_(0, math.sqrt(2. / (4*1*1)))

        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.apply(weights_init)

    def single_forward(self, x, action):
        y1 = self.action_encoder1(action).view(16, self.classes_in, 5, 5)
        y2 = self.action_encoder2(action).view(32, 16, 5, 5)
        if self.deep:
            y3 = self.action_encoder3(action).view(64, 32, 5, 5)
            y4 = self.action_encoder4(action).view(32, 64, 5, 5)
            y5 = self.action_encoder5(action).view(64, 32, 5, 5)
            y6 = self.action_encoder6(action).view(32, 64, 5, 5)
        y7 = self.action_encoder7(action).view(self.classes_out, 32, 1, 1)

        result1 = F.relu(self.bn1(F.conv2d(x, y1, stride=1, padding=4, dilation=2)))
        result1 = self.bn2(F.conv2d(result1, y2, stride=1, padding=2))
        if self.deep:
            result2 = F.avg_pool2d(result1, kernel_size=2, stride=2)
            result2 = F.relu(self.bn3(F.conv2d(result2, y3, stride=1, padding=4, dilation=2)))
            result2 = self.up(self.bn4(F.conv2d(result2, y4, stride=1, padding=2)))
            result = result1 + result2
            result = F.relu(self.bn5(F.conv2d(result, y5, stride=1, padding=4, dilation=2)))
            result = self.bn6(F.conv2d(result, y6, stride=1, padding=2))
        else:
            result = result1
        result = F.conv2d(result, y7, stride=1, padding=0)
        return result


    def forward(self, x, action):
        batch_size, channels, height, width = x.size()
        x = x.contiguous().view(1, batch_size*channels, height, width)
        y1 = self.action_encoder1(action).contiguous().view(batch_size*16, self.classes_in, 5, 5)
        y2 = self.action_encoder2(action).contiguous().view(batch_size*32, 16, 5, 5)
        if self.deep:
            y3 = self.action_encoder3(action).contiguous().view(batch_size*64, 32, 5, 5)
            y4 = self.action_encoder4(action).contiguous().view(batch_size*32, 64, 5, 5)
            y5 = self.action_encoder5(action).contiguous().view(batch_size*64, 32, 5, 5)
            y6 = self.action_encoder6(action).contiguous().view(batch_size*32, 64, 5, 5)
        y7 = self.action_encoder7(action).contiguous().view(batch_size*self.classes_out, 32, 1, 1)

        result1 = F.conv2d(x, y1, stride=1, padding=4, dilation=2, groups=batch_size).contiguous().view(batch_size, 16, height, width)
        result1 = F.relu(self.bn1(result1)).contiguous().view(1, batch_size*16, height, width)

        result1 = F.conv2d(result1, y2, stride=1, padding=2, groups=batch_size).contiguous().view(batch_size, 32, height, width)
        result1 = self.bn2(result1).contiguous().view(1, batch_size*32, height, width)

        if self.deep:
            result2 = F.avg_pool2d(result1, kernel_size=2, stride=2)

            result2 = F.conv2d(result2, y3, stride=1, padding=4, dilation=2, groups=batch_size).contiguous().view(batch_size, 64, int(height/2), int(width/2))
            result2 = F.relu(self.bn3(result2)).contiguous().view(1, batch_size*64, int(height/2), int(width/2))

            result2 = F.conv2d(result2, y4, stride=1, padding=2, groups=batch_size).contiguous().view(batch_size, 32, int(height/2), int(width/2))
            result2 = self.up(self.bn4(result2)).contiguous().view(1, batch_size*32, height, width)

            result = result1 + result2

            result = F.conv2d(result, y5, stride=1, padding=4, dilation=2, groups=batch_size).contiguous().view(batch_size, 64, height, width)
            result = F.relu(self.bn5(result)).contiguous().view(1, batch_size*64, height, width)

            result = F.conv2d(result, y6, stride=1, padding=2, groups=batch_size).contiguous().view(batch_size, 32, height, width)
            result = self.bn6(result).contiguous().view(1, batch_size*32, height, width)

        else:
            result = result1
        result = F.conv2d(result, y7, stride=1, padding=0, groups=batch_size).contiguous().view(batch_size, self.classes_out, height, width)
        # result = torch.stack([self.single_forward(x[batch_id].unsqueeze(0), action[batch_id]).squeeze(0) for batch_id in range(action.size(0))], dim = 0)

        x = self.softmax(result)
        # y = self.logsoftmax(result)
        return x

if __name__ == '__main__':
    x = Variable(torch.rand(16, 24, 256, 256)).cuda()
    action = Variable(torch.rand(16, 32)).cuda()
    f = fcn(24, 4).cuda()
    f = torch.nn.DataParallel(f, device_ids=[0, 1])
    y1, y2 = f(x, action)
    print(y1.size(), y2.size())