import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class PRED(nn.Module):
    def __init__(self, classes, num_actions):
        super(PRED, self).__init__()
        self.classes = classes
        self.num_actions = num_actions

        self.bn0 = nn.BatchNorm2d(classes)
        self.bn0.weight.data.fill_(1)
        self.bn0.bias.data.zero_()


        self.action_encoder1 = nn.Linear(num_actions, 16*classes*5*5)
        self.action_encoder1.weight.data.normal_(0, math.sqrt(2. / (16*5*5)))
        # self.action_encoder1.weight.data[:] += 1 / (4*5*5)

        self.action_encoder2 = nn.Linear(num_actions, 16*16*5*5)
        self.action_encoder2.weight.data.normal_(0, math.sqrt(2. / (16*5*5)))
        # self.action_encoder2.weight.data[:] += 1 / (16*5*5)


        self.bn1 = nn.BatchNorm2d(16)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()

        self.bn2 = nn.BatchNorm2d(16)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()

        self.action_encoder3 = nn.Linear(num_actions, 32*16*5*5)
        self.action_encoder3.weight.data.normal_(0, math.sqrt(2. / (32*5*5)))
        # self.action_encoder3.weight.data[:] += 1 / (16*5*5)


        self.action_encoder4 = nn.Linear(num_actions, 32*32*5*5)
        self.action_encoder4.weight.data.normal_(0, math.sqrt(2. / (32*5*5)))
        # self.action_encoder4.weight.data[:] += 1 / (32*5*5)


        self.bn3 = nn.BatchNorm2d(32)
        self.bn3.weight.data.fill_(1)
        self.bn3.bias.data.zero_()

        self.bn4 = nn.BatchNorm2d(32)
        self.bn4.weight.data.fill_(1)
        self.bn4.bias.data.zero_()

        self.action_encoder5 = nn.Linear(num_actions, 64*32*5*5)
        self.action_encoder5.weight.data.normal_(0, math.sqrt(2. / (64*5*5)))
        # self.action_encoder5.weight.data[:] += 1 / (32*5*5)

        self.action_encoder6 = nn.Linear(num_actions, 64*64*5*5)
        self.action_encoder6.weight.data.normal_(0, math.sqrt(2. / (64*5*5)))
        # self.action_encoder6.weight.data[:] += 1 / (64*5*5)

        self.bn5 = nn.BatchNorm2d(64)
        self.bn5.weight.data.fill_(1)
        self.bn5.bias.data.zero_()

        self.bn6 = nn.BatchNorm2d(64)
        self.bn6.weight.data.fill_(1)
        self.bn6.bias.data.zero_()

        self.up = nn.UpsamplingBilinear2d(scale_factor = 2)

        self.action_encoder7 = nn.Linear(num_actions, 4*64*1*1)
        self.action_encoder7.weight.data.normal_(0, math.sqrt(2. / (4*1*1)))
        # self.action_encoder7.weight.data[:] += 1 / (64*5*5)

    def single_forward(self, x, action):
        # one_hot = torch.zeros(1, self.num_actions)
        # one_hot[0, action] = 1
        # one_hot = Variable(one_hot, requires_grad = False)
        # if torch.cuda.is_available():
        #     one_hot = one_hot.cuda()
        y1 = self.action_encoder1(one_hot).view(16, self.classes, 5, 5)
        y2 = self.action_encoder2(one_hot).view(16, 16, 5, 5)
        y3 = self.action_encoder3(one_hot).view(32, 16, 5, 5)
        y4 = self.action_encoder4(one_hot).view(32, 32, 5, 5)
        y5 = self.action_encoder5(one_hot).view(64, 32, 5, 5)
        y6 = self.action_encoder6(one_hot).view(64, 64, 5, 5)
        y7 = self.action_encoder7(one_hot).view(4, 64, 1, 1)

        result = F.relu(self.bn1(F.conv2d(self.bn0(x), y1, stride = 1, padding = 4, dilation = 2)))
        result = self.bn2(F.conv2d(result, y2, stride = 1, padding = 2))
        result = F.avg_pool2d(result, kernel_size = 2, stride = 2)
        result = F.relu(self.bn3(F.conv2d(result, y3, stride = 1, padding = 4, dilation = 2)))
        result = self.bn4(F.conv2d(result, y4, stride = 1, padding = 2))
        result = F.relu(self.bn5(F.conv2d(result, y5, stride = 1, padding = 4, dilation = 2)))
        result = self.bn6(F.conv2d(result, y6, stride = 1, padding = 2))
        result = self.up(F.conv2d(result, y7, stride = 1, padding = 0))
        return result


    def forward(self, x, action):        
        result = torch.stack([self.single_forward(x[batch_id].unsqueeze(0), action[batch_id]).squeeze(0) for batch_id in range(action.size(0))], dim = 0)
        return result + x