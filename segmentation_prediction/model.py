import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import drn
import math
from utils import fill_up_weights

class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=False, use_torch_up=False, num_actions = 6):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        pmodel = nn.DataParallel(model)
        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()

    def forward(self, x):
        x = self.base(x)
        feature_map = self.seg(x)
        return feature_map

class UP_Samper(nn.Module):
    def __init__(self):
        super(UP_Samper, self).__init__()
        self.up = nn.ConvTranspose2d(4, 4, 16, stride=8, padding=4,
                                output_padding=0, groups=4, bias=False)
        fill_up_weights(self.up)

    def forward(self, feature_map):
        y = self.up(feature_map)
        return y

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

        self.action_encoder7 = nn.Linear(num_actions, classes*64*1*1)
        self.action_encoder7.weight.data.normal_(0, math.sqrt(2. / (4*1*1)))
        # self.action_encoder7.weight.data[:] += 1 / (64*5*5)

    def single_forward(self, x, action):
        one_hot = torch.zeros(1, self.num_actions)
        one_hot[0, action] = 1
        one_hot = Variable(one_hot, requires_grad = False)
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
        y1 = self.action_encoder1(one_hot).view(16, self.classes, 5, 5)
        y2 = self.action_encoder2(one_hot).view(16, 16, 5, 5)
        y3 = self.action_encoder3(one_hot).view(32, 16, 5, 5)
        y4 = self.action_encoder4(one_hot).view(32, 32, 5, 5)
        y5 = self.action_encoder5(one_hot).view(64, 32, 5, 5)
        y6 = self.action_encoder6(one_hot).view(64, 64, 5, 5)
        y7 = self.action_encoder7(one_hot).view(self.classes, 64, 1, 1)

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

class FURTHER(nn.Module):
    def __init__(self):
        super(FURTHER, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 5, stride = 1, padding = 2)
        self.conv1.weight.data.normal_(0, math.sqrt(2. / (16*5*5)))

        self.conv2 = nn.Conv2d(16, 32, 3, stride = 1, padding = 1)
        self.conv2.weight.data.normal_(0, math.sqrt(2. / (32*3*3)))

        self.conv3 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
        self.conv3.weight.data.normal_(0, math.sqrt(2. / (64*3*3)))

        self.fc1 = nn.Linear(1024, 1024)
        self.fc1.weight.data.normal_(0, math.sqrt(2. / 1024))

        self.fc2 = nn.Linear(1024, 256)
        self.fc2.weight.data.normal_(0, math.sqrt(2. / 256))

        self.fc3_1 = nn.Linear(256, 64)
        self.fc3_1.weight.data.normal_(0, math.sqrt(2. / 64))

        self.fc3_2 = nn.Linear(256, 64)
        self.fc3_2.weight.data.normal_(0, math.sqrt(2. / 64))

        # self.fc3_3 = nn.Linear(256, 64)
        # self.fc3_3.weight.data.normal_(0, math.sqrt(2. / 64))

        self.fc4_1 = nn.Linear(64, 16)
        self.fc4_1.weight.data.normal_(0, math.sqrt(2. / 16))

        self.fc4_2 = nn.Linear(64, 16)
        self.fc4_2.weight.data.normal_(0, math.sqrt(2. / 16))

        # self.fc4_3 = nn.Linear(64, 16)
        # self.fc4_3.weight.data.normal_(0, math.sqrt(2. / 16))

        self.fc5_1 = nn.Linear(16, 1)
        self.fc5_1.weight.data.normal_(0, math.sqrt(2. / 16))

        self.fc5_2 = nn.Linear(16, 1)
        self.fc5_2.weight.data.normal_(0, math.sqrt(2. / 16))

        # self.fc5_3 = nn.Linear(16, 1)
        # self.fc5_3.weight.data.normal_(0, math.sqrt(2. / 1))

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