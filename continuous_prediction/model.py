import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import drn
import dla
import math
import cv2
from utils import fill_up_weights, weights_init, from_variable_to_numpy

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

        self.action_encoder7 = nn.Linear(num_actions, 4*64*1*1)
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

class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.args = args
        self.encoder = nn.Linear(args.hidden_dim + args.info_dim, args.hidden_dim)
        self.fc1 = nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc4 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.lstm = nn.LSTMCell(args.hidden_dim, args.hidden_dim)
        self.out_encoder = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.apply(weights_init)

    def forward(self, feature_encoding, action, hx, cx):
        encoding = F.relu(self.encoder(torch.cat([feature_encoding, action], dim = 1)))
        x = F.relu(self.fc1(torch.cat([encoding, hx], dim = 1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        hx, cx = self.lstm(x, (hx, cx))
        hx = self.out_encoder(hx)
        return hx, cx

class FURTHER_continuous(nn.Module):
    def __init__(self, args):
        super(FURTHER_continuous, self).__init__()
        self.fc_coll_1 = nn.Linear(args.hidden_dim + args.info_dim, 128)
        self.fc_coll_2 = nn.Linear(128 + args.info_dim, 32)
        self.fc_coll_3 = nn.Linear(32 + args.info_dim, 1)

        self.fc_off_1 = nn.Linear(args.hidden_dim + args.info_dim, 128)
        self.fc_off_2 = nn.Linear(128 + args.info_dim, 32)
        self.fc_off_3 = nn.Linear(32 + args.info_dim, 1)

        self.fc_dist_1 = nn.Linear(args.hidden_dim + args.info_dim, 128)
        self.fc_dist_2 = nn.Linear(128 + args.info_dim, 32)
        self.fc_dist_3 = nn.Linear(32 + args.info_dim, 1)

        self.apply(weights_init)

    def forward(self, feature, action):
        x = torch.cat([feature, action], dim = 1)

        coll_prob = F.relu(self.fc_coll_1(x))
        coll_prob = F.relu(self.fc_coll_2(torch.cat([coll_prob, action], dim = 1)))
        coll_prob = F.sigmoid(self.fc_coll_3(torch.cat([coll_prob, action], dim = 1)))

        offroad_prob = F.relu(self.fc_off_1(x))
        offroad_prob = F.relu(self.fc_off_2(torch.cat([offroad_prob, action], dim = 1)))
        offroad_prob = F.sigmoid(self.fc_off_3(torch.cat([offroad_prob, action], dim = 1)))

        dist = F.relu(self.fc_dist_1(x))
        dist = F.relu(self.fc_dist_2(torch.cat([dist, action], dim = 1)))
        dist = self.fc_dist_3(torch.cat([dist, action], dim = 1))

        return coll_prob, offroad_prob, dist

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, 9)
        
    def forward(self, x):
        x = from_variable_to_numpy(x)
        x = [torch.from_numpy(cv2.resize(x[i].transpose(1, 2, 0), (84, 84)).transpose(2, 0, 1)) for i in range(x.shape[0])]
        x = Variable(torch.stack(x, dim = 0), requires_grad = False)
        if torch.cuda.is_available():
            x = x.cuda()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        res = self.fc5(x)
        return res

class hybrid_net(nn.Module):
    def __init__(self, args):
        super(hybrid_net, self).__init__()
        self.args = args

        if self.args.continuous:
            self.action_encoder = nn.Linear(2, args.info_dim)

        if args.use_seg:
            self.feature_encoder = DRNSeg('drn_d_22', args.semantic_classes)
            self.up_sampler = UP_Samper()
            self.predictor = PRED(args.semantic_classes, args.num_actions)
            self.further = FURTHER()
        else:
            self.dla = dla.dla46x_c(pretrained = True)
            self.feature_encoder = nn.Linear(256, args.hidden_dim)
            self.predictor = RNN(args)
            self.further = FURTHER_continuous(args)

    def forward(self, obs, action, hx = None, cx = None):
        if self.args.continuous:
            action = self.action_encoder(action)

        if self.args.use_seg:
            feature_map = self.feature_encoder(obs)
            segmentation = self.up_sampler(feature_map)
            prediction = self.predictor(feature_map, action)
            return feature_map, segmentation, prediction, self.further(feature_map)
        else:
            feature = self.feature_encoder(self.dla(obs))
            hx, cx = self.predictor(feature, action, hx, cx)
            return hx, cx, self.further(hx, action)

if __name__ == '__main__':
    dqn = DQN().cuda()
    obs = Variable(torch.zeros(16, 3, 256, 256)).cuda()
    act = Variable(torch.zeros(16, 1)).type(torch.LongTensor).cuda()
    x = dqn(obs).detach().max(1)[0]
    print(x.size())
    # dla = dla.dla46x_c(pretrained = True).cuda()
    # obs = Variable(torch.zeros(16, 3, 256, 256)).cuda()
    # res = dla(obs)
    # print(res.size())