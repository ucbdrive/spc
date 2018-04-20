import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import os
import drn2
from seg import fill_up_weights, draw_from_pred
from scipy.misc import imsave
from scipy.misc.pilutil import imshow
import matplotlib.pyplot as plt

def Focal_Loss(probs, target, reduce=True):
    # probs : batch * num_class
    # target : batch,
    loss = -1.0 * (1-probs).pow(1) * torch.log(probs)
    batch_size = int(probs.size()[0])
    loss = loss[torch.arange(batch_size).long().cuda(), target.long()]
    if reduce == True:
        loss = loss.sum()/(batch_size*1.0)
    return loss

def Focal_Loss_Regress(probs, target, reduce=True):
    # probs: batch * num_class
    # target: batch * num_class
    target_class = (torch.max(target, -1)[1]).view(-1,1)
    target = target * 2 - 1
    res1 = -1.0*(torch.log(probs) * target)
    weight = Variable(torch.arange(19)).view(1,19).repeat(probs.size()[0], 1).cuda().float()
    weight = 0.1*torch.abs(weight - target_class.repeat(1, 19).float())+1
    loss = (weight * res1).sum(-1)
    if reduce == True:
        loss = loss.sum()/(probs.size()[0]*19.0)
    return loss

class DRNSeg(nn.Module):
    def __init__(self, model_name, classes, pretrained_model=None,
                 pretrained=False, use_torch_up=False, num_actions = 6):
        super(DRNSeg, self).__init__()
        model = drn2.__dict__.get(model_name)(
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
        self.up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                output_padding=0, groups=classes, bias=False)
        fill_up_weights(self.up)

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return F.log_softmax(y, dim = 2), x # , dim = 2

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param
        for param in self.up.parameters():
            yield param

class PRED(nn.Module):
    def __init__(self):
        super(PRED, self).__init__()
        self.action_encoder1 = nn.Linear(6, 16*4*5*5)
        self.action_encoder1.weight.data.normal_(0, math.sqrt(2. / (16*5*5)))

        self.action_encoder2 = nn.Linear(6, 16*16*5*5)
        self.action_encoder2.weight.data.normal_(0, math.sqrt(2. / (16*5*5)))

        self.bn1 = nn.BatchNorm2d(16)
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()

        self.bn2 = nn.BatchNorm2d(16)
        self.bn2.weight.data.fill_(1)
        self.bn2.bias.data.zero_()

        self.action_encoder3 = nn.Linear(6, 32*16*5*5)
        self.action_encoder3.weight.data.normal_(0, math.sqrt(2. / (32*5*5)))

        self.action_encoder4 = nn.Linear(6, 32*32*5*5)
        self.action_encoder4.weight.data.normal_(0, math.sqrt(2. / (32*5*5)))

        self.bn3 = nn.BatchNorm2d(32)
        self.bn3.weight.data.fill_(1)
        self.bn3.bias.data.zero_()

        self.bn4 = nn.BatchNorm2d(32)
        self.bn4.weight.data.fill_(1)
        self.bn4.bias.data.zero_()

        self.action_encoder5 = nn.Linear(6, 64*32*5*5)
        self.action_encoder5.weight.data.normal_(0, math.sqrt(2. / (64*5*5)))

        self.action_encoder6 = nn.Linear(6, 64*64*5*5)
        self.action_encoder6.weight.data.normal_(0, math.sqrt(2. / (64*5*5)))

        self.bn5 = nn.BatchNorm2d(64)
        self.bn5.weight.data.fill_(1)
        self.bn5.bias.data.zero_()

        self.bn6 = nn.BatchNorm2d(64)
        self.bn6.weight.data.fill_(1)
        self.bn6.bias.data.zero_()

        self.action_encoder7 = nn.Linear(6, 4*64*1*1)
        self.action_encoder7.weight.data.normal_(0, math.sqrt(2. / (4*1*1)))

    def forward(self, x, action):
        one_hot = torch.zeros(1, 6)
        one_hot[0, action] = 1
        one_hot = Variable(one_hot, requires_grad = False)
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
        y1 = self.action_encoder1(one_hot).view(16, 4, 5, 5)
        y2 = self.action_encoder2(one_hot).view(16, 16, 5, 5)
        y3 = self.action_encoder3(one_hot).view(32, 16, 5, 5)
        y4 = self.action_encoder4(one_hot).view(32, 32, 5, 5)
        y5 = self.action_encoder5(one_hot).view(64, 32, 5, 5)
        y6 = self.action_encoder6(one_hot).view(64, 64, 5, 5)
        y7 = self.action_encoder7(one_hot).view(4, 64, 1, 1)

        result = F.relu(self.bn1(F.conv2d(x, y1, stride = 1, padding = 4, dilation = 2)))
        result = self.bn2(F.conv2d(result, y2, stride = 1, padding = 2))
        result = F.relu(self.bn3(F.conv2d(result, y3, stride = 1, padding = 4, dilation = 2)))
        result = self.bn4(F.conv2d(result, y4, stride = 1, padding = 2))
        result = F.relu(self.bn5(F.conv2d(result, y5, stride = 1, padding = 4, dilation = 2)))
        result = self.bn6(F.conv2d(result, y6, stride = 1, padding = 2))
        result = F.conv2d(result, y7, stride = 1, padding = 0)
        return result

class FURTHER(nn.Module):
    def __init__(self):
        super(FURTHER, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 5, stride = 1, padding = 2)
        self.conv1.weight.data.normal_(0, math.sqrt(2. / (16*5*5)))

        self.conv2 = nn.Conv2d(16, 32, 3, stride = 1, padding = 1)
        self.conv2.weight.data.normal_(0, math.sqrt(2. / (32*3*3)))

        self.conv3 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
        self.conv3.weight.data.normal_(0, math.sqrt(2. / (64*3*3)))

        self.fc1 = nn.Linear(4480, 1024)
        self.fc1.weight.data.normal_(0, math.sqrt(2. / 1024))

        self.fc2 = nn.Linear(1024, 256)
        self.fc2.weight.data.normal_(0, math.sqrt(2. / 256))

        self.fc3 = nn.Linear(256, 64)
        self.fc3.weight.data.normal_(0, math.sqrt(2. / 64))

        self.fc4 = nn.Linear(64, 16)
        self.fc4.weight.data.normal_(0, math.sqrt(2. / 16))

        self.fc5_1 = nn.Linear(16, 1)
        self.fc5_1.weight.data.normal_(0, math.sqrt(2. / 1))

        self.fc5_2 = nn.Linear(16, 1)
        self.fc5_2.weight.data.normal_(0, math.sqrt(2. / 1))

        self.fc5_3 = nn.Linear(16, 1)
        self.fc5_3.weight.data.normal_(0, math.sqrt(2. / 1))

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 2, stride = 2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 2, stride = 2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size = 2, stride = 2)) # 1x64x7x10
        x = x.view(x.size(0), -1) # 4480
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        offroad = F.sigmoid(self.fc5_1(x)).view(1)
        collision = F.sigmoid(self.fc5_2(x)).view(1)
        dist = F.sigmoid(self.fc5_3(x)).view(1)
        return offroad, collision, dist

def latest_model(dir = 'pred_models'):
    model_list = os.listdir(dir)
    return os.path.join(dir, sorted(model_list, key = lambda x: int(x[5:-4]))[-1])

num_steps = 12
train = True
# LOAD = True

if __name__ == '__main__':
    if os.path.exists('pred_log.txt'):
        os.system('rm pred_log.txt')
    if not os.path.isdir('pred_models'):
        os.mkdir('pred_models')
    model = DRNSeg('drn_d_22', 4)
    predictor = PRED()
    further = FURTHER()
    inputs = torch.autograd.Variable(torch.ones(1, 9, 480, 640), requires_grad = False)
    target = torch.autograd.Variable(torch.ones(13, 480, 640), requires_grad = False).type(torch.LongTensor)
    target_off = torch.autograd.Variable(torch.zeros(13), requires_grad = False)
    target_coll = torch.autograd.Variable(torch.zeros(13), requires_grad = False)
    target_dis = torch.autograd.Variable(torch.zeros(13), requires_grad = False)
    NLL = nn.NLLLoss2d()
    BCE = nn.BCELoss()
    L1 = nn.SmoothL1Loss()
    optimizer = torch.optim.SGD(list(model.optim_parameters()) + list(predictor.parameters()) + list(further.parameters()),
                                0.0001,
                                momentum=0.5,
                                weight_decay=1e-1)
    if torch.cuda.is_available():
        model = model.cuda()
        predictor = predictor.cuda()
        further = further.cuda()
        inputs = inputs.cuda()
        target = target.cuda()
        target_off = target_off.cuda()
        target_coll = target_coll.cuda()
        target_dis = target_dis.cuda()
        NLL = NLL.cuda()
        BCE = BCE.cuda()
        L1 = L1.cuda()
    
    if train:
        losses = 0
        epoch = 0
        while True: # while
            LOSS = np.zeros((num_steps + 1, 4))
            for i in range(2000): # 2000
                data = np.load('dataset2/%d.npz' % i)
                inputs[0] = torch.from_numpy(data['obs'])
                action = data['action']
                target[:] = torch.from_numpy(data['seg']).type(torch.LongTensor)
                target_off[:] = torch.from_numpy(data['off'])
                target_coll[:] = torch.from_numpy(data['coll'])
                target_dis[:] = torch.from_numpy(data['dist'])
                output, feature_map = model(inputs)
                off, coll, dis = further(feature_map)
                loss = NLL(output, target[0].view(1, 480, 640)) + BCE(off, target_off[0]) / 10.0 + BCE(coll, target_coll[0]) / 10.0 + L1(dis, target_dis[0]) / 0.4
                LOSS[0, 0] += NLL(output, target[0].view(1, 480, 640)).data.cpu().numpy()[0]
                LOSS[0, 1] += BCE(off, target_off[0]).data.cpu().numpy()[0] / 10.0
                LOSS[0, 2] += BCE(coll, target_coll[0]).data.cpu().numpy()[0] / 10.0
                LOSS[0, 3] += L1(dis, target_dis[0]).data.cpu().numpy()[0] / 0.4
                gamma = 1
                for j in range(1, num_steps + 1):
                    gamma *= 0.9
                    feature_map = predictor(feature_map, action[j - 1])
                    off, coll, dis = further(feature_map)
                    output = F.log_softmax(model.up(feature_map), dim = 2) # dim = 2
                    loss += gamma * (NLL(output, target[j].view(1, 480, 640)) + BCE(off, target_off[j]) / 10.0 + BCE(coll, target_coll[j]) / 10.0 + L1(dis, target_dis[j]) / 0.4)
                    LOSS[j, 0] += NLL(output, target[j].view(1, 480, 640)).data.cpu().numpy()[0]
                    LOSS[j, 1] += BCE(off, target_off[j]).data.cpu().numpy()[0] / 10.0
                    LOSS[j, 2] += BCE(coll, target_coll[j]).data.cpu().numpy()[0] / 10.0
                    LOSS[j, 3] += L1(dis, target_dis[j]).data.cpu().numpy()[0] / 0.4
                losses += loss.data.cpu().numpy()[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print(LOSS / 100.0)
                    print('iteration %d, mean loss %f' % (i + 1, losses / 100.0))
                    with open('pred_log.txt', 'a') as f:
                        f.write('%s\nIteration %d, mean loss %f\n' % (str(LOSS / 100.0), i + 1, losses / 100.0))
                    losses = 0
                    LOSS = np.zeros((num_steps + 1, 4))
                if (i + 1) % 1000 == 0:
                    torch.save(model.state_dict(), os.path.join('pred_models', 'model_epoch%d.dat' % epoch))
                    torch.save(predictor.state_dict(), os.path.join('pred_models', 'predictor_epoch%d.dat' % epoch))
                    torch.save(further.state_dict(), os.path.join('pred_models', 'further_epoch%d.dat' % epoch))
            epoch += 1