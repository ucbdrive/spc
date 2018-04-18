import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import os
import drn2
from single_conv import PRED as single_pred
from single_conv import UP
from seg import fill_up_weights, draw_from_pred
from scipy.misc import imsave
from scipy.misc.pilutil import imshow
import matplotlib.pyplot as plt

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
        self.action_encoder1 = nn.Linear(6, 400)
        self.action_encoder1.weight.data.normal_(0, math.sqrt(2. / 400))
        self.action_encoder2 = nn.Linear(6, 400)
        self.action_encoder2.weight.data.normal_(0, math.sqrt(2. / 400))
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(4)

    def forward(self, x, action):
        one_hot = torch.zeros(1, 6)
        one_hot[0, action] = 1
        one_hot = Variable(one_hot, requires_grad = False)
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
        y1 = self.action_encoder1(one_hot).view(4, 4, 5, 5)
        y2 = self.action_encoder2(one_hot).view(4, 4, 5, 5)
        result = F.relu(self.bn1(F.conv2d(x, y1, stride = 1, padding = 2))) + F.relu(self.bn1(F.conv2d(x, y1, stride = 1, padding = 4, dilation = 2)))
        result = self.bn2(F.conv2d(result, y2, stride = 1, padding = 2))
        return result

num_steps = 12
train = True

if __name__ == '__main__':
    if os.path.exists('pred_log.txt'):
        os.system('rm pred_log.txt')
    if not os.path.isdir('pred_models'):
        os.mkdir('pred_models')
    model = DRNSeg('drn_d_22', 4)
    predictor = PRED()
    inputs = torch.autograd.Variable(torch.ones(1, 9, 480, 640), requires_grad = False)
    target = torch.autograd.Variable(torch.ones(12, 480, 640), requires_grad = False).type(torch.LongTensor)
    criterion = nn.NLLLoss2d()
    optimizer = torch.optim.SGD(list(model.optim_parameters()) + list(predictor.parameters()),
                                0.01,
                                momentum=0.9,
                                weight_decay=1e-4)
    if torch.cuda.is_available():
        model = model.cuda()
        predictor = predictor.cuda()
        inputs = inputs.cuda()
        target = target.cuda()
        criterion = criterion.cuda()
    
    if train:
        losses = 0
        epoch = 0
        while True: # while
            for i in range(2000): # 2000
                data = np.load('dataset2/%d.npz' % i)
                inputs[0] = torch.from_numpy(data['obs'])
                action = data['action']
                target[:] = torch.from_numpy(data['seg']).type(torch.LongTensor)
                _, feature_map = model(inputs)
                loss = 0
                gamma = 1
                for j in range(num_steps):
                    feature_map = predictor(feature_map, action[j])
                    output = F.log_softmax(model.up(feature_map), dim = 2) # dim = 2
                    loss += gamma * criterion(output, target[j].view(1, 480, 640))
                    gamma *= 0.9
                losses += loss.data.cpu().numpy()[0]
                print(loss.data.cpu().numpy()[0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print('iteration %d, mean loss %f' % (i + 1, losses / 100.0))
                    with open('pred_log.txt', 'a') as f:
                        f.write('iteration %d, mean loss %f\n' % (i + 1, losses / 100.0))
                    losses = 0
                if (i + 1) % 1000 == 0:
                    torch.save(model.state_dict(), os.path.join('pred_models', 'epoch%d.dat' % epoch))
            epoch += 1