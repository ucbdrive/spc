import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import os

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class PRED(nn.Module):
    def __init__(self):
        super(PRED, self).__init__()
        self.action_encoder = nn.Linear(6, 400)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x, action):
        one_hot = torch.zeros(1, 6)
        one_hot[0, action] = 1
        one_hot = Variable(one_hot, requires_grad = False)
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
        y = F.sigmoid(self.action_encoder(one_hot).view(4, 4, 5, 5))
        result = self.bn(F.relu(F.conv2d(x, y, stride = 1, padding = 2)))
        result = F.conv2d(result, y, stride = 1, padding = 2)
        return result

class UP(nn.Module):
    def __init__(self):
        super(UP, self).__init__()
        self.up = nn.ConvTranspose2d(4, 4, 16, stride=8, padding=4,
                                output_padding=0, groups=4, bias=False)
        fill_up_weights(self.up)

    def forward(self, x):
        return F.log_softmax(self.up(x), dim = 2)

def select(dataset, idx):
    miniset = list(filter(lambda x: ('episode%d_' % idx) in x, dataset))
    return sorted(miniset, key = lambda x: int(x[:-4].split('_')[1][4:]))

if __name__ == '__main__':
    if os.path.exists('double_conv_log.txt'):
        os.system('rm double_conv_log.txt')
    if not os.path.isdir('double_conv_models'):
        os.mkdir('double_conv_models')
    np.random.seed(0)
    idx_list = list(range(2000))
    np.random.shuffle(idx_list)

    dataset = os.listdir('dataset')

    model = PRED()
    up = UP()
    inputs = Variable(torch.zeros(1, 4, 60, 80), requires_grad = False)
    target = Variable(torch.zeros(1, 480, 640), requires_grad = False).type(torch.LongTensor)
    criterion = nn.NLLLoss2d()
    optimizer = torch.optim.SGD(list(model.parameters()) + list(up.parameters()),
                                0.01,
                                momentum=0.9,
                                weight_decay=1e-4)
    if torch.cuda.is_available():
        model = model.cuda()
        up = up.cuda()
        inputs = inputs.cuda()
        target = target.cuda()
        criterion = criterion.cuda()

    cnt = 0
    for ooo in range(100):
        for idx in idx_list:
            cnt += 1
            data = select(dataset, idx)
            losses = 0
            for i in range(len(data) - 1):
                inputs[:] = torch.from_numpy(np.load(os.path.join('dataset', data[i])))
                action = int(data[i + 1][-5])
                tar = Variable(torch.from_numpy(np.load(os.path.join('dataset', data[i + 1]))), requires_grad = False)
                if torch.cuda.is_available():
                    tar = tar.cuda()
                _, pred = torch.max(up(tar), 1)
                target[0] = pred.type(torch.LongTensor)
                output = up(model(inputs, action))
                loss = criterion(output, target)
                if torch.cuda.is_available():
                    losses += loss.data.cpu().numpy()
                else:
                    losses += loss.data.numpy()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            mean_loss = losses / (len(data) - 1)
            print('dataset %d, mean loss %f' % (idx, mean_loss))
            with open('log.txt', 'a') as f:
                f.write('dataset %d, mean loss %f' % (idx, mean_loss))
            if cnt % 100 == 0:
                torch.save(model.state_dict(), os.path.join('double_conv_models', 'epoch%d.dat' % cnt))



