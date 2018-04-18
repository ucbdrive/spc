import torch
import torch.nn as nn
import math
import numpy as np
import drn
import data_transforms as transforms
import os
from scipy.misc import imsave
from scipy.misc.pilutil import imshow
import matplotlib.pyplot as plt

train = False

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

def up_sampler(classes, use_torch_up=False):
    if use_torch_up:
        up = nn.UpsamplingBilinear2d(scale_factor=8)
    else:
        up = nn.ConvTranspose2d(classes, classes, 16, stride=8, padding=4,
                                output_padding=0, groups=classes, bias=False)
        fill_up_weights(up)
    return up

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
        self.softmax = nn.LogSoftmax()
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        self.up0 = up_sampler(classes)
        self.up1 = up_sampler(classes)
        self.up2 = up_sampler(classes)
        self.up3 = up_sampler(classes)
        self.up4 = up_sampler(classes)
        self.up5 = up_sampler(classes)

    def forward(self, x, action):
        x = self.base(x)
        x = self.seg(x)
        if action == 0:
            y = self.up0(x)
        elif action == 1:
            y = self.up1(x)
        elif action == 2:
            y = self.up2(x)
        elif action == 3:
            y = self.up3(x)
        elif action == 4:
            y = self.up4(x)
        elif action == 5:
            y = self.up5(x)
        return self.softmax(y), x

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param

def draw_from_pred(pred):
    pred = pred.data.cpu().numpy()[0]
    illustration = np.zeros((480, 640, 3)).astype(np.uint8)
    illustration[:, :, 0] = 255
    illustration[pred == 1] = np.array([0, 255, 0])
    illustration[pred == 2] = np.array([0, 0, 0])
    illustration[pred == 3] = np.array([0, 0, 255])
    return illustration

if __name__ == '__main__':
    if os.path.exists('log.txt'):
        os.system('rm log.txt')
    model = DRNSeg('drn_d_22', 4)
    inputs = torch.autograd.Variable(torch.ones(1, 3, 480, 640), requires_grad = False)
    target = torch.autograd.Variable(torch.ones(1, 480, 640), requires_grad = False).type(torch.LongTensor)
    criterion = nn.NLLLoss2d()
    optimizer = torch.optim.SGD(model.optim_parameters(),
                                0.01,
                                momentum=0.9,
                                weight_decay=1e-4)
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        target = target.cuda()
        criterion = criterion.cuda()

    if train:
        losses = 0
        epoch = 0
        while True:
            for i in range(10000):
                data = np.load('/home/cxy/semantic_pred/dataset/%d.npz' % i)
                inputs[0] = torch.from_numpy(data['obs'].transpose(2, 0, 1))
                action = data['action'][0]
                target[0] = torch.from_numpy(data['seg']).type(torch.LongTensor)
                output, _ = model(inputs, action)
                loss = criterion(output, target)
                losses += loss.data.cpu().numpy()[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print('iteration %d, mean loss %f' % (i + 1, losses / 100.0))
                    with open('log.txt', 'a') as f:
                        f.write('iteration %d, mean loss %f\n' % (i + 1, losses / 100.0))
                    losses = 0
                if (i + 1) % 1000 == 0:
                    torch.save(model.state_dict(), os.path.join('models', 'epoch%d.dat' % epoch))
            epoch += 1
    else:
        model.load_state_dict(torch.load('models/epoch5.dat'))
        print(model.up1.weight)
        data = np.load('/home/cxy/semantic_pred/dataset/50.npz')  
        inputs[0] = torch.from_numpy(data['obs'].transpose(2, 0, 1))
        preds = []
        for action in range(6):
            plt.subplot(2, 3, action + 1)
            output, _ = model(inputs, action)
            _, pred = torch.max(output, 1)
            preds.append(pred)
            plt.imshow(draw_from_pred(pred))
        plt.savefig('pred.png', dpi = 300)



'''
    output, x = model(x, 0) # output: 1x4x480x640, x: 1x4x60x80
    loss = criterion(output, y)
    print(loss)
    print(output)
'''