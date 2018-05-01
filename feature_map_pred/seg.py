import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import drn
import os
import random
from scipy.misc import imsave
from scipy.misc.pilutil import imshow
import matplotlib.pyplot as plt
from py_TORCS import torcs_envs

train = True
batch_size = 16
seed = 233

def naive_driver(info):
    if info['angle'] > 0.5 or (info['trackPos'] < -1 and info['angle'] > 0):
        return 0
    elif info['angle'] < -0.5 or (info['trackPos'] > 3 and info['angle'] < 0):
        return 2
    return 1

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
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
        self.up = up_sampler(classes)

    def forward(self, x):
        x = self.base(x)
        x = self.seg(x)
        y = self.up(x)
        return F.log_softmax(y, dim = 1)

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.seg.parameters():
            yield param
        for param in self.up.parameters():
            yield param

def draw_from_pred(pred):
    pred = pred.data.cpu().numpy()
    illustration = np.zeros((480, 640, 3)).astype(np.uint8)
    illustration[:, :, 0] = 255
    illustration[pred == 1] = np.array([0, 255, 0])
    illustration[pred == 2] = np.array([0, 0, 0])
    illustration[pred == 3] = np.array([0, 0, 255])
    return illustration

def reduce(pred):
    pred[pred != 2] = 0
    pred[pred == 2] = 1
    return pred

if __name__ == '__main__':
    random.seed(seed)
    torch.manual_seed(seed)
    if os.path.exists('seg_log.txt'):
        os.system('rm seg_log.txt')
    model = DRNSeg('drn_d_22', 4)
    if not train:
        batch_size = 1
    inputs = torch.autograd.Variable(torch.ones(batch_size, 3, 480, 640), requires_grad = False)
    target = torch.autograd.Variable(torch.ones(batch_size, 480, 640), requires_grad = False).type(torch.LongTensor)
    criterion = nn.NLLLoss2d()
    optimizer = torch.optim.Adam(model.optim_parameters(), 0.001, amsgrad = True)
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
        target = target.cuda()
        criterion = criterion.cuda()

    env = torcs_envs(num = 1, isServer = 0, mkey_start = 899).get_envs()[0]
    obs = env.reset()
    obs, reward, done, info = env.step(1)
    obs = (obs.transpose(2, 0, 1) - 112.62289744791671) / 56.1524832523

    if train:
        model.train()
        losses = 0
        epoch = 0
        all_obs = np.zeros((batch_size, 480, 640, 3))
        while True:
            for i in range(batch_size):
                action = random.randint(0, 5) if random.random() < 0.5 else naive_driver(info)
                obs, reward, done, info = env.step(action)
                all_obs[i] = obs
                obs = (obs.transpose(2, 0, 1) - 112.62289744791671) / 56.1524832523
                inputs[i] = torch.from_numpy(obs)
                target[i] = torch.from_numpy(env.get_segmentation().astype(np.uint8))
                if done or reward <= -2.5:
                    obs = env.reset()
                    obs, reward, done, info = env.step(1)
                    obs = (obs.transpose(2, 0, 1) - 112.62289744791671) / 56.1524832523
            output = model(inputs)
            loss = criterion(output, target)
            print('iteration %d, loss %f' % (i + 1, loss.data.cpu().numpy()))
            with open('seg_log.txt', 'a') as f:
                f.write('iteration %d, loss %f\n' % (i + 1, loss.data.cpu().numpy()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch += 1
            if (epoch + 1) % 100 == 0:
                torch.save(model.state_dict(), os.path.join('models', 'epoch%d.dat' % epoch))
            if (epoch + 1) % 20 == 0:
                pred = torch.argmax(output, dim = 1)
                print('Saving images.')
                for i in range(batch_size):
                    imsave('images/%d.png' % i, np.concatenate((all_obs[i], draw_from_pred(target[i]), draw_from_pred(pred[i])), axis = 1))

    else:
        # model.load_state_dict(torch.load('models/epoch15329.dat'))
        for i in range(5):
            action = random.randint(0, 5) if random.random() < 0.5 else naive_driver(info)
            obs, reward, done, info = env.step(action)
            obs = (obs.transpose(2, 0, 1) - 112.62289744791671) / 56.1524832523
            inputs[0] = torch.from_numpy(obs)
            output = model(inputs)
            pred = torch.argmax(output, dim = 1)
            imsave('images/%d.png' % i, draw_from_pred(pred[0]))
        # preds = []
        # for action in range(6):
        #     plt.subplot(2, 3, action + 1)
        #     output, _ = model(inputs, action)
        #     _, pred = torch.max(output, 1)
        #     preds.append(pred)
        #     plt.imshow(draw_from_pred(pred))
        # plt.savefig('pred.png', dpi = 300)
    env.close()


'''
    output, x = model(x, 0) # output: 1x4x480x640, x: 1x4x60x80
    loss = criterion(output, y)
    print(loss)
    print(output)
'''