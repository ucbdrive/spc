import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import dla

BatchNorm = nn.BatchNorm2d


def set_bn(bn):
    global BatchNorm
    BatchNorm = bn
    dla.BatchNorm = bn


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


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


class CAB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels * 2, in_channels * 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(
            in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, w):
        # w = F.adaptive_avg_pool2d(w, (x.size(2), x.size(3)))
        y = torch.cat((x, w), 1)
        y = F.adaptive_avg_pool2d(y, 1)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y)
        z = torch.mul(x, y)
        z = z + w
        return z


class RRB(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(RRB, self).__init__()
        self.conv0 = nn.Conv2d(
            in_channels,  out_channels, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.conv0(x)
        y = self.conv1(x)
        y = self.bn(y)
        y = self.relu(y)
        y = self.conv2(y)
        z = x + y
        z = self.relu(z)
        return z


class IDAUp(nn.Module):

    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2d(c, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    BatchNorm(out_dim),
                    nn.ReLU(inplace=True))
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(
                    out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                    output_padding=0, groups=out_dim, bias=False)
                fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            # node = nn.Sequential(
            #     nn.Conv2d(out_dim * 2, out_dim,
            #               kernel_size=node_kernel, stride=1,
            #               padding=node_kernel // 2, bias=False),
            #     BatchNorm(out_dim),
            #     nn.ReLU(inplace=True))
            # setattr(self, 'node_' + str(i), node)
            setattr(self, 'node_CAB_' + str(i), CAB(out_dim, out_dim))
            setattr(self, 'node_RRBL_' + str(i), RRB(out_dim, out_dim))
            setattr(self, 'node_RRBR_' + str(i), RRB(out_dim, out_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            CAB = getattr(self, 'node_CAB_' + str(i))
            RRBL = getattr(self, 'node_RRBL_' + str(i))
            RRBR = getattr(self, 'node_RRBR_' + str(i))
            x = RRBR(CAB(RRBL(x), layers[i]))
            y.append(x)
        return x, y


class DLAUp(nn.Module):

    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y
        return x


class DLASeg(nn.Module):

    def __init__(self, base_name, classes,
                 pretrained_base=None, down_ratio=2):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.base = dla.__dict__[base_name](pretrained=pretrained_base,
                                            return_levels=True)
        channels = self.base.channels
        up_factor = 2 ** self.first_level
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)
        self.fc = nn.Sequential(
            nn.Conv2d(channels[self.first_level], classes, kernel_size=1,
                      stride=1, padding=0, bias=True), 
            nn.Upsample(scale_factor=up_factor)
        )
        
        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x[self.first_level:])
        x = self.fc(x)
        return x


def dla34up_cr(classes, pretrained_base=None, **kwargs):
    model = DLASeg('dla34', classes, pretrained_base=pretrained_base, **kwargs)
    return model


# def dla60up(classes, pretrained_base=None, **kwargs):
#     model = DLASeg('dla60', classes, pretrained_base=pretrained_base, **kwargs)
#     return model


# def dla102up(classes, pretrained_base=None, **kwargs):
#     model = DLASeg('dla102', classes,
#                    pretrained_base=pretrained_base, **kwargs)
#     return model


# def dla169up(classes, pretrained_base=None, **kwargs):
#     model = DLASeg('dla169', classes,
#                    pretrained_base=pretrained_base, **kwargs)
#     return model
