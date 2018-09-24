import math

import numpy as np
import torch
from torch import nn

import sedla as dla

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


class SELayer(nn.Module):

    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                               stride=stride, padding=kernel_size // 2, bias=False)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                               stride=1, padding=kernel_size // 2, bias=False)
        self.bn2 = BatchNorm(planes)
        self.stride = stride
        self.se = SELayer(planes, reduction)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out = self.relu(out)
        return out


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
            node = BasicBlock(out_dim * 2, out_dim)
            setattr(self, 'node_' + str(i), node)

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
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
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
        
        # if up_factor > 1:
        #     up = nn.ConvTranspose2d(classes, classes, up_factor * 2,
        #                             stride=up_factor, padding=up_factor // 2,
        #                             output_padding=0, groups=classes,
        #                             bias=False)
        #     fill_up_weights(up)
        #     up.weight.requires_grad = False
        # else:
        #     up = Identity()
        # self.up = up

        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
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


def sedla34up(classes, pretrained_base=None, **kwargs):
    model = DLASeg('dla34', classes, pretrained_base=None, **kwargs)
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
