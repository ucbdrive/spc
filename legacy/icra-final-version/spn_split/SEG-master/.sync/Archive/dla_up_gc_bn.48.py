import math

import numpy as np
import torch
from torch import nn

import dla_bn as dla
from sync_batchnorm import SynchronizedBatchNorm2d
BatchNorm = SynchronizedBatchNorm2d

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


class GlobalConvolutionV1(nn.Module):

    def __init__(self, c, o, lk, sk):
        super(GlobalConvolutionV1, self).__init__()
        self.gcl = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1),
            nn.Conv2d(c, c, kernel_size=(1, lk), padding=(0, lk // 2), stride=1))
        self.gcr = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, lk), padding=(0, lk // 2), stride=1),
            nn.Conv2d(c, c, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1))

        self.br = nn.Sequential(
            BatchNorm(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=sk, stride=1,
                      padding=sk // 2, bias=False),
            BatchNorm(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, o, kernel_size=1, stride=1,
                      padding=0, bias=False),
            BatchNorm(o),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.gcl(x) + self.gcr(x)
        x = self.br(x)
        return x


class IDAUpV1(nn.Module):

    def __init__(self, large_kernel, small_kernel, out_dim, channels, up_factors):
        super(IDAUpV1, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            # if c == out_dim:
            #     proj = Identity()
            # else:
            #     proj = nn.Sequential(
            #         nn.Conv2d(c, out_dim,
            #                   kernel_size=1, stride=1, bias=False),
            #         BatchNorm(out_dim),
            #         nn.ReLU(inplace=True))
            proj = GlobalConvolutionV1(c, out_dim, large_kernel, small_kernel)
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
            # node = GlobalConvolution(out_dim * 2, out_dim, large_kernel, small_kernel)
            node = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim,
                          kernel_size=small_kernel, stride=1,
                          padding=small_kernel // 2, bias=False),
                BatchNorm(out_dim),
                nn.ReLU(inplace=True))
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

class GlobalConvolutionV2(nn.Module):

    def __init__(self, c, o, lk, sk):
        super(GlobalConvolutionV2, self).__init__()
        self.gcl = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1),
            nn.Conv2d(c, c, kernel_size=(1, lk), padding=(0, lk // 2), stride=1))
        self.gcr = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, lk), padding=(0, lk // 2), stride=1),
            nn.Conv2d(c, c, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1))

        self.br = nn.Sequential(
            BatchNorm(c + c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c + c, o, kernel_size=1, stride=1,
                      padding=0, bias=False),
            BatchNorm(o),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, kernel_size=sk, stride=1,
                      padding=sk // 2, bias=False),
            BatchNorm(o),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = torch.cat([self.gcl(x), self.gcr(x)], dim=1)
        x = self.br(x)
        return x


class IDAUpV2(nn.Module):

    def __init__(self, large_kernel, small_kernel, out_dim, channels, up_factors):
        super(IDAUpV2, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            # if c == out_dim:
            #     proj = Identity()
            # else:
            #     proj = nn.Sequential(
            #         nn.Conv2d(c, out_dim,
            #                   kernel_size=1, stride=1, bias=False),
            #         BatchNorm(out_dim),
            #         nn.ReLU(inplace=True))
            proj = GlobalConvolutionV2(c, out_dim, large_kernel, small_kernel)
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
            node = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim,
                          kernel_size=1, stride=1,
                          padding=0, bias=False),
                BatchNorm(out_dim),
                nn.ReLU(inplace=True))
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


class BoundaryRefinement(nn.Module):

  def __init__(self, planes, kernel_size):
    super(BoundaryRefinement, self).__init__()
    self.conv1 = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                           stride=1, padding=kernel_size // 2, bias=False)
    self.bn1 = BatchNorm(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size,
                           stride=1, padding=kernel_size // 2, bias=False)
    self.bn2 = BatchNorm(planes)

  def forward(self, x, residual=None):
    if residual is None:
      residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    out += residual
    out = self.relu(out)

    return out
    

class GlobalConvolutionV3(nn.Module):

    def __init__(self, c, o, lk, sk):
        super(GlobalConvolutionV3, self).__init__()
        self.gcl = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1),
            nn.Conv2d(c, c, kernel_size=(1, lk), padding=(0, lk // 2), stride=1))
        self.gcr = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, lk), padding=(0, lk // 2), stride=1),
            nn.Conv2d(c, c, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1))

        self.proj = nn.Sequential(
            nn.Conv2d(c, o, kernel_size=1, stride=1,
                      padding=0, bias=False),
            BatchNorm(o),
            nn.ReLU(inplace=True))

        self.br = BoundaryRefinement(o, sk)

    def forward(self, x):
        # x = torch.cat([self.gcl(x), self.gcr(x)], dim=1)
        x = self.gcl(x) + self.gcr(x)
        x = self.proj(x)
        x = self.br(x)
        return x


class IDAUpV3(nn.Module):

    def __init__(self, large_kernel, small_kernel, out_dim, channels, up_factors):
        super(IDAUpV3, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            proj = GlobalConvolutionV3(c, out_dim, large_kernel, small_kernel)
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
            node = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim,
                          kernel_size=1, stride=1,
                          padding=0, bias=False),
                BatchNorm(out_dim),
                nn.ReLU(inplace=True))
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


class GlobalConvolutionV4(nn.Module):

    def __init__(self, c, o, lk, sk):
        super(GlobalConvolutionV4, self).__init__()
        self.gcl = nn.Sequential(
            nn.Conv2d(c, o, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1),
            nn.Conv2d(o, o, kernel_size=(1, lk), padding=(0, lk // 2), stride=1))
        self.gcr = nn.Sequential(
            nn.Conv2d(c, o, kernel_size=(1, lk), padding=(0, lk // 2), stride=1),
            nn.Conv2d(o, o, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1))

        self.br = BoundaryRefinement(o, sk)

    def forward(self, x):
        x = self.gcl(x) + self.gcr(x)
        x = self.br(x)
        return x


class IDAUpV4(nn.Module):

    def __init__(self, large_kernel, small_kernel, out_dim, channels, up_factors):
        super(IDAUpV4, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            proj = GlobalConvolutionV4(c, out_dim, large_kernel, small_kernel)
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
            node = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim,
                          kernel_size=1, stride=1,
                          padding=0, bias=False),
                BatchNorm(out_dim),
                nn.ReLU(inplace=True))
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


# class GlobalConvolutionV5(nn.Module):

#     def __init__(self, c, o, lk, sk):
#         super(GlobalConvolutionV5, self).__init__()
#         self.gcl = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1),
#             nn.Conv2d(c, c, kernel_size=(1, lk), padding=(0, lk // 2), stride=1))
#         self.gcr = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=(1, lk), padding=(0, lk // 2), stride=1),
#             nn.Conv2d(c, c, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1))

#         self.proj = nn.Sequential(
#             nn.Conv2d(c + c, o, kernel_size=1, stride=1,
#                       padding=0, bias=False),
#             BatchNorm(o),
#             nn.ReLU(inplace=True))

#         self.br = BoundaryRefinement(o, sk)

#     def forward(self, x):
#         # x = self.gcl(x) + self.gcr(x)
#         x = torch.cat([self.gcl(x), self.gcr(x)], dim=1)
#         x = self.br(self.proj(x))
#         return x


# class IDAUpV5(nn.Module):

#     def __init__(self, large_kernel, small_kernel, out_dim, channels, up_factors):
#         super(IDAUpV5, self).__init__()
#         self.channels = channels
#         self.out_dim = out_dim
#         for i, c in enumerate(channels):
#             f = int(up_factors[i])
#             if f == 1:
#                 up = Identity()
#                 proj = GlobalConvolutionV5(c, out_dim, large_kernel, small_kernel)
#             else:
#                 proj = GlobalConvolutionV5(c, out_dim * f * f, large_kernel, small_kernel)
#                 up = nn.PixelShuffle(f)
#                 # up = nn.ConvTranspose2d(
#                 #     out_dim, out_dim, f * 2, stride=f, padding=f // 2,
#                 #     output_padding=0, groups=out_dim, bias=False)
#                 # fill_up_weights(up)
#             setattr(self, 'proj_' + str(i), proj)
#             setattr(self, 'up_' + str(i), up)

#         for i in range(1, len(channels)):
#             node = nn.Sequential(
#                 nn.Conv2d(out_dim * 2, out_dim,
#                           kernel_size=1, stride=1,
#                           padding=0, bias=False),
#                 BatchNorm(out_dim),
#                 nn.ReLU(inplace=True))
#             setattr(self, 'node_' + str(i), node)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, BatchNorm):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def forward(self, layers):
#         assert len(self.channels) == len(layers), \
#             '{} vs {} layers'.format(len(self.channels), len(layers))
#         layers = list(layers)
#         for i, l in enumerate(layers):
#             upsample = getattr(self, 'up_' + str(i))
#             project = getattr(self, 'proj_' + str(i))
#             layers[i] = upsample(project(l))
#         x = layers[0]
#         y = []
#         for i in range(1, len(layers)):
#             node = getattr(self, 'node_' + str(i))
#             x = node(torch.cat([x, layers[i]], 1))
#             y.append(x)
#         return x, y

class GlobalConvolutionV5(nn.Module):

    def __init__(self, c, o, lk, sk):
        super(GlobalConvolutionV5, self).__init__()
        self.gcl = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1),
            nn.Conv2d(c, c, kernel_size=(1, lk), padding=(0, lk // 2), stride=1))
        self.gcr = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=(1, lk), padding=(0, lk // 2), stride=1),
            nn.Conv2d(c, c, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1))

        self.proj = nn.Sequential(
            nn.Conv2d(c, o, kernel_size=1, stride=1,
                      padding=0, bias=False),
            BatchNorm(o),
            nn.ReLU(inplace=True))

        self.br = BoundaryRefinement(o, sk)

    def forward(self, x):
        x = self.gcl(x) + self.gcr(x)
        x = self.proj(x)
        x = self.br(x)
        return x



class IDAUpV5(nn.Module):

    def __init__(self, large_kernel, small_kernel, out_dim, channels, up_factors):
        super(IDAUpV5, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = GlobalConvolutionV5(c, out_dim, large_kernel, small_kernel)
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
            node = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim,
                          kernel_size=1, stride=1,
                          padding=0, bias=False),
                BatchNorm(out_dim),
                nn.ReLU(inplace=True))
            setattr(self, 'node_' + str(i), node)


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


class GlobalConvolutionV6(nn.Module):

    def __init__(self, c, o, lk, sk):
        super(GlobalConvolutionV6, self).__init__()
        self.gcl = nn.Sequential(
            nn.Conv2d(c, o, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1),
            nn.Conv2d(o, o, kernel_size=(1, lk), padding=(0, lk // 2), stride=1))
        self.gcr = nn.Sequential(
            nn.Conv2d(c, o, kernel_size=(1, lk), padding=(0, lk // 2), stride=1),
            nn.Conv2d(o, o, kernel_size=(lk, 1), padding=(lk // 2, 0), stride=1))

        self.proj = nn.Sequential(
            nn.Conv2d(o, o, kernel_size=sk, stride=1,
                      padding=sk // 2, bias=False),
            BatchNorm(o),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.gcl(x) + self.gcr(x)
        x = self.proj(x)
        return x



class IDAUpV6(nn.Module):

    def __init__(self, large_kernel, small_kernel, out_dim, channels, up_factors):
        super(IDAUpV6, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = GlobalConvolutionV6(c, out_dim, large_kernel, small_kernel)
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
            setattr(self, 'node_' + str(i), BoundaryRefinement(out_dim, small_kernel))


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
            x = node(x + layers[i])
            y.append(x)
        return x, y



class DLAUp(nn.Module):

    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None, 
                 version='v1', large_kernel=15, small_kernel=3):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            if version == 'v1':
                setattr(self, 'ida_{}'.format(i),
                        IDAUpV1(large_kernel, small_kernel, channels[j], in_channels[j:],
                              scales[j:] // scales[j]))
            elif version == 'v2':
                setattr(self, 'ida_{}'.format(i),
                        IDAUpV2(large_kernel, small_kernel, channels[j], in_channels[j:],
                              scales[j:] // scales[j]))
            elif version == 'v3':
                setattr(self, 'ida_{}'.format(i),
                        IDAUpV3(large_kernel, small_kernel, channels[j], in_channels[j:],
                              scales[j:] // scales[j]))
            elif version == 'v4':
                setattr(self, 'ida_{}'.format(i),
                        IDAUpV4(large_kernel, small_kernel, channels[j], in_channels[j:],
                              scales[j:] // scales[j]))
            elif version == 'v5':
                setattr(self, 'ida_{}'.format(i),
                        IDAUpV5(large_kernel, small_kernel, channels[j], in_channels[j:],
                              scales[j:] // scales[j]))
            elif version == 'v6':
                setattr(self, 'ida_{}'.format(i),
                        IDAUpV6(large_kernel, small_kernel, channels[j], in_channels[j:],
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

    def __init__(self, base_name, classes, pretrained_base, down_ratio, 
                 version, large_kernel=15, small_kernel=3):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.base = dla.__dict__[base_name](pretrained=pretrained_base,
                                            return_levels=True)
        channels = self.base.channels
        up_factor = 2 ** self.first_level
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales, version=version, 
                            large_kernel=large_kernel, small_kernel=small_kernel)
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
        # x = self.up(x)
        return x

def dla34up_gc_bn_v1(classes, pretrained_base=None, **kwargs):
    return DLASeg('dla34', classes, pretrained_base=pretrained_base, version='v1', **kwargs)

def dla34up_gc_bn_v2(classes, pretrained_base=None, **kwargs):
    return DLASeg('dla34', classes, pretrained_base=pretrained_base, version='v2', **kwargs)

def dla34up_gc_bn_v3(classes, pretrained_base=None, **kwargs):
    return DLASeg('dla34', classes, pretrained_base=pretrained_base, version='v3', **kwargs)

def dla34up_gc_bn_v4(classes, pretrained_base=None, **kwargs):
    return DLASeg('dla34', classes, pretrained_base=pretrained_base, version='v4', **kwargs)

def dla34up_gc_bn_v5(classes, pretrained_base=None, **kwargs):
    return DLASeg('dla34', classes, pretrained_base=pretrained_base, version='v5', **kwargs)

def dla34up_gc_bn_v6(classes, pretrained_base=None, **kwargs):
    return DLASeg('dla34', classes, pretrained_base=pretrained_base, version='v6', **kwargs)

# def dla60up_gc_bn(classes, pretrained_base=None, **kwargs):
#     model = DLASeg('dla60', classes, pretrained_base=pretrained_base, **kwargs)
#     return model


# def dla102up_gc_bn(classes, pretrained_base=None, **kwargs):
#     model = DLASeg('dla102', classes,
#                    pretrained_base=pretrained_base, **kwargs)
#     return model


# def dla169up_gc_bn(classes, pretrained_base=None, **kwargs):
#     model = DLASeg('dla169', classes,
#                    pretrained_base=pretrained_base, **kwargs)
#     return model
