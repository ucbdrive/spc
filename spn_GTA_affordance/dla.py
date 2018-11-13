import math

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo

BatchNorm = nn.BatchNorm2d

webroot = 'https://tigress-web.princeton.edu/~fy/dla/public_models/imagenet/'

model_urls = {
    'dla34': webroot + 'dla34-6ba26179.pth',
    'dla46_c': webroot + 'dla46_c-37675969.pth',
    'dla46x_c': webroot + 'dla46x_c-893b8958.pth',
    'dla60x_c': webroot + 'dla60x_c-5de0ad33.pth',
    'dla60': webroot + 'dla60-e2f4df06.pth',
    'dla60x': webroot + 'dla60x-3062c917.pth',
    'dla102': webroot + 'dla102-d94d9790.pth',
    'dla102x': webroot + 'dla102x-ad62be81.pth',
    'dla102x2': webroot + 'dla102x2-262837b6.pth',
    'dla169': webroot + 'dla169-0914e092.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Identity(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Identity, self).__init__()
        assert in_channels == out_channels
        pass

    def forward(self, x):
        return x


class BasicTree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, linear_root=True, root_dim=0):
        super(BasicTree, self).__init__()
        # self.root = None
        # if levels == 0:
        #     self.tree1 = self.tree2 = None
        #     self.root = block(in_channels, out_channels, stride)
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride)
            self.tree2 = block(out_channels, out_channels, 1)
        else:
            self.tree1 = BasicTree(levels - 1, block, in_channels,
                                   out_channels, stride, root_dim=0)
            self.tree2 = BasicTree(levels - 1, block, out_channels,
                                   out_channels,
                                   root_dim=root_dim + out_channels)
        if levels == 1:
            # root_in_dim = out_channels * (levels + 1)
            # root_in_dim += in_channels
            root_layers = [nn.Conv2d(root_dim, out_channels,
                                     kernel_size=1, stride=1, bias=False),
                           BatchNorm(out_channels)]
            if not linear_root:
                root_layers.append(nn.ReLU(inplace=True))
            self.root = nn.Sequential(*root_layers)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        assert self.downsample is None or x.size(2) % 2 == 0
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = torch.cat([x1, x2], 1)
            for i in range(len(children)):
                x = torch.cat([x, children[i]],1)
            x = self.root(x)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class RRoot(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 scale_residual=False):
        super(RRoot, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.scale_residual = scale_residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.scale_residual:
            for c in children:
                if c.size(1) == x.size(1):
                    x += c
        else:
            x += children[0]
        x = self.relu(x)

        return x


class RTree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 scale_residual=False, dilation=1):
        super(RTree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = RTree(levels - 1, block, in_channels, out_channels,
                               stride, root_dim=0,
                               root_kernel_size=root_kernel_size,
                               scale_residual=scale_residual,
                               dilation=dilation)
            self.tree2 = RTree(levels - 1, block, out_channels, out_channels,
                               root_dim=root_dim + out_channels,
                               root_kernel_size=root_kernel_size,
                               scale_residual=scale_residual,
                               dilation=dilation)
        if levels == 1:
            self.root = RRoot(root_dim, out_channels, root_kernel_size)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)

        linear_root = False
        if not residual_root:
            self.level2 = BasicTree(
                levels[2], block, channels[1], channels[2], 2,
                level_root=False, linear_root=linear_root)
            self.level3 = BasicTree(
                levels[3], block, channels[2], channels[3], 2,
                level_root=True, linear_root=linear_root)
            self.level4 = BasicTree(
                levels[4], block, channels[3], channels[4], 2,
                level_root=True, linear_root=linear_root)
            self.level5 = BasicTree(
                levels[5], block, channels[4], channels[5], 2,
                level_root=True, linear_root=linear_root)
        else:
            self.level2 = RTree(levels[2], block, channels[1], channels[2], 2,
                                level_root=False,
                                scale_residual=False)
            self.level3 = RTree(levels[3], block, channels[2], channels[3], 2,
                                level_root=True, scale_residual=False)
            self.level4 = RTree(levels[4], block, channels[3], channels[4], 2,
                                level_root=True, scale_residual=False)
            self.level5 = RTree(levels[5], block, channels[4], channels[5], 2,
                                level_root=True, scale_residual=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            # x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x


def dla34(pretrained=False, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dla34']))
    return model


def dla46_c(pretrained=False, **kwargs):  # DLA-46-C
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=Bottleneck, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dla46_c']))
    return model


def dla46x_c(pretrained=False, **kwargs):  # DLA-X-46-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dla46x_c']))
    return model


def dla60x_c(pretrained=False, **kwargs):  # DLA-X-60-C
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 64, 64, 128, 256],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dla60x_c']))
    return model


def dla60(pretrained=False, **kwargs):  # DLA-60
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dla60']))
    return model


def dla60x(pretrained=False, **kwargs):  # DLA-X-60
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dla60x']))
    return model


def dla102(pretrained=False, **kwargs):  # DLA-102
    Bottleneck.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dla102']))
    return model


def dla102x(pretrained=False, **kwargs):  # DLA-X-102
    BottleneckX.expansion = 2
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dla102x']))
    return model


def dla102x2(pretrained=False, **kwargs):  # DLA-X-102 64
    BottleneckX.cardinality = 64
    model = DLA([1, 1, 1, 3, 4, 1], [16, 32, 128, 256, 512, 1024],
                block=BottleneckX, residual_root=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dla102x2']))
    return model


def dla169(pretrained=False, **kwargs):  # DLA-169
    Bottleneck.expansion = 2
    model = DLA([1, 1, 2, 3, 5, 1], [16, 32, 128, 256, 512, 1024],
                block=Bottleneck, residual_root=True)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dla169']))
    return model

if __name__ == '__main__':
    net = dla46x_c(pretrained=False, return_levels=True)
    x = torch.rand(2, 3, 256, 256)
    y = net(x)
    print(len(y))
    for t in y:
        print(t.shape)
