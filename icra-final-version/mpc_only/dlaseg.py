import math
import numpy as np
import dla
from dla_up import DLAUp, Identity, fill_up_weights, BatchNorm
import torch.nn as nn
from utils import weights_init

class DLASeg(nn.Module):
    def __init__(self, args, down_ratio=4):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.base = dla.__dict__[args.drn_model](pretrained=args.pretrained, return_levels=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)
        self.fc = nn.Sequential(
            nn.Conv2d(channels[self.first_level], args.classes, kernel_size=1,
                      stride=1, padding=0, bias=True)
        )
        up_factor = 2 ** self.first_level
        if up_factor == 4:
            up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(args.classes, args.classes, 3, padding=1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(args.classes, args.classes, 5, padding=2)
            )
            up.apply(weights_init)
        elif up_factor > 1:
            up = nn.ConvTranspose2d(args.classes, args.classes, up_factor * 2,
                                    stride=up_factor, padding=up_factor // 2,
                                    output_padding=0, groups=args.classes,
                                    bias=False)
            fill_up_weights(up)
            up.weight.requires_grad = False
        else:
            up = Identity()
        self.up = up
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        for m in self.fc.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, mode=None):
        if mode == 'up':
            x = self.up(x)
            y = self.logsoftmax(x)
            x = self.softmax(x)
            return x, y
        else:
            x = self.base(x)
            x = self.dla_up(x[self.first_level:])
            x = self.fc(x)
            xx = x
            x = self.up(x)
            y = self.logsoftmax(x)
            x = self.softmax(x)
            return xx, x, y

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.dla_up.parameters():
            yield param
        for param in self.fc.parameters():
            yield param

if __name__ == '__main__':
    class dummy():
        def __init__(self):
            self.pretrained = None
            self.drn_model = 'dla46x_c'
            self.classes = 4
    a = DLASeg(dummy())