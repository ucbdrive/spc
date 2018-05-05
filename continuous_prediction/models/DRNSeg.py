import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import models.drn as drn

class DRNSeg(nn.Module):
    def __init__(self, model_name, 
                classes=4,
                pretrained=False):
        super(DRNSeg, self).__init__()
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000)
        self.base = nn.Sequential(*list(model.children())[:-2])

        self.seg = nn.Conv2d(model.out_dim, classes,
                             kernel_size=1, bias=True)
        m = self.seg
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        self.seg.weight.data.normal_(0, math.sqrt(2. / n))
        self.seg.bias.data.zero_()

    def forward(self, x):
        x = self.base(x)
        feature_map = self.seg(x)
        return feature_map
