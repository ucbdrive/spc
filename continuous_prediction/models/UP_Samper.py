import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from utils import fill_up_weights

class UP_Samper(nn.Module):
    def __init__(self):
        super(UP_Samper, self).__init__()
        self.up = nn.ConvTranspose2d(4, 4, 16, stride=8, padding=4,
                                output_padding=0, groups=4, bias=False)
        fill_up_weights(self.up)

    def forward(self, feature_map):
        y = self.up(feature_map)
        return y