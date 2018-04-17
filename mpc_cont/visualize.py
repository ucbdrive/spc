import torch
import torchvision.utils as vutils
import numpy as np
from model import *
from tensorboardX import SummaryWriter

model = ConvLSTMNet(3,3,9)
writer = SummaryWriter()
for name, param in model.named_parameters():
    writer.add_histogram(name, param.clone().cpu().data.numpy(), 1)
writer.export_scalars_to_json("./all_scalars.json")
writer.close()
