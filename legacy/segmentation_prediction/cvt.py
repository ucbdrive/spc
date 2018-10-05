import torch
from collections import OrderedDict
from model import DRNSeg, UP_Samper, PRED, FURTHER

state_dict = torch.load('pretrained_models/seg.dat')
model_state_dict = OrderedDict()
up_state_dict = OrderedDict()
for k, v in state_dict.items():
    print(k)
    if k[0:2] != 'up':
        model_state_dict[k] = v
    else:
        up_state_dict[k] = v
torch.save(model_state_dict, 'pretrained_models/model.dat')
torch.save(up_state_dict, 'pretrained_models/up.dat')