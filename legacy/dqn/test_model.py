from run_dqn_atari import atari_model
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb

dqn = atari_model(12,9,4)
x = Variable(torch.randn(1,12,256,256)).cuda()
dqn = dqn.cuda()
pdb.set_trace()
outs = dqn(x)

