import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

ce = nn.CrossEntropyLoss(reduce=False)
x = torch.randn((1, 4, 256,256)).view(1,
t = np.random.randint(4, size=(256*256,)).reshape((256,256, 1))
t = torch.from_numpy(t)
loss = ce(x, t)
import pdb
pdb.set_trace()
