from model import *

net = ConvLSTMMulti(3, True, 4, True, True, 'drn_d_22', 4, 1024, 64)
imgs = Variable(torch.randn(1,15, 12, 256, 256))
act = Variable(torch.randn(1, 15, 3))
y = net(imgs, act, 15)
pdb.set_trace()
