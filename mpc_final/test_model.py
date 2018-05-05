from model import *

net = ConvLSTMMulti(3, True, 3, True, True, 'drn_d_22', 4, 1024, 64)
imgs = Variable(torch.randn(1,15, 9, 256, 256))
act = Variable(torch.randn(1, 15, 3))
y = net(imgs, act)
pdb.set_trace() 
