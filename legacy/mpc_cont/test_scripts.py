from model import *

x = Variable(torch.randn(3,4,3,256,256)).cuda().contiguous()
act = Variable(torch.randn(3,4,9)).cuda().contiguous()
sp = Variable(torch.randn(3,4,2)).cuda().contiguous()
net = ConvLSTMMulti(3,3,9).cuda()
feat = net.get_feature(x)
print(x.size(), feat.size())
#outs = net(x, act, sp)
#speed = Variable(torch.randn(1,2)).cuda()
#outs = net.sample_action(x[0,0,:,:,:].view(1,3,256,256), 0, speed=speed, num_time=10)
