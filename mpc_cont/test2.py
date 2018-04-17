from model import *
from mpc_utils import *
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim

def get_e_i(s):
    s = s.split('_')
    e = int(s[0][7:])
    s = s[1].split('.')
    i = int(s[0][5:])
    return e * 10000 + i

train_net = ConvLSTMMulti(3,3, 3, True, multi_info=False, with_posinfo=False, use_pos_class=False, with_speed=True, with_pos=True, frame_history_len=3)
    
mpc_buffer = MPCBuffer(10000, 3, 12, 3)
use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor
    train_net = train_net.cuda()
else:
    dtype = torch.FloatTensor
params = [param for param in train_net.parameters() if param.requires_grad]
optimizer = optim.Adam(params, lr=0.001)

dataset = sorted(os.listdir('dataset'), key = get_e_i)
for data_path in dataset:
    data = np.load(os.path.join('dataset', data_path))
    obs = data['obs']
    action = data['action']
    done = data['done']
    coll_flag = data['coll_flag']
    offroad_flag = data['offroad_flag']
    speed = data['speed']
    angle = data['angle']
    trackPos = data['trackPos']
    ret = mpc_buffer.store_frame(obs.transpose(1,2,0))
    mpc_buffer.store_effect(ret, action, done, coll_flag, offroad_flag, speed, angle, trackPos)

for tt in range(1):
    x = list(mpc_buffer.sample(5))
    act_batch     = Variable(torch.from_numpy(x[0]), requires_grad=False).type(dtype)
    coll_batch    = Variable(torch.from_numpy(x[1]), requires_grad=False).type(dtype)
    speed_batch   = Variable(torch.from_numpy(x[2]), requires_grad=False).type(dtype)
    offroad_batch = Variable(torch.from_numpy(x[3]), requires_grad=False).type(dtype)
    pos_batch     = Variable(torch.from_numpy(x[7])[:, 0, :], requires_grad=False).type(dtype)
    img_batch     = Variable(torch.from_numpy(x[5])[:, 0, :, :, :], requires_grad=False).type(dtype)

    losses = 0
    for j in range(5):
        speed_np = np.zeros((1, 2))
        speed_np[0, 0] = speed_batch[j][0][0]
        speed_np = Variable(torch.from_numpy(speed_np), requires_grad=False).type(dtype)
        action = train_net.sample_action(img_batch, speed=speed_np, pos=pos_batch, target_coll=coll_batch, target_off=offroad_batch, num_time=12)
        losses += nn.L2Loss(action, act_batch[j])
    optimizer.zero_grad()
    losses.backward()
    optimizer.step()


'''
for i in range(15):
    ret = mpc_buffer.store_frame(np.random.rand(256, 256, 3))
    mpc_buffer.store_effect(ret, np.random.rand(3), 0, 0, 0, 1, 0, 0)

x = list(mpc_buffer.sample(3))
act_batch     = Variable(torch.from_numpy(x[0]), requires_grad=False).type(dtype)
coll_batch    = Variable(torch.from_numpy(x[1]), requires_grad=False).type(dtype)
speed_batch   = Variable(torch.from_numpy(x[2]), requires_grad=False).type(dtype)
offroad_batch = Variable(torch.from_numpy(x[3]), requires_grad=False).type(dtype)
pos_batch     = Variable(torch.from_numpy(x[7]), requires_grad=False).type(dtype)
img_batch     = Variable(torch.from_numpy(x[5]), requires_grad=False).type(dtype)/255.0
print(act_batch.size())
print(coll_batch.size())
print(speed_batch.size())
print(offroad_batch.size())
print(pos_batch.size())
print(img_batch.size())
'''