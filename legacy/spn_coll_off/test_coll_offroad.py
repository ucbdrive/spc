import sys
sys.path.append('/media/shared/pyTORCS/py_TORCS')
sys.path.append('/media/shared/pyTORCS/')
from py_TORCS import torcs_envs
import torch
torch.manual_seed(0)
import random
random.seed(0)
from mpc_utils import *
import torch.nn as nn
import pdb
import numpy as np
np.random.seed(0)
from torcs_wrapper import *
from torch.utils.data import Dataset, DataLoader
import os
import pickle as pkl
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import pdb
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 32, 1, stride=1, padding=0)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv4(x), kernel_size=2, stride=2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class COData(Dataset):
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir

    def __len__(self):
        files = os.listdir(self.data_dir)
        return len(files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, str(idx).zfill(9)+'.pkl')
        label_name = os.path.join(self.label_dir, str(idx).zfill(9)+'.pkl')
        img = pkl.load(open(img_name, 'rb')).reshape((256, 256))#.unsqueeze(0)
        img = np.expand_dims(img, axis=0)
        label = pkl.load(open(label_name,'rb')).reshape((-1))
        return img, label
        
if __name__ == '__main__':
    game_config = '/media/shared/pyTORCS/game_config/michigan.xml'
    id = -1
    envs = torcs_envs(num = 1, game_config = game_config, mkey_start = 817 + id, screen_id = 160 + id,
                      isServer = 1, continuous = True, resize = True)
    env = envs.get_envs()[0]
    env = TorcsWrapper(env, random_reset = False, continuous = True)
    model = Model().cuda().float()
    if os.path.isdir('data') == False:
        os.mkdir('data')
    if os.path.isdir('label') == False:
        os.mkdir('label')
    data = COData('data', 'label')
    obs = env.reset()
    opt = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    loader = DataLoader(dataset=data, batch_size=32, shuffle=True, num_workers=0)
    for i in range(100000):
        if random.random() < 1 - i * 0.00001:
            action = np.random.rand(2)*2-1
        else:
            action = naive_driver(info, True)
        _, _, done, info = env.step(action)
        seg = env.env.get_segmentation().reshape((256, 256, 1))
        pkl.dump(seg, open('data/'+str(i).zfill(9)+'.pkl', 'wb'))
        off = np.array([int(info['off_flag'])])
        pkl.dump(off, open('label/'+str(i).zfill(9)+'.pkl', 'wb'))
        # print(info['pos'], info['trackPos'], info['off_flag'])
        if done:
            _ = env.reset()
            tn, fp, fn, tp = 0, 0, 0, 0
            for i_batch, sample_batched in enumerate(loader):
                img = Variable(sample_batched[0].cuda()).float()
                label = Variable(sample_batched[1].cuda()).long().squeeze(-1)
                pred = model(img)
                loss = nn.CrossEntropyLoss()(pred, label)
                opt.zero_grad()
                loss.backward()
                opt.step()
                pred_np = pred.data.cpu().numpy().reshape(-1, 2)
                label_np = label.data.cpu().numpy().reshape(-1)
                tn1, fp1, fn1, tp1 = confusion_matrix(np.argmax(pred_np, axis=-1), label_np, labels=[0, 1]).ravel()
                tn += tn1
                fp += fp1
                fn += fn1
                tp += tp1
            acc = (tn + tp)/ (tn + fp + fn + tp) * 100.0
            print(' accuracy is %0.2f%%' % acc) 
