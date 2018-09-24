import torch
import torch.optim as optim
from dqn_utils import *
import numpy as np
import cv2
import random
from model import *
from torch.autograd import Variable
import pdb
import os

class DQNAgent:
    def __init__(self, inc, frame_history_len=4, num_actions=11, lr=0.0001, exploration=None, save_path=None, change_model=False, img_h=0, img_w=0, device='cuda', buffer_size=100000):
        self.dqn_net = atari_model(inc * frame_history_len, num_actions, h=img_h, w=img_w, change=change_model).to(device).float().train()
        self.target_q_net = atari_model(inc * frame_history_len, num_actions, w=img_w, h=img_h, change=change_model).to(device).float().train() 
        self.optimizer = optim.Adam(self.dqn_net.parameters(), lr = lr, amsgrad=True)
        self.replay_buffer = ReplayBuffer(buffer_size, frame_history_len)
        self.frame_history_len = frame_history_len
        self.num_actions = num_actions
        self.ret = 0
        self.exploration = exploration
        self.num_param_updates = 0
        self.save_path = save_path
        self.target_update_freq = 4#
        self.device = device
        if os.path.isdir(save_path) == False:
            os.mkdir(save_path)
        if os.path.isdir(os.path.join(save_path, 'dqn')) == False:
            os.mkdir(os.path.join(save_path, 'dqn'))
        if os.path.isdir(os.path.join(save_path, 'dqn', 'model')) == False:
            os.mkdir(os.path.join(save_path, 'dqn', 'model'))
        if os.path.isdir(os.path.join(save_path, 'dqn', 'optimizer')) == False:
            os.mkdir(os.path.join(save_path, 'dqn', 'optimizer'))
        self.model_path = os.path.join(save_path, 'dqn', 'model')
        self.optim_path = os.path.join(save_path, 'dqn', 'optimizer')

    def load_model(self):
        num = 0
        try:
            model_path, optim_path = self.model_path, self.optim_path
            file_list = sorted(os.listdir(model_path))
            file_name = os.path.join(model_path, 'model_0.pt')
            state_dict = torch.load(os.path.join(file_name), map_location=self.device)
            self.dqn_net.load_state_dict(state_dict)
            self.target_q_net.load_state_dict(state_dict)
            optim_list = sorted(os.listdir(optim_path))
            optim_name = os.path.join(optim_path, 'optim_0.pt')
            num = 1000000
            print('load model ', file_name)
        except:
            num = 100
            print('Fail loading model state')
        try:
            self.optimizer.load_state_dict(torch.load(optim_name, map_location=self.device))
        except:
            print('Fail loading optimizer state')
        return num
     
    def sample_action(self, obs, t):
        if t == 1e7:
            dqn_obs = obs
        else:
            self.ret = self.replay_buffer.store_frame(obs)
            dqn_obs = self.replay_buffer.encode_recent_observation()
        sample = random.random()
        eps_threshold = self.exploration.value(t)
        if sample > eps_threshold:
            dqn_obs = torch.from_numpy(dqn_obs / 255.0).to(self.device).float().unsqueeze(0)
            action = int(self.dqn_net(Variable(dqn_obs, requires_grad=False)).data.max(1)[1].cpu().numpy())
        else:
            action = random.randrange(self.num_actions)
        return int(action)

    def store_effect(self, action, reward, done):
        self.replay_buffer.store_effect(self.ret, action, reward, done)

    def train_model(self, batch_size, save_num=None):
        if self.device == 'cuda':
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = self.replay_buffer.sample(batch_size)
        obs_t_batch     = Variable(torch.from_numpy(obs_t_batch).type(dtype))
        act_batch       = Variable(torch.from_numpy(act_batch).long()).to(self.device)
        rew_batch       = Variable(torch.from_numpy(rew_batch)).to(self.device)
        obs_tp1_batch   = Variable(torch.from_numpy(obs_tp1_batch).type(dtype))
        done_mask       = Variable(torch.from_numpy(done_mask)).type(dtype)
       
        q_a_values = self.dqn_net(obs_t_batch).gather(1, act_batch.unsqueeze(1))
        q_a_values_tp1 = self.target_q_net(obs_tp1_batch).detach().max(1)[0]
        target_values = rew_batch + (0.9 * (1-done_mask) * q_a_values_tp1)
        dqn_loss = nn.MSELoss()(q_a_values, target_values.view(q_a_values.size()))#((target_values.view(q_a_values.size()) - q_a_values)**2).mean()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        self.num_param_updates += 1
        
        if self.num_param_updates % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.dqn_net.state_dict())
            try:
                os.rename(self.model_path+'/model_0.pt', self.model_path+'/model_0.pt.old')
                os.rename(self.optim_path+'/optim_0.pt', self.model_path+'/optim_0.pt.old')
            except:
                pass
            
            torch.save(self.target_q_net.to('cpu').state_dict(), self.model_path+'/model_'+str(0)+'.pt')
            torch.save(self.optimizer.state_dict(), self.optim_path+'/optim_'+str(0)+'.pt')
            self.target_q_net.to(self.device)
            
            #print('checkpoint saved')
