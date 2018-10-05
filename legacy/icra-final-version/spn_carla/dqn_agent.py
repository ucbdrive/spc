import torch
import torch.optim as optim
from dqn_utils import *
import os
import numpy as np
import cv2
import random
from model import *
from torch.autograd import Variable

class DQNAgent:
    def __init__(self, args, exploration=None, save_path=None):
        frame_history_len = args.frame_history_len
        num_actions = args.num_dqn_action
        lr = args.lr
        self.dqn_net = atari_model(3 * frame_history_len, num_actions, frame_history_len).cuda().float().train()
        self.target_q_net = atari_model(3 * frame_history_len, num_actions, frame_history_len).cuda().float().eval()
        self.optimizer = optim.Adam(self.dqn_net.parameters(), lr = lr, amsgrad=True)
        self.replay_buffer = ReplayBuffer(100000, frame_history_len)
        self.frame_history_len = frame_history_len
        self.num_actions = num_actions
        self.ret = 0
        self.exploration = exploration
        self.num_param_updates = 0
        self.save_path = save_path
        self.target_update_freq = args.target_update_freq
        if os.path.isdir(os.path.join(save_path, 'dqn')) == False:
            os.mkdir(os.path.join(save_path, 'dqn'))
        if os.path.isdir(os.path.join(save_path, 'dqn', 'model')) == False:
            os.mkdir(os.path.join(save_path, 'dqn', 'model'))
        if os.path.isdir(os.path.join(save_path, 'dqn', 'optimizer')) == False:
            os.mkdir(os.path.join(save_path, 'dqn', 'optimizer'))
        self.model_path = os.path.join(save_path, 'dqn', 'model')
        self.optim_path = os.path.join(save_path, 'dqn', 'optimizer')

    def load_model(self, model_name=None):
        model_path, optim_path = self.model_path, self.optim_path
        file_list = sorted(os.listdir(model_path))
        if len(file_list) == 0:
            print('no model to resume!')
        else:
            file_name = os.path.join(model_path, file_list[-1])
            if model_name is None:
                model_name = file_name
            else:
                model_name = os.path.join(model_path, model_name)
            self.dqn_net.load_state_dict(torch.load(os.path.join(model_name)))
            self.target_q_net.load_state_dict(torch.load(os.path.join(model_name)))
        
        optim_list = sorted(os.listdir(optim_path))
        if len(optim_list) == 0:
            print('no optimizer to resume!')
        else:
            optim_name = os.path.join(optim_path, optim_list[-1])
            self.optimizer.load_state_dict(torch.load(os.path.join(optim_name)))
        
    def sample_action(self, obs, t):
        self.ret = self.replay_buffer.store_frame(cv2.resize(obs, (84, 84)))
        dqn_obs = self.replay_buffer.encode_recent_observation()
        sample = random.random()
        eps_threshold = self.exploration.value(t)
        if sample > eps_threshold:
            with torch.no_grad():
                dqn_obs = torch.from_numpy(dqn_obs).cuda().float().unsqueeze(0) / 255.0
                action = int(self.dqn_net(Variable(dqn_obs)).data.max(1)[1].cpu().numpy())
        else:
            action = random.randrange(self.num_actions)
        return int(action)

    def store_effect(self, action, reward, done):
        self.replay_buffer.store_effect(self.ret, action, reward, done)

    def train_model(self, batch_size, save_num=None):
        dtype = torch.cuda.FloatTensor
        obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = self.replay_buffer.sample(batch_size)
        obs_t_batch     = Variable(torch.from_numpy(obs_t_batch).type(dtype) / 255.0)
        act_batch       = Variable(torch.from_numpy(act_batch).long()).cuda()
        rew_batch       = Variable(torch.from_numpy(rew_batch)).cuda()
        obs_tp1_batch   = Variable(torch.from_numpy(obs_tp1_batch).type(dtype) / 255.0)
        done_mask       = Variable(torch.from_numpy(done_mask)).type(dtype)
       
        q_a_values = self.dqn_net(obs_t_batch).gather(1, act_batch.unsqueeze(1))
        q_a_values_tp1 = self.target_q_net(obs_tp1_batch).detach().max(1)[0]
        target_values = rew_batch + (0.99 * (1-done_mask) * q_a_values_tp1)
        dqn_loss = ((target_values.view(q_a_values.size()) - q_a_values)**2).mean()
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        self.num_param_updates += 1
        
        if self.num_param_updates % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.dqn_net.state_dict())
            torch.save(self.target_q_net.state_dict(), self.model_path+'/model_'+str(save_num)+'.pt')
            torch.save(self.optimizer.state_dict(), self.optim_path+'/optimizer.pt')
