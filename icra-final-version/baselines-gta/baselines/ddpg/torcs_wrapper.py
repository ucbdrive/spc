from baselines.ddpg.utils import *
import cv2
import math
import numpy as np
import pdb
import copy
from gym.spaces import Box
from baselines.ddpg.model import *
import torch
from args import init_parser
import argparse
from torch.autograd import Variable
import pdb

def naive_driver(info, continuous):
    if info['angle'] > 0.5 or (info['trackPos'] < -1 and info['angle'] > 0):
        return np.array([1.0, 0.1]) if continuous else 0
    elif info['angle'] < -0.5 or (info['trackPos'] > 3 and info['angle'] < 0):
        return np.array([1.0, -0.1]) if continuous else 2
    return np.array([1.0, 0.0]) if continuous else 1


class dummy():
    def __init__(self):
        self.pretrained = None
        self.drn_model = 'dla46x_c'
        self.classes = 4
        self.frame_history_len = 3
        self.pred_step = 10
        self.num_total_act = 2
        self.one_hot = True
        self.use_collision = True
        self.use_offroad = True
        self.use_distance = True
        self.use_seg = True
        self.lstm2 = True
        self.hidden_dim = 1024
        self.info_dim = 32
        self.use_lstm = False
        self.use_otherlane = False
        self.use_pos = False
        self.use_angle = False
        self.use_speed = False
        self.use_xyz = False
        self.use_dqn = False
        self.num_dqn_action = 10
		
class TorcsWrapper:
    def __init__(self, env, imsize=(84, 84), random_reset = True, continuous = True):
        self.dummy = dummy()
        self.segnet = ConvLSTMMulti(self.dummy)
        self.segnet.eval()
        state_dict = torch.load('/media/shared/xiangyuchen/spn-icra/baselines5/baselines/ddpg/trained_model_torcs.pth')
        self.segnet.load_state_dict(state_dict)
        self.segnet = torch.nn.DataParallel(self.segnet).cuda().float()
        self.env = env
        self.imsize = imsize
        self.random_reset = random_reset
        self.continuous = continuous
        self.doneCond = DoneCondition(10)
        self.epi_len = 0
        self.coll_cnt = 0
        self.done_cnt = 0
        self.last_done = np.array([467, 12.54])
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)

    def process_frame(self, obs):
        obs = cv2.resize(obs, (256, 256)).reshape((256, 256, 3)).transpose(2,0,1)
        obs = Variable(torch.from_numpy(obs)).unsqueeze(0).cuda().float()
        seg = self.segnet(obs).data.cpu().numpy().squeeze(0).transpose(1,2,0)
        seg = seg*255
        seg = cv2.resize(seg, (84, 84), interpolation = cv2.INTER_NEAREST)
        return seg

    def reset(self, rand_reset=True, restart=False):
        obs = self.env.reset()
        _, _, _, info = self.env.step(naive_driver(self.env.get_info(), self.continuous))
        current_pos = np.array(info['pos'][:2])
        dist = np.sqrt(np.sum((current_pos-self.last_done)**2.0))
        while dist > 100 and not restart:
            obs, _, _, info = self.env.step(naive_driver(self.env.get_info(), self.continuous))
            current_pos = np.array(info['pos'][:2])
            dist = np.sqrt(np.sum((current_pos-self.last_done)**2.0))
        info = self.env.get_info()
        off_flag = int(info['trackPos'] >=3 or info['trackPos'] <= -1)
        coll_flag = int(abs(info['trackPos']) > 7.0)
        info['off_flag'] = off_flag
        info['coll_flag'] = coll_flag
        self.doneCond = DoneCondition(10)
        self.epi_len = 0
        self.coll_cnt = 0
        self.done_cnt = 0
        obs = self.process_frame(obs)
        return obs, info

    def _seed(self, seed):
        return

    def seed(self, seed):
        return

    def step(self, action):
        if self.continuous:
            real_action = copy.deepcopy(action)
            real_action[0] = real_action[0] * 0.5 + 0.5
            # if real_action[1] > 0.1:
            #     real_action[1] = 0.1
            # elif real_action[1] < -0.1:
            #     real_action[1] = -0.1
            real_action[1] = real_action[1] * 0.1
        else:
            real_action = copy.deepcopy(action)
        self.epi_len += 1
        obs, reward, real_done, info = self.env.step(real_action)
        dist_this = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])))
        reward_with_pos = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])) - np.abs(info['trackPos']) / 9.0) / 40.0
        reward_without_pos = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle']))) / 40.0
        done = self.doneCond.isdone(info['trackPos'], dist_this, info['pos'], info['angle']) or self.epi_len > 1000 or self.done_cnt > 30 
        # off_flag = int(info['trackPos'] >= 7 or info['trackPos'] <= -4)
        # coll_flag = int(info['trackPos'] >= 20 or info['trackPos'] <= -6.5)
        off_flag = int(info['trackPos'] >= 3 or info['trackPos'] <= -1)
        coll_flag = int(abs(info['trackPos']) > 7.0)
        if coll_flag:
            self.coll_cnt += 1
        else:
            self.coll_cnt = 0
        if info['trackPos'] <= -4.0 or info['trackPos'] >= 5.0:
            self.done_cnt += 1
        else:
            self.done_cnt = 0
        obs = self.process_frame(obs)#cv2.resize(obs, self.imsize)
        reward = {}
        reward['with_pos'] = reward_with_pos
        reward['without_pos'] = reward_without_pos
        info['off_flag'] = off_flag
        info['coll_flag'] = coll_flag
        reward2 = reward_without_pos - off_flag * 0.1 - coll_flag * 1.0
        reward['true_reward'] = reward2
        if done:
            self.last_done = np.array(info['pos'][:2])
        return obs, reward, done, info

    def close(self):
        self.env.close()
