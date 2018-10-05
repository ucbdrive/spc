import cv2
import math
import numpy as np
import pdb
import math
import copy
from gym.spaces import Box
import sys
sys.path.append('/media/xinleipan/data/git/pyTORCS')
sys.path.append('/media/xinleipan/data/git/pyTORCS/py_TORCS')
from py_TORCS import torcs_envs

class DoneCondition:
    def __init__(self, size):
        self.size = size
        self.off_cnt = 0
        self.pos = []

    def isdone(self, pos, dist, posxyz, angle):
        if pos <= -6.2 and dist < 0:
            self.off_cnt += 1
        elif pos > -6.2 or dist > 0:
            self.off_cnt = 0
        if self.off_cnt > self.size:
            return True
        if np.abs(pos) >= 21.0:
            return True
        self.pos.append(list(posxyz))
        real_pos = np.concatenate(self.pos[-100:])
        real_pos = real_pos.reshape(-1,3)
        std = np.sum(np.std(real_pos, 0))
        if std < 2.0 and len(self.pos) > 100:
            self.pos = []
            return True
        return False 

class ObsBuffer:
    def __init__(self, frame_history_len=3):
        self.frame_history_len = frame_history_len
        self.last_obs_all = []

    def store_frame(self, frame):
        obs_np = frame
        if len(self.last_obs_all) < self.frame_history_len:
            self.last_obs_all = []
            for ii in range(self.frame_history_len):
                self.last_obs_all.append(obs_np)
        else:
            self.last_obs_all = self.last_obs_all[1:] + [obs_np]
        return np.concatenate(self.last_obs_all, 2)

    def clear(self):
        self.last_obs_all = []
        return

class TorcsWrapper:
    def __init__(self, env, imsize=(84, 84), low_dim = False, use_pos=False, env_id=0, use_collision=False):
        self.env = env# torcs_envs(num = 1, game_config = '/media/xinleipan/data/git/pyTORCS/game_config/michigan.xml', mkey_start=817+105, \
                      #        screen_id = 160+105, isServer=True, continuous = True, resize = True).get_envs()[0]
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        if low_dim:
            self.observation_space = Box(low=-1, high=1, shape=(2,), dtype=np.uint8)
        else:
            self.observation_space = Box(low=0, high=255, shape=(84, 84, 9), dtype=np.uint8)
        self.reward_range = (-math.inf, math.inf)
        self.use_pos = use_pos
        self.use_collision = use_collision
        
        self.imsize = imsize
        self.doneCond = DoneCondition(30)
        self.epi_len = 0
        self.low_dim = low_dim
        self.coll_cnt = 0
        self.obsbuffer = ObsBuffer()
        self.target_speed = np.random.uniform(15, 35)

    def get_state(self):
        state = np.zeros(2)
        info = self.env.get_info()
        state[0] = np.clip(info['trackPos'] / 9.0, -1, 1)
        state[1] = np.clip(info['angle'] / math.pi * 2, -1, 1)
        return state

    def seed(self, seed):
        pass

    def reset(self):
        obs = self.env.reset()
        self.doneCond = DoneCondition(30)
        self.target_speed = np.random.uniform(15, 35)
        self.epi_len = 0
        self.coll_cnt = 0
        if self.low_dim:
            return self.get_state()
        else:   
            obs = cv2.resize(obs, self.imsize)
            self.obsbuffer.clear()
            return self.obsbuffer.store_frame(obs)
         
    def step(self, action):
        real_action = copy.deepcopy(action)
        real_action[0] = real_action[0] * 0.5 + 0.5
        #if abs(real_action[1]) < 0.8:
        #    real_action[1] = 0
        self.epi_len += 1
        obs, reward, done, info = self.env.step(real_action)
        with open('speed_log.txt', 'a') as f:
            f.write('target %0.4f speed %0.4f\n' % (self.target_speed, info["speed"]))
        if self.low_dim:
            obs = self.get_state()
        else:
            obs = cv2.resize(obs, self.imsize)
            obs = self.obsbuffer.store_frame(obs)
        dist_this = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])))
        reward_with_pos = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])) - np.abs(info['trackPos']) / 9.0) / 40.0
        reward_without_pos = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle']))) / 40.0
        done = done or self.doneCond.isdone(info['trackPos'], dist_this, info['pos'], info['angle']) or self.epi_len > 1000
        
        off_flag = int(info['trackPos'] >= 3 or info['trackPos'] <= -1)
        coll_flag = int(abs(info['trackPos'] + info['angle']) > 6.5 or reward <= -2.5 or (info['damage'] > 0 and info['angle'] > 0.5 and info['speed'] < 15))
        reward = -0.02 * (info['speed'] - self.target_speed) ** 2
        info['off_flag'] = off_flag
        info['coll_flag'] = coll_flag
        return obs, reward, done, info

    def close(self):
        self.env.close()
