from utils import *
import cv2
import math
import numpy as np
import pdb
import copy

def naive_driver(info, continuous):
    if info['angle'] > 0.5 or (info['trackPos'] < -1 and info['angle'] > 0):
        return np.array([1.0, 0.1]) if continuous else 0
    elif info['angle'] < -0.5 or (info['trackPos'] > 3 and info['angle'] < 0):
        return np.array([1.0, -0.1]) if continuous else 2
    return np.array([1.0, 0.0]) if continuous else 1

class TorcsWrapper:
    def __init__(self, env, imsize=(256, 256), random_reset = True, continuous = True):
        self.env = env
        self.imsize = imsize
        self.random_reset = random_reset
        self.continuous = continuous
        self.doneCond = DoneCondition(10)
        self.epi_len = 0
        self.coll_cnt = 0

    def reset(self, rand_reset=True):
        obs = self.env.reset()
        if self.random_reset and rand_reset:
            rand_step = np.random.randint(500)
            for i in range(rand_step):
                obs, _, _, _ = self.env.step(naive_driver(self.env.get_info(), self.continuous))
        info = self.env.get_info()
        off_flag = int(info['trackPos'] >=3 or info['trackPos'] <= -1)
        coll_flag = int(abs(info['trackPos']) > 7.0)
        info['off_flag'] = off_flag
        info['coll_flag'] = coll_flag
        self.doneCond = DoneCondition(10)
        self.epi_len = 0
        self.coll_cnt = 0
        return cv2.resize(obs, self.imsize), info
         
    def step(self, action):
        real_action = copy.deepcopy(action)
        real_action[0] = real_action[0] * 0.5 + 0.5
        self.epi_len += 1
        obs, reward, real_done, info = self.env.step(real_action)
        dist_this = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])))
        reward_with_pos = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])) - np.abs(info['trackPos']) / 9.0) / 40.0
        reward_without_pos = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle']))) / 40.0
        done = self.doneCond.isdone(info['trackPos'], dist_this, info['pos'], info['angle']) or self.epi_len > 1000
        
        off_flag = int(info['trackPos'] >= 3 or info['trackPos'] <= -1)
        coll_flag = int(abs(info['trackPos']) > 7.0)
        if coll_flag:
            self.coll_cnt += 1
        else:
            self.coll_cnt = 0
        obs = cv2.resize(obs, self.imsize)
        reward = {}
        reward['with_pos'] = reward_with_pos
        reward['without_pos'] = reward_without_pos
        info['off_flag'] = off_flag
        info['coll_flag'] = coll_flag
        return obs, reward, done, info

    def close(self):
        self.env.close()
