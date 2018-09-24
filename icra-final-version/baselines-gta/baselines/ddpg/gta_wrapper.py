import cv2
import math
import numpy as np
import pdb
import time
import copy
import torch
from gym.spaces import Box
from torch.autograd import Variable
import baselines.ddpg.sedla_up as dlaup
from baselines.ddpg.utils import draw_from_pred_gta

def naive_driver(info, continuous):
    if info['angle'] > 0.5 or (info['trackPos'] < -1 and info['angle'] > 0):
        return np.array([1.0, 0.1]) if continuous else 0
    elif info['angle'] < -0.5 or (info['trackPos'] > 3 and info['angle'] < 0):
        return np.array([1.0, -0.1]) if continuous else 2
    return np.array([1.0, 0.0]) if continuous else 1

class GTAWrapper:
    def __init__(self, env, imsize=(84, 84), random_reset = True, continuous = True):
        self.env = env
        self.imsize = imsize
        self.random_reset = random_reset
        self.continuous = continuous
        self.epi_len = 0
        self.coll_cnt = 0
        self.speed_list = []
        self.num_off = 0
        self.obs = None
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = Box(low=0, high=1, shape=(84, 84, 32), dtype=np.float32)
        self.model = dlaup.__dict__.get('sedla34up')(19, 'imagenet', down_ratio=2)
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.load_state_dict(torch.load('viper_dla34up_best.pth.tar')['state_dict'], strict=False)

    def get_segmentation(self):
    	# obs = Variable(torch.from_numpy(np.transpose(self.obs, (2, 0, 1)))).cuda().unsqueeze(0).float()
        #print('get segmentation')
        seg = self.model(self.obs)
        seg = seg.squeeze(0).data.cpu().numpy()
        seg = np.transpose(seg, (1, 2, 0))
        #print('get segmentation done')
        seg = cv2.resize(seg, self.imsize)
        # cv2.imwrite('seg.png', seg)
        # print('seg return')
        return seg

    def reset(self, rand_reset=True):
        self.env.done = True
        self.num_off = 0
        obs, info = self.env.reset()
        print(obs.shape)
        self.epi_len = 0
        self.coll_cnt = 0
        self.speed_list = []
        obs = obs[:-15,:,:]
        self.obs = cv2.resize(obs, (640, 352))
        self.obs = Variable(torch.from_numpy(np.transpose(self.obs, (2, 0, 1)))).cuda().unsqueeze(0).float()

        info['angle'] = 0.1
        try:
            info['pos'] = info['location']
        except:
            info['pos'] = [1, 420, 900]
        try:
            road_dir = np.arctan(info['roadinfo'][9]/info['roadinfo'][8])
            velo_dir = np.arctan(info['roadinfo'][6]/info['roadinfo'][5])
            road_velo_angle = road_dir-velo_dir
            angle_sign = 1 if road_velo_angle > 0 else -1
            road_velo_angle = min(np.abs(road_velo_angle), np.abs(np.pi / 2 - np.abs(road_velo_angle)))
        except:
            road_velo_angle = np.pi/2.0+0.1
            angle_sign = 1
        info['angle'] = 0 #angle_sign * road_velo_angle
        off_flag = 1-int(info['roadinfo'][0])
        try:
            road_center = 0.5*(np.array(info['roadinfo'][11:13])+np.array(info['roadinfo'][14:16]))
            road_width = np.sqrt(np.sum((np.array(info['roadinfo'][11:13]-np.array(info['roadinfo'][14:16])))**2.0))
            trackPos = np.sqrt(np.sum((np.array(info['location'][:2])-road_center)**2.0))
            left_dist = np.sqrt(np.sum((np.array(info['location'][:2])-np.array(info['roadinfo'][11:13]))**2.0))
            right_dist = np.sqrt(np.sum((np.array(info['location'][:2])-np.array(info['roadinfo'][14:16]))**2.0))
            if left_dist > right_dist:
                dist_sign = -1
            else:
                dist_sign = 1
        except:
            dist_sign = 1
            road_width = 5
            trackPos = 5 if off_flag else 2
        info['trackPos'] = trackPos*dist_sign
        off_flag = False
        coll_flag = off_flag#int(abs(info['trackPos']) > 7.0)
        info['off_flag'] = bool(coll_flag)#float(trackPos)/float(road_width) > 2.0
        info['coll_flag'] = bool(coll_flag)
        return self.get_segmentation(), info
         

    def step(self, action):
        real_action = copy.deepcopy(action)
        this_action = np.zeros(3)
        this_action[0] = real_action[0]*0.5 + 0.5
        this_action[1] = 0
        this_action[2] = real_action[1]
        self.epi_len += 1
        obs, info, real_done = self.env.step(this_action)
        self.obs = cv2.resize(obs, (640,352))
        # cv2.imwrite('visualize/'+str(self.epi_len).zfill(4)+'.png', self.obs)
        self.obs = Variable(torch.from_numpy(np.transpose(self.obs, (2, 0, 1)))).cuda().unsqueeze(0).float()
        seg = self.get_segmentation()
        # seg = draw_from_pred_gta(seg)
        # cv2.imwrite('visualize/'+str(self.epi_len).zfill(4)+'_seg.png', seg)
        self.speed_list.append(info['speed'])
        if np.mean(self.speed_list[-10:]) < 0.5 and len(self.speed_list) > 30:
            done = True
        else:
            done = False

        info['angle'] = 0#0.1
        try:
            info['pos'] = info['location']
        except:
            info['pos'] = [1, 420, 900]
        if True:
            try:
                road_dir = np.arctan(info['roadinfo'][9]/info['roadinfo'][8])
                velo_dir = np.arctan(info['roadinfo'][6]/info['roadinfo'][5])
                road_velo_angle = road_dir-velo_dir
                angle_sign = 1 if road_velo_angle > 0 else -1
                road_velo_angle = min(np.abs(road_velo_angle), np.abs(np.pi / 2 - np.abs(road_velo_angle)))
            except:
                road_velo_angle = np.pi/2.0#+0.1
                angle_sign = 1
        else:#xcept:
            road_velo_angle = 0
            angle_sign = 1
        info['angle'] = angle_sign * road_velo_angle
        off_flag = 1-int(info['roadinfo'][0])
        try:
            road_center = 0.5*(np.array(info['roadinfo'][11:13])+np.array(info['roadinfo'][14:16]))
            road_width = np.sqrt(np.sum((np.array(info['roadinfo'][11:13]-np.array(info['roadinfo'][14:16])))**2.0))
            trackPos = np.sqrt(np.sum((np.array(info['location'][:2])-road_center)**2.0))
            left_dist = np.sqrt(np.sum((np.array(info['location'][:2])-np.array(info['roadinfo'][11:13]))**2.0))
            right_dist = np.sqrt(np.sum((np.array(info['location'][:2])-np.array(info['roadinfo'][14:16]))**2.0))
            if left_dist > right_dist:
                dist_sign = -1
            else:
                dist_sign = 1
        except:
            dist_sign = 1
            road_width = 5
            trackPos = 5 if off_flag else 2
        if off_flag:
            self.num_off += 1
        else:
            self.num_off = 0
        info['trackPos'] = trackPos*dist_sign
        if self.epi_len < 10:
            off_flag = False
        coll_flag = int(info['reward']==-1)
        if coll_flag:
            self.coll_cnt += 1
        else:
            self.coll_cnt = 0
        dist_this = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])))
        reward_with_pos = info['speed'] * (np.cos(info['angle']))/40.0# - np.abs(np.sin(info['angle'])))/40.0# - np.abs(info['trackPos']) / road_width) / 40.0
        reward_without_pos = reward_with_pos# - np.abs(np.sin(info['angle']))) / 40.0
        done = done or self.num_off > 15
        if self.epi_len < 20:
            done = False
        obs = obs[:-15,:,:]
        obs = cv2.resize(obs, self.imsize)
        reward = dict()
        reward['with_pos'] = info['speed'] if coll_flag == False else 0
        reward['without_pos'] = info['speed'] if coll_flag == False else 0
        reward['true_reward'] = reward['with_pos']
        info['off_flag'] = bool(off_flag)#float(trackPos)/float(road_width) > 2.0
        info['coll_flag'] = bool(coll_flag)
        return self.get_segmentation(), reward, done, info

    def close(self):
        self.env.close()
