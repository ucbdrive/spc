from __future__ import division, print_function
from TORCS_ctrl import mem_init, env_start, env_action_continuous, env_terminate, env_get_state, mem_cleanup, getRGBImage, get_written, get_segmentation
import time
import os
from subprocess import Popen
import copy
import math
import itertools
import numpy as np
import cv2
import gym
from gym.spaces import Box, Discrete

max_steps = 1000
max_damage = 90000
max_stuck_count = 50
game_config_default = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'game_config', 'michigan.xml')
image_height, image_width = 480, 640

# accel, brake, steer
action_map = [[1, 0, 1],
              [1, 0, 0],
              [1, 0, -1],
              [0, 0, 1],
              [0, 0, 0],
              [0, 0, -1],
              [0, 1, 1],
              [0, 1, 0],
              [0, 1, -1]]

def naive_driver(info):
    if info['angle'] > 0.5 or (info['trackPos'] < -1 and info['angle'] > 0):
        return 0
    elif info['angle'] < -0.5 or (info['trackPos'] > 3 and info['angle'] < 0):
        return 2
    return 1

class DoneCondition(object):
    def __init__(self, size):
        super(DoneCondition, self).__init__()
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

class torcs_env(gym.Env):
    def __init__(self):
        super(torcs_env, self).__init__()
        self.game_is_on, self.memory_is_on = False, False

    def init(self, ID = 0,
                   game_config=game_config_default, 
                   mkey=817, 
                   auto_back=0, 
                   isServer=0, 
                   screen_id=160, 
                   continuous=False, 
                   resize=False,
                   low_dim=False,
                   reward_with_pos=True):
        self.game_config = game_config
        self.mkey = mkey + ID
        self.auto_back = auto_back
        self.isServer = isServer
        self.screen_id = screen_id + ID
        self.continuous = continuous
        self.resize = resize
        self.low_dim = low_dim
        self.reward_with_pos = reward_with_pos

        if self.continuous:
            self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            self.action_space = Discrete(3)
        if low_dim:
            self.observation_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            if self.resize:
                self.observation_space = Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
            else:
                self.observation_space = Box(low=0, high=255, shape=(640, 480, 3), dtype=np.uint8)
        self.reward_range = (-math.inf, math.inf)

        if self.isServer:
            os.environ['DISPLAY'] = ":%d" % self.screen_id
            self.xvfb = Popen(['Xvfb', os.environ['DISPLAY'], "-screen", "0", "%dx%dx24" % (image_width, image_height)])
    
    def reset(self):
        self.close()
        self.start()
        #for i in range(np.random.randint(300)):
        #    self.step_discrete(naive_driver(self.get_info()))
        return self.render()

    def start(self):
        self.damage = 0
        self.timestep = 0
        self.stuck_count = 0
        self.doneCond = DoneCondition(30)
        if self.game_is_on:
            self._close()
        self.pid = env_start(self.auto_back, self.mkey, self.isServer, 0, self.game_config)
        self.game_is_on = True

        if not self.memory_is_on:
            self.shmid, self.shm = mem_init(self.mkey)
            self.wait_for_ack()
        self.memory_is_on = True

    def get_reward_and_terminal(self, info):
        if self.reward_with_pos:
            reward = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])) - np.abs(info['trackPos']) / 9.0) / 40.0
        else:
            reward = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle']))) / 40.0

        dist_this = info['speed'] * (np.cos(info['angle']) - np.abs(np.sin(info['angle'])))
        terminal = info['end'] or self.timestep > max_steps or self.doneCond.isdone(info['trackPos'], dist_this, info['pos'], info['angle'])
        # self.damage = info['next_damage']

        return reward, terminal

    def step(self, action):
        self.timestep += 1
        if self.continuous:
            self.step_continuous(max(action[0], 0), -min(action[0], 0), action[1])
        else:
            self.step_discrete(action)
        info = self.get_info()
        return (self.render(),) + self.get_reward_and_terminal(info) + (info,)

    def step_discrete(self, action):
        self.step_continuous(*action_map[action])

    def step_continuous(self, accel, brake, steer):
        env_action_continuous(self.shm, accel, brake, steer)
        self.wait_for_ack()

    def get_info(self):
        end, dist, speed, angle, damage, pos, segtype, radius, frontCarNum, frontDist, posX, posY, posZ, width = env_get_state(self.shm)
        is_stuck = abs(angle) > math.pi * 50 / 180 and speed < 1
        off_flag = int(pos >= 3 or pos <= -1)
        coll_flag = int(abs(pos + angle) > 6.5 or (damage > 0 and angle > 0.5 and speed < 15))
        return {'speed': speed, 'angle': angle, 'trackPos': pos, 'trackWidth': width, 'damage': self.damage, 'next_damage': damage, 'is_stuck': is_stuck, 'pos': (posX, posY, posZ), 'end': end, 'off_flag': off_flag, 'coll_flag': coll_flag}

    def render(self):
        img = np.flip(getRGBImage(self.shm), 0)
        if self.resize:
            img = cv2.resize(img, (256, 256))
        return img

    def get_segmentation(self):
        RGB_image = self.render()
        # 1: asphalt, 2:sky, 0:others
        grass, asphalt, sky = get_segmentation(self.shm)
        grass, asphalt, sky = np.flip(grass, 0), np.flip(asphalt, 0), np.flip(sky, 0)

        if self.resize:
            grass, asphalt, sky = cv2.resize(grass, (256, 256)), cv2.resize(asphalt, (256, 256)), cv2.resize(sky, (256, 256))

        seg_result = np.zeros((RGB_image.shape[:-1]))
        seg_grass = np.any(RGB_image != grass, axis = 2)
        seg_result[seg_grass] = 1
        seg_asphalt = np.any(RGB_image != asphalt, axis = 2)
        seg_result[seg_asphalt] = 2
        seg_sky = np.any(RGB_image != sky, axis = 2)
        seg_result[seg_sky] = 3

        if 0:
            illustration = np.zeros(RGB_image.shape)
            illustration[:, :, 0] = 255
            illustration[seg_grass] = np.array([0, 255, 0])
            illustration[seg_asphalt] = np.array([0, 0, 0])
            illustration[seg_sky] = np.array([0, 0, 255])
            cv2.imwrite('/home/cxy/pyTORCS/segmentation_illustration.png', illustration)

        return seg_result

    def close(self):
        if self.game_is_on:
            env_terminate(self.pid, self.shm)
            self.game_is_on = False

        if self.memory_is_on:
            mem_cleanup(self.shmid, self.shm)
            self.memory_is_on = False

    def __del__(self):
        if self.isServer:
            self.xvfb.terminate()

    def wait_for_ack(self):
        count = 0
        while get_written(self.shm) == 0:
            count += 1
            time.sleep(0.001)
            if count > 10000:
                raise Exception('Error', 'Lost connection to torcs')

if __name__ == '__main__':
    try:
        env1 = torcs_env(mkey = 817)
        # env2 = torcs_env(mkey = 818)
        obs = env1.reset()
        # reward, state, terminate = env2.reset()
        for i in range(100):
            obs, reward, done, info = env1.step(1)
            # reward, state, terminate = env2.step_continuous(1, 0, 0)
            time.sleep(0.01)
        env1.close()
    except:
        env1.close()
    # env2.terminate()
