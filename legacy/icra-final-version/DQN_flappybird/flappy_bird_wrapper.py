from Flappy_Bird_with_Segmentation.environment import FlappyBirdSegEnv
import cv2
import numpy as np
from gym.spaces import Box, Discrete


class flappy_bird_wrapper(object):
    def __init__(self):
        super(flappy_bird_wrapper, self).__init__()
        self.action_space = Discrete(2)
        self.observation_space = Box(low=0, high=255, shape=(80, 80, 3), dtype=np.uint8)
        self.env = FlappyBirdSegEnv()

    def reset(self):
        self.timestep = 0
        obs, info = self.env.reset()
        # obs = cv2.resize(obs, (128, 256))
        info['segmentation'] = cv2.resize(info['segmentation'], (128, 256), interpolation=cv2.INTER_NEAREST)
        return obs, info

    def step(self, action):
        self.timestep += 1
        action = int(action)
        obs, reward, terminal, info = self.env.step(action)
        # obs = cv2.resize(obs, (128, 256))
        info['segmentation'] = cv2.resize(info['segmentation'], (128, 256), interpolation=cv2.INTER_NEAREST)
        return obs, reward, terminal or self.timestep > 100, info
