from Flappy_Bird_with_Segmentation.environment import FlappyBirdSegEnv
import cv2


class flappy_bird_wrapper(object):
    def __init__(self):
        super(flappy_bird_wrapper, self).__init__()
        self.env = FlappyBirdSegEnv()

    def reset(self):
        self.timestep = 0
        self.real_obs, info = self.env.reset()
        obs = cv2.resize(self.real_obs, (128, 256))
        info['segmentation'] = cv2.resize(info['segmentation'], (128, 256), interpolation=cv2.INTER_NEAREST)
        return obs, info

    def step(self, action):
        self.timestep += 1
        action = int(action)
        self.real_obs, reward, terminal, info = self.env.step(action)
        obs = cv2.resize(self.real_obs, (128, 256))
        info['segmentation'] = cv2.resize(info['segmentation'], (128, 256), interpolation=cv2.INTER_NEAREST)
        return obs, reward, terminal or self.timestep > 1000, info
