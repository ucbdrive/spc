import cv2
import gym
import numpy as np
from gym.spaces.box import Box
import gym
import sys
import os
env_path = os.path.realpath('/media/xinleipan/data/py-torcs')
sys.path.append(env_path)

def create_env(env_id, reward_ben=False, config='quickrace_discrete_single.xml', rescale=False, continuous=False):
    if env_id == 'torcs-v0':
        import py_torcs
        if reward_ben == False and continuous == False:
            env = py_torcs.TorcsEnv("discrete_improved", server=True, detailed_info=True, game_config=env_path+'/game_config/'+config)
        elif reward_ben == True and continuous == False:
            env = py_torcs.TorcsEnv("discrete_improved", custom_reward="reward_ben", server=True, detailed_info=True, game_config=env_path+'/game_config/'+config)
        elif continuous == True:
            env = py_torcs.TorcsEnv("continuous", custom_reward="reward_ben", server=True, detailed_info=True, game_config=env_path+'/game_config/'+config)
        if rescale==True:
            env = AtariRescale42x42(env)
    elif env_id == 'torcs-v1':
        from py_TORCS import torcs_env
        env = torcs_env(game_config = '/home/qizhicai/multitask/pyTORCS/game_config/' + config)
    elif env_id == 'torcs-v2':
        from py_TORCS import torcs_envs
        env = torcs_envs(num = 1, game_config = '/home/cxy/pyTORCS/game_config/' + config).get_envs()[0]
    else:
        env = gym.make(env_id)
        env = AtariRescale42x42(env)
    return env


def _process_frame42_grid(frame):
    frame = cv2.resize(frame, (256, 256))
    frame = frame.astype(np.float32)*255
    return frame

def _process_frame42(frame):
    frame = cv2.resize(frame, (256, 256))
    frame = frame.astype(np.float32)
    return frame

class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 255.0, [256, 256, 3])
        
    def _observation(self, observation):
        return _process_frame42(observation)
