import os
import numpy as np
import cv2
from carla.client import make_carla_client
from carla_env import carla_env

def get_action(control):
    print(control.throttle, control.brake, control.steer)
    return np.array([control.throttle - control.brake, control.steer])

def save_data(obs, seg, action, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    cv2.imwrite(os.path.join(path, 'obs.png'), cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(path, 'seg.png'), seg)
    np.save(os.path.join(path, 'action.npy'), action)

folder = 'imitation_data2'

if __name__ == '__main__':
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with make_carla_client('localhost', 2000) as client:
        print('\033[1;32mCarla client connected\033[0m')
        env = carla_env(client, True)
        for episode in range(1, 11):
            episode_folder = os.path.join(folder, 'episode_%d' % episode)
            if not os.path.isdir(episode_folder):
                os.mkdir(episode_folder)
            (obs, seg), info = env.reset()
            action = get_action(info['expert_control'])
            for step in range(1, 1001):
                (obs, seg), reward, done, info = env.step(info['expert_control'], expert=True)
                action = get_action(info['expert_control'])
                save_data(obs, seg, action, os.path.join(episode_folder, 'step_%d' % step))
                if done:
                    break
