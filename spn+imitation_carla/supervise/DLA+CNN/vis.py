from __future__ import division, print_function
import numpy as np
import torch
import torch.nn as nn
import os
from model import DLANET
import cv2
from carla.client import make_carla_client
from carla_env import carla_env

actions = {
    0: [0.5, 0.0],
    1: [0.0, -0.5],
    2: [0.0, 0.5],
    3: [-1.0, 0.0]
}

folder = 'test_results'

if __name__ == '__main__':
    model = DLANET()
    model = nn.DataParallel(model).cuda().eval()
    model.load_state_dict(torch.load(os.path.join('trained_models', 'epoch_%d.pth' % 10)))

    if not os.path.isdir(folder):
        os.makedirs(folder)
        
    with make_carla_client('localhost', 2000) as client:
        print('\033[1;32mCarla client connected\033[0m')
        env = carla_env(client, True)
        for episode in range(1, 11):
            episode_folder = os.path.join(folder, 'episode_%d' % episode)
            if not os.path.isdir(episode_folder):
                os.mkdir(episode_folder)
            video = cv2.VideoWriter(os.path.join(episode_folder, 'video.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 24.0, (256, 256), True)
            (obs, seg), info = env.reset()
            video.write(obs)
            obs = torch.from_numpy(np.expand_dims(obs.transpose(2, 0, 1), axis=0)).float() / 255.0
            with torch.no_grad():
                logits, _ = model(obs)
                pred = torch.argmax(logits, dim=1).long()
            action = np.array(actions[int(pred[0])])
            for step in range(1, 1001):
                (obs, seg), reward, done, info = env.step(action)
                video.write(obs)
                obs = torch.from_numpy(np.expand_dims(obs.transpose(2, 0, 1), axis=0)).float() / 255.0
                with torch.no_grad():
                    logits, _ = model(obs)
                    pred = torch.argmax(logits, dim=1).long()
                action = np.array(actions[int(pred[0])])
                if done:
                    break
            video.release()
