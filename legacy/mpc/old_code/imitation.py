import numpy as np
from envs import create_atari_env
import os
import pickle as pkl
import cv2

''' collect imitation data '''
env = create_atari_env('torcs-v0', reward_ben=True, config='quickrace_discrete_single.xml', rescale=False)
try:
    os.mkdir('imitation')
except:
    pass
obs = env.reset()
obs, reward, done, info = env.step(4)
try:
    os.mkdir('imitation/data')
    os.mkdir('imitation/action')
except:
    pass
prev_action = 4
dist = 5
for i in range(4000000):
    pos = info['trackPos']
    sp = info['speed']
    if pos <=-1*dist:
        if sp <= 3.0:
            action = 0
        else:
            action = 3
    elif pos <= -2 and pos > -1*dist:
        if prev_action!=3 and prev_action!=0:
            if sp <= 3.0:
                action = 0
            else:
                action = 3
        else:
            action = 4
    elif pos >= 2 and pos <= dist:
        if prev_action not in [2,5]:
            if sp <= 3.0:
                action = 2
            else:
                action = 5
        else:
            action = 4
    elif pos >= dist:
        if sp <= 3.0:
            action = 2
        else:
            action = 5
    else:
        if sp <= 3:
            action = 1
        else:
            action = 4
    if prev_action in [0,2,3,5] and action == prev_action:
        action = 4
    cv2.imwrite('imitation/data/'+str(i).zfill(9)+'.png', obs)
    pkl.dump(action, open('imitation/action/'+str(i).zfill(9)+'.pkl','wb'))
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
        obs, reward, done, info = env.step(4) 
    
