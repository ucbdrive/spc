from envs import create_atari_env
import pdb
import os
import cv2

env = create_atari_env('torcs-v0', reward_ben=True, config='quickrace_discrete_single.xml', rescale=False)
obs = env.reset()
for i in range(10000):
    cv2.imwrite('obs.png', obs)
    action = input('action ..')
    try:
        obs, _, done, info = env.step(int(action))
        print(info['pos'])
    except:
        pass
