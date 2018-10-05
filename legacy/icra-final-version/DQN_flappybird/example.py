import os
os.chdir('..')
from py_TORCS import torcs_envs
import time
#USE_SERVER = 1

if __name__ == '__main__':
    game_config = '/home/xiaocw/git/pyTORCS/game_config/michigan.xml'
    envs = torcs_envs(num = 1, game_config=game_config, isServer = 1, continuous=False, resize=True)
    env1 = envs.get_envs()[0]
    obs1 = env1.reset()
    for i in range(100):
        obs1, reward1, done1, info1 = env1.step(0)
        print(info1['trackPos'])
        time.sleep(0.01)
    env1.close()
