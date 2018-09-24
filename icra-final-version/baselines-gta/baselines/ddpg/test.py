import numpy as np
from time import sleep
from gta_env import *
from gta_wrapper import *

env = GtaEnv(autodrive=None)
env = GTAWrapper(env)
env.reset()
env.reset()
for i in range(200):
    env.step(np.array([1, 0]))
    sleep(0.1)
env.close()