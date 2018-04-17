from envs import create_atari_env
import pdb
import os

env = create_atari_env('torcs-v0', reward_ben=True, config='quickrace_discrete_single.xml', rescale=False)
from test_policy import *
test_policy(env, 40000000, use_pos_class=False)
