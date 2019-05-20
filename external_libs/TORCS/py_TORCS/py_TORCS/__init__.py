from gym.envs.registration import register

register(
    id='TORCS-v0',
    entry_point='py_TORCS.py_TORCS:torcs_env',
)