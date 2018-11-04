from carla.client import make_carla_client
from carla_env import carla_env

action_map = {'w': [1.0, 0.0],
              'a': [0.0, -1.0],
              's': [-1.0, 0.0],
              'd': [0.0, 1.0]}

with make_carla_client('localhost', 2000) as client:
    print('\033[1;32mCarla client connected\033[0m')
    env = carla_env(client, True)
    env.reset()
    for i in range(1000):
        env.step(action_map[input()])
