import gym
import gym_gridworld
import numpy as np
from gym import spaces


def create_gridworld_env(plan, lidar=False, random_start=False, render=False, seed=0):
    env = gym.make('gridworld-v0')
    env.init_states(plan)
    env.random_start = random_start

    if render:
        env.verbose = True
        _ = env._reset(True)
    if lidar:
        env = LidarWrapper(env)

    env.seed(seed)
    
    return env

_dir8 = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1)
]

_dir_distance_coeff = [
    1, 1.414, 1, 1.414,
    1, 1.414, 1, 1.414,
]


def _in_range(pos, bound):
    return 0 <= pos[0] < bound and 0 <= pos[1] < bound


def _lidar_8(grid_map, pos, density=7):
    grid_map = (grid_map == 1)
    obs = []
    pos = tuple(pos)

    assert not grid_map[pos]

    for idx, d in enumerate(_dir8):
        distance = 0.5
        new_pos = (pos[0] + d[0], pos[1] + d[1])
        while _in_range(new_pos, 7) and not grid_map[new_pos]:
            distance += 1
            new_pos = (new_pos[0] + d[0], new_pos[1] + d[1])
        distance *= _dir_distance_coeff[idx]
        obs.append(distance)

    return np.array(obs).astype(np.float)


class LidarWrapper(gym.Wrapper):
    def __init__(self, env, density=7):
        super().__init__(env)
        assert self.env.current_grid_map.shape == (density, density)

        self.observation_space = spaces.Box(low=0, high=16, shape=(density,))
        self.density = density

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        #if action == 0 or info['success'] == False:
        #    reward = -0.01
        return _lidar_8(self.env.current_grid_map, self.env.agent_state, self.density), reward, done, info

    def _reset(self, x=None, y = None):
        if x is None or y is None:
            x = np.random.randint(self.density)
            y = np.random.randint(self.density)
        while self.env.current_grid_map[x,y] == 1 or self.env.current_grid_map[x,y] == 3:
            x = np.random.randint(self.density)
            y = np.random.randint(self.density)
        self.env.reset(x=x, y=y)
        return _lidar_8(self.env.current_grid_map, self.env.agent_state, self.density)
"""
class LidarWrapper(gym.Wrapper):
    def __init__(self, env, density=7):
        super().__init__(env)
        assert density == density
        assert self.env.current_grid_map.shape == (density, density)

        self.observation_space = spaces.Box(low=0, high=16, shape=(density,))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _lidar_8(self.env.current_grid_map, self.env.agent_state), reward, done, info

    def reset(self, x = 0, y = 0):
        self.env._reset(x=x, y=y)
        return _lidar_8(self.env.current_grid_map, self.env.agent_state)   
"""
