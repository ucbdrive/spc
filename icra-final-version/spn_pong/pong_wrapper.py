import gym
import numpy as np
import os
import cv2


def enlarge(obs):
    obs = obs[34: 194, :, :]
    # Paddle
    x, y = np.where(np.all(obs == np.array([92, 186, 92]), axis=-1))
    paddle = len(x) > 0
    if paddle:
        paddle_x_min, paddle_x_max, paddle_y_min, paddle_y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        paddle_y_max += 6

    # Ball
    x, y = np.where(np.all(obs == np.array([236, 236, 236]), axis=-1))
    ball = len(x) > 0
    if ball:
        ball_x_min, ball_x_max, ball_y_min, ball_y_max = np.min(x), np.max(x), np.min(y), np.max(y)
    
    # Opponent
    x, y = np.where(np.all(obs == np.array([213, 130, 74]), axis=-1))
    opponent = len(x) > 0
    if opponent:
        opponent_x_min, opponent_x_max, opponent_y_min, opponent_y_max = np.min(x), np.max(x), np.min(y), np.max(y)
        opponent_y_min -= 6

    # Paddle
    if paddle:
        obs[paddle_x_min: paddle_x_max+1, paddle_y_min: paddle_y_max+1] = np.array([92, 186, 92])
    
    # Opponent
    if opponent:
        obs[opponent_x_min: opponent_x_max+1, opponent_y_min: opponent_y_max+1] = np.array([213, 130, 74])
    
    # Ball
    if ball:
        obs[max(0, ball_x_min-3): min(ball_x_max+3+1, 160), max(0, ball_y_min): min(ball_y_max+1, 160)] = np.array([236, 236, 236])
        obs[max(0, ball_x_min): min(ball_x_max+1, 160), max(0, ball_y_min-3): min(ball_y_max+3+1, 160)] = np.array([236, 236, 236])
        obs[max(0, ball_x_min-2): min(ball_x_max+2+1, 160), max(0, ball_y_min-2): min(ball_y_max+2+1, 160)] = np.array([236, 236, 236])

    if os.path.isdir('/tmp'):
        cv2.imwrite('/tmp/pong.png', cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
    return obs


def crop_and_seg(obs):
    seg = np.zeros((160, 160), dtype=np.uint8)

    # Background
    x, y = np.where(np.all(obs == np.array([144, 72, 17]), axis=-1))
    seg[x, y] = 0

    # Paddle
    x, y = np.where(np.all(obs == np.array([92, 186, 92]), axis=-1))
    seg[x, y] = 1

    # Ball
    x, y = np.where(np.all(obs == np.array([236, 236, 236]), axis=-1))
    seg[x, y] = 2

    # Opponent
    x, y = np.where(np.all(obs == np.array([213, 130, 74]), axis=-1))
    seg[x, y] = 3

    if os.path.isdir('/tmp'):
        cv2.imwrite('/tmp/pong.png', cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))

    return obs, seg


def reward_and_done(seg):
    x, y = np.where(seg == 2)
    if len(x) == 0:
        return 0, True
    ball_x_min, ball_x_max, ball_y_min, ball_y_max = np.min(x), np.max(x), np.min(y), np.max(y)
    if ball_y_max < 138:
        return 0, False
    if ball_y_min >= 141:
        return -1, ball_y_max > 156
    x, _ = np.where(seg == 1)
    paddle_x_min, paddle_x_max = np.min(x), np.max(x)
    if ball_x_max-2 <= paddle_x_min or ball_x_min+2 >= paddle_x_max:
        return -1, False
    else:
        return 1, False


def eligible(seg):
    return np.sum(seg == 3) * np.sum(seg == 2) > 0


class pong_wrapper(object):
    def __init__(self):
        super(pong_wrapper, self).__init__()
        self.env = gym.make('Pong-v0')

    def reset(self):
        self.timestep = 0
        obs = self.env.reset()
        obs = enlarge(obs)
        obs, seg = crop_and_seg(obs)
        while not eligible(seg):
            obs, _, _, _ = self.env.step(0)
            obs = enlarge(obs)
            obs, seg = crop_and_seg(obs)
        return obs, seg

    def close(self):
        return self.env.close()

    def step(self, action):
        self.timestep += 1
        action_map = {0: 0, 1: 2, -1: 3}
        action = action_map[int(action)]
        obs, _, _, info = self.env.step(action)
        obs = enlarge(obs)
        obs, seg = crop_and_seg(obs)
        reward, done = reward_and_done(seg)
        done = done or self.timestep >= 1000
        return (obs, seg), reward, done, info

if __name__ == '__main__':
    env = pong_wrapper()
    env.reset()
    for i in range(1000):
        action = int(input())
        (obs, seg), reward, done, info = env.step(action)
        cv2.imwrite('/tmp/obs.png', cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
        print(reward)
        if done:
            obs, seg = env.reset()
    env.close()