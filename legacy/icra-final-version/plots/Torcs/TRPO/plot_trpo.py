# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import copy
import matplotlib.pyplot as plt

def smooth(array, m=0):
    _array = copy.deepcopy(array)
    n = _array.shape[0]
    for i in range(1, n):
        _array[i] = np.mean(array[max(0, i - m): min(n, i + m + 1)])
    return _array

def plot_trpo(fname):
    steps = []
    rewards = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            if line[:11] == '| EpLenMean':
                step = float(line[20:-2])
                steps.append(step)
            elif line[:11] == '| EpRewMean':
                reward = float(line[20:-2])
                rewards.append(reward)

    for i in range(1, len(steps)):
        steps[i] = steps[i-1] + steps[i]
    with open('TRPO.txt', 'w') as f:
        for i in range(1, len(steps)):
            f.write('Step %d Reward %0.2f\n' % (steps[i], rewards[i]))
    steps = np.array(steps)
    rewards = np.array(rewards)
    rewards = smooth(rewards)

    plt.figure(figsize = (8, 5))

    plt.plot(steps, rewards)
    plt.xlabel('step')
    plt.ylabel('reward')
    plt.tight_layout()
    plt.savefig('trpo_reward.png', dpi = 300)

if __name__ == '__main__':
    plot_trpo('trpo_log.txt')