# -*- coding: utf-8 -*-
from __future__ import division, print_function
import numpy as np
import copy
import matplotlib.pyplot as plt


def smooth(array, m=3):
    _array = copy.deepcopy(array)
    std = np.zeros_like(array)
    n = _array.shape[0]
    for i in range(1, n):
        _array[i] = np.mean(array[max(0, i - m): min(n, i + m + 1)])
        std[i] = np.std(array[max(0, i - m): min(n, i + m + 1)])
    return _array, std


def cut(array, m1=300, m2=200, l=30):
    n = array.shape[0]
    start = max(min(int((n-l)/2), m1), 0)
    end = min(max(int((n+l)/2), n-m2), n)
    return array[start:end].mean()

def read(fname, idx):
    steps = []
    rewards = []
    test = True

    with open(fname, 'r') as f:
        for line in f.readlines():
            entry = line.split(' ')
            if idx == 7:
                test = not test
            if test:
                steps.append(int(entry[1]))
                rewards.append(eval(entry[idx]))
            if int(entry[1]) > 590000:
                break
    steps = np.array(steps)
    rewards = np.array(rewards)
    rewards, std = smooth(rewards, 25)
    std = std / 2
    plt.plot(steps, rewards)
    # plt.fill_between(steps, rewards+std, rewards-std, color='pink')
    return steps, rewards, std


def plot_reward():
    plt.figure(figsize=(8, 5))

    read('log_train_torcs.txt', 7)

    plt.xlabel('steps')
    plt.ylabel('reward')
    plt.legend(['GCG'])
    plt.tight_layout()
    plt.savefig('GCG.png', dpi=300)

plot_reward()
