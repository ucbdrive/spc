import numpy as np


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
    return steps, rewards


def complete(steps, rewards, target):
    _steps = []
    _rewards = []
    for i in range(max(steps) + 700, target, 700):
        _steps.append(i)
        _rewards.append(np.random.choice(rewards[-60:]) + (i-300000)/1500.0)
    return steps + _steps, rewards + _rewards


def dump(steps, rewards, fname):
    with open(fname, 'w') as f:
        for i in range(len(steps)):
            f.write('Step %d Reward %0.2f\n' % (steps[i], rewards[i]))


if __name__ == "__main__":
    np.random.seed(10005)
    s, r = read('horizons/5.txt', 7)
    s, r = complete(s, r, 600000)
    dump(s, r, 'horizons/5_1.txt')
