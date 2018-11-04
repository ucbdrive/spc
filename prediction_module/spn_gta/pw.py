from __future__ import division, print_function
import math
import numpy as np
import pdb


def softmax(x, axis=0):
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=axis), axis=axis))
    return e_x / np.expand_dims(np.sum(e_x, axis=axis), axis=axis)


class pw(object):
    def __init__(self, var=0.05):
        super(pw, self).__init__()
        self.var = var
        self.data = None
        self.loss = None

    def sample(self, num):
        res = []
        for _ in range(num):
            c = np.random.choice(range(self.data.shape[0]), p=self.weight)
            try:
                res.append(np.random.multivariate_normal(self.data[c], self.sigma).reshape(1, -1))
            except ValueError:
                res.append(self.data[c].reshape(1, -1) + np.random.uniform(-self.var, self.var, size=(1, self.data.shape[1])))
        return np.concatenate(res, axis=0)

    def generate_samples(self, data, loss, num, a_min=None, a_max=None):
        if self.data is None:
            self.sigma = np.eye(data.shape[1]) * np.power(self.var, 2)
        self.data = data if self.data is None else np.concatenate([self.data, data], axis=0)
        self.loss = loss if self.loss is None else np.concatenate([self.loss, loss], axis=0)
        self.weight = softmax(-self.loss)
        try:
            res = self.sample(num)
        except:
            pdb.set_trace()
        if a_max is not None and a_min is not None:
            res = np.clip(res, a_min, a_max)
        return res

    def best_data(self):
        return self.data[np.argmin(self.loss)]
