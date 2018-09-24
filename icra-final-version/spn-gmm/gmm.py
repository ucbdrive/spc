from __future__ import division, print_function
import math
import numpy as np


def softmax(x, axis=0):
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=axis), axis=axis))
    return e_x / np.expand_dims(np.sum(e_x, axis=axis), axis=axis)


def N(x, mu, sigma):
    if x.ndim == 1:
        x = np.expand_dims(x, axis=0)
    k = x.shape[1]
    x = x - mu
    if np.linalg.det(sigma) < 1e-4:
        x_2 = sigma[0, 0] * np.square(x[0, 0]) + 2 * sigma[0, 1] * x[0, 0] * x[0, 1] + sigma[1, 1] * np.square(x[0, 1])
        sigma_2 = np.sum(np.diagonal(sigma))
        if math.isnan(sigma_2) or abs(sigma_2) < 0.0001:
            return 1
        else:
            return float(np.exp(-0.5 * x_2 / sigma_2) / np.sqrt(2 * math.pi * sigma_2))
    else:
        return float(np.exp(-0.5 * x.dot(np.linalg.inv(sigma)).dot(x.T)) / np.sqrt((2 * math.pi) ** k * np.linalg.det(sigma)))


class GMM(object):
    def __init__(self, n_components=1, iterations=20, threshold=0.01):
        super(GMM, self).__init__()
        assert isinstance(n_components, int) and n_components > 0
        self.n_components = n_components
        self.iterations = iterations
        self.threshold = threshold

    def fit_gaussian(self, data, weight):
        t = np.sum(weight)
        if math.isnan(t) or t < 0.0001:
            weight = np.ones(weight.shape) / weight.shape[0]
        weight = np.expand_dims(weight / np.sum(weight), axis=1)
        mu = np.expand_dims(np.sum(data * weight, axis=0), axis=0)
        data = data - mu
        sigma = data.T.dot(data * weight)
        return mu, sigma

    def E_step(self, data, weight):
        self.gamma = np.array([[self.pi[j] * N(data[i], self.mu[j], self.sigma[j]) for j in range(self.n_components)] for i in range(data.shape[0])])
        self.gamma = self.gamma / np.expand_dims(np.sum(self.gamma, axis=1), axis=1)

    def M_step(self, data, weight):
        self.pi = np.sum(self.gamma * np.expand_dims(weight, axis=1), axis=0)
        self.pi = self.pi / np.sum(self.pi)
        self.mu, self.sigma = zip(*map(lambda x: self.fit_gaussian(data, weight * self.gamma[:, x] / np.sum(weight * self.gamma[:, x])), range(self.n_components)))

    def fit(self, data, weight):
        weight = weight / np.sum(weight)

        _max, _min = np.expand_dims(np.max(data, axis=0), axis=0), np.expand_dims(np.min(data, axis=0), axis=0)
        self.mu = np.random.rand(self.n_components, data.shape[1]) * (_max - _min) + _min
        distance = np.array([[np.linalg.norm(data[i] - self.mu[j]) for j in range(self.n_components)] for i in range(data.shape[0])])
        assignment = np.argmin(distance, axis=1)
        selections = [np.where(assignment == i)[0] for i in range(self.n_components)]
        while min(list(map(lambda x: len(x), selections))) <= 1:
            self.mu = np.random.rand(self.n_components, data.shape[1]) * (_max - _min) + _min
            distance = np.array([[np.linalg.norm(data[i] - self.mu[j]) for j in range(self.n_components)] for i in range(data.shape[0])])
            assignment = np.argmin(distance, axis=1)
            selections = [np.where(assignment == i)[0] for i in range(self.n_components)]
        self.pi = np.array(list(map(lambda x: len(x), selections)))
        self.mu, self.sigma = zip(*map(lambda x: self.fit_gaussian(data[x], weight[x]), selections))

        for i in range(self.iterations):
            mu = np.array(self.mu)
            self.E_step(data, weight)
            self.M_step(data, weight)
            if np.linalg.norm(self.mu - mu) < self.threshold:
                break
        return self

    def sample(self, num):
        res = []
        for _ in range(num):
            c = np.random.choice(range(self.n_components), p=self.pi)
            try:
                res.append(np.random.multivariate_normal(self.mu[c][0], self.sigma[c]))
            except ValueError:
                res.append(self.mu[c][0].reshape(1, -1) + np.random.rand(1, self.mu[c].shape[1]) * 0.05)
        return np.concatenate(res, axis=0)

    def score(self, data, weight):
        weight = weight / np.sum(weight)
        scores = np.sum(np.sum(np.array([[self.pi[j] * N(data[i], self.mu[j], self.sigma[j]) for j in range(self.n_components)] for i in range(data.shape[0])]), axis=1) * weight)
        return scores * len(weight)

    def _n_parameters(self):
        n_features = self.mu[0].shape[1]
        cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def aic(self, data, weight):
        return -2 * self.score(data, weight) * data.shape[0] + 2 * self._n_parameters()

    def bic(self, data, weight):
        return (-2 * self.score(data, weight) * data.shape[0] + self._n_parameters() * np.log(data.shape[0]))


def show(data, mu, sigma, gamma=None):
    colors = ['r', 'b', 'g']
    if gamma is None:
        distance = np.array([[np.linalg.norm(data[i] - mu[j]) for j in range(len(mu))] for i in range(data.shape[0])])
        assignment = np.argmin(distance, axis=1)
    else:
        assignment = np.argmax(gamma, axis=1)
    plt.figure()
    ax = plt.subplot(111, aspect='equal')
    for i in range(DATA_SIZE):
        ax.scatter(data[i, 0], data[i, 1], color=colors[assignment[i]])
    for i in range(CLUSTERS):
        ax.scatter(mu[i][0, 0], mu[i][0, 1], marker='x', color=colors[i])
        lambda_, v = np.linalg.eig(sigma[i])
        lambda_ = np.sqrt(lambda_)
        ell = Ellipse((mu[i][0, 0], mu[i][0, 1]),
                      width=lambda_[0]*2, height=lambda_[1]*2,
                      angle=-np.rad2deg(np.arctan2(v[0, 1], v[0, 0])), edgecolor=colors[i])
        ell.set_facecolor('none')
        ax.add_artist(ell)
    plt.tight_layout()
    plt.show()


class hyper_GMM(object):
    def __init__(self, k=6, criterion='aic'):
        super(hyper_GMM, self).__init__()
        self.k = k
        self.criterion = criterion
        self.data = None
        self.loss = None

    def generate_samples(self, data, loss, num, a_min=None, a_max=None):
        self.data = data if self.data is None else np.concatenate([self.data, data], axis=0)
        self.loss = loss if self.loss is None else np.concatenate([self.loss, loss], axis=0)
        weight = softmax(-self.loss)
        models = [GMM(n_components=i+1).fit(self.data, weight) for i in range(self.k)]
        scores = np.array(list(map(lambda x: x.aic(self.data, weight) if self.criterion == 'aic' else x.bic(self.data, weight), models)))
        res = models[np.argmin(scores)].sample(num)
        if a_max is not None and a_min is not None:
            res = np.clip(res, a_min, a_max)
        return res

    def best_data(self):
        return self.data[np.argmin(self.loss)]

if __name__ == '__main__':
    seed = np.random.randint(65536)  # 28357
    np.random.seed(seed)
    print("Seed was:", seed)
    DATA_SIZE = 10
    CLUSTERS = 2
    data = np.random.rand(DATA_SIZE, 2) * 2 - 1
    from matplotlib.patches import Ellipse
    import matplotlib.pyplot as plt
    weight = np.ones(DATA_SIZE) / DATA_SIZE

    # gmm = GMM(CLUSTERS)
    # mu, sigma = gmm.fit(data, weight)
    # show(data, mu, sigma, gmm.gamma)
    # new_points = gmm.sample(DATA_SIZE*10)
    # for i in range(DATA_SIZE*10):
    #     plt.scatter(new_points[i][0], new_points[i][1], color='g')
    # plt.show()
    # print(gmm.score(data, weight))
    # print(gmm.aic(data, weight))
    # print(gmm.bic(data, weight))

    hg = hyper_GMM()
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
    for i in range(4):
        data = hg.generate_samples(data, weight, DATA_SIZE, -1, 1)
        plt.scatter(hg.data[:, 0], hg.data[:, 1])
        plt.show()