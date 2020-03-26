from os.path import expanduser

import numpy as np
from joblib import Memory
from scipy.special import logsumexp

import matplotlib.pyplot as plt


class Sampler():
    def __init__(self, mean, cov, p):
        k, d = mean.shape
        k, d, d = cov.shape
        k = p.shape
        self.mean = mean
        self.cov = cov
        self.icov = np.concatenate([np.linalg.inv(cov)[None, :, :] for cov in self.cov], axis=0)
        det = np.array([np.linalg.det(cov) for cov in self.cov])
        self.norm = np.sqrt((2 * np.pi) ** d * det)
        self.p = p

    def __call__(self, n):
        k, d = self.mean.shape
        indices = np.random.choice(k, n, p=self.p)
        pos = np.zeros((n, d), dtype=np.float32)
        for i in range(k):
            mask = indices == i
            size = mask.sum()
            pos[mask] = np.random.multivariate_normal(self.mean[i], self.cov[i], size=size)
        return pos

    def log_prob(self, x):
        # b, d = x.shape
        diff = x[:, None, :] - self.mean[None, :]  # b, k, d
        return np.sum(self.p[None, :] * np.exp(-np.einsum('bkd,kde,bke->bk',
                                                          [diff, self.icov, diff]) / 2) / self.norm, axis=1)


def compute_distance(x, y):
    x2 = np.sum(x ** 2, axis=1) / 2
    y2 = np.sum(y ** 2, axis=1) / 2
    return x2[:, None] + y2[None, :] - x @ y.T


def var_norm(x, axis=None):
    return np.max(x, axis=axis) - np.min(x, axis=axis)


def evaluate_potential(q, distance, eps):
    return - eps * logsumexp((- distance + q[None, :]) / eps, axis=1)


def sinkhorn(x, y, sampling=1., alternate=True, resample=True, pin_potential=False,
             eps=1., step_size=1., avg_step_size=1.,
             n_iter=1000, record_every=1):
    n, m = x.shape[0], y.shape[0]

    record = {'iter': [], 'f': [], 'g': []}

    distance = compute_distance(x, y)
    distanceT = np.array(distance.T)

    sn = int(sampling * n)
    sm = int(sampling * m)

    # potential representation
    q = np.full((m,), fill_value=-float('inf'), )
    avg_q = np.full((m,), fill_value=-float('inf'), )
    p = np.full((n,), fill_value=-float('inf'), )
    avg_p = np.full((n,), fill_value=-float('inf'), )

    if sampling == 1:
        resample = False

    if not resample:
        if sn < n:
            x_idx = np.random.permutation(n)[:sn]
        else:
            x_idx = slice(None)
        if sm < m:
            y_idx = np.random.permutation(m)[:sm]
        else:
            y_idx = slice(None)
    else:
        x_idx, y_idx = None, None

    float_step_size = isinstance(step_size, float)
    float_avg_step_size = isinstance(avg_step_size, float)

    for i in range(n_iter):
        if sn < n:
            if resample:
                x_idx = np.random.permutation(n)[:sn]
        else:
            x_idx = slice(None)
        if sm < m:
            if resample:
                y_idx = np.random.permutation(m)[:sm]
        else:
            y_idx = slice(None)

        if float_step_size:
            step_size_ = step_size
        elif step_size == '1/t':
            step_size_ = 1 / (i + 1)
        elif step_size == '1/sqrt(t)':
            step_size_ = 1 / np.sqrt(i + 1)
        else:
            raise ValueError
        if float_avg_step_size:
            avg_step_size_ = avg_step_size
        elif avg_step_size == '1/t':
            avg_step_size_ = 1 / (i + 1)
        elif avg_step_size == '1/sqrt(t)':
            avg_step_size_ = 1 / np.sqrt(i + 1)
        else:
            raise ValueError

        if i > 0:
            g = evaluate_potential(p, distanceT[y_idx], eps)
            if not alternate:
                f = evaluate_potential(q, distance[x_idx], eps)
            else:
                f = None
        else:
            g = np.zeros((sm,))
            if not alternate:
                f = np.zeros((sn,))
            else:
                f = None


        if step_size_ != 1:
            update = np.log(step_size_) - np.log(sm) + g / eps
            q += eps * np.log(1 - step_size_)
            q[y_idx] = eps * np.logaddexp(update, q[y_idx] / eps)
            if pin_potential:
                q += evaluate_potential(q, distance[0], eps)
        else:
            q[y_idx] = - eps * np.log(sm) + g

        if alternate:
            f = evaluate_potential(q, distance[x_idx], eps)

        if step_size_ != 1:
            update = np.log(step_size_) - np.log(sn) + f / eps
            p += eps * np.log(1 - step_size_)
            p[x_idx] = eps * np.logaddexp(update, p[x_idx] / eps)
            if pin_potential:
                p += evaluate_potential(p, distanceT[0], eps)

        else:
            p[x_idx] = - eps * np.log(sn) + f

        if avg_step_size_ != 1:
            update = np.log(avg_step_size_) + q[y_idx] / eps
            avg_q += eps * np.log(1 - avg_step_size_)
            avg_q[y_idx] = eps * np.logaddexp(update, avg_q[y_idx] / eps)

            update = np.log(avg_step_size_) + p[x_idx] / eps
            avg_p += eps * np.log(1 - avg_step_size_)
            avg_p[x_idx] = eps * np.logaddexp(update, avg_p[x_idx] / eps)
        else:
            avg_p[:] = p
            avg_q[:] = q

        if i % record_every == 0:
            print(i)
            if i == 0:
                g = np.zeros(m, dtype=x.dtype)
                f = np.zeros(n, dtype=x.dtype)
            else:
                g = evaluate_potential(avg_p, distanceT, eps)
                f = evaluate_potential(avg_q, distance, eps)
            record['f'].append(f)
            record['g'].append(g)
            record['iter'].append(i)
    g = evaluate_potential(avg_p, distanceT, eps)
    f = evaluate_potential(avg_q, distance, eps)
    for key in record:
        record[key] = np.array(record[key])
    return f, g, record


def run():
    np.random.seed(10)
    n = 3
    m = 3
    sampling = 2 / 3
    eps = 1
    x_sampler = Sampler(mean=np.array([[1.], [2], [3]]), cov=np.array([[[.1]], [[.1]], [[.1]]]),
                        p=np.ones(3) / 3)
    y_sampler = Sampler(mean=np.array([[0.], [3], [5]]), cov=np.array([[[.1]], [[.1]], [[.4]]]),
                        p=np.ones(3) / 3)

    x = x_sampler(n)
    y = y_sampler(m)

    mem = Memory(expanduser('~/cache'), verbose=0)

    fref, gref, refrecords = mem.cache(sinkhorn)(x, y, n_iter=1000, sampling=1., eps=eps)

    for pin_potential in [False]:
        for avg_step_size in ['1/sqrt(t)']:
            f, g, records = mem.cache(sinkhorn)(x, y, n_iter=int(1e6), record_every=1000,
                                                step_size='1/sqrt(t)',
                                                avg_step_size=avg_step_size,
                                                sampling=sampling, pin_potential=pin_potential,
                                                eps=eps)
            norm = var_norm(refrecords['f'][-1][None, :] - records['f'], axis=1)

            plt.plot(range(len(norm)), norm, label=f'pin_pot{pin_potential} avg{avg_step_size}')
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    run()
