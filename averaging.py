import math
from os.path import expanduser

import numpy as np
import torch
from joblib import Memory


# from matplotlib import rc
# import matplotlib
# matplotlib.rcParams['backend'] = 'pdf'
# rc('text', usetex=True)
# import matplotlib.pyplot as plt


def sample_from(x, m):
    n = x.shape[0]
    indices = torch.from_numpy(np.random.permutation(n)[:m])
    loga = torch.full((m,), fill_value=-math.log(m))
    return loga, x[indices]


def compute_distance(x, y):
    return (torch.sum((x[:, None, :] ** 2 + y[None, :, :] ** 2 - 2 * x[:, None, :] * y[None, :, :]), dim=2)) / 2


def evaluate_potential(log_pot: torch.tensor, pos: torch.tensor, x: torch.tensor, eps):
    distance = compute_distance(x, pos)
    return - eps * torch.logsumexp((- distance + log_pot[None, :]) / eps, dim=1)


def var_norm(x):
    return x.max() - x.min()


def sinkhorn(loga, x, logb, y, eps, n_iter=1000, simultaneous=False):
    n = x.shape[0]
    distance = compute_distance(x, y)
    g = torch.zeros((n,), dtype=x.dtype)
    f = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
    for i in range(n_iter):
        ff = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
        f_diff = f - ff
        if simultaneous:
            f_used = f
        else:
            f_used = ff
        gg = - eps * torch.logsumexp((- distance.transpose(0, 1) + f_used[None, :]) / eps
                                     + loga[None, :], dim=1)
        g_diff = g - gg
        f, g = ff, gg
        tol = f_diff.max() - f_diff.min() + g_diff.max() - g_diff.min()
        if tol < 1e-7:
            return (g + eps * logb, y), (f + eps * logb, x)
    print('Not converged')
    return (g + eps * logb, y), (f + eps * logb, x)


class Sampler():
    def __init__(self, mean: torch.tensor, cov: torch.tensor, p: torch.tensor):
        k, d = mean.shape
        k, d, d = cov.shape
        k = p.shape
        self.mean = mean
        self.cov = cov
        self.icov = torch.cat([torch.inverse(cov)[None, :, :] for cov in self.cov], dim=0)
        det = torch.tensor([torch.det(cov) for cov in self.cov])
        self.norm = torch.sqrt((2 * math.pi) ** d * det)
        self.p = p

    def _call(self, n):
        k, d = self.mean.shape
        indices = np.random.choice(k, n, p=self.p.numpy())
        pos = np.zeros((n, d), dtype=np.float32)
        for i in range(k):
            mask = indices == i
            size = mask.sum()
            pos[mask] = np.random.multivariate_normal(self.mean[i], self.cov[i], size=size)
        logweight = np.full_like(pos[:, 0], fill_value=-math.log(n))
        return torch.from_numpy(pos), torch.from_numpy(logweight)

    def __call__(self, n, fake=False):
        if fake:
            if not hasattr(self, 'pos_'):
                self.pos_, self.logweight_ = self._call(n)
            return self.pos_, self.logweight_
        else:
            return self._call(n)

    def log_prob(self, x):
        # b, d = x.shape
        diff = x[:, None, :] - self.mean[None, :]  # b, k, d
        return torch.sum(self.p[None, :] * torch.exp(-torch.einsum('bkd,kde,bke->bk',
                                                                   [diff, self.icov, diff]) / 2) / self.norm, dim=1)


def averaging():
    n = 2
    m = 1
    T = 100
    eps = 1e-4

    torch.manual_seed(10)
    np.random.seed(10)

    x_sampler = Sampler(mean=torch.tensor([[1.], [2], [3]]), cov=torch.tensor([[[.1]], [[.1]], [[.1]]]),
                        p=torch.ones(3) / 3)
    y_sampler = Sampler(mean=torch.tensor([[0.], [3], [5]]), cov=torch.tensor([[[.1]], [[.1]], [[.4]]]),
                        p=torch.ones(3) / 3)

    y, logb = y_sampler(n)
    x, loga = x_sampler(n)

    (fw, fpos), (gw, gpos) = sinkhorn(loga, x, logb, y, eps=eps, n_iter=1000)
    fref = evaluate_potential(fw, fpos, x, eps)
    gref = evaluate_potential(gw, gpos, y, eps)
    g0 = gref[0]
    gref = gref - g0
    fref = fref + g0
    fs, gs = [], []
    for i in range(T):
        this_loga, this_x = sample_from(x, m)
        this_logb, this_y = sample_from(y, m)
        (fw, fpos), (gw, gpos) = sinkhorn(this_loga, this_x, this_logb, this_y, eps=eps, n_iter=100)
        this_fs = evaluate_potential(fw, fpos, x, eps)
        this_gs = evaluate_potential(gw, gpos, y, eps)
        g0 = this_gs[0]
        this_gs = this_gs - g0
        this_fs += g0
        fs.append(this_fs)
        gs.append(this_gs)
    gs = torch.cat([g[None, :] for g in gs])
    print(gref)
    print(gs)
    g = - eps * (torch.logsumexp(- gs / eps, dim=0) - math.log(T))
    print(g)
    distance = compute_distance(x, y)
    f = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
    gmean = torch.mean(gs, dim=0)
    fmean = - eps * torch.logsumexp((- distance + gmean[None, :]) / eps + logb[None, :], dim=1)
    wref = torch.sum(fref * torch.exp(loga) + gref * torch.exp(logb))
    w = torch.sum(f * torch.exp(loga) + g * torch.exp(logb))
    wmean = torch.sum(fmean * torch.exp(loga) + gmean * torch.exp(logb))
    print(var_norm(g - gref))
    print(var_norm(gmean - gref))
    print(wref)
    print(w)
    print(wmean)


if __name__ == '__main__':
    averaging()
    # simple()
    # main()
    # plot()
    # one_dimensional_exp()
