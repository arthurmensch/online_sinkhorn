import math
import os
from collections import defaultdict
from os.path import expanduser

import joblib
import numpy as np
import torch

import matplotlib.pyplot as plt
from joblib import Memory, delayed, Parallel
import seaborn as sns
import pandas as pd

# from matplotlib import rc
# import matplotlib
# matplotlib.rcParams['backend'] = 'pdf'
# rc('text', usetex=True)
# import matplotlib.pyplot as plt
from sklearn.utils import check_random_state

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


def sample_from_finite(x, m, random_state=None):
    random_state = check_random_state(random_state)
    n = x.shape[0]
    indices = torch.from_numpy(random_state.permutation(n)[:m])
    loga = torch.full((m,), fill_value=-math.log(m))
    return indices, loga


def var_norm(x):
    return x.max() - x.min()


def compute_distance(x, y):
    return (torch.sum((x[:, None, :] ** 2 + y[None, :, :] ** 2 - 2 * x[:, None, :] * y[None, :, :]), dim=2)) / 2


def evaluate_potential(log_pot: torch.tensor, pos: torch.tensor, x: torch.tensor, eps):
    distance = compute_distance(x, pos)
    return - eps * torch.logsumexp((- distance + log_pot[None, :]) / eps, dim=1)


def evaluate_potential_finite(log_pot: torch.tensor, idx: torch.tensor, distance, eps):
    return - eps * torch.logsumexp((- distance[idx] + log_pot[None, :]) / eps, dim=1)


def online_sinkhorn(x, y, eps, m, n_iter=100, last_transform=True, alternated=False,
                    random_state=None, averaging='primal', ieta=.5, eta=1.):
    random_state = check_random_state(random_state)
    torch.manual_seed(random_state.randint(100000))

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    n = x.shape[0]
    p = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)
    q = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)

    avg_p = p
    avg_q = q

    distance = compute_distance(x, y)
    errors = defaultdict(list)
    u_eta = not isinstance(eta, float)
    u_ieta = not isinstance(ieta, float)

    for i in range(0, n_iter):
        # Update f
        y_idx, logb = sample_from_finite(y, m, random_state=random_state)
        x_idx, loga = sample_from_finite(x, m, random_state=random_state)
        f = None
        if i > 0:
            g = evaluate_potential_finite(p, y_idx, distance.transpose(0, 1), eps)
            f = evaluate_potential_finite(q, x_idx, distance, eps)
        else:
            g = torch.zeros(m, dtype=x.dtype)
            f = torch.zeros(m, dtype=x.dtype)
            f, g = f[x_idx], g[y_idx]
        if u_ieta:
            if ieta == '1/t':
                ieta_ = torch.tensor(1 / (i + 1))
            elif ieta == '1/sqrt(t)':
                ieta_ = torch.tensor(1 / math.sqrt(i + 1))
            else:
                raise ValueError
        else:
            ieta_ = torch.tensor(ieta)

        q += eps * torch.log(- ieta_ + 1)
        update = eps * torch.log(ieta_) + logb * eps + g
        q[y_idx] = eps * torch.logsumexp(torch.cat([q[y_idx][None, :], update[None, :]], dim=0) / eps, dim=0)

        f = evaluate_potential_finite(q, x_idx, distance, eps)
        p += eps * torch.log(- ieta_ + 1)
        update = eps * torch.log(ieta_) + loga * eps + f
        p[x_idx] = eps * torch.logsumexp(torch.cat([p[x_idx][None, :], update[None, :]], dim=0) / eps, dim=0)

        if averaging != 'dual':
            if u_eta:
                if eta == '1/t':
                    eta_ = torch.tensor(1 / (i + 1))
                elif eta == '1/sqrt(t)':
                    eta_ = torch.tensor(1 / math.sqrt(i + 1))
                elif eta == '1/t^3/4':
                    eta_ = torch.tensor(1 / math.pow(i + 1, 1 / 4))
                else:
                    raise ValueError
            else:
                eta_ = torch.tensor(eta)
            if averaging == 'dual':
                avg_q += eps * torch.log(- eta_ + 1)
                update = eps * torch.log(eta_) + q[y_idx]
                avg_q[y_idx] = eps * torch.logsumexp(torch.cat([update[None, :], avg_q[y_idx][None, :]]) / eps, dim=0)

                avg_p += eps * torch.log(- eta_ + 1)
                update = eps * torch.log(eta_) + p[x_idx]
                avg_p[x_idx] = eps * torch.logsumexp(torch.cat([update[None, :], avg_p[x_idx][None, :]]) / eps, dim=0)
        else:
            avg_q = avg_q
            avg_p = avg_p
        if i % 1 == 0:
            ff = evaluate_potential_finite(avg_q, slice(None), distance, eps)
            gg = evaluate_potential_finite(avg_p, slice(None), distance.transpose(0, 1), eps)
            w = (ff.mean() + gg.mean())
            errors['ff'].append(ff.numpy().tolist())
            errors['gg'].append(gg.numpy().tolist())
            errors['w'].append(w.item())
            errors['iter'].append(i)
    return ff.numpy(), gg.numpy(), errors


def sinkhorn(x, y, eps, n_iter=1000):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    n = x.shape[0]
    loga = torch.full((n,), fill_value=-math.log(n), dtype=torch.float64)
    logb = torch.full((n,), fill_value=-math.log(n), dtype=torch.float64)
    distance = compute_distance(x, y)
    g = torch.zeros((n,), dtype=torch.float64)
    f = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
    errors = defaultdict(list)
    for i in range(n_iter):
        ff = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
        gg = - eps * torch.logsumexp((- distance.transpose(0, 1) + ff[None, :]) / eps + loga[None, :], dim=1)
        f, g = ff, gg
        w = (ff.mean() + gg.mean())
        errors['ff'].append(ff.numpy().tolist())
        errors['gg'].append(gg.numpy().tolist())
        errors['w'].append(w.item())
        errors['iter'].append(i)
    return (g + eps * logb, y), (f + eps * logb, x), errors


def simple():
    n = 10
    eps = 1

    x_sampler = Sampler(mean=torch.tensor([[1.], [2], [3]]), cov=torch.tensor([[[.1]], [[.1]], [[.1]]]),
                        p=torch.ones(3) / 3)
    y_sampler = Sampler(mean=torch.tensor([[0.], [3], [5]]), cov=torch.tensor([[[.1]], [[.1]], [[.4]]]),
                        p=torch.ones(3) / 3)

    y, logb = y_sampler(n)
    x, loga = x_sampler(n)
    y = y.numpy()
    x = x.numpy()

    mem = Memory(location=expanduser('~/cache'))
    fref, gref = mem.cache(sinkhorn)(x, y, eps=eps, n_iter=100, simultaneous=True)
    results = Parallel(n_jobs=18)(delayed(online_sinkhorn())(eta=eta, ieta=ieta, m=m, last_transform=last_transform,
                                                             averaging=averaging,
                                                             random_state=random_state)
                                  for m in [5]
                                  for eta in ['1/sqrt(t)']
                                  for random_state in [1]
                                  for averaging in ['dual', 'none']
                                  for last_transform in [True]
                                  for ieta in ['1/sqrt(t)'])
    if not os.path.exists(expanduser('~/output/online_sinkhorn')):
        os.makedirs(expanduser('~/output/online_sinkhorn'))
    joblib.dump(results, expanduser('~/output/online_sinkhorn/results_from_ref.pkl'))


def plot():
    # results = joblib.load(expanduser('~/output/online_sinkhorn/results_1e6_2.pkl'))
    # results = joblib.load(expanduser('~/output/online_sinkhorn/results_1e5.pkl'))
    results = joblib.load(expanduser('~/output/online_sinkhorn/results_from_ref.pkl'))
    df = []
    for result in results:
        for i in range(len(result['errors']['iter'])):
            df.append(dict(ieta=result["ieta"], eta=result["eta"], last_transform=result["last_transform"],
                           first_transform=result["first_transform"], random_state=result["random_state"],
                           ie=f'{result["ieta"]}_{result["eta"]}_{result["first_transform"]}_{result["last_transform"]}',
                           m=result["m"], iter=result['errors']['iter'][i],
                           averaging=result['averaging'],
                           value=abs(result['errors']['werr'][i])))
    df = pd.DataFrame(df)
    grid = sns.FacetGrid(data=df, col="eta", row="ieta", hue='averaging')
    grid.map(plt.plot, "iter", "value")
    for ax in grid.axes.ravel():
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim([1e-7, 1])
        ax.set_xlim([1e2, 1e3])
    grid.add_legend()

    plt.show()


if __name__ == '__main__':
    simple()
    # main()
    plot()
    # one_dimensional_exp()
