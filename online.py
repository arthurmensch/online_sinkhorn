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

from matplotlib import rc
import matplotlib
matplotlib.rcParams['backend'] = 'pdf'
rc('text', usetex=True)
import matplotlib.pyplot as plt

def sample_from(x, m):
    n = x.shape[0]
    indices = torch.from_numpy(np.random.permutation(n)[:m])
    loga = torch.full((m,), fill_value=-math.log(m))
    return x[indices], loga


def sample_from_finite(x, m, ):
    n = x.shape[0]
    indices = torch.from_numpy(np.random.permutation(n)[:m])
    loga = torch.full((m,), fill_value=-math.log(m))
    return indices, loga


def compute_distance(x, y):
    return (torch.sum((x[:, None, :] ** 2 + y[None, :, :] ** 2 - 2 * x[:, None, :] * y[None, :, :]), dim=2)) / 2


def evaluate_potential(log_pot: torch.tensor, pos: torch.tensor, x: torch.tensor, eps):
    distance = compute_distance(x, pos)
    return - eps * torch.logsumexp((- distance + log_pot[None, :]) / eps, dim=1)


def stochasic_sinkhorn(x, y, eps, m, n_iter=100, step_size='sqrt'):
    hatf = torch.full((m * n_iter,), fill_value=-float('inf'))
    ahatf = torch.full((m * n_iter,), fill_value=-float('inf'))
    posy = torch.zeros(m * n_iter, 2)
    hatg = torch.full((m * n_iter,), fill_value=-float('inf'))
    ahatg = torch.full((m * n_iter,), fill_value=-float('inf'))
    posx = torch.zeros(m * n_iter, 2)
    sum_eta = 0
    for i in range(0, n_iter):
        if step_size == 'sqrt':
            eta = torch.tensor(1. / math.sqrt(i + 1))
        elif step_size == 'constant':
            eta = torch.tensor(0.01)

        # Update f
        y_, logb = sample_from(y, m)
        if i > 0:
            g = evaluate_potential(hatf[:i * m], posx[:i * m], y_, eps)
        else:
            g = torch.zeros(m)
        hatg[:i * m] += eps * torch.log(1 - eta)
        hatg[i * m:(i + 1) * m] = eps * math.log(eta) + logb * eps + g
        posy[i * m:(i + 1) * m] = y_

        update = eps * math.log(eta) + hatg[:(i + 1) * m]
        ahatg[:(i + 1) * m] = eps * torch.logsumexp(torch.cat([update[None, :], ahatg[:(i + 1) * m][None, :]]) / eps,
                                                    dim=0)

        # Update g
        x_, loga = sample_from(x, m)
        f = evaluate_potential(hatg[:(i + 1) * m], posy[:(i + 1) * m], x_, eps)
        hatf[:i * m] += eps * torch.log(1 - eta)
        hatf[i * m:(i + 1) * m] = eps * math.log(eta) + loga * eps + f
        posx[i * m:(i + 1) * m] = x_

        update = eps * math.log(eta) + hatf[:(i + 1) * m]
        ahatf[:(i + 1) * m] = eps * torch.logsumexp(torch.cat([update[None, :], ahatf[:(i + 1) * m][None, :]]) / eps,
                                                    dim=0)

        sum_eta += eta

    ahatg -= torch.log(sum_eta) * eps
    ahatf -= torch.log(sum_eta) * eps
    f = evaluate_potential(ahatg, posy, x, eps)
    return evaluate_potential(ahatg, posy, x, eps), evaluate_potential(ahatf, posx, y, eps)


def evaluate_potential_finite(log_pot: torch.tensor, idx: torch.tensor, distance, eps):
    return - eps * torch.logsumexp((- distance[idx] + log_pot[None, :]) / eps, dim=1)


def stochastic_sinkhorn_finite(x, y, fref, gref, wref, eps, m, n_iter=100,
                               averaging='none', step_size='1/t', eta=1.,
                               scheme='alternated'):
    n = x.shape[0]
    p = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)
    q = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)

    if averaging == 'dual':
        avg_p = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)
        avg_q = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)
    elif averaging == 'primal':
        avg_ff = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)
        avg_gg = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)

    distance = compute_distance(x, y)
    trajs = []
    errors = defaultdict(list)
    sum_eta = 0
    period_sum_eta = 0
    for i in range(0, n_iter):
        if step_size == 'constant':
            eta = torch.tensor(eta)
        elif step_size == "1/2":
            eta = torch.tensor(1 / (i + 1))
        else:
            eta = torch.tensor(math.pow(i + 1, -.5))
        sum_eta += eta
        period_sum_eta += eta

        # Update f
        y_idx, logb = sample_from_finite(y, m)
        x_idx, loga = sample_from_finite(x, m)

        if i > 0:
            g = evaluate_potential_finite(p, y_idx, distance.transpose(0, 1), eps)
        else:
            g = torch.zeros(m, dtype=x.dtype)

        if scheme != 'alternated':
            if i > 0:
                f = evaluate_potential_finite(q, x_idx, distance, eps)
            else:
                f = torch.zeros(m, dtype=x.dtype)

        q += eps * torch.log(1 - eta)
        update = eps * math.log(eta) + logb * eps + g
        q[y_idx] = eps * torch.logsumexp(torch.cat([q[y_idx][None, :], update[None, :]], dim=0) / eps, dim=0)

        if averaging == 'dual':
            avg_q = eps * torch.logsumexp(torch.cat([q[None, :] + math.log(eta) * eps, avg_q[None, :]]) / eps, dim=0)

        # Update g
        if scheme == 'alternated':
            f = evaluate_potential_finite(q, x_idx, distance, eps)

        p += eps * torch.log(1 - eta)
        update = eps * math.log(eta) + loga * eps + f
        p[x_idx] = eps * torch.logsumexp(torch.cat([p[x_idx][None, :], update[None, :]], dim=0) / eps, dim=0)
        if averaging == 'dual':
            avg_p = eps * torch.logsumexp(torch.cat([p[None, :] + math.log(eta) * eps, avg_p[None, :]]) / eps, dim=0)

        if i % 100 == 0:
            if averaging == 'primal':
                this_ff = evaluate_potential_finite(q, slice(None), distance, eps)
                this_gg = evaluate_potential_finite(p, slice(None), distance.transpose(0, 1), eps)
                avg_ff = eps * torch.logsumexp(torch.cat([this_ff[None, :]
                                                          + math.log(period_sum_eta) * eps, avg_ff[None, :]]) / eps,
                                               dim=0)
                avg_gg = eps * torch.logsumexp(torch.cat([this_gg[None, :]
                                                          + math.log(period_sum_eta) * eps, avg_gg[None, :]]) / eps,
                                               dim=0)
                ff = avg_ff - eps * torch.log(sum_eta)
                gg = avg_gg - eps * torch.log(sum_eta)
                period_sum_eta = 0
            elif averaging == 'dual':
                ff = evaluate_potential_finite(avg_p - eps * torch.log(sum_eta), slice(None), distance, eps)
                gg = evaluate_potential_finite(avg_q - eps * torch.log(sum_eta), slice(None), distance.transpose(0, 1),
                                               eps)
            else:
                ff = evaluate_potential_finite(q, slice(None), distance, eps)
                gg = evaluate_potential_finite(p, slice(None), distance.transpose(0, 1), eps)

            w = ff.mean() + gg.mean()

            trajs.append([ff, gg, w.item()])

            lya = torch.abs(
                torch.sqrt(torch.exp(fref - ff).mean() * torch.exp(gref - gg).mean()) - 1)

            errors['f - fref'].append(var_norm(ff - fref))
            errors['g - gref'].append(var_norm(gg - gref))
            errors['lya'].append(lya)
            errors['w - wref'].append(torch.abs(w - wref))
            string = f"iter:{i} "
            for k, v in errors.items():
                string += f'[{k}]:{v[-1]:.3e} '
            print(string)
    return ff, gg, errors, trajs


def stochastic_sinkhorn_finite_simple(x, y, fref, gref, wref, eps, m, n_iter=100,
                                      ieta=.5, eta=1.):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    fref = torch.from_numpy(fref)
    gref = torch.from_numpy(gref)
    n = x.shape[0]
    p = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)
    q = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)

    avg_p = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)
    avg_q = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)

    distance = compute_distance(x, y)
    trajs = []
    errors = defaultdict(list)
    u_eta = not isinstance(eta, float)
    u_ieta = not isinstance(ieta, float)
    for i in range(0, n_iter):
        # Update f
        y_idx, logb = sample_from_finite(y, m)
        x_idx, loga = sample_from_finite(x, m)

        if i > 0:
            g = evaluate_potential_finite(p, y_idx, distance.transpose(0, 1), eps)
            f = evaluate_potential_finite(q, x_idx, distance, eps)
        else:
            g = torch.zeros(m, dtype=x.dtype)
            f = torch.zeros(m, dtype=x.dtype)
        if u_ieta:
            if ieta == '1/t':
                ieta_ = torch.tensor(1 / (i + 1))
            elif ieta == '1/sqrt(t)':
                ieta_ = torch.tensor(1 / math.sqrt(i + 1))
            else:
                raise ValueError
        else:
            ieta_ = torch.tensor(ieta)
        if u_eta:
            if eta == '1/t':
                eta_ = torch.tensor(1 / (i + 1))
            elif eta == '1/sqrt(t)':
                eta_ = torch.tensor(1 / math.sqrt(i + 1))
            else:
                raise ValueError
        else:
            eta_ = torch.tensor(eta)
        q += eps * torch.log(- ieta_ + 1)
        update = eps * torch.log(ieta_) + logb * eps + g
        q[y_idx] = eps * torch.logsumexp(torch.cat([q[y_idx][None, :], update[None, :]], dim=0) / eps, dim=0)

        p += eps * torch.log(- ieta_ + 1)
        update = eps * torch.log(ieta_) + loga * eps + f
        p[x_idx] = eps * torch.logsumexp(torch.cat([p[x_idx][None, :], update[None, :]], dim=0) / eps, dim=0)

        # eta_= torch.tensor(1.)
        avg_q += eps * torch.log(- eta_ + 1)
        update = eps * torch.log(eta_) + logb * eps + q[y_idx]
        avg_q[y_idx] = eps * torch.logsumexp(torch.cat([update[None, :], avg_q[y_idx][None, :]]) / eps, dim=0)

        avg_p += eps * torch.log(- eta_ + 1)
        update = eps * torch.log(eta_) + loga * eps + p[x_idx]
        avg_p[x_idx] = eps * torch.logsumexp(torch.cat([update[None, :], avg_p[x_idx][None, :]]) / eps, dim=0)

        if i % 100 == 0:
            ff = evaluate_potential_finite(avg_q, slice(None), distance, eps)
            gg = evaluate_potential_finite(avg_p, slice(None), distance.transpose(0, 1), eps)
            fff = evaluate_potential_finite(gg + math.log(1 / n) * eps, slice(None), distance, eps)
            ggg = evaluate_potential_finite(ff + math.log(1 / n) * eps, slice(None), distance.transpose(0, 1), eps)
            ff = (ff + fff) / 2
            gg = (gg + ggg) / 2
            w = ff.mean() + gg.mean()

            errors['varnorm'].append(var_norm(ff - fref) + var_norm(gg - gref).item())
            errors['rel varnorm'].append(
                ((var_norm(ff - fref) + var_norm(gg - gref)) / (var_norm(fref) + var_norm(fref))).item())
            errors['werr'].append(torch.abs(w - wref).item())
            errors['rel werr'].append(torch.abs((w - wref) / wref).item())
            errors['iter'].append(i)
            string = f"iter:{i} "
            for k, v in errors.items():
                string += f'[{k}]:{v[-1]:.3e} '
            print(string)
    ff = evaluate_potential_finite(avg_p, slice(None), distance, eps)
    gg = evaluate_potential_finite(avg_q, slice(None), distance.transpose(0, 1), eps)
    return ff.numpy(), gg.numpy(), errors


def var_norm(x):
    return x.max() - x.min()


def sinkhorn(x, y, eps, n_iter=1000, l=1.):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    n = x.shape[0]
    loga = torch.full((n,), fill_value=-math.log(n), dtype=torch.float64)
    logb = torch.full((n,), fill_value=-math.log(n), dtype=torch.float64)
    distance = compute_distance(x, y)
    g = torch.zeros((n,), dtype=torch.float64)
    f = torch.zeros((n,), dtype=torch.float64)
    for i in range(n_iter):
        ff = - l * eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
        f0 = ff[0].item()
        ff -= f0
        f_diff = f - ff

        gg = - l * eps * torch.logsumexp((- distance.transpose(0, 1) + ff[None, :]) / eps + loga[None, :], dim=1)
        g_diff = g - gg
        tol = f_diff.max() - f_diff.min() + g_diff.max() - g_diff.min()
        f, g = ff, gg
    print('tol', tol)
    return f.numpy(), g.numpy()


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


def sampling_sinkhorn(x_sampler, y_sampler, eps, m, grid, n_iter=100, step_size='sqrt'):
    posx = torch.zeros(m * n_iter, 1)
    q = torch.full((m * n_iter,), fill_value=-float('inf'))
    p = torch.full((m * n_iter,), fill_value=-float('inf'))

    posy = torch.zeros(m * n_iter, 1)
    fevals = []
    gevals = []
    for i in range(0, n_iter):
        if step_size == 'sqrt':
            eta = torch.tensor(i + 1.).pow(torch.tensor(-0.51))
        elif step_size == 'linear':
            eta = torch.tensor(1. / (i + 1))
        elif step_size == 'constant':
            eta = torch.tensor(.9)

        # Update f
        y_, logb = y_sampler(m)
        if i > 0:
            g = evaluate_potential(p[:i * m], posx[:i * m], y_, eps)
        else:
            g = torch.zeros(m)
        q[:i * m] += eps * torch.log(1 - eta)
        q[i * m:(i + 1) * m] = eps * math.log(eta) + logb * eps + g
        posy[i * m:(i + 1) * m] = y_

        # Update g
        x_, loga = x_sampler(m)
        f = evaluate_potential(q[:(i + 1) * m], posy[:(i + 1) * m], x_, eps)
        p[:i * m] += eps * torch.log(1 - eta)
        p[i * m:(i + 1) * m] = eps * math.log(eta) + loga * eps + f
        posx[i * m:(i + 1) * m] = x_

        if i % 10 == 0:
            feval = evaluate_potential(q[:(i + 1) * m], posy[:(i + 1) * m], grid, eps)
            geval = evaluate_potential(p[:(i + 1) * m], posx[:(i + 1) * m], grid, eps)
            # plan = (x_sampler.log_prob(grid)[:, None] + feval[:, None] / eps + y_sampler.log_prob(grid)[None, :]
            #         + geval[None, :] / eps - C / eps)
            # print(torch.logsumexp(plan.view(-1), dim=0))
            fevals.append(feval)
            gevals.append(geval)
    return p, posx, q, posy, fevals, gevals


def one_dimensional_exp():
    eps = 1e-3

    grid = torch.linspace(-4, 12, 500)[:, None]
    C = compute_distance(grid, grid)

    x_sampler = Sampler(mean=torch.tensor([[1.], [2], [3]]), cov=torch.tensor([[[.1]], [[.1]], [[.1]]]),
                        p=torch.ones(3) / 3)
    y_sampler = Sampler(mean=torch.tensor([[0.], [3], [5]]), cov=torch.tensor([[[.1]], [[.1]], [[.4]]]),
                        p=torch.ones(3) / 3)

    lpx = x_sampler.log_prob(grid)
    lpy = y_sampler.log_prob(grid)
    lpx -= torch.logsumexp(lpx, dim=0)
    lpy -= torch.logsumexp(lpy, dim=0)
    px = torch.exp(lpx)
    py = torch.exp(lpy)

    fevals = []
    gevals = []
    labels = []
    plans = []

    n_samples = 1000
    x, loga = x_sampler(n_samples)
    y, logb = y_sampler(n_samples)
    f, g = sinkhorn(x, y, eps=eps, n_iter=100)
    distance = compute_distance(grid, y)
    feval = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
    distance = compute_distance(grid, x)
    geval = - eps * torch.logsumexp((- distance + f[None, :]) / eps + loga[None, :], dim=1)

    plan = (lpx[:, None] + feval[:, None] / eps + lpy[None, :]
            + geval[None, :] / eps - C / eps)

    plans.append((plan, grid, grid))

    fevals.append(feval)
    gevals.append(geval)
    labels.append(f'Sinkhorn n={n_samples}')

    m = 10
    hatf, posx, hatg, posy, sto_fevals, sto_gevals = sampling_sinkhorn(x_sampler, y_sampler, m=m, eps=eps,
                                                                       n_iter=100,
                                                                       step_size='constant', grid=grid)
    feval = evaluate_potential(hatg, posy, grid, eps)
    geval = evaluate_potential(hatf, posx, grid, eps)
    plan = (lpx[:, None] + feval[:, None] / eps + lpy[None, :]
            + geval[None, :] / eps - C / eps)
    plans.append((plan, grid, grid))

    fevals.append(feval)
    gevals.append(geval)
    labels.append(f'Online Sinkhorn m={m}')
    fevals = torch.cat([feval[None, :] for feval in fevals], dim=0)
    gevals = torch.cat([geval[None, :] for geval in gevals], dim=0)

    fig, axes = plt.subplots(4, 1, figsize=(8, 12))
    fig_plan, axes_plan = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].plot(grid, px, label='alpha')
    axes[0].plot(grid, py, label='beta')
    axes[0].legend()
    for i, (label, feval, geval, (plan, x, y)) in enumerate(zip(labels, fevals, gevals, plans)):
        axes[1].plot(grid, feval, label=label)
        axes[2].plot(grid, geval, label=label)
        plan = plan.numpy()
        axes_plan[i].contourf(y[:, 0], x[:, 0], plan, levels=30)
    # axes_plan[1].add_colorbar()
    colors = plt.cm.get_cmap('Blues')(np.linspace(0.2, 1, len(sto_fevals)))
    axes[3].set_prop_cycle('color', colors)
    for eval in sto_fevals:
        axes[3].plot(grid, eval, label=label)
    axes[2].legend()
    axes[0].set_title('Distributions')
    axes[1].set_title('Potential f')
    axes[2].set_title('Potential g')
    plt.show()


def main():
    n = 2
    m = 1
    eps = 1

    torch.manual_seed(100)
    np.random.seed(100)

    y = np.random.randn(n, 10)
    x = np.random.randn(n, 10) + 2

    mem = Memory(location=expanduser('~/cache'))
    print('===========================================True=====================================')
    fref, gref = mem.cache(sinkhorn)(x, y, eps=eps, n_iter=100, l=1.)
    wref = fref.mean() + gref.mean()
    print('wref', wref)
    print('===========================================Finite stochastic=====================================')
    results = []

    def single(eta, ieta, m):
        torch.manual_seed(100)
        np.random.seed(100)
        smd_f, smd_g, errors = mem.cache(stochastic_sinkhorn_finite_simple)(x, y, fref=fref,
                                                                            gref=gref, ieta=ieta,
                                                                            eta=eta,
                                                                            eps=eps, wref=wref, m=m,
                                                                            n_iter=int(1e6), )
        return dict(m=m, ieta=ieta, eta=eta, name='', errors=errors)
    results = Parallel(n_jobs=18)(delayed(single)(eta, ieta, m)
                                 for m in [1, 2]
                                 for eta in [1., '1/sqrt(t)', '1/t']
                                 for ieta in [1., '1/sqrt(t)', '1/t'])
    if not os.path.exists(expanduser('~/output/online_sinkhorn')):
        os.makedirs(expanduser('~/output/online_sinkhorn'))
    joblib.dump(results, expanduser('~/output/online_sinkhorn/results_1e6_2.pkl'))


def plot():
    # results = joblib.load(expanduser('~/output/online_sinkhorn/results_1e6_2.pkl'))
    results = joblib.load(expanduser('~/output/online_sinkhorn/results_1e5.pkl'))
    df = []
    for result in results:
        for i in range(len(result['errors']['iter'])):
            df.append(dict(ieta=result["ieta"], eta=result["eta"],
                           ie=f'{result["ieta"]}_{result["eta"]}',
                           m=result["m"], iter=result['errors']['iter'][i],
                      value=result['errors']['rel werr'][i]))
    df = pd.DataFrame(df)
    grid = sns.FacetGrid(data=df, col="eta", row="ieta", hue='m')
    grid.map(plt.plot, "iter", "value")
    for ax in grid.axes.ravel():
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim([1e-15, 1])
        ax.set_xlim([1e2, 1e6])
    grid.add_legend()

    grid = sns.FacetGrid(data=df, col="m", hue='ie')
    grid.map(plt.plot, "iter", "value")
    for ax in grid.axes.ravel():
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_ylim([1e-15, 1])
        ax.set_xlim([1e2, 1e6])
    grid.add_legend()

    plt.show()


if __name__ == '__main__':
    # main()
    plot()
    # one_dimensional_exp()
