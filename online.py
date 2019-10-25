import math

import numpy as np
import torch

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
    hatf = torch.zeros(m * n_iter)
    posy = torch.zeros(m * n_iter, 2)
    hatg = torch.zeros(m * n_iter)
    posx = torch.zeros(m * n_iter, 2)

    for i in range(0, n_iter):
        if step_size == 'sqrt':
            eta = torch.tensor(1. / math.sqrt(i + 1))
        elif step_size == 'constant':
            eta = torch.tensor(1.)

        # Update f
        y_, logb = sample_from(y, m)
        if i > 0:
            g = evaluate_potential(hatf[:i * m], posx[:i * m], y_, eps)
        else:
            g = torch.zeros(m)
        hatg[:i * m] += eps * torch.log(1 - eta)
        hatg[i * m:(i + 1) * m] = eps * math.log(eta) + logb * eps + g
        posy[i * m:(i + 1) * m] = y_

        # Update g
        x_, loga = sample_from(x, m)
        f = evaluate_potential(hatg[:(i + 1) * m], posy[:(i + 1) * m], x_, eps)
        hatf[:i * m] += eps * torch.log(1 - eta)
        hatf[i * m:(i + 1) * m] = eps * math.log(eta) + loga * eps + f
        posx[i * m:(i + 1) * m] = x_
    return evaluate_potential(hatg, posy, x, eps), evaluate_potential(hatf, posx, y, eps)


def evaluate_potential_finite(log_pot: torch.tensor, idx: torch.tensor, distance, eps):
    return - eps * torch.logsumexp((- distance[idx] + log_pot[None, :]) / eps, dim=1)


def stochasic_sinkhorn_finite(x, y, eps, m, n_iter=100, step_size='sqrt'):
    n = x.shape[0]
    hatf = torch.full((n,), fill_value=-float('inf'))
    hatg = torch.full((n,), fill_value=-float('inf'))
    avg_f = torch.zeros_like(hatf)
    avg_g = torch.zeros_like(hatg)
    distance = compute_distance(x, y)
    sum_eta = 0
    for i in range(0, n_iter):
        if step_size == 'sqrt':
            eta = torch.tensor(1. / math.sqrt(i + 1))
        elif step_size == 'constant':
            eta = torch.tensor(1.)

        # Update f
        y_idx, logb = sample_from_finite(y, m)
        if i > 0:
            g = evaluate_potential_finite(hatf, y_idx, distance.transpose(0, 1), eps)
        else:
            g = torch.zeros(m)
        hatg += eps * torch.log(1 - eta)
        update = eps * math.log(eta) + logb * eps + g
        hatg[y_idx] = eps * torch.logsumexp(torch.cat([hatg[y_idx][None, :], update[None, :]], dim=0) / eps, dim=0)
        # Update g
        x_idx, loga = sample_from_finite(x, m)
        f = evaluate_potential_finite(hatg, x_idx, distance, eps)
        hatf += eps * torch.log(1 - eta)
        update = eps * math.log(eta) + loga * eps + f
        hatf[x_idx] = eps * torch.logsumexp(torch.cat([hatf[x_idx][None, :], update[None, :]], dim=0) / eps, dim=0)
        sum_eta += eta
    avg_g /= sum_eta
    avg_f /= sum_eta
    return (evaluate_potential_finite(hatg, slice(None), distance, eps), \
            evaluate_potential_finite(hatf, slice(None), distance.transpose(0, 1), eps))


def sinkhorn(x, y, eps, n_iter=100):
    n = x.shape[0]
    loga = torch.full((n,), fill_value=-math.log(n))
    logb = torch.full((n,), fill_value=-math.log(n))
    distance = compute_distance(x, y)
    g = torch.zeros((n,))
    f = torch.zeros((n,))
    for i in range(n_iter):
        # print(f, g)
        fn = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
        gn = - eps * torch.logsumexp((- distance.transpose(0, 1) + fn[None, :]) / eps + loga[None, :], dim=1)
        f_diff = f - fn
        g_diff = g - gn
        f = fn
        g = gn
        tol = f_diff.max() - f_diff.min() + g_diff.max() - g_diff.min()
    return f, g


def main():
    n = 10
    m = 2
    eps = 1

    torch.manual_seed(10)
    np.random.seed(10)

    y = torch.randn(n, 2)
    x = torch.randn((n, 2))
    print('===========================================True=====================================')
    f, g = sinkhorn(x, y, eps=eps, n_iter=2000)
    print(f.mean() + g.mean())
    # print('===========================================Stochastic=====================================')
    # torch.manual_seed(100)
    # np.random.seed(100)
    # smd_f, smd_g = stochasic_sinkhorn(x, y, eps=eps, m=m, n_iter=1000, step_size='sqrt')
    # print(smd_f.mean() + smd_g.mean())
    print('===========================================Finite stochastic=====================================')
    torch.manual_seed(100)
    np.random.seed(100)
    smd_f_finite, smd_g_finite = stochasic_sinkhorn_finite(x, y, eps=eps, m=m, n_iter=2000, step_size='sqrt')
    print((f - smd_f_finite), (g - smd_g_finite))
    print(smd_f_finite.mean() + smd_g_finite.mean())


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

    def __call__(self, n):
        k, d = self.mean.shape
        indices = np.random.choice(k, n, p=self.p.numpy())
        pos = np.zeros((n, d), dtype=np.float32)
        for i in range(k):
            mask = indices == i
            size = mask.sum()
            pos[mask] = np.random.multivariate_normal(self.mean[i], self.cov[i], size=size)
        logweight = np.full_like(pos[:, 0], fill_value=-math.log(n))
        return torch.from_numpy(pos), torch.from_numpy(logweight)

    def log_prob(self, x):
        # b, d = x.shape
        diff = x[:, None, :] - self.mean[None, :]  # b, k, d
        return torch.sum(self.p[None, :] * torch.exp(-torch.einsum('bkd,kde,bke->bk',
                                                                   [diff, self.icov, diff]) / 2) / self.norm, dim=1)


def sampling_sinkhorn(x_sampler, y_sampler, eps, m, grid, n_iter=100, step_size='sqrt'):
    hatf = torch.zeros(m * n_iter)
    posx = torch.zeros(m * n_iter, 2)
    hatg = torch.zeros(m * n_iter)
    posy = torch.zeros(m * n_iter, 2)
    w = 0
    fevals = []
    for i in range(0, n_iter):
        if step_size == 'sqrt':
            eta = torch.tensor(1. / math.sqrt(i + 1))
        elif step_size == 'linear':
            eta = torch.tensor(1. / (i + 1))
        elif step_size == 'constant':
            eta = torch.tensor(1.)

        # Update f
        y_, logb = y_sampler(m)
        if i > 0:
            g = evaluate_potential(hatf[:i * m], posx[:i * m], y_, eps)
        else:
            g = torch.zeros(m)
        hatg[:i * m] += eps * torch.log(1 - eta)
        hatg[i * m:(i + 1) * m] = eps * math.log(eta) + logb * eps + g
        posy[i * m:(i + 1) * m] = y_

        # Update g
        x_, loga = x_sampler(m)
        f = evaluate_potential(hatg[:(i + 1) * m], posy[:(i + 1) * m], x_, eps)
        hatf[:i * m] += eps * torch.log(1 - eta)
        hatf[i * m:(i + 1) * m] = eps * math.log(eta) + loga * eps + f
        posx[i * m:(i + 1) * m] = x_
        w *= 1 - eta
        w += eta * (hatf[i * m:(i + 1) * m].mean() + hatg[i * m:(i + 1) * m].mean())

        if i % 10 == 0:
            fevals.append(evaluate_potential(hatg, posy, grid, eps))

    return hatf, posx, hatg, posy, w, fevals


def one_dimensional_exp():
    eps = 1e-1

    grid = torch.linspace(-1, 7, 100)[:, None]

    x_sampler = Sampler(mean=torch.tensor([[1.], [2], [3]]), cov=torch.tensor([[[.1]], [[.1]], [[.1]]]),
                        p=torch.ones(3) / 3)
    y_sampler = Sampler(mean=torch.tensor([[0.], [3], [5]]), cov=torch.tensor([[[.1]], [[.1]], [[.4]]]),
                        p=torch.ones(3) / 3)

    x_sampler = Sampler(mean=torch.tensor([[0.]]), cov=torch.tensor([[[.1]]]), p=torch.ones(1))
    y_sampler = Sampler(mean=torch.tensor([[2.]]), cov=torch.tensor([[[.1]]]), p=torch.ones(1))

    px = torch.exp(x_sampler.log_prob(grid))
    py = torch.exp(y_sampler.log_prob(grid))

    fevals = []
    gevals = []
    labels = []
    ws = []
    for n_samples in [1, 10, 100, 1000]:
        x, loga = x_sampler(n_samples)
        y, logb = y_sampler(n_samples)
        f, g = sinkhorn(x, y, eps=eps, n_iter=100)
        ws.append(torch.mean(f) + torch.mean(g))
        distance = compute_distance(grid, y)
        feval = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
        distance = compute_distance(grid, x)
        geval = - eps * torch.logsumexp((- distance + f[None, :]) / eps + loga[None, :], dim=1)
        fevals.append(feval)
        gevals.append(geval)
        labels.append(f'Sinkhorn n={n_samples}')
    for n_samples in [100]:
        hatf, posx, hatg, posy, w, evals = sampling_sinkhorn(x_sampler, y_sampler, m=n_samples, eps=eps, n_iter=100,
                                                             step_size='sqrt', grid=grid)
        ws.append(w)
        feval = evaluate_potential(hatg, posy, grid, eps)
        geval = evaluate_potential(hatf, posx, grid, eps)
        fevals.append(feval)
        gevals.append(geval)
        labels.append(f'Online Sinkhorn n={n_samples}')
    print(ws)
    fevals = torch.cat([feval[None, :] for feval in fevals], dim=0)
    gevals = torch.cat([geval[None, :] for geval in gevals], dim=0)

    fig, axes = plt.subplots(4, 1, figsize=(8, 12))
    axes[0].plot(grid, px, label='alpha')
    axes[0].plot(grid, py, label='beta')
    axes[0].legend()
    for label, feval, geval in zip(labels, fevals, gevals):
        axes[1].plot(grid, feval, label=label)
        axes[2].plot(grid, geval, label=label)
    colors = plt.cm.get_cmap('Blues')(np.linspace(0.2, 1, len(evals)))
    axes[3].set_prop_cycle('color', colors)
    for eval in evals:
        axes[3].plot(grid, eval, label=label)
    axes[2].legend()
    axes[0].set_title('Distributions')
    axes[1].set_title('Potential f')
    axes[2].set_title('Potential g')
    plt.show()


if __name__ == '__main__':
    main()
    # one_dimensional_exp()
