import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.optim import Adam, SGD

from online import Sampler


def compute_distance(x, y):
    return ((x ** 2).sum(dim=1)[:, None] + (y ** 2).sum(dim=1)[None, :] - 2 * x @ y.transpose(0, 1)) / 2


class Potential(nn.Module):
    def __init__(self, in_features, n_samples, eps=1e-3):
        super(Potential, self).__init__()
        self.pos = Parameter(torch.linspace(-1, 7, n_samples)[:, None])
        self.eps = eps
        self.logprob = Parameter(torch.full((n_samples,), fill_value=-math.log(n_samples)))
        self.pot = Parameter(torch.zeros(n_samples))

    def forward(self, x: torch.tensor):
        logprob = self.logprob - torch.logsumexp(self.logprob, dim=0)
        distance = compute_distance(x, self.pos)
        term = logprob[None, :] + (self.pot[None, :] - distance) / self.eps
        return - self.eps * torch.logsumexp(term, dim=1)


class MLPPotential(nn.Module):
    def __init__(self, in_features, n_samples, eps=1e-3):
        super(MLPPotential, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, x):
        return self.mlp(x)[:, 0]


def dual_objective(f, g, x, y, loga, logb, eps):
    evalf = f(x)
    evalg = g(y)
    C = compute_distance(x, y)
    penalty = eps * torch.logsumexp(
        ((loga + evalf / eps)[:, None] + (logb + evalg / eps)[None, :] - C / eps).view(-1), dim=0)
    return - torch.sum(torch.exp(loga) * evalf) - torch.sum(torch.exp(logb) * evalg) + penalty


def sinkhorn(x, y, loga, logb, eps, n_iter=1000):
    n = x.shape[0]
    m = y.shape[0]
    distance = compute_distance(x, y)
    g = torch.zeros((m,))
    f = torch.zeros((n,))
    for i in range(n_iter + 1):
        ff = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
        gg = - eps * torch.logsumexp((- distance.transpose(0, 1) + ff[None, :]) / eps + loga[None, :], dim=1)
        f_diff = f - ff
        g_diff = g - gg
        tol = f_diff.max() - f_diff.min() + g_diff.max() - g_diff.min()
        # f = (ff + f) / 2
        # g = (gg + g) / 2
        f = ff
        g = gg
        # print('tol', tol)
    return f, g


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    eps = 1
    n = 10
    m = 20
    x = torch.randn(n, 1)
    y = torch.randn(m, 1)
    loga = torch.full((n,), fill_value=-math.log(n))
    logb = torch.full((m,), fill_value=-math.log(m))

    ff, gg = sinkhorn(x, y, loga, logb, eps=eps)

    C = compute_distance(x, y)
    penalty = eps * torch.exp((loga + ff / eps)[:, None] + (logb + gg / eps)[None, :] - C / eps)
    print(penalty.sum())
    print(torch.sum(torch.exp(loga) * ff) + torch.sum(torch.exp(logb) * gg))

    f = MLPPotential(1, n, eps)
    g = MLPPotential(1, m, eps)
    optimizer = Adam(list(f.parameters()) + list(g.parameters()), lr=1e-1)
    for i in range(1):
        obj = dual_objective(f, g, x, y, loga, logb, eps)
        optimizer.zero_grad()
        obj.backward()
        optimizer.step()
        print(obj.item())


def one_dimensional_exp():
    eps = 1

    torch.manual_seed(100)
    np.random.seed(100)

    grid = torch.linspace(-1, 7, 100)[:, None]

    x_sampler = Sampler(mean=torch.tensor([[1.], [2], [3]]), cov=torch.tensor([[[.1]], [[.1]], [[.1]]]),
                        p=torch.ones(3) / 3)
    y_sampler = Sampler(mean=torch.tensor([[0.], [3], [5]]), cov=torch.tensor([[[.1]], [[.1]], [[.4]]]),
                        p=torch.ones(3) / 3)
    #
    # x_sampler = Sampler(mean=torch.tensor([[0.]]), cov=torch.tensor([[[.1]]]), p=torch.ones(1))
    # y_sampler = Sampler(mean=torch.tensor([[2.]]), cov=torch.tensor([[[.1]]]), p=torch.ones(1))

    px = torch.exp(x_sampler.log_prob(grid))
    py = torch.exp(y_sampler.log_prob(grid))

    fevals = []
    gevals = []
    labels = []
    ws = []
    n_samples = 100
    x, loga = x_sampler(n_samples)
    y, logb = y_sampler(n_samples)
    f, g = sinkhorn(x, y, loga, logb, eps=eps, n_iter=100)
    print('Wasserstein', torch.mean(f) + torch.mean(g))
    distance = compute_distance(grid, y)
    feval = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
    distance = compute_distance(grid, x)
    geval = - eps * torch.logsumexp((- distance + f[None, :]) / eps + loga[None, :], dim=1)
    fevals.append(feval)
    gevals.append(geval)
    labels.append(f'Sinkhorn n={n_samples}')

    # Sampling Sinkhorn
    n_samples = 1000
    F = Potential(1, n_samples, eps)
    G = Potential(1, n_samples, eps)

    # G.pot.data = f.clone()
    # G.logprob.data = loga.clone()
    # G.pos.data = x.clone()
    # F.pot.data = g.clone()
    # F.logprob.data = logb.clone()
    # F.pos.data = y.clone()
    #
    optimizer = Adam(list(F.parameters()) + list(G.parameters()), lr=1e-3)
    for i in range(1000):
        x, loga = x_sampler(100)
        y, logb = y_sampler(100)
        obj = dual_objective(F, G, x, y, loga, logb, eps)
        print(obj.item())
        optimizer.zero_grad()
        obj.backward()
        optimizer.step()
    with torch.no_grad():
        feval = F(grid)
        geval = G(grid)

    fevals.append(feval)
    gevals.append(geval)
    labels.append(f'Online Sinkhorn n={n_samples}')

    fevals = torch.cat([feval[None, :] for feval in fevals], dim=0)
    gevals = torch.cat([geval[None, :] for geval in gevals], dim=0)

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    axes[0].plot(grid, px, label='alpha')
    axes[0].plot(grid, py, label='beta')
    axes[0].legend()
    for label, feval, geval in zip(labels, fevals, gevals):
        axes[1].plot(grid, feval, label=label)
        axes[2].plot(grid, geval, label=label)
    # colors = plt.cm.get_cmap('Blues')(np.linspace(0.2, 1, len(sto_fevals)))
    # axes[3].set_prop_cycle('color', colors)
    # for eval in sto_fevals:
    #     axes[3].plot(grid, eval, label=label)
    axes[2].legend()
    axes[0].set_title('Distributions')
    axes[1].set_title('Potential f')
    axes[2].set_title('Potential g')
    plt.show()


if __name__ == '__main__':
    one_dimensional_exp()
