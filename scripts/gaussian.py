import matplotlib.pyplot as plt
import torch

from onlikhorn.algorithm import sinkhorn
from onlikhorn.dataset import GaussianSampler

import numpy as np


def main():
    A = torch.diag(torch.tensor([.1]))
    mu = torch.tensor([3])
    B = torch.diag(torch.tensor([1.]))
    nu = torch.tensor([-1])

    x_sampler = GaussianSampler(mu, A)
    y_sampler = GaussianSampler(nu, B)

    epsilon = 1e-2

    F, G = sinkhorn_gaussian(x_sampler, y_sampler, epsilon=epsilon)
    x, la, _ = x_sampler(10000)
    y, lb, _ = y_sampler(10000)
    Ft, Gt, _ = sinkhorn(x, la, y, lb, n_iter=10, epsilon=epsilon, verbose=True, save_trace=True)
    z = torch.linspace(-5, 5, 100)[:, None]
    f = F(z)
    g = G(z)

    ft = Ft(z)
    gt = Gt(z)

    lf = x_sampler.log_prob(z)
    lg = y_sampler.log_prob(z)

    fig, (ax, ax_pot, ax_points) = plt.subplots(1, 3, sharex=True)
    ax_pot.plot(z, f, label='f')
    ax_pot.plot(z, g, label='g')
    ax_pot.plot(z, ft, label='ft')
    ax_pot.plot(z, gt, label='gt')
    ax.plot(z, lf, label='log(alpha)')
    ax.plot(z, lg, label='log(beta)')
    ax_points.scatter(x, torch.ones(len(x)), marker='+', label='alpha')
    ax_points.scatter(y, 2 * torch.ones(len(y)), marker='+', label='beta')
    ax.legend()
    ax_pot.legend()
    plt.show()


if __name__ == '__main__':
    main()
