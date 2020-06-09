import math
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from joblib import Memory
from matplotlib import gridspec

import matplotlib as mpl
from matplotlib import rc

from onlikhorn.dataset import get_output_dir

mpl.rcParams['font.size'] = 7
mpl.rcParams['backend'] = 'pdf'
rc('text', usetex=True)

pt_width = 397.48499
pt_per_inch = 72.27
width = pt_width / pt_per_inch

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

def sample_from(x, m):
    n = x.shape[0]
    indices = torch.from_numpy(np.random.permutation(n)[:m])
    loga = torch.full((m,), fill_value=-math.log(m))
    return x[indices], loga


def compute_distance(x, y):
    return (torch.sum((x[:, None, :] ** 2 + y[None, :, :] ** 2 - 2 * x[:, None, :] * y[None, :, :]), dim=2)) / 2


def evaluate_potential(log_pot: torch.tensor, pos: torch.tensor, x: torch.tensor, eps):
    distance = compute_distance(x, pos)
    return - eps * torch.logsumexp((- distance + log_pot[None, :]) / eps, dim=1)


def evaluate_kernel(weights, pos, x, sigma):
    C = torch.exp(-compute_distance(x, pos) / 2 / (sigma ** 2))
    return torch.sum(weights[None, :] * C, dim=1)


def var_norm(x):
    return x.max() - x.min()


def stochastic_sinkhorn(x, y, eps, m, n_iter=100, step_size='sqrt'):
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


def sinkhorn(x, y, eps, n_iter=1000, simultaneous=False):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    n = x.shape[0]
    loga = torch.full((n,), fill_value=-math.log(n), dtype=torch.float64)
    logb = torch.full((n,), fill_value=-math.log(n), dtype=torch.float64)
    distance = compute_distance(x, y)
    g = torch.zeros((n,), dtype=torch.float64)
    f = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
    for i in range(n_iter):
        ff = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
        f_diff = f - ff
        if simultaneous:
            f_used = f
        else:
            f_used = ff
        gg = - eps * torch.logsumexp((- distance.transpose(0, 1) + f_used[None, :]) / eps + loga[None, :], dim=1)
        g_diff = g - gg
        tol = f_diff.max() - f_diff.min() + g_diff.max() - g_diff.min()
        f, g = ff, gg
        print('tol', tol)
    return f.numpy(), g.numpy()


def sampling_sinkhorn(x_sampler, y_sampler, eps, m, grid, n_iter=100):
    posx = torch.zeros(m * n_iter, 1)
    q = torch.full((m * n_iter,), fill_value=-float('inf'))
    p = torch.full((m * n_iter,), fill_value=-float('inf'))

    posy = torch.zeros(m * n_iter, 1)
    fevals = []
    gevals = []
    for i in range(0, n_iter):
        eta = torch.tensor(i + 1.).pow(torch.tensor(-0.5))

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
            fevals.append(feval)
            gevals.append(geval)
    return p, posx, q, posy, fevals, gevals


def rkhs_sinkhorn(x_sampler, y_sampler, eps, m, grid, n_iter=1000, sigma=.1):
    posx = torch.zeros(m * n_iter, 1)
    alpha = torch.full((m * n_iter,), fill_value=0)

    posy = torch.zeros(m * n_iter, 1)
    fevals = []
    gevals = []
    for i in range(0, n_iter):
        eta = torch.tensor(i + 1.).pow(torch.tensor(-0.5)) * 0.01

        # Update f
        x_, loga = x_sampler(m)
        y_, logb = y_sampler(m)

        if i > 0:
            f = evaluate_kernel(alpha[:i * m], posx[:i * m], x_, sigma)
            g = evaluate_kernel(alpha[:i * m], posy[:i * m], y_, sigma)
        else:
            g = torch.zeros(m)
            f = torch.zeros(m)
        C = torch.sum((x_ - y_) ** 2, dim=1) / 2
        alpha[i * m:(i + 1) * m] = eta * (- torch.exp((f + g - C) / eps) + 1)
        posy[i * m:(i + 1) * m] = y_
        posx[i * m:(i + 1) * m] = x_
        if i % 10 == 0:
            feval = evaluate_kernel(alpha[:(i + 1) * m], posx[:(i + 1) * m], grid, sigma)
            geval = evaluate_kernel(alpha[:(i + 1) * m], posy[:(i + 1) * m], grid, sigma)
            fevals.append(feval)
            gevals.append(geval)
    return alpha, posx, posy, fevals, gevals


def one_dimensional_exp():
    eps = 1e-1

    grid = torch.linspace(-4, 12, 500)[:, None]
    C = compute_distance(grid, grid)

    x_sampler = Sampler(mean=torch.tensor([[1.], [2], [3]]), cov=torch.tensor([[[.1]], [[.1]], [[.1]]]),
                        p=torch.ones(3) / 3)
    y_sampler = Sampler(mean=torch.tensor([[0.], [3], [5]]), cov=torch.tensor([[[.1]], [[.1]], [[.4]]]),
                        p=torch.ones(3) / 3)
    torch.manual_seed(100)
    np.random.seed(100)
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

    mem = Memory(location=expanduser('~/cache'))
    n_samples = 5000
    x, loga = x_sampler(n_samples)
    y, logb = y_sampler(n_samples)
    f, g = mem.cache(sinkhorn)(x.numpy(), y.numpy(), eps=eps, n_iter=100)
    f = torch.from_numpy(f).float()
    g = torch.from_numpy(g).float()
    distance = compute_distance(grid, y)
    feval = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
    distance = compute_distance(grid, x)
    geval = - eps * torch.logsumexp((- distance + f[None, :]) / eps + loga[None, :], dim=1)

    plan = (lpx[:, None] + feval[:, None] / eps + lpy[None, :]
            + geval[None, :] / eps - C / eps)

    plans.append((plan, grid, grid))

    fevals.append(feval)
    gevals.append(geval)
    labels.append(f'True potential')

    m = 50
    hatf, posx, hatg, posy, sto_fevals, sto_gevals = sampling_sinkhorn(x_sampler, y_sampler, m=m, eps=eps,
                                                                       n_iter=100,
                                                                       grid=grid)
    feval = evaluate_potential(hatg, posy, grid, eps)
    geval = evaluate_potential(hatf, posx, grid, eps)
    plan = (lpx[:, None] + feval[:, None] / eps + lpy[None, :]
            + geval[None, :] / eps - C / eps)
    plans.append((plan, grid, grid))

    fevals.append(feval)
    gevals.append(geval)
    labels.append(f'Online Sinkhorn')

    alpha, posx, posy, _, _ = rkhs_sinkhorn(x_sampler, y_sampler, m=m, eps=eps,
                                                                 n_iter=100, sigma=1,
                                                                 grid=grid)
    feval = evaluate_kernel(alpha, posx, grid, sigma=1)
    geval = evaluate_kernel(alpha, posy, grid, sigma=1)
    plan = (lpx[:, None] + feval[:, None] / eps + lpy[None, :]
            + geval[None, :] / eps - C / eps)
    plans.append((plan, grid, grid))
    fevals.append(feval)
    gevals.append(geval)
    labels.append(f'RKHS')

    fevals = torch.cat([feval[None, :] for feval in fevals], dim=0)
    gevals = torch.cat([geval[None, :] for geval in gevals], dim=0)

    fig = plt.figure(figsize=(width, .21 * width))
    gs = gridspec.GridSpec(ncols=6, nrows=1, width_ratios=[1, 1.5, 1.5, .8, .8, .8], figure=fig)
    plt.subplots_adjust(right=0.97, left=0.05, bottom=0.27, top=0.85)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(grid, px, label=r'$\alpha$')
    ax0.plot(grid, py, label=r'$\beta$')
    ax0.legend(frameon=False)
    # ax0.axis('off')
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])
    ax4 = fig.add_subplot(gs[4])
    ax5 = fig.add_subplot(gs[5])

    for i, (label, feval, geval, (plan, x, y)) in enumerate(zip(labels, fevals, gevals, plans)):
        if label == 'True potential':
            ax1.plot(grid, feval, label=label, zorder=100 if label == 'True Potential' else 1,
                     linewidth=3 if label == 'True potential' else 1, color='C3')
            ax2.plot(grid, geval, label=None, zorder=100 if label == 'True Potential' else 1,
                     linewidth=3 if label == 'True potential' else 1, color='C3')
            plan = plan.numpy()
            ax3.contour(y[:, 0], x[:, 0], plan, levels=30)
        elif label == 'RKHS':
            ax1.plot(grid, feval, label=label, linewidth=2, color='C4')
            ax2.plot(grid, geval, label=None, linewidth=2, color='C4')
            plan = plan.numpy()
            ax4.contour(y[:, 0], x[:, 0], plan, levels=30)
        else:
            plan = plan.numpy()
            ax5.contour(y[:, 0], x[:, 0], plan, levels=30)
    ax0.set_title('Distributions')
    ax1.set_title('Estimated $f$')
    ax2.set_title('Estimated $g$')
    ax3.set_title('True OT plan')
    ax4.set_title('RKHS')
    ax5.set_title('O-S')
    # ax4.set_title('Estimated OT plan')
    colors = plt.cm.get_cmap('Blues')(np.linspace(0.2, 1, len(sto_fevals[::2])))
    for i, eval in enumerate(sto_fevals[::2]):
        ax1.plot(grid, eval, color=colors[i],
                 linewidth=2, label=f'O-S $n_t={i * 10 * 2 * 50}$' if i % 2 == 0 else None,
                 zorder=1)
    for i, eval in enumerate(sto_gevals[::2]):
        ax2.plot(grid, eval, color=colors[i],
                 linewidth=2, label=None,
                 zorder=1)
    for ax in (ax1, ax2, ax3):
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.tick_params(axis='both', which='minor', labelsize=5)
        ax.minorticks_on()
    ax1.legend(frameon=False, bbox_to_anchor=(-1, -0.53), ncol=5, loc='lower left')
    # ax2.legend(frameon=False, bbox_to_anchor=(0., 1), loc='upper left')
    sns.despine(fig)
    for ax in [ax3, ax4, ax5]:
        ax.axis('off')
    ax0.axes.get_yaxis().set_visible(False)
    plt.savefig(join(get_output_dir(), 'continuous.pdf'))
    plt.show()


if __name__ == '__main__':
    # Figure 2
    one_dimensional_exp()
