import math
from typing import List, Union, Optional

import numpy as np
import torch
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

def safe_log(x):
    if x == 0:
        return - float('inf')
    else:
        return np.log(x)

def from_numpy(*vectors, device='cpu'):
    return [torch.from_numpy(vector).float().to(device) for vector in vectors]


def make_gmm_1d():
    x_sampler = GMMSampler(mean=torch.tensor([[1.], [2], [3]]), cov=torch.tensor([[[.1]], [[.1]], [[.1]]]),
                           p=torch.ones(3) / 3)
    y_sampler = GMMSampler(mean=torch.tensor([[0.], [3], [5]]), cov=torch.tensor([[[.1]], [[.1]], [[.4]]]),
                           p=torch.ones(3) / 3)

    return x_sampler, y_sampler


def make_gmm_2d():
    cov_x = torch.eye(2) * .1, torch.eye(2) * .1, torch.eye(2) * .4
    cov_y = torch.eye(2) * .1, torch.eye(2) * .1, torch.eye(2) * .1
    cov_x = torch.cat([cov[None, :, :,] for cov in cov_x], dim=0)
    cov_y = torch.cat([cov[None, :, :] for cov in cov_y], dim=0)
    x_sampler = GMMSampler(mean=torch.tensor([[1., 0], [2, 1.], [0., 1.]]), cov=cov_x,
                           p=torch.ones(3) / 3)
    y_sampler = GMMSampler(mean=torch.tensor([[0., -2], [2, -1], [3, 0]]), cov=cov_y,
                           p=torch.ones(3) / 3)

    return x_sampler, y_sampler


def make_gmm_2d_simple():
    cov_x = [torch.eye(2) * .1]
    cov_y = [torch.eye(2) * .1]
    cov_x = torch.cat([cov[None, :, :,] for cov in cov_x], dim=0)
    cov_y = torch.cat([cov[None, :, :] for cov in cov_y], dim=0)
    x_sampler = GMMSampler(mean=torch.tensor([[1., 0]]), cov=cov_x,
                           p=torch.ones(1))
    y_sampler = GMMSampler(mean=torch.tensor([[0., -2]]), cov=cov_y,
                           p=torch.ones(1))

    return x_sampler, y_sampler


class GMMSampler:
    def __init__(self, mean: torch.tensor, cov: torch.tensor, p: torch.tensor):
        k, d = mean.shape
        k, d, d = cov.shape
        k = p.shape
        self.dimension = d
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
        return torch.from_numpy(pos), torch.from_numpy(logweight), None

    def log_prob(self, x):
        # b, d = x.shape
        diff = x[:, None, :] - self.mean[None, :]  # b, k, d
        return torch.log(torch.sum(self.p[None, :] * torch.exp(-torch.einsum('bkd,kde,bke->bk',
                                                                   [diff, self.icov, diff]) / 2) / self.norm, dim=1))


class Subsampler:
    def __init__(self, positions: torch.tensor, weights: torch.tensor, cycle=True):
        self.positions = positions
        self.cycle = cycle
        self.weights = weights

        if self.cycle:
            self.idx = np.arange(len(self.positions), dtype=np.long)
            self.positions, self.weights, self.idx = shuffle(self.positions, self.weights, self.idx)
        self.cursor = 0

    @property
    def dimension(self):
        return self.positions.shape[1]

    def __call__(self, n):
        if n >= len(self.positions):
            return self.positions, self.weights, self.idx.tolist()
        if not self.cycle:
            idx = np.random.permutation(len(self.positions))[:n].tolist()
            weights = self.weights[idx]
            positions = self.positions[idx]
        else:
            new_cursor = self.cursor + n
            if new_cursor >= len(self.positions):
                idx = self.idx[self.cursor:].copy()
                positions = self.positions[self.cursor:].clone()
                weights = self.weights[self.cursor:].clone()
                self.positions, self.weights, self.idx = shuffle(self.positions, self.weights, self.idx)
                reset_cursor = new_cursor - len(self.positions)
                idx = np.concatenate([idx, self.idx[:reset_cursor]], axis=0)
                positions = torch.cat([positions, self.positions[:reset_cursor]], dim=0)
                weights = torch.cat([weights, self.weights[:reset_cursor]], dim=0)
                self.cursor = reset_cursor
            else:
                idx = self.idx[self.cursor:new_cursor].copy()
                positions = self.positions[self.cursor:new_cursor].clone()
                weights = self.weights[self.cursor:new_cursor].clone()
                self.cursor = new_cursor
        weights -= torch.logsumexp(weights, dim=0)
        return positions, weights, idx.tolist()


def var_norm(potential, other_potential, positions):
    v = potential(positions) - other_potential(positions)
    return v.max() - v.min()


def compute_distance(x, y):
    x2 = torch.sum(x ** 2, dim=1)
    y2 = torch.sum(y ** 2, dim=1)
    return .5 * (x2[:, None] + y2[None, :] - 2 * x @ y.transpose(0, 1))


def logaddexp(x, y):
    return torch.logsumexp(torch.cat([x[None, :], y[None, :]], dim=0), dim=0)


def check_idx(n, idx):
    if isinstance(idx, slice):
        return np.arange(n)[idx].tolist()
    elif isinstance(idx, list):
        return idx
    else:
        raise ValueError


class BasePotential:
    def __init__(self, positions: torch.Tensor, weights: torch.tensor, epsilon=1.):
        self.positions = positions
        self.weights = weights
        self.epsilon = epsilon
        self.seen = slice(None)

    def add_weight(self, weight):
        self.weights[self.seen] += weight

    def __call__(self, positions: torch.tensor):
        """Evaluation"""
        C = compute_distance(positions, self.positions[self.seen])
        return - self.epsilon * torch.logsumexp((self.weights[None, self.seen] - C) / self.epsilon, dim=1)

    def refit(self, other_potential):
        x = self.positions[self.seen]
        self.weights[self.seen] = other_potential(x) - np.log(len(x))


class FinitePotential(BasePotential):
    def __init__(self, positions: torch.Tensor, weights: Optional[torch.Tensor] = None, epsilon=1.):
        weights_provided = isinstance(weights, torch.Tensor)
        if not weights_provided:
            weights = torch.full_like(positions[:, 0], fill_value=-float('inf'))
        super(FinitePotential, self).__init__(positions, weights, epsilon)
        if weights_provided:
            self.seen = slice(None)
        else:
            self.seen = []
            self.seen_set = set()

    def push(self, idx, weights, override=False):
        idx = check_idx(len(self.weights), idx)
        if self.seen != slice(None):  # Still a set
            self.seen_set.update(set(idx))
            if len(self.seen_set) == len(self.weights):
                self.seen = slice(None)
            else:
                self.seen = list(self.seen_set)
        if override:
            self.weights[idx] = weights
        else:
            if isinstance(weights, float):
                weights = torch.full_like(self.weights[idx], fill_value=weights)
            self.weights[idx] = logaddexp(self.weights[idx], weights)


class InfinitePotential(BasePotential):
    def __init__(self, length, dimension, epsilon=1.):
        weights = torch.full((length, ), fill_value=-float('inf'))
        positions = torch.zeros((length, dimension))
        super(InfinitePotential, self).__init__(positions, weights, epsilon)
        self.length = length
        self.cursor = 0
        self.seen = slice(0, 0)

    def add_weight(self, weight):
        self.weights[self.seen] += weight

    def push(self, positions, weights):
        old_cursor = self.cursor
        self.cursor += len(positions)
        seen = min(self.length, self.seen.stop + len(positions))
        self.seen = slice(0, seen)
        if self.cursor > self.length:
            new_cursor = self.cursor - self.length
            cut = self.length - old_cursor
            self.cursor = new_cursor
            to_slices = [slice(old_cursor, self.length), slice(0, new_cursor)]
            from_slices = [slice(0, cut), slice(cut, None)]
        else:
            to_slices = [slice(old_cursor, self.cursor)]
            from_slices = [slice(None)]
        for t, f in zip(to_slices, from_slices):
            self.positions[t] = positions[f]
            if isinstance(weights, float):
                self.weights[t] = weights
            else:
                self.weights[t] = weights[f]

    def trim(self,):
        max_weight = self.weights[self.seen].max()
        large_enough = self.weights[self.seen] > max_weight - 10
        n = large_enough.float().sum().int().item()
        print(n)
        if n > 0:
            self.weights[:n] = self.weights[self.seen][large_enough]
            self.positions[:n] = self.positions[self.seen][large_enough]
            self.weights[n:].fill_(-float('inf'))
            self.seen = slice(0, n)
            self.cursor = n


def subsampled_sinkhorn(x, la, y, lb, n_iter=100, batch_size: int = 10):
    x_sampler = Subsampler(x, la)
    y_sampler = Subsampler(y, lb)
    x, la, xidx = x_sampler(batch_size)
    y, lb, yidx = y_sampler(batch_size)
    return sinkhorn(x, la, y, lb, n_iter)


def sinkhorn(x, la, y, lb, n_iter=100, epsilon=1.):
    F = FinitePotential(y, lb.clone(), epsilon=epsilon)
    G = FinitePotential(x, la.clone(), epsilon=epsilon)

    for _ in range(n_iter):
        F.push(slice(None), G(y) + lb, override=True)
        G.push(slice(None), F(x) + la, override=True)
    anchor = F(torch.zeros_like(x[[0]]))
    F.add_weight(anchor)
    G.add_weight(-anchor)
    return F, G


def online_sinkhorn(x_sampler=None, y_sampler=None, x=None, la=None, y=None, lb=None, use_finite=True,
                    epsilon=1., length=100000, trim_every=None,
                    refit=False,
                    n_iter=100,
                    batch_sizes: Optional[Union[List[int], int]] = 10,
                    lrs: Union[List[float], float] = .1):
    if not isinstance(batch_sizes, int) or not isinstance(batch_sizes, int):
        if not isinstance(batch_sizes, int):
            n_iter = len(batch_sizes)
        else:
            n_iter = len(lrs)
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes for _ in range(n_iter)]
    if isinstance(lrs, (float, int)):
        lrs = [lrs for _ in range(n_iter)]
    assert(n_iter == len(lrs) == len(batch_sizes))

    if use_finite:
        x_sampler = Subsampler(x, la)
        y_sampler = Subsampler(y, lb)
        F = FinitePotential(y, epsilon=epsilon)
        G = FinitePotential(x, epsilon=epsilon)
    else:
        if x_sampler is None:
            x_sampler = Subsampler(x, la)
        if y_sampler is None:
            y_sampler = Subsampler(y, lb)
        F = InfinitePotential(length=length, dimension=y_sampler.dimension, epsilon=epsilon)
        G = InfinitePotential(length=length, dimension=x_sampler.dimension, epsilon=epsilon)

    # Init
    x, la, xidx = x_sampler(batch_sizes[0])
    y, lb, yidx = y_sampler(batch_sizes[0])
    F.push(yidx if use_finite else y, la)
    G.push(xidx if use_finite else x, lb)

    for i in range(1, n_iter):
        y, lb, yidx = y_sampler(batch_sizes[i])
        if refit:
            F.push(yidx if use_finite else y, -float('inf'))
            F.refit(G)
        else:
            F.add_weight(safe_log(1 - lrs[i]))
            F.push(yidx if use_finite else y, np.log(lrs[i]) + G(y) + lb)
        x, la, xidx = x_sampler(batch_sizes[i])
        if refit:
            G.push(xidx if use_finite else x, -float('inf'))
            G.refit(F)
        else:
            G.add_weight(safe_log(1 - lrs[i]))
            G.push(xidx if use_finite else x, np.log(lrs[i]) + F(x) + la)
        if not use_finite and trim_every is not None and i % trim_every == 0:
            G.trim()
            F.trim()
    anchor = F(torch.zeros_like(x[[0]]))
    F.add_weight(anchor)
    G.add_weight(-anchor)
    return F, G


def random_sinkhorn(x_sampler=None, y_sampler=None, x=None, la=None, y=None, lb=None, use_finite=True, n_iter=100,
                    epsilon=1,
                    batch_sizes: Union[List[int], int] = 10):
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes for _ in range(n_iter)]
    else:
        n_iter = len(batch_sizes)
    if use_finite:
        x_sampler = Subsampler(x, la)
        y_sampler = Subsampler(y, lb)
    F, G = None, None
    for i in range(n_iter):
        x, la, _ = x_sampler(batch_sizes[i])
        y, lb, _ = y_sampler(batch_sizes[i])
        eG = 0 if i == 0 else G(y)
        F = FinitePotential(y, eG + lb, epsilon=epsilon)
        eF = 0 if i == 0 else F(x)
        G = FinitePotential(x, eF + la, epsilon=epsilon)
    return F, G


def simple_test():
    n_iter = 200
    ref_iter = 500
    epsilon = 1e-2
    ref_grid = 400

    x_sampler, y_sampler = make_gmm_1d()
    x, la, _ = x_sampler(ref_grid)
    y, lb, _ = y_sampler(ref_grid)
    F, G = sinkhorn(x, la, y, lb, n_iter=ref_iter, epsilon=epsilon)

    batch_sizes = np.ceil(10 * np.float_power(np.arange(n_iter, dtype=float) + 1, 1)).astype(int)
    lrs = (1 * np.float_power(np.arange(n_iter, dtype=float) + 1, 0))
    lim = np.where(batch_sizes >= ref_grid)[0][0]
    lrs[lim:] = 1
    batch_sizes[lim:] = ref_grid
    batch_sizes = batch_sizes.tolist()
    lrs = lrs.tolist()

    aF, aG = sinkhorn(x, la, y, lb, n_iter=n_iter - lim, epsilon=epsilon)

    oF, oG = online_sinkhorn(x=x, y=y, la=la, lb=lb, x_sampler=x_sampler, y_sampler=y_sampler, use_finite=True, batch_sizes=batch_sizes, lrs=lrs, refit=True, epsilon=epsilon)
    print(var_norm(oF, F, x) + var_norm(oG, G, y))
    print(var_norm(aF, F, x) + var_norm(aG, G, y))

    import matplotlib.pyplot as plt
    x, _ = torch.sort(x[:, 0])
    y, _ = torch.sort(y[:, 0])
    f1 = oF(x[:, None])
    f2 = F(x[:, None])
    g1 = oG(y[:, None])
    g2 = G(y[:, None])
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax1.plot(x, f1)
    ax1.plot(x, f2)
    ax2.plot(x, g1)
    ax2.plot(x, g2)
    X = torch.linspace(-4, 4, 100)
    llX = x_sampler.log_prob(X[:, None])
    Y = torch.linspace(-4, 4, 100)
    llY = y_sampler.log_prob(Y[:, None])
    ax0.plot(X, llX)
    ax0.plot(Y, llY)
    plt.show()


def make_2d_grid():
    X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    Z = np.concatenate([X[:, :, None], Y[:, :, None]], axis=2).reshape(-1, 2)
    Z = torch.from_numpy(Z).float()
    return Z


def compute_grad(potential, z):
    z = z.clone()
    z.requires_grad = True
    grad, = torch.autograd.grad(potential(z).sum(), (z,))
    return - grad.detach()

def plot_2d():
    n_iter = 200
    ref_iter = 100
    epsilon = 1e-4
    ref_grid = 1000
    lr_exp = .5
    batch_exp = 0  # > 2*(1 - lr_exp) for convergence if not refit,
    batch_size = 100
    lr = 1
    refit = False
    use_finite = False

    x_sampler, y_sampler = make_gmm_2d()

    x, la, _ = x_sampler(ref_grid)
    y, lb, _ = y_sampler(ref_grid)
    F, G = sinkhorn(x, la, y, lb, n_iter=ref_iter, epsilon=epsilon)

    batch_sizes = np.ceil(batch_size * np.float_power(np.arange(n_iter, dtype=float) + 1, batch_exp)).astype(int)
    lrs = (lr * np.float_power(np.arange(n_iter, dtype=float) + 1, -lr_exp))
    batch_sizes = batch_sizes.tolist()
    lrs = lrs.tolist()

    oF, oG = online_sinkhorn(x=x, y=y, la=la, lb=lb, x_sampler=x_sampler, y_sampler=y_sampler, use_finite=use_finite,
                             batch_sizes=batch_sizes, lrs=lrs, refit=refit, epsilon=epsilon)

    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 18))

    z = make_2d_grid()
    llx = x_sampler.log_prob(z)
    lly = y_sampler.log_prob(z)

    grad_f = compute_grad(F, x)
    grad_g = compute_grad(G, y)

    grad_fgrid = compute_grad(F, z)
    grad_ggrid = compute_grad(G, z)

    shape = (100, 100)
    axes[0, 0].contour(z[:, 0].view(shape), z[:, 1].view(shape), llx.view(shape), zorder=0, levels=30)
    axes[0, 0].scatter(x[:, 0], x[:, 1], 2, zorder=10)
    axes[0, 0].quiver(x[:, 0], x[:, 1], grad_f[:, 0], grad_f[:, 1], zorder=20)

    axes[1, 0].contour(z[:, 0].view(shape), z[:, 1].view(shape), llx.view(shape), zorder=0, levels=30)
    axes[1, 0].quiver(z[:, 0], z[:, 1], grad_fgrid[:, 0] * llx.exp(), grad_fgrid[:, 1] * llx.exp(), zorder=20)

    axes[0, 1].contour(z[:, 0].view(shape), z[:, 1].view(shape), lly.view(shape), zorder=0, levels=30)
    axes[0, 1].scatter(y[:, 0], y[:, 1], 2, zorder=10)
    axes[0, 1].quiver(y[:, 0], y[:, 1], grad_g[:, 0], grad_g[:, 1], zorder=20)

    axes[1, 1].contour(z[:, 0].view(shape), z[:, 1].view(shape), lly.view(shape), zorder=0, levels=30)
    axes[1, 1].quiver(z[:, 0], z[:, 1], grad_ggrid[:, 0] * lly.exp(), grad_ggrid[:, 1] * lly.exp(), zorder=20)


    plt.show()

plot_2d()