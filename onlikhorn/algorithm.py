from typing import Optional, Union, List

import numpy as np
import torch
from onlikhorn.data import Subsampler


def safe_log(x: float):
    if x == 0:
        return - float('inf')
    else:
        return np.log(x)


def var_norm(v):
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

        self.n_calls_ = 0

    def add_weight(self, weight):
        self.weights[self.seen] += weight

    def __call__(self, positions: torch.tensor):
        """Evaluation"""
        C = compute_distance(positions, self.positions[self.seen])
        self.n_calls_ += C.shape[0] * C.shape[1]
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


def sinkhorn(x, la, y, lb, n_iter=100, epsilon=1., save_trace=True):
    F = FinitePotential(y, lb.clone(), epsilon=epsilon)
    G = FinitePotential(x, la.clone(), epsilon=epsilon)

    trace = []
    for n_iter in range(n_iter):
        eG = G(y)
        if save_trace:
            fixed_err = var_norm(eG + lb - F.weights)
            w = (eG * lb.exp()).sum()
        else:
            fixed_err, w = None, None
        F.push(slice(None), eG + lb, override=True)
        eF = F(x)
        if save_trace:
            fixed_err += var_norm(eF + la - G.weights)
            w += (eF * la.exp()).sum()
            trace.append(dict(fixed_err=fixed_err, w=w, n_iter=n_iter, n_calls=F.n_calls_ + G.n_calls_))
        G.push(slice(None), F(x) + la, override=True)
    anchor = F(torch.zeros_like(x[[0]]))
    F.add_weight(anchor)
    G.add_weight(-anchor)
    if save_trace:
        return F, G, trace
    else:
        return F, G


def online_sinkhorn(x_sampler=None, y_sampler=None,
                    x=None, la=None, y=None, lb=None, use_finite=True,
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