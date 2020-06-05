import warnings
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
    def __init__(self, positions: torch.Tensor, weights: torch.tensor, epsilon=1., force_count_compute=False):
        self.positions = positions
        self.weights = weights
        self.epsilon = epsilon
        self.seen = slice(None)

        self.force_count_compute = force_count_compute

        self.n_calls_ = 0

    def add_weight(self, weight):
        self.weights[self.seen] += weight

    @property
    def n_samples_(self):
        return len(self.positions[self.seen])

    def __call__(self, positions: torch.tensor = None, C=None, free=False, free_scaling=False, free_compute=False):
        """Evaluation"""
        if free:  # hacks as the implementation of memory persistence is not finished
            free_scaling = True
            free_compute = True
        if C is None:
            C = compute_distance(positions, self.positions[self.seen])
        elif not self.force_count_compute:
            free_compute = True

        if not free_compute:
            self.n_calls_ += C.shape[0] * C.shape[1] * self.positions.shape[1]
        if not free_scaling:
            self.n_calls_ += C.shape[0] * C.shape[1]

        return - self.epsilon * torch.logsumexp((self.weights[None, self.seen] - C) / self.epsilon, dim=1)

    def to(self, device):
        self.weights = self.weights.to(device)
        self.positions = self.positions.to(device)
        return self

    def cpu(self):
        return self.to('cpu')

    @property
    def full(self):
        return self.seen == slice(None)

    def refit(self, F, C=None):
        x = self.positions[self.seen]
        la = np.log(x.shape[0])
        eF = F(x, C=C, free_compute=True) + la
        fixed_err = var_norm(eF - self.weights[self.seen])
        self.weights[self.seen] = eF
        return fixed_err


class FinitePotential(BasePotential):
    def __init__(self, positions: torch.Tensor, weights: Optional[torch.Tensor] = None, epsilon=1.,
                 force_count_compute=False):
        weights_provided = isinstance(weights, torch.Tensor)
        if not weights_provided:
            weights = torch.full_like(positions[:, 0], fill_value=-float('inf'))
        super(FinitePotential, self).__init__(positions, weights, epsilon, force_count_compute)
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
    def __init__(self, max_length, dimension, epsilon=1.):
        weights = torch.full((max_length,), fill_value=-float('inf'))
        positions = torch.zeros((max_length, dimension))
        super(InfinitePotential, self).__init__(positions, weights, epsilon)
        self.max_length = max_length
        self.cursor = 0
        self.seen = slice(0, 0)

    def add_weight(self, weight):
        self.weights[self.seen] += weight

    @property
    def full(self):
        return self.seen.stop == self.max_length

    def push(self, positions, weights):
        old_cursor = self.cursor
        self.cursor += len(positions)
        seen = min(self.max_length, self.seen.stop + len(positions))
        self.seen = slice(0, seen)
        if self.cursor > self.max_length:
            new_cursor = self.cursor - self.max_length
            cut = self.max_length - old_cursor
            self.cursor = new_cursor
            to_slices = [slice(old_cursor, self.max_length), slice(0, new_cursor)]
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

    def trim(self, ):
        max_weight = self.weights[self.seen].max()
        large_enough = self.weights[self.seen] > max_weight - 10
        n = large_enough.float().sum().int().item()
        if n > 0:
            self.weights[:n] = self.weights[self.seen][large_enough]
            self.positions[:n] = self.positions[self.seen][large_enough]
            self.weights[n:].fill_(-float('inf'))
            self.seen = slice(0, n)
            self.cursor = n


def subsampled_sinkhorn(x, la, y, lb, n_iter=100, batch_size: int = 10, epsilon=1, save_trace=False, ref=None,
                        precompute_C=True, count_recompute=False, max_calls=None,):
    x_sampler = Subsampler(x, la)
    y_sampler = Subsampler(y, lb)
    x, la, xidx = x_sampler(batch_size)
    y, lb, yidx = y_sampler(batch_size)
    return sinkhorn(x, la, y, lb, n_iter, epsilon, save_trace=save_trace, ref=ref, precompute_C=precompute_C,
                    count_recompute=count_recompute, max_calls=max_calls)


def sinkhorn(x, la, y, lb, n_iter=100, epsilon=1., save_trace=False, F=None, G=None, precompute_C=True, trace=None,
             precompute_for_free=False, count_recompute=False, max_calls=None, verbose=True,
             ref=None,
             start_iter=0):
    if F is None:
        F = FinitePotential(y, lb.clone(), epsilon=epsilon, force_count_compute=count_recompute)
    if G is None:
        G = FinitePotential(x, la.clone(), epsilon=epsilon, force_count_compute=count_recompute)

    if precompute_C:
        Cxy = compute_distance(x, y)
        Cyx = Cxy.T
        if not precompute_for_free and not count_recompute:  # Count compute only once
            F.n_calls_ += x.shape[0] * y.shape[0] * x.shape[1]
    else:
        Cxy, Cyx = None, None

    trace, ref = check_trace(save_trace, trace=trace, ref=ref, ref_needed=False)

    for i in range(start_iter, n_iter):
        eG = G(positions=y, C=Cyx)
        if save_trace:
            fixed_err = var_norm(eG + lb - F.weights)
            w = (eG * lb.exp()).sum()
        else:
            fixed_err, w = None, None
        F.push(slice(None), eG + lb, override=True)
        eF = F(positions=x, C=Cxy)
        n_calls = F.n_calls_ + G.n_calls_
        n_samples = F.n_samples_ + G.n_samples_
        if max_calls is not None and n_calls > max_calls:
            break
        if save_trace:
            fixed_err += var_norm(eF + la - G.weights)
            w += (eF * la.exp()).sum()
            this_trace = dict(n_iter=i + 1, n_calls=n_calls, n_samples=n_samples,
                              fixed_err=fixed_err.item(), w=w.item(), algorithm='full')
            for name, (fr, xr, gr, yr) in ref.items():
                this_trace[f'ref_err_{name}'] = (var_norm(F(xr, free=True) - fr) + var_norm(G(yr, free=True) - gr)).item()
            trace.append(this_trace)
        else:
            this_trace = dict(n_iter=i + 1, n_calls=n_calls)
        if verbose:
            print(' '.join(f'{k}:{v}' for k, v in this_trace.items()))
        G.push(slice(None), eF + la, override=True)
    anchor = F(torch.zeros_like(x[[0]]))
    F.add_weight(anchor)
    G.add_weight(-anchor)
    if save_trace:
        return F, G, trace
    else:
        return F, G


def online_sinkhorn(x_sampler=None, y_sampler=None,
                    x=None, la=None, y=None, lb=None, use_finite=True,
                    epsilon=1., max_length=100000, trim_every=None,
                    refit=False,
                    n_iter=100, force_full=True,
                    batch_sizes: Optional[Union[List[int], int]] = 10, max_calls=None,
                    lrs: Union[List[float], float] = .1, save_trace=False, ref=None):
    if not isinstance(batch_sizes, int) or not isinstance(batch_sizes, int):
        if not isinstance(batch_sizes, int):
            n_iter = len(batch_sizes)
        else:
            n_iter = len(lrs)
    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes for _ in range(n_iter)]
    if isinstance(lrs, (float, int)):
        lrs = [lrs for _ in range(n_iter)]
    assert (n_iter == len(lrs) == len(batch_sizes))

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
        F = InfinitePotential(max_length=max_length, dimension=y_sampler.dimension, epsilon=epsilon).to(x_sampler.device)
        G = InfinitePotential(max_length=max_length, dimension=x_sampler.dimension, epsilon=epsilon).to(y_sampler.device)

    if force_full:  # save for later
        xf, laf, yf, lbf = x, la, y, lb
    else:
        xf, laf, yf, lbf = None, None, None, None
    # Init
    x, la, xidx = x_sampler(batch_sizes[0])
    y, lb, yidx = y_sampler(batch_sizes[0])
    F.push(yidx if use_finite else y, la)
    G.push(xidx if use_finite else x, lb)

    trace, ref = check_trace(save_trace, ref=ref, ref_needed=True)

    for i in range(n_iter):
        n_calls = F.n_calls_ + G.n_calls_
        n_samples = F.n_samples_ + G.n_samples_
        if max_calls is not None and n_calls > max_calls:
            break
        if save_trace:
            this_trace = dict(n_iter=i, n_calls=n_calls, n_samples=n_samples, algorithm='online')
            for name, (fr, xr, gr, yr) in ref.items():
                f = F(xr, free=True)
                g = G(yr, free=True)
                this_trace[f'ref_err_{name}'] = (var_norm(f - fr) + var_norm(g - gr)).item()

                gg = FinitePotential(xr, fr - np.log(len(fr)))(yr)
                ff = FinitePotential(yr, gr - np.log(len(gr)))(xr)
                this_trace[f'var_err_{name}'] = var_norm(f - ff) + var_norm(g - gg)
                
            trace.append(this_trace)
            print(' '.join(f'{k}:{v}' for k, v in this_trace.items()))
        if use_finite and force_full:
            if G.full and F.full:  # Force full iterations once every point has been observed
                res = sinkhorn(xf, laf, yf, lbf, F=F, G=G, save_trace=save_trace, trace=trace, start_iter=i,
                               n_iter=n_iter - 1, ref=ref, precompute_C=True, precompute_for_free=True,
                               # A better implementation would not require to recompute C
                               epsilon=epsilon)
                if save_trace:
                    F, G, trace = res
                else:
                    F, G = res
                break

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
    if save_trace:
        return F.cpu(), G.cpu(), trace
    else:
        return F.cpu(), G.cpu()


def check_trace(save_trace=False, trace=None, ref=None, ref_needed=False):
    if save_trace:
        if trace is None:
            trace = []
        if ref is None:
            if ref_needed:
                raise ValueError('Must provide ref when asking to save trace')
            else:
                ref = {}
    else:
        trace = None
    return trace, ref


def random_sinkhorn(x_sampler=None, y_sampler=None, x=None, la=None, y=None, lb=None, use_finite=True, n_iter=100,
                    epsilon=1, max_calls=None,
                    batch_sizes: Union[List[int], int] = 10, save_trace=False, ref=None):
    trace, ref = check_trace(save_trace, ref=ref, ref_needed=True)

    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes for _ in range(n_iter)]
    else:
        n_iter = len(batch_sizes)
    if use_finite:
        x_sampler = Subsampler(x, la)
        y_sampler = Subsampler(y, lb)
    F, G = None, None
    n_calls = 0
    for i in range(n_iter):
        if max_calls is not None and n_calls > max_calls:
            break
        x, la, _ = x_sampler(batch_sizes[i])
        y, lb, _ = y_sampler(batch_sizes[i])
        eG = 0 if i == 0 else G(y)
        F = FinitePotential(y, eG + lb, epsilon=epsilon)
        eF = 0 if i == 0 else F(x)
        G = FinitePotential(x, eF + la, epsilon=epsilon)
        n_samples = F.n_samples_ + G.n_samples_
        n_calls += F.n_calls_ + G.n_calls_
        if save_trace:
            this_trace = dict(n_iter=i + 1, n_calls=n_calls, n_samples=n_samples, algorithm='random')
            for name, (fr, xr, gr, yr) in ref.items():
                this_trace[f'ref_err_{name}'] = (var_norm(F(xr, free=True) - fr) + var_norm(G(yr, free=True) - gr)).item()
            print(' '.join(f'{k}:{v}' for k, v in this_trace.items()))
            trace.append(this_trace)
    if save_trace:
        return F, G, trace
    else:
        return F, G


def schedule(batch_exp, batch_size, lr, lr_exp, max_length, n_iter, refit, iota=.1):
    batch_sizes = np.ceil(batch_size * np.float_power(np.linspace(1., n_iter / 10, n_iter), batch_exp)).astype(int)  # Those are important hyperparameters...
    batch_sizes[batch_sizes > max_length] = max_length
    batch_sizes = batch_sizes.tolist()
    if lr_exp == 'auto':
        if not refit:
            lr_exp = min(max(0, 1 - batch_exp / 2 + iota), 1)
        else:
            lr_exp = min(max(0, 1 - (batch_exp + 1) / 2 + iota), 1)
    lrs = (lr * np.float_power(np.linspace(1, n_iter / 10, n_iter), -lr_exp)).tolist()
    return batch_sizes, lrs, lr_exp
