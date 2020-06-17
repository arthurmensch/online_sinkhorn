import time
import warnings
from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from pykeops.torch import LazyTensor

from onlikhorn.data import Subsampler

import time

def safe_log(x: float):
    if x == 0:
        return - float('inf')
    else:
        return np.log(x)


def var_norm(v):
    return v.max() - v.min()


def compute_distance(x, y, lazy=False):
    if lazy:
        x = LazyTensor(x[:, None, :])
        y = LazyTensor(y[None, :, :])
        return (((x - y) ** 2) / 2).sum(dim=2)
    else:
        x2 = torch.sum(x ** 2, dim=1)
        y2 = torch.sum(y ** 2, dim=1)
        return .5 * (x2[:, None] + y2[None, :] - 2 * x @ y.transpose(0, 1))[..., None]


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

    @property
    def n_samples_(self):
        return len(self.positions[self.seen])

    def __call__(self, positions: torch.tensor = None, C=None, free=False, return_C=False):
        """Evaluation"""
        lazy = self.positions.device.type == 'cuda' and C is None and not return_C
        if C is None:
            C = compute_distance(positions, self.positions[self.seen], lazy=lazy)
            if not free:
                self.n_calls_ += C.shape[0] * C.shape[1] * self.positions.shape[1]
        if not free:
            self.n_calls_ += C.shape[0] * C.shape[1] * self.positions.shape[1]
        weights = self.weights[None, self.seen, None]
        if lazy:
            weights = LazyTensor(weights)
        weighted_C = (weights - C) / self.epsilon
        e = - self.epsilon * weighted_C.logsumexp(dim=1)[..., 0]
        if not return_C:
            return e
        else:
            return e, C

    def to(self, device):
        self.weights = self.weights.to(device)
        self.positions = self.positions.to(device)
        return self

    @property
    def device(self):
        return self.positions.device

    def cpu(self):
        return self.to('cpu')

    @property
    def full(self):
        return self.seen == slice(None)

    def refit(self, F, C=None):
        x = self.positions[self.seen]
        la = np.log(x.shape[0])
        eF = F(x, C=C) + la
        fixed_err = var_norm(eF - self.weights[self.seen])
        self.weights[self.seen] = eF
        return fixed_err


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
                        precompute_C=True, max_calls=None, trace_every=1):
    if batch_size is not None and (batch_size != len(x) or batch_size != len(y)):
        x_sampler = Subsampler(x, la)
        y_sampler = Subsampler(y, lb)
        x, la, xidx = x_sampler(batch_size)
        y, lb, yidx = y_sampler(batch_size)
    return sinkhorn(x, la, y, lb, n_iter, epsilon, save_trace=save_trace, ref=ref, precompute_C=precompute_C,
                    max_calls=max_calls, trace_every=trace_every)


def sinkhorn(x, la, y, lb, n_iter=100, epsilon=1., save_trace=False, F=None, G=None,
             precompute_C: Union[bool, Tuple[torch.Tensor, torch.Tensor]] = True,
             trace=None,
             max_calls=None, verbose=True, trace_every=1,
             ref=None,
             start_iter=0, start_time=0):
    eval_time = 0
    t0 = time.perf_counter()
    if F is None:
        F = FinitePotential(y, lb.clone(), epsilon=epsilon)
    if G is None:
        G = FinitePotential(x, la.clone(), epsilon=epsilon)

    if n_iter is None:
        assert max_calls is not None
        n_iter = int(1e6)

    if precompute_C is not False:
        if precompute_C is True:
            Cxy = compute_distance(x, y, lazy=False)
            Cyx = Cxy.transpose(0, 1)
            F.n_calls_ += x.shape[0] * y.shape[0] * x.shape[1]
        else:
            Cxy = precompute_C
            Cyx = Cxy.transpose(0, 1)
    else:
        Cxy, Cyx = None, None

    trace, ref = check_trace(save_trace, trace=trace, ref=ref, ref_needed=False)

    call_trace = 0
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
        if save_trace and n_calls >= call_trace:
            eval_t0 = time.perf_counter()
            fixed_err += var_norm(eF + la - G.weights)
            w += (eF * la.exp()).sum()
            this_trace = dict(n_iter=i + 1, n_calls=n_calls, n_samples=n_samples,
                              fixed_err=fixed_err.item(), w=w.item(), algorithm='full')
            fixed_err, ref_err = evaluate(F, G, epsilon, ref)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            eval_time += time.perf_counter() - eval_t0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            this_trace['time'] = time.perf_counter() - t0 - eval_time + start_time
            for name, err in fixed_err.items():
                this_trace[f'fixed_err_{name}'] = err
            for name, err in ref_err.items():
                this_trace[f'ref_err_{name}'] = err

            trace.append(this_trace)
            call_trace = n_calls + trace_every
            if verbose:
                print(' '.join(f'{k}:{v:.2e}' if type(v) in [int, float] else f'{k}:{v}' for k, v in this_trace.items()))
        G.push(slice(None), eF + la, override=True)
    anchor = F(torch.zeros_like(x[[0]]))
    F.add_weight(anchor)
    G.add_weight(-anchor)
    if save_trace:
        return F, G, trace
    else:
        return F, G


def scatter(target: torch.tensor, xidx: List[int], yidx: List[int], value: torch.tensor):
    idx = np.concatenate([a[..., None] for a in np.meshgrid(xidx, yidx, indexing='ij')], axis=2).reshape(-1, 2)
    target[idx[:, 0], idx[:, 1]] = value.reshape(-1)


def online_sinkhorn(x_sampler=None, y_sampler=None,
                    x=None, la=None, y=None, lb=None, use_finite=True,
                    epsilon=1., max_length=100000, trim_every=None,
                    refit=False, precompute_C=False,
                    n_iter=100, force_full=False,
                    batch_sizes: Optional[Union[List[int], int]] = 10, max_calls=None, verbose=True,
                    start_time=0,
                    trace_every=1,
                    lrs: Union[List[float], float] = .1, save_trace=False, ref=None):
    eval_time = 0
    t0 = time.perf_counter()

    if n_iter is None:
        assert max_calls is not None
        n_iter = int(1e6)

    if force_full:
        assert use_finite

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
        F = FinitePotential(y, epsilon=epsilon).to(x_sampler.device)
        G = FinitePotential(x, epsilon=epsilon).to(y_sampler.device)
    else:
        if x_sampler is None:
            x_sampler = Subsampler(x, la)
        if y_sampler is None:
            y_sampler = Subsampler(y, lb)
        F = InfinitePotential(max_length=max_length, dimension=y_sampler.dimension, epsilon=epsilon).to(
            x_sampler.device)
        G = InfinitePotential(max_length=max_length, dimension=x_sampler.dimension, epsilon=epsilon).to(
            y_sampler.device)

    if force_full:  # save for later
        xf, laf, yf, lbf = x, la, y, lb
        if precompute_C:
            C = torch.empty((xf.shape[0], yf.shape[0], 1), device=xf.device)
        else:
            C = None
    else:
        xf, laf, yf, lbf = None, None, None, None
        C = None
    # Init
    x, la, xidx = x_sampler(batch_sizes[0])
    y, lb, yidx = y_sampler(batch_sizes[0])
    if force_full and precompute_C:
        this_C = compute_distance(x, y, lazy=False)
        scatter(C[..., 0], xidx, yidx, this_C[..., 0].transpose(0, 1))

    F.push(yidx if use_finite else y, la)
    G.push(xidx if use_finite else x, lb)

    trace, ref = check_trace(save_trace, ref=ref, ref_needed=True)

    call_trace = 0
    for i in range(n_iter):
        n_calls = F.n_calls_ + G.n_calls_
        n_samples = F.n_samples_ + G.n_samples_
        if max_calls is not None and n_calls > max_calls:
            break
        if save_trace and n_calls >= call_trace:
            eval_t0 = time.perf_counter()
            this_trace = dict(n_iter=i, n_calls=n_calls, n_samples=n_samples, algorithm='online')
            fixed_err, ref_err = evaluate(F, G, epsilon, ref)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            eval_time += time.perf_counter() - eval_t0
            for name, err in fixed_err.items():
                this_trace[f'fixed_err_{name}'] = err
            for name, err in ref_err.items():
                this_trace[f'ref_err_{name}'] = err
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            this_trace['time'] = time.perf_counter() - t0 - eval_time + start_time
            trace.append(this_trace)
            call_trace = n_calls + trace_every
            if verbose:
                print(' '.join(f'{k}:{v:.2e}' if type(v) in [int, float] else f'{k}:{v}' for k, v in this_trace.items()))
        y, lb, yidx = y_sampler(batch_sizes[i])
        if refit:
            F.push(yidx if use_finite else y, -float('inf'))
            F.refit(G)
        else:
            F.add_weight(safe_log(1 - lrs[i]))
            if force_full and precompute_C:
                eG, this_C = G(y, return_C=True)
                xidx = check_idx(len(G.positions), G.seen)
                scatter(C[..., 0], xidx, yidx, this_C[..., 0].transpose(0, 1))
            else:
                eG = G(y)
            F.push(yidx if use_finite else y, np.log(lrs[i]) + eG + lb)
        x, la, xidx = x_sampler(batch_sizes[i])
        if refit:
            G.push(xidx if use_finite else x, -float('inf'))
            G.refit(F)
        else:
            G.add_weight(safe_log(1 - lrs[i]))
            if force_full and precompute_C:
                eF, this_C = F(x, return_C=True)
                yidx = check_idx(len(F.positions), F.seen)
                scatter(C[..., 0], xidx, yidx, this_C[..., 0])
            else:
                eF = F(x)
            G.push(xidx if use_finite else x, np.log(lrs[i]) + eF + la)
        if not use_finite and trim_every is not None and i % trim_every == 0:
            G.trim()
            F.trim()

        if force_full and G.full and F.full:
                start_time = time.perf_counter() - t0 - eval_time + start_time
                res = sinkhorn(xf, laf, yf, lbf, F=F, G=G, save_trace=save_trace, trace=trace, start_iter=i,
                               n_iter=n_iter - 1, ref=ref, precompute_C=C if precompute_C else False,
                               max_calls=max_calls, trace_every=trace_every, start_time=start_time,
                               epsilon=epsilon)
                if save_trace:
                    F, G, trace = res
                else:
                    F, G = res
                break

    anchor = F(torch.zeros_like(x[[0]]))
    F.add_weight(anchor)
    G.add_weight(-anchor)
    if save_trace:
        return F, G, trace
    else:
        return F, G


def evaluate(F, G, epsilon, ref):
    ref_err = {}
    fixed_err = {}
    for name, (fr, xr, gr, yr) in ref.items():
        f = F(xr, free=True)
        g = G(yr, free=True)
        if fr is not None and gr is not None:
            ref_err[name] = (var_norm(f - fr) + var_norm(g - gr)).item()

        gg = FinitePotential(xr, f - np.log(len(f)), epsilon=epsilon)(yr)
        ff = FinitePotential(yr, g - np.log(len(g)), epsilon=epsilon)(xr)
        fixed_err[name] = (var_norm(f - ff) + var_norm(g - gg)).item()
    return fixed_err, ref_err


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
                    epsilon=1, max_calls=None, start_time=0,
                    batch_sizes: Union[List[int], int] = 10, save_trace=False, ref=None, verbose=True,
                    trace_every=1):
    eval_time = 0
    t0 = time.perf_counter()
    trace, ref = check_trace(save_trace, ref=ref, ref_needed=True)

    if n_iter is None:
        assert max_calls is not None
        n_iter = int(1e6)

    if isinstance(batch_sizes, int):
        batch_sizes = [batch_sizes for _ in range(n_iter)]
    else:
        n_iter = len(batch_sizes)
    if use_finite:
        x_sampler = Subsampler(x, la)
        y_sampler = Subsampler(y, lb)
    F, G = None, None
    n_calls = 0
    call_trace = 0
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
        if save_trace and n_calls >= call_trace:
            this_trace = dict(n_iter=i + 1, n_calls=n_calls, n_samples=n_samples, algorithm='random')
            eval_t0 = time.perf_counter()
            fixed_err, ref_err = evaluate(F, G, epsilon, ref)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            eval_time += time.perf_counter() - eval_t0
            for name, err in fixed_err.items():
                this_trace[f'fixed_err_{name}'] = err
            for name, err in ref_err.items():
                this_trace[f'ref_err_{name}'] = err
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            this_trace['time'] = time.perf_counter() - t0 - eval_time + start_time
            trace.append(this_trace)
            call_trace = n_calls + trace_every
            if verbose:
                print(' '.join(f'{k}:{v:.2e}' if type(v) in [int, float] else f'{k}:{v}' for k, v in this_trace.items()))
    if save_trace:
        return F, G, trace
    else:
        return F, G


def schedule(batch_exp, batch_size, lr, lr_exp, max_length, n_iter, refit, iota=.1):
    batch_sizes = np.ceil(batch_size * np.float_power(1 + 0.1 * np.arange(n_iter, dtype=float), batch_exp)).astype(
        int)  # Those are important hyperparameters...
    batch_sizes[batch_sizes > max_length] = max_length
    batch_sizes = batch_sizes.tolist()
    if lr_exp == 'auto':
        if not refit:
            lr_exp = min(max(0, 1 - batch_exp / 2 + iota), 1)
        else:
            lr_exp = min(max(0, 1 - (batch_exp + 1) / 2 + iota), 1)
    lrs = (lr * np.float_power(1 + 0.1 * np.arange(n_iter, dtype=float), -lr_exp)).tolist()
    return batch_sizes, lrs, lr_exp
