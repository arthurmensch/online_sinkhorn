import os
import sys
import tempfile
from os.path import join

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver

# import pykeops
# pykeops.bin_folder = tempfile.mkdtemp()
# sys.path.append(pykeops.bin_folder)

from onlikhorn.algorithm import sinkhorn, online_sinkhorn, random_sinkhorn, subsampled_sinkhorn, schedule
from onlikhorn.cache import torch_cached
from onlikhorn.dataset import get_output_dir, make_data
from onlikhorn.gaussian import sinkhorn_gaussian

exp_name = 'online_grid_big_4'
exp = Experiment(exp_name)
exp_dir = join(get_output_dir(), exp_name)
exp.observers = [FileStorageObserver(exp_dir)]


@exp.config
def config():
    data_source = 'gmm_10d'
    n_samples = int(1e5)  # ref
    max_length = int(1e7)
    device = 'cuda'

    # Overrided
    batch_size = int(1e5)
    seed = 0
    epsilon = 1e-2

    method = 'sinkhorn'
    batch_exp = 1.
    lr_exp = 'auto'
    lr = 1
    refit = True

    n_iter = None
    max_calls = 1e12

    compare_with_ref = False

    precompute_C = False
    force_full = False


@exp.named_config
def debug():
    data_source = 'dragon'
    n_samples = 10000
    max_length = 10000
    device = 'cuda'

    n_iter = int(1e4)
    max_calls = 1e12

    batch_size = 100
    seed = 0
    epsilon = 1e-3

    method = 'online'
    force_full = True
    precompute_C = False
    batch_exp = 0.5
    lr_exp = 0
    lr = 1
    refit = True

    compare_with_ref = False


@exp.named_config
def quiver():
    data_source = 'gmm_2d'
    n_samples = 1000
    n_iter = 10000
    max_length = 10000
    device = 'cpu'
    max_calls = int(5e8)

    # Overrided
    batch_size = 100
    seed = 0
    epsilon = 1e-2

    method = 'online'
    batch_exp = 0
    lr_exp = 1
    lr = 1
    refit = True


@exp.named_config
def gaussian():
    data_source = 'gaussian_10d'
    n_samples = 1000
    n_iter = 200
    max_length = 10000
    device = 'cuda'

    # Overrided
    batch_size = 100
    seed = 0
    epsilon = 1

    # method = 'online_as_warmup'
    method = 'online'
    # method = 'sinkhorn_precompute'
    batch_exp = 0
    lr_exp = 1
    lr = 1
    refit = False


@exp.named_config
def debug_dragon():
    data_source = 'dragon'
    n_samples = 1000
    n_iter = 200
    max_length = 10000
    device = 'cuda'

    # Overrided
    batch_size = 100
    seed = 0
    epsilon = 1e-1

    # method = 'online_as_warmup'
    method = 'sinkhorn_precompute'
    batch_exp = 0
    lr_exp = 1 / 2
    lr = 1
    refit = True


@exp.main
def run(data_source, n_samples, epsilon, n_iter, device, method, max_calls, compare_with_ref,
        precompute_C, force_full, batch_exp, batch_size, lr, lr_exp, max_length, refit, _seed, _run):
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    output_dir = join(exp.observers[0].dir, 'artifacts')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x, la, y, lb, x_sampler, y_sampler = make_data(data_source, n_samples)

    x, y, la, lb = x.to(device), y.to(device), la.to(device), lb.to(device)
    x_sampler.to(device)
    y_sampler.to(device)

    if n_iter is None:
        n_iter = int(1e4)

    xr, lar, yr, lbr, _, _ = make_data(data_source, n_samples)
    xr, yr, lar, lbr = xr.to(device), yr.to(device), lar.to(device), lbr.to(device)
    if compare_with_ref:
        F, G = torch_cached(sinkhorn)(x, la, y, lb, n_iter=n_iter, epsilon=epsilon, save_trace=False,
                                      verbose=False, max_calls=max_calls * 4, trace_every=max_calls * 4 // 100,
                                      count_recompute=True)
        if 'gaussian' in data_source:
            Fr, Gr = sinkhorn_gaussian(x_sampler, y_sampler, epsilon=epsilon)  # Exact reference
        else:
            Fr, Gr = torch_cached(sinkhorn)(xr, lar, yr, lbr, n_iter=n_iter, epsilon=epsilon, save_trace=False,
                                            verbose=False, max_calls=max_calls * 4, trace_every=max_calls * 4 // 100,
                                            count_recompute=True)
        fr = Fr(xr)
        gr = Gr(yr)
        f = F(x)
        g = G(y)
    else:
        fr, gr, f, g = None, None, None, None

    ref = {'test': (fr, xr, gr, yr), 'train': (f, x, g, y)}

    if method == 'sinkhorn':
        n_iter = min(n_iter, int(2e3))  # Faster
        F, G, trace = subsampled_sinkhorn(x, la, y, lb, n_iter=n_iter, batch_size=batch_size,
                                          max_calls=max_calls, precompute_C=precompute_C,
                                          trace_every=max_calls // 100,
                                          epsilon=epsilon, save_trace=True, ref=ref)
    elif method == 'random':
        F, G, trace = random_sinkhorn(x_sampler=x_sampler, y_sampler=y_sampler, n_iter=n_iter,
                                      epsilon=epsilon, save_trace=True, ref=ref, use_finite=False,
                                      batch_sizes=batch_size,
                                      trace_every=max_calls // 100,
                                      max_calls=max_calls)
    elif method == 'online':
        batch_sizes, lrs, lr_exp = schedule(batch_exp, batch_size, lr, lr_exp, max_length, n_iter, refit)
        print(f'Using lr_exp={lr_exp}')
        _run.info['lr_exp'] = lr_exp
        if method == 'online':
            F, G, trace = online_sinkhorn(x=x, la=la, y=y, lb=lb, x_sampler=x_sampler, y_sampler=y_sampler, batch_sizes=batch_sizes,
                                          refit=refit, force_full=force_full, precompute_C=precompute_C,
                                          trace_every=max_calls // 1000,
                                          lrs=lrs, n_iter=n_iter, use_finite=force_full, max_length=max_length,
                                          epsilon=epsilon, save_trace=True, ref=ref, max_calls=max_calls)
    else:
        raise ValueError

    torch.save(dict(x=x, la=la, y=y, lb=lb, F=F, G=G, trace=trace), join(output_dir, 'results.pkl'))


if __name__ == '__main__':
    exp.run_commandline()
