import os
from os.path import join

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver

from onlikhorn.algorithm import sinkhorn, online_sinkhorn, random_sinkhorn, subsampled_sinkhorn, schedule
from onlikhorn.cache import torch_cached
from onlikhorn.dataset import get_output_dir, make_data

exp_name = 'online_grid6'
exp = Experiment(exp_name)
exp_dir = join(get_output_dir(), exp_name)
exp.observers = [FileStorageObserver(exp_dir)]


@exp.config
def config():
    data_source = 'gmm_1d'
    n_samples = 10000
    n_iter = 1000
    max_length = 100000
    device = 'cuda'

    # Overrided
    batch_size = 100
    seed = 0
    epsilon = 1e-2

    method = 'sinkhorn'
    batch_exp = 0
    lr_exp = 1
    lr = 1
    refit = True

@exp.named_config
def long():
    data_source = 'gmm_1d'
    n_samples = 10000
    n_iter = 1000
    max_length = 100000
    device = 'cuda'

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
def debug():
    data_source = 'dragon'
    n_samples = 1000
    n_iter = 4
    max_length = 10000
    device = 'cpu'

    # Overrided
    batch_size = 10
    seed = 0
    epsilon = 1e-2

    method = 'online'
    batch_exp = 0
    lr_exp = 0
    lr = 1
    refit = True


@exp.main
def run(data_source, n_samples, epsilon, n_iter, device, method,
        batch_exp, batch_size, lr, lr_exp, max_length, refit, _seed, _run):
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    output_dir = join(exp.observers[0].dir, 'artifacts')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x, la, y, lb, x_sampler, y_sampler = make_data(data_source, n_samples)

    x, y, la, lb = x.to(device), y.to(device), la.to(device), lb.to(device)
    x_sampler.to(device)
    y_sampler.to(device)

    F, G, trace = torch_cached(sinkhorn)(x, la, y, lb, n_iter=2 * n_iter, epsilon=epsilon, save_trace=True,
                                         verbose=False,
                                         count_recompute=True)
    xr, lar, yr, lbr, _, _ = make_data(data_source, n_samples)
    xr, yr, lar, lbr = xr.to(device), yr.to(device), lar.to(device), lbr.to(device)
    Fr, Gr, tracer = torch_cached(sinkhorn)(xr, lar, yr, lbr, n_iter=2 * n_iter, epsilon=epsilon, save_trace=True,
                                            verbose=False,
                                            count_recompute=True)
    ref = {'train': (F(x), x, G(y), y), 'test': (Fr(xr), xr, G(yr), yr)}
    max_calls = tracer[-1]['n_calls']

    if method == 'sinkhorn_precompute':
        F, G, trace = sinkhorn(x, la, y, lb, n_iter=n_iter, epsilon=epsilon, save_trace=True, ref=ref,
                               max_calls=max_calls)
    elif method == 'sinkhorn':
        F, G, trace = sinkhorn(x, la, y, lb, n_iter=n_iter, epsilon=epsilon, save_trace=True, ref=ref,
                               max_calls=max_calls,
                               count_recompute=True)
    elif method == 'subsampled':
        F, G, trace = subsampled_sinkhorn(x, la, y, lb, n_iter=n_iter, batch_size=batch_size,
                                          max_calls=max_calls,
                                          epsilon=epsilon, save_trace=True, ref=ref, count_recompute=True)
    elif method == 'random':
        F, G, trace = random_sinkhorn(x_sampler=x_sampler, y_sampler=y_sampler, n_iter=int(n_iter * len(x) / batch_size),
                                      epsilon=epsilon, save_trace=True, ref=ref, use_finite=False,
                                      batch_sizes=batch_size,
                                      max_calls=max_calls)
    elif method in ['online', 'online_on_finite', 'online_as_warmup']:
        batch_sizes, lrs, lr_exp = schedule(batch_exp, batch_size, lr, lr_exp, max_length, n_iter, refit)
        print(f'Using lr_exp={lr_exp}')
        _run.info['lr_exp'] = lr_exp
        if method == 'online':
            F, G, trace = online_sinkhorn(x_sampler=x_sampler, y_sampler=y_sampler, batch_sizes=batch_sizes,
                                          refit=refit, force_full=False,
                                          lrs=lrs, n_iter=n_iter, use_finite=False, max_length=max_length,
                                          epsilon=epsilon, save_trace=True, ref=ref, max_calls=max_calls)
        elif method == 'online_on_finite':
            F, G, trace = online_sinkhorn(x=x, la=la, y=y, lb=lb, batch_sizes=batch_sizes,
                                          refit=refit, force_full=True,
                                          lrs=lrs, n_iter=n_iter, use_finite=False, max_length=max_length,
                                          max_calls=max_calls,
                                          epsilon=epsilon, save_trace=True, ref=ref)
        elif method == 'online_as_warmup':
            F, G, trace = online_sinkhorn(x=x, la=la, y=y, lb=lb, batch_sizes=batch_sizes,
                                          refit=refit, force_full=True,
                                          lrs=lrs, n_iter=n_iter, use_finite=True, max_length=max_length,
                                          max_calls=max_calls,
                                          epsilon=epsilon, save_trace=True, ref=ref)
        else:
            raise ValueError
    else:
        raise ValueError

    torch.save(dict(x=x, la=la, y=y, lb=lb, F=F, G=G, trace=trace), join(output_dir, 'results.pkl'))


if __name__ == '__main__':
    exp.run_commandline()
