import os
from os.path import join

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver

from onlikhorn.algorithm import sinkhorn, online_sinkhorn, random_sinkhorn, subsampled_sinkhorn, schedule
from onlikhorn.dataset import get_output_dir, make_data

exp_name = 'online'
exp = Experiment('online')
exp_dir = join(get_output_dir(), exp_name)
exp.observers = [FileStorageObserver(exp_dir)]


@exp.config
def config():
    data_source = 'dragon'
    n_samples = 10000
    n_iter = 1000
    seed = 0
    epsilon = 1e-2
    device = 'cuda'

    method = 'online'

    lr_exp = .5
    batch_exp = 0  # > 2*(1 - lr_exp) for convergence if not refit,
    batch_size = 100
    lr = 1
    max_length = 100000
    refit = False
    force_full = False


@exp.main
def run(data_source, n_samples, epsilon, n_iter, device, method, force_full,
        batch_exp, batch_size, lr, lr_exp, max_length, refit, _seed):
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    output_dir = join(exp.observers[0].dir, 'artifacts')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x, la, y, lb, x_sampler, y_sampler = make_data(data_source, n_samples)
    x, y, la, lb = x.to(device), y.to(device), la.to(device), lb.to(device)
    x_sampler.to(device)
    y_sampler.to(device)

    if method == 'sinkhorn':
        F, G, trace = sinkhorn(x, la, y, lb, n_iter=n_iter, epsilon=epsilon, save_trace=True)
    elif method == 'subsampled_sinkhorn':
        F, G, trace = subsampled_sinkhorn(x, la, y, lb, n_iter=n_iter, batch_size=batch_size,
                                          epsilon=epsilon, save_trace=True)
    elif method == 'random_sinkhorn':
        F, G, trace = random_sinkhorn(x, la, y, lb, n_iter=n_iter, epsilon=epsilon, save_trace=True)
    elif method == 'online_sinkhorn':
        batch_sizes, lrs = schedule(batch_exp, batch_size, lr, lr_exp, max_length, n_iter)

        F, G, trace = online_sinkhorn(x_sampler=x_sampler, y_sampler=y_sampler, batch_sizes=batch_sizes,
                                      refit=refit, force_full=force_full,
                                      lrs=lrs, n_iter=n_iter, use_finite=False, max_length=max_length,
                                      epsilon=epsilon, save_trace=True)
    else:
        raise ValueError

    torch.save(dict(x=x, la=la, y=y, lb=lb, F=F, G=G, trace=trace),
               join(exp_dir,
                    f'data_{data_source}_n_samples_{n_samples}_n_iter_{n_iter}_epsilon_{epsilon}_seed_{_seed}.pkl'))


if __name__ == '__main__':
    exp.run_commandline()
