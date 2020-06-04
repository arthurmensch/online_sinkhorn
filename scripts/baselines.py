import os
from os.path import join

import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver

from onlikhorn.algorithm import sinkhorn
from onlikhorn.dataset import get_output_dir, make_data

exp_name = 'baselines'
exp = Experiment('baselines')
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


@exp.main
def run(data_source, n_samples, epsilon, n_iter, device, _seed):
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    output_dir = join(exp.observers[0].dir, 'artifacts')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x, la, y, lb, x_sampler, y_sampler = make_data(data_source, n_samples)

    x, y, la, lb = x.to(device), y.to(device), la.to(device), lb.to(device)

    F, G, trace = sinkhorn(x, la, y, lb, n_iter=n_iter, epsilon=epsilon, save_trace=True)
    torch.save(dict(x=x, la=la, y=y, lb=lb, F=F, G=G, trace=trace),
               join(exp_dir,
                    f'data_{data_source}_n_samples_{n_samples}_n_iter_{n_iter}_epsilon_{epsilon}_seed_{_seed}.pkl'))


if __name__ == '__main__':
    exp.run_commandline()
