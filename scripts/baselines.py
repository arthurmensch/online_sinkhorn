import os
from os.path import join

import joblib
import torch

from onlikhorn.dataset import get_output_dir, make_dragon, make_sphere, make_gmm_1d, make_gmm
from sacred import Experiment
from sacred.observers import FileStorageObserver

from onlikhorn.algorithm import sinkhorn

import numpy as np

exp_name = 'baselines'
exp = Experiment('baselines')
exp_dir = join(get_output_dir(), exp_name)
exp.observers = [FileStorageObserver(exp_dir)]


@exp.config
def config():
    data_source = 'dragon'
    n_samples = 10000
    n_iter = 10
    seed = 0
    epsilon = 1e-2


@exp.main
def run(data_source, n_samples, epsilon, n_iter, _seed):
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    output_dir = join(exp.observers[0].dir, 'artifacts')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if data_source == 'dragon':
        x, la = make_sphere()
        y, lb = make_dragon()
    else:
        if data_source == 'gmm_1d':
            x_sampler, y_sampler = make_gmm_1d()
        elif data_source == 'gmm_2d':
            x_sampler, y_sampler = make_gmm_1d()
        elif data_source == 'gmm_10d':
            x_sampler, y_sampler = make_gmm(10, 10)
        else:
            raise ValueError
        x, la, _ = x_sampler(n_samples)
        y, lb, _ = y_sampler(n_samples)

    F, G, trace = sinkhorn(x, la, y, lb, n_iter=n_iter, epsilon=epsilon, save_trace=True)
    print(trace)
    torch.save(dict(x=x, la=la, y=y, lb=lb, F=F, G=G, trace=trace), join(exp_dir, f'data_{data_source}_n_samples_{n_samples}_n_iter_{n_iter}_epsilon_{epsilon}.pkl'))


if __name__ == '__main__':
    exp.run_commandline()
