import json
import os
from os.path import join

import torch

import pandas as pd

from onlikhorn.dataset import get_output_dir, make_gmm_2d

import numpy as np

output_dir = join(get_output_dir(), 'online_grid5')

import matplotlib.pyplot as plt

def gather():
    traces = []

    for exp_dir in os.listdir(output_dir):
        try:
            conf = json.load(open(join(output_dir, exp_dir, 'config.json'), 'r'))
            run = json.load(open(join(output_dir, exp_dir, 'run.json'), 'r'))
            status = run['status']
        except:
            continue
        try:
            trace = torch.load(join(output_dir, exp_dir, 'artifacts', 'results.pkl'), map_location=torch.device('cpu'))['trace']
            print(len(trace))
        except:
            print(f'No trace for {exp_dir}, {status}')
            print(conf)
            continue
        trace = pd.DataFrame(trace)
        for k, v in conf.items():
            if k not in ['n_iter', 'n_samples']:
                trace[k] = v
            trace['exp_dir'] = exp_dir
            trace['status'] = status
        traces.append(trace)
    traces = pd.concat(traces)
    traces.to_pickle(join(output_dir, 'all.pkl'))


def make_2d_grid(shape):
    X, Y = np.meshgrid(np.linspace(-5, 5, shape[0]), np.linspace(-5, 5, shape[1]))
    Z = np.concatenate([X[:, :, None], Y[:, :, None]], axis=2).reshape(-1, 2)
    Z = torch.from_numpy(Z).float()
    return Z


def compute_grad(potential, z):
    z = z.clone()
    z.requires_grad = True
    grad, = torch.autograd.grad(potential(z).sum(), (z,))
    return - grad.detach()


def get_ids():
    import pandas as pd

    df = pd.read_pickle(join(output_dir, 'all.pkl'))

    df['data_source'].value_counts()

    q = df.query('data_source == "gmm_2d"').groupby(by='method')['exp_dir'].first()
    print(q)


def plot(exp_dir):
    res = torch.load(join(output_dir, str(exp_dir), 'artifacts',
                            'results.pkl'), map_location=torch.device('cpu'))
    F, x, G, y = res['F'], res['x'], res['G'], res['y']

    x_sampler, y_sampler = make_gmm_2d()

    shape = (100, 100)
    z = make_2d_grid(shape)
    llx = x_sampler.log_prob(z)
    lly = y_sampler.log_prob(z)

    grad_f = compute_grad(F, x)
    grad_g = compute_grad(G, y)

    grad_fgrid = compute_grad(F, z)
    grad_ggrid = compute_grad(G, z)

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)

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


gather()
# plot(25)