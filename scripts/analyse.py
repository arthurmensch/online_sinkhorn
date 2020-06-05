import json
import os
from os.path import join

import torch

import pandas as pd

from onlikhorn.dataset import get_output_dir, make_gmm_2d

import numpy as np

# output_dir = join(get_output_dir(), 'online_grid5')



import matplotlib.pyplot as plt

def gather(output_dirs):
    traces = []

    for output_dir in output_dirs:
        for exp_dir in os.listdir(output_dir):
            try:
                conf = json.load(open(join(output_dir, exp_dir, 'config.json'), 'r'))
                run = json.load(open(join(output_dir, exp_dir, 'run.json'), 'r'))
                status = run['status']
            except:
                print(f'No trace for {exp_dir}')
                continue
            try:
                trace = torch.load(join(output_dir, exp_dir, 'artifacts', 'results.pkl'), map_location=torch.device('cpu'))['trace']
            except:
                print(f'No trace for {exp_dir}, {status}')
                continue
            print(output_dir, exp_dir, conf, len(trace))
            trace = pd.DataFrame(trace)
            for k, v in conf.items():
                if k not in ['n_iter', 'n_samples']:
                    trace[k] = v
                else:
                    if k == 'n_iter':
                        trace['total_n_iter'] = v
                trace['exp_dir'] = exp_dir
                trace['status'] = status
            traces.append(trace)
    traces = pd.concat(traces)
    traces.to_pickle(join(get_output_dir(), 'all.pkl'))
    return traces


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


def get_ids(df):
    df['data_source'].value_counts()

    q = df.query('data_source == "gmm_2d"').groupby(by='method')['exp_dir'].first()
    print(q)


def plot_quiver(output_dir, exp_dir):
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


def plot_warmup(df):

    import matplotlib.pyplot as plt
    import seaborn as sns
    df = df.query('method != "online"')
    df = df.query('method != "random"')
    df = df.query('method != "sinkhorn"')
    # df = df.query('lr_exp == "auto" | method != "online_as_warmup"')
    df = df.query('(refit == False & batch_exp == .5 & lr_exp == 0) | method != "online_as_warmup"')
    df = df.query('method != "subsampled"')

    df = df.query('data_source in ["gmm_2d", "gmm_10d", "dragon", ]')
    df = df.query('epsilon == 1e-3')

    pk = ['data_source', 'epsilon', 'method', 'refit', 'batch_exp', 'lr_exp', 'batch_size', 'n_iter']
    df = df.groupby(by=pk).agg(['mean', 'std']).reset_index('n_iter')

    plot_err = True
    NAMES = {'dragon': 'Stanford 3D', 'gmm_10d': '10D GMM', 'gmm_2d': '2D GMM'}
    fig, axes = plt.subplots(1, 3, figsize=(6, 1.8))
    for i, ((data_source, epsilon), df2) in enumerate(df.groupby(['data_source', 'epsilon'])):
        for index, df3 in df2.groupby(['method', 'refit', 'batch_exp', 'lr_exp', 'batch_size']):
            n_calls = df3['n_calls']
            train = df3['ref_err_train']
            test = df3['ref_err_test']
            err = df3['fixed_err']
            if plot_err:
                train = err
            if index[0] == 'sinkhorn_precompute':
                label = 'Standard\nSinkhorn'
            else:
                label = 'Online\nSinkhorn\nwarmup'
            axes[i].plot(n_calls['mean'], train['mean'], label=label)
            if index[0] != 'sinkhorn_precompute':
                axes[i].fill_between(n_calls['mean'], train['mean'] - train['std'], train['mean'] + train['std'],
                                     alpha=0.2)
        axes[i].annotate(NAMES[data_source], xy=(.5, .8), xycoords="axes fraction",
                         ha='center', va='bottom')
    axes[0].annotate('Computat.', xy=(-.3, -.13), xycoords="axes fraction",
                     ha='center', va='bottom')
    axes[2].legend(loc='center left', frameon=False, bbox_to_anchor=(.7, 0.5), ncol=1)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.tick_params(axis='both', which='minor', labelsize=7)
    if plot_err:
        axes[0].set_ylabel('||T(f)-g|| +|| T(g)-f||')
    else:
        axes[0].set_ylabel('|| f - f*|| + || g -g*||')
    sns.despine(fig)
    fig.subplots_adjust(right=0.75)
    fig.savefig('online+full.pdf')
    plt.show()

output_dirs = [join(get_output_dir(), 'online_grid10'), join(get_output_dir(), 'online_grid11')]
# df = gather(output_dirs)
df = pd.read_pickle(join(join(get_output_dir(), 'online_grid9'), 'all.pkl'))
get_ids(df)
plot_warmup(df)
# plot(25)