import os.path
import random
from os.path import expanduser, join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from joblib import Parallel, delayed, Memory
from matplotlib import rc
from torch.nn import Parameter

from gamesrl.games import QuadraticGame
from gamesrl.numpy.games import make_positive_matrix
from gamesrl.train import mirror_prox_nash

matplotlib.rcParams['backend'] = 'svg'
matplotlib.rcParams['svg.fonttype'] = 'none'
rc('text', usetex=True)

output_dir = expanduser('~/output/games_rl/toy-2')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class SimpleStrategy(nn.Module):
    def __init__(self, x: float):
        super().__init__()
        self.x = Parameter(torch.tensor([x]))

    def forward(self):
        return self.x


class Callback:
    def __init__(self):
        self.xs = []
        self.ys = []
        self.computations = []

    def __call__(self, strategies, computations):
        self.xs.append(strategies[0].x.data.item())
        self.ys.append(strategies[1].x.data.item())
        self.computations.append(computations)


n_jobs = 4
seed = 2000
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

A = torch.from_numpy(make_positive_matrix(n_players=2, n_actions=1, skewness=0.999)).float()
game = QuadraticGame(A, activation='softplus')


def solve(extrapolation=True, sampling='all', averaging=True, variance_reduction=False, adam=False, seed=None,
          step_size=.1, n_iter=1000):

    strategies = [SimpleStrategy(1.5), SimpleStrategy(1.2)]
    callback = Callback()
    mirror_prox_nash(game, strategies, extrapolation=extrapolation, averaging=averaging, sampling=sampling,
                     variance_reduction=variance_reduction, step_size=step_size, n_iter=n_iter, callback=callback,
                     eval_every=False, seed=seed)
    return callback.xs, callback.ys, callback.computations


def make_trace():
    exps = [dict(name='Full extra-gradient', extrapolation=True, sampling='all', variance_reduction=False),
            dict(name=r'\textbf{Cyclic extra-gradient}', extrapolation=True, sampling='alternated', variance_reduction=False),
            # dict(name='Subsampled extragradient (w/ VR)', extrapolation=True, sampling=1, variance_reduction=True),
            dict(name=r'\textbf{Doubly-stochastic}' + '\n' + r'\textbf{extra-gradient}', extrapolation=True, sampling=1, variance_reduction=False)
            ]
    n_repeat = 10
    seeds = np.random.randint(0, 10000, size=n_repeat)
    step_sizes = [.2]   # np.logspace(-3, 0, 16)
    mem = Memory(location=expanduser('~/cache'))
    res = Parallel(n_jobs=4, verbose=10)(
        delayed(solve)(variance_reduction=exp['variance_reduction'],
                       extrapolation=exp['extrapolation'],
                       sampling=exp['sampling'], averaging=True, adam=False, n_iter=1000,
                       seed=this_seed, step_size=step_size)
        for exp in exps for this_seed in seeds for step_size in step_sizes)
    df = []
    for exp in exps:
        for this_seed in seeds:
            for step_size in step_sizes:
                xs, ys, cs = res[0]
                for i, (x, y, c) in enumerate(zip(xs, ys, cs)):
                    df.append(dict(x=x, y=y, c=c, step=i, seed=this_seed, step_size=step_size, name=exp['name']))
                res = res[1:]

    df = pd.DataFrame(df)
    df.set_index(['name', 'step_size', 'seed', 'step'], inplace=True)
    df = df.groupby(['name', 'step_size', 'step']).mean()

    df['distance'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2)


    def score_fn(rec):
        weights = np.float_power(1, rec['c'][-1] - rec['c'])
        score = np.sum(rec['distance'] * weights) / np.sum(weights)
        return score

    df['score'] = df.groupby(['name', 'step_size']).apply(score_fn)
    idxmin = df['score'].groupby(['name', 'step']).idxmin()

    df = df.loc[idxmin]
    df.reset_index('step_size', inplace=True)

    df.to_pickle(join(output_dir, 'records.pkl'))


def plot():
    df = pd.read_pickle(join(output_dir, 'records.pkl'))

    X, Y = np.meshgrid(np.linspace(-1.2, 1.8, 20, dtype=np.float32),
                       np.linspace(-0.5, 2.5, 20, dtype=np.float32))

    def get_vector_field(X, Y):
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        gX, gY = torch.zeros_like(X), torch.zeros_like(Y)
        for i in range(len(X)):
            for j in range(len(Y)):
                x = Parameter(X[i, [j]])
                y = Parameter(Y[i, [j]])
                lx, ly = game([x, y])
                gX[i, j] = torch.autograd.grad(lx, (x,), retain_graph=True)[0]
                gY[i, j] = torch.autograd.grad(ly, (y,))[0]
        return gX.numpy(), gY.numpy()

    gX, gY = get_vector_field(X, Y)

    scale = 50
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(397.48499 / scale, 100 / scale), constrained_layout=False)
    plt.subplots_adjust(left=0., right=0.98, top=0.98, bottom=0.12, wspace=0.33)
    ax1.quiver(X, Y, -gX * 3, -gY * 3, width=3e-3, color='.5')
    for name, rec in df.groupby(['name']):
        ax1.plot(rec['x'], rec['y'], markersize=2, label=name, linewidth=2, alpha=0.8)
        ax2.plot(rec['c'], rec['distance'], markersize=2, label=name, linewidth=2, alpha=0.8)

    ax2.legend(fontsize=10, frameon=False, loc='upper right', bbox_to_anchor=(1.12, 1.08))
    ax1.set_ylim([-0.5, 2.5])
    ax1.set_xlim([-1.2, 1.8])
    ax2.set_ylabel('Distance to Nash')
    ax2.annotate('Computations', xy=(0, 0), xytext=(0, -7), textcoords='offset points',
                 fontsize=8,
                 xycoords='axes fraction', ha='right', va='top', zorder=10000)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    ax2.tick_params(axis='both', which='minot', labelsize=5)
    ax1.annotate('Nash', xy=(0, 0), xytext=(6, -6), textcoords='offset points', xycoords='data', zorder=10000,
                 fontsize=11)
    ax1.annotate(r'$\theta_0$', xytext=(6, -6), textcoords='offset points', xy=(1.5, 1.2), xycoords='data',
                 fontsize=11)
    ax1.annotate('Trajectory', xy=(0.5, 0), xytext=(-3, -5), textcoords='offset points',
                 xycoords='axes fraction', ha='center', va='top', fontsize=11)
    ax2.set_yscale('log')
    ax2.set_ylim([3e-2, 6])
    ax1.axis('off')
    ax0.axis('off')
    sns.despine(fig, [ax2])
    # plt.savefig(join(output_dir, 'figure.pdf'), transparent=True)
    plt.savefig(join(output_dir, 'figure.svg'), transparent=True)
    plt.show()


# make_trace()
plot()
