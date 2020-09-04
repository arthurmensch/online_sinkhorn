import json
import os
from os.path import join

import torch

import pandas as pd
from onlikhorn.algorithm import compute_grad

from onlikhorn.dataset import get_output_dir, make_gmm_2d

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib import rc
mpl.rcParams['font.size'] = 7
mpl.rcParams['backend'] = 'pdf'
rc('text', usetex=True)

pt_width = 397.48499
pt_per_inch = 72.27
width = pt_width / pt_per_inch

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


def get_ids(df):
    df['data_source'].value_counts()

    q = df.query('data_source == "gmm_2d"').groupby(by='method')['exp_dir'].first()
    print(q)


def plot_quiver(output_dir, exp_dir):
    res = torch.load(join(output_dir, str(exp_dir), 'artifacts', 'results.pkl'), map_location=torch.device('cpu'))
    F, x, G, y = res['F'], res['x'], res['G'], res['y']
    print(len(x))
    torch.manual_seed(122)
    np.random.seed(122)
    x_sampler, y_sampler = make_gmm_2d()
    xr, lar, _ = x_sampler(1000)
    yr, lbr, _ = y_sampler(1000)
    shape = (100, 100)
    z = make_2d_grid(shape)
    llx = x_sampler.log_prob(z)
    lly = y_sampler.log_prob(z)

    grad_f = compute_grad(F, xr)
    grad_g = compute_grad(G, yr)

    grad_fgrid = compute_grad(F, z)
    grad_ggrid = compute_grad(G, z)

    config = json.load(open(join(output_dir, str(exp_dir), 'config.json'), 'r'))
    method = config['method']
    n_samples = config['n_samples']
    batch_exp = config['batch_exp']
    print(batch_exp)
    batch_size = config['batch_size']
    max_calls = config['max_calls']
    annotate = False
    if method == 'sinkhorn':
        name = rf'Sinkhorn N={n_samples},\quad$10^{{{np.log10(max_calls):.0f}}}$ computations'
        if n_samples == 10000:
            annotate = True
    else:
        name = rf'Online Sinkhorn $n(t) \propto {batch_size} t^{{{batch_exp:.1f}}}$,\quad'\
               rf'$10^{{{np.log10(max_calls):.0f}}}$ computations'

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(width, width * .5))

    axes[0].contour(z[:, 0].view(shape), z[:, 1].view(shape), llx.view(shape), zorder=0, levels=7, cmap='Reds')
    axes[0].scatter(xr[:, 0], xr[:, 1], 2, zorder=20, color='C3')
    axes[0].quiver(xr[:, 0], xr[:, 1], grad_f[:, 0], grad_f[:, 1], zorder=30, scale=50)

    axes[0].contour(z[:, 0].view(shape), z[:, 1].view(shape), lly.view(shape), zorder=0, levels=7, cmap='Blues')
    axes[0].scatter(yr[:, 0], yr[:, 1], 2, zorder=10, color='C0')
    if annotate:
        axes[0].annotate('Reg. OT displacement field\n'
                         'on empirical samples', xy=(.5, 0), xycoords='axes fraction',
                         textcoords='offset points', xytext=(0, -10), ha='center', va='top', fontsize=9)
    # axes[0, 0].quiver(y[:, 0], y[:, 1], grad_g[:, 0], grad_g[:, 1], zorder=20)

    scale = .3
    axes[1].contour(z[:, 0].view(shape), z[:, 1].view(shape), llx.view(shape), zorder=0, levels=7, cmap='Reds')
    axes[1].quiver(z[:, 0], z[:, 1], grad_fgrid[:, 0] * llx.exp(), grad_fgrid[:, 1] * llx.exp(), scale=3, zorder=20)

    axes[1].contour(z[:, 0].view(shape), z[:, 1].view(shape), lly.view(shape), zorder=0, levels=7, cmap='Blues')
    if annotate:
        axes[1].annotate('Reg. OT displacement field\n on a regular grid', xy=(.5, 0), xycoords='axes fraction',
                         textcoords='offset points', xytext=(0, -10), ha='center', va='top', fontsize=9)
    # axes[1, 0].quiver(z[:, 0], z[:, 1], grad_ggrid[:, 0] * lly.exp(), grad_ggrid[:, 1] * lly.exp(), zorder=20)
    for ax in axes:
        ax.axis('off')
    fig.suptitle(name, fontsize=9)
    fig.subplots_adjust(bottom=0.2)
    plt.savefig(join(output_dir, f'quiver_{name}.pdf'))

def plot_warmup(df, ytype='err', epsilon=1e-2):
    df = df.query('(method in ["sinkhorn_precompute", "online_as_warmup"]) & '
                  '(data_source in ["gmm_2d", "gmm_10d", "dragon"]) & '
                  'batch_size == 100 & '
                  f'epsilon == {epsilon} & '
                  '((refit == False & batch_exp == .5 & lr_exp == 0) | method != "online_as_warmup")')

    # df.loc[df['fixed_err'].isna(), 'fixed_err'] = df.loc[df['fixed_err'].isna(), 'var_err_train'].values

    pk = ['data_source', 'epsilon', 'method', 'refit', 'batch_exp', 'lr_exp', 'batch_size', 'n_iter']
    df = df.groupby(by=pk).agg(['mean', 'std']).reset_index('n_iter')

    NAMES = {'dragon': 'Stanford 3D', 'gmm_10d': '10D GMM', 'gmm_2d': '2D GMM'}
    fig, axes = plt.subplots(1, 3, figsize=(0.7 * width, width * 0.2 * 0.7))
    order = {'gmm_2d': 0, 'gmm_10d': 1, 'dragon': 2}
    speedups = []
    for (data_source, epsilon), df2 in df.groupby(['data_source', 'epsilon']):
        iter_at_prec = {}
        for index, df3 in df2.groupby(['method', 'refit', 'batch_exp', 'lr_exp', 'batch_size']):
            n_calls = df3['n_calls']
            if ytype == 'train':
                y = df3['ref_err_train']
            elif ytype == 'test':
                y = df3['ref_err_test']
            elif ytype == 'err':
                y = df3['fixed_err']
            else:
                raise ValueError
            if index[0] == 'sinkhorn_precompute':
                label = 'Standard\nSinkhorn'
            else:
                label = 'Online\nSinkhorn\nwarmup'
            axes[order[data_source]].plot(n_calls['mean'], y['mean'], label=label, linewidth=2, alpha=0.8, color='C3' if index[0] == 'sinkhorn_precompute'
            else 'C0')
            # if index[0] != 'sinkhorn_precompute':
            #     axes[i].fill_between(n_calls['mean'], y['mean'] - y['std'], y['mean'] + y['std'],
            #                          alpha=0.2)
            try:
                iter_at_prec[index[0]] = n_calls['mean'].iloc[np.where(y['mean'] < 1e-3)[0][0]]
            except IndexError:
                iter_at_prec[index[0]] = np.float('inf')
        axes[order[data_source]].annotate(NAMES[data_source], xy=(.5, .8), xycoords="axes fraction",
                         ha='center', va='bottom')
        speedups.append(dict(data_source=data_source, epsilon=epsilon,
                             speedup=iter_at_prec['sinkhorn_precompute'] / iter_at_prec['online_as_warmup'],
                             ytype=ytype))
    axes[0].annotate('Comput.', xy=(-.27, -.32), xycoords="axes fraction",
                     ha='center', va='bottom')
    if ytype == 'err' and epsilon == 1e-3:
        axes[0].set_ylim([1e-5, 0.5])
        axes[0].set_ylim([1e-6, 0.1])
        axes[0].set_ylim([1e-4, 0.1])
    axes[2].legend(loc='center left', frameon=False, bbox_to_anchor=(.95, 0.5), ncol=1)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.tick_params(axis='both', which='minor', labelsize=5)
        ax.minorticks_on()
        # ax.set_xlim([0.8e9, 1.2e11])
        # ax.set_xticks([1e9, 1e10, 1e11])
    if ytype == 'err':
        axes[0].set_ylabel(r'$\Vert T(\hat f){-} \hat g\Vert_{\textrm{var}}$', fontsize=5)
    elif ytype == 'train':
        axes[0].set_ylabel(r'$\Vert \hat f {-} f^\star\Vert_{\textrm{var}}$', fontsize=5)
    else:
        axes[0].set_ylabel('$\Vert f {-} f_0^\star\Vert_{\textrm{var}}$', fontsize=5)
    sns.despine(fig)
    fig.subplots_adjust(right=0.8, bottom=0.23)
    fig.savefig(join(get_output_dir(), f'online+full_{epsilon}_{ytype}.pdf'))
    return speedups


def plot_online(df, ytype='test', epsilon=1e-2, refit=False, name=''):
    df = df.query('(method in ["online", "sinkhorn", "subsampled"]) & data_source != "dragon" &'
                  f'(method != "online" | (refit ==  {refit} & lr_exp == "auto")) &'
                  f'epsilon == {epsilon}')

    if refit: # Remove saturated memory
        df.loc[(df['method'] == 'online') & (df['n_samples'] == 40000), ['fixed_err_train', 'fixed_err_test', 'fixed_err']] = np.nan

    pk = ['data_source', 'epsilon', 'method', 'refit', 'batch_exp', 'lr_exp', 'batch_size', 'n_iter']
    df = df.groupby(by=pk).agg(['mean', 'std']).reset_index('n_iter')

    NAMES = {'gmm_1d': '1D GMM', 'gmm_10d': '10D GMM', 'gmm_2d': '2D GMM'}

    df1 = df
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.2))
    order = {'gmm_1d': 0, 'gmm_10d': 2, 'gmm_2d': 1}
    for (data_source, epsilon), df2 in df1.groupby(['data_source', 'epsilon']):
        iter_at_prec = {}
        for index, df3 in df2.groupby(['method', 'refit', 'batch_exp', 'lr_exp', 'batch_size']):
            n_calls = df3['n_calls']
            if ytype == 'train':
                y = df3['fixed_err_train']
            elif ytype == 'test':
                y = df3['fixed_err_test']
            elif ytype == 'err':
                y = df3['fixed_err']
            else:
                raise ValueError
            if index[0] == 'sinkhorn':
                label = f'Sinkhorn $n = 10^5$'
            elif index[0] == 'subsampled':
                label = f'Sinkhorn $n = {index[-1]}$'
            else:
                if index[2] == 0:
                    label = f'O-S $n(t) = {index[-1]}$'
                else:
                    label = f'O-S $n(t) \propto {index[-1]} t^{{{index[2]}}}$'
            axes[order[data_source]].plot(n_calls['mean'], y['mean'], label=label, linewidth=2, alpha=0.8)

        axes[order[data_source]].annotate(NAMES[data_source], xy=(.5, .83), xycoords="axes fraction",
                         ha='center', va='bottom')
    axes[0].annotate('Computations', xy=(-.3, -.25), xycoords="axes fraction",
                     ha='center', va='bottom')
    axes[1].legend(loc='center left', frameon=False, bbox_to_anchor=(1.03, 0.5), ncol=1, fontsize=7)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.tick_params(axis='both', which='minor', labelsize=5)
        ax.minorticks_on()
    if ytype == 'err':
        axes[0].set_ylabel(r'$\Vert T(\hat f){-}\hat g\Vert_{\textrm{var}}$', fontsize=5)
    elif ytype == 'train':
        axes[0].set_ylabel(r'$\Vert \hat f {-} f^\star\Vert_{\textrm{var}}$', fontsize=5)
    else:
        axes[0].set_ylabel(r'$\Vert \hat f {-} f_0^\star\Vert_{\textrm{var}}$', fontsize=5)
    sns.despine(fig)
    fig.subplots_adjust(right=0.75, bottom=0.21)
    fig.savefig(join(get_output_dir(), f'online_{epsilon}_{refit}_{ytype}_{name}.pdf'))


def plot_random(df, ytype='test', epsilon=1e-2, name=''):
    df = df.query('(method in ["random", "sinkhorn", "subsampled"]) & data_source != "dragon" &'
                  f'epsilon == {epsilon}')

    pk = ['data_source', 'epsilon', 'method', 'refit', 'batch_exp', 'lr_exp', 'batch_size', 'n_iter']
    df = df.groupby(by=pk).agg(['mean', 'std']).reset_index('n_iter')

    NAMES = {'gmm_1d': '1D GMM', 'gmm_10d': '10D GMM', 'gmm_2d': '2D GMM'}

    df1 = df
    fig, axes = plt.subplots(1, 3, figsize=(width, width * 0.2))
    order = {'gmm_1d': 0, 'gmm_10d': 2, 'gmm_2d': 1}
    for (data_source, epsilon), df2 in df1.groupby(['data_source', 'epsilon']):
        iter_at_prec = {}
        for index, df3 in df2.groupby(['method', 'refit', 'batch_exp', 'lr_exp', 'batch_size']):
            n_calls = df3['n_calls']
            if ytype == 'train':
                y = df3['ref_err_train']
            elif ytype == 'test':
                y = df3['ref_err_test']
            elif ytype == 'err':
                y = df3['fixed_err']
            else:
                raise ValueError
            if index[0] == 'sinkhorn':
                label = 'Sinkhorn $n = 10^4$'
            elif index[0] == 'subsampled':
                label = f'Sinkhorn $n = {index[-1]}$'
            else:
                if index[2] == 0:
                    label = f'R-S $n = {index[-1]}$'
                else:
                    label = f'R-S $n = {index[-1]}$'
            axes[order[data_source]].plot(n_calls['mean'], y['mean'], label=label, linewidth=2, alpha=0.8)

        axes[order[data_source]].annotate(NAMES[data_source], xy=(.5, .83), xycoords="axes fraction",
                         ha='center', va='bottom')
    axes[0].annotate('Computations', xy=(-.3, -.25), xycoords="axes fraction",
                     ha='center', va='bottom')
    axes[2].legend(loc='center left', frameon=False, bbox_to_anchor=(1.03, 0.5), ncol=1)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.tick_params(axis='both', which='minor', labelsize=5)
        ax.minorticks_on()
    if ytype == 'err':
        axes[0].set_ylabel(r'$\Vert T(\hat f){-}\hat g\Vert_{\textrm{var}}$', fontsize=5)
    elif ytype == 'train':
        axes[0].set_ylabel(r'$\Vert \hat f {-} f^\star\Vert_{\textrm{var}}$', fontsize=5)
    else:
        axes[0].set_ylabel(r'$\Vert \hat f {-} f_0^\star\Vert_{\textrm{var}}$', fontsize=5)
    sns.despine(fig)
    fig.subplots_adjust(right=0.75, bottom=0.21)
    fig.savefig(join(get_output_dir(), f'random_{epsilon}_{ytype}_{name}.pdf'))


def plot_gaussian(df, ytype='test', epsilon=1e-2, refit=False):
    df = df.query('(method in ["online", "sinkhorn", "subsampled"]) & data_source != "dragon" &'
                  f'(method != "online" | (refit ==  {refit} & lr_exp == "auto")) &'
                  f'epsilon == {epsilon}')

    if refit: # Remove saturated memory
        df.loc[(df['method'] == 'online') & (df['n_samples'] == 40000), ['ref_err_test', 'fixed_err']] = np.nan

    pk = ['data_source', 'epsilon', 'method', 'refit', 'batch_exp', 'lr_exp', 'batch_size', 'n_iter']
    df = df.groupby(by=pk).agg(['mean', 'std']).reset_index('n_iter')

    NAMES = {'gaussian_2d': '2D Gaussian', 'gaussian_10d': '10D Gaussian'}

    df1 = df
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.2))
    order = {'gaussian_2d': 0, 'gaussian_10d': 1}
    for (data_source, epsilon), df2 in df1.groupby(['data_source', 'epsilon']):
        iter_at_prec = {}
        for index, df3 in df2.groupby(['method', 'refit', 'batch_exp', 'lr_exp', 'batch_size']):
            n_calls = df3['n_calls']
            if ytype == 'train':
                y = df3['ref_err_train']
            elif ytype == 'test':
                y = df3['ref_err_test']
            elif ytype == 'err':
                y = df3['fixed_err']
            else:
                raise ValueError
            if index[0] == 'sinkhorn':
                label = 'Sinkhorn $n = 10^4$'
            elif index[0] == 'subsampled':
                label = f'Sinkhorn $n = {index[-1]}$'
            else:
                if index[2] == 0:
                    label = f'O-S $n(t) = 100$'
                else:
                    label = f'O-S $n(t) \propto t^{{{index[2]}}}$'
            axes[order[data_source]].plot(n_calls['mean'], y['mean'], label=label, linewidth=2, alpha=0.8)

        axes[order[data_source]].annotate(NAMES[data_source], xy=(.5, .83), xycoords="axes fraction",
                         ha='center', va='bottom')
    axes[0].annotate('Computations', xy=(-.2, -.25), xycoords="axes fraction",
                     ha='center', va='bottom')
    axes[1].legend(loc='center left', frameon=False, bbox_to_anchor=(1.03, 0.5), ncol=1)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.tick_params(axis='both', which='minor', labelsize=5)
        ax.minorticks_on()
    if ytype == 'err':
        axes[0].set_ylabel(r'$\Vert T(\hat f){-}\hat g\Vert_{\textrm{var}}$', fontsize=5)
    elif ytype == 'train':
        axes[0].set_ylabel(r'$\Vert \hat f {-} f^\star\Vert_{\textrm{var}}$', fontsize=5)
    else:
        axes[0].set_ylabel(r'$\Vert \hat f {-} f^\star\Vert_{\textrm{var}}$', fontsize=5)
    sns.despine(fig)
    fig.subplots_adjust(right=0.75, bottom=0.21)
    fig.savefig(join(get_output_dir(), f'online_{epsilon}_{refit}_{ytype}_gaussian.pdf'))

pipeline = ['gather']

# if 'gather' in pipeline:
#     output_dirs = [join(get_output_dir(), 'online_grid12')]
#     df = gather(output_dirs)
#     df.to_pickle(join(get_output_dir(), 'all_warmup.pkl'))

if 'gather' in pipeline:
    output_dirs = [join(get_output_dir(), 'online_grid_big_5')]
    df = gather(output_dirs)
    df.to_pickle(join(get_output_dir(), 'all_big_5.pkl'))

# Figure 1
if 'figure_1' in pipeline:
    output_dirs = [join(get_output_dir(), 'online_grid10'), join(get_output_dir(), 'online_grid11')]
    df = pd.read_pickle(join(get_output_dir(), 'all_big.pkl'))
    for ytype in ['test']:
        for epsilon in np.logspace(-4, -1, 4):
            for refit in [False, True]:
                plot_online(df, refit=refit, epsilon=epsilon, ytype=ytype, name='big')
    del df

if 'random' in pipeline:
    output_dirs = [join(get_output_dir(), 'online_grid10'), join(get_output_dir(), 'online_grid11')]
    df = pd.read_pickle(join(get_output_dir(), 'all.pkl'))
    for ytype in ['test']:
        for epsilon in np.logspace(-4, -1, 4):
            plot_random(df, epsilon=epsilon, ytype=ytype)
    del df
# #
# Figure 3
if 'figure_3' in pipeline:
    speedups = []
    df = pd.read_pickle(join(get_output_dir(), 'all_warmup.pkl'))
    for epsilon in np.logspace(-4, -1, 4):
        for ytype in ['err', 'train']:
            speedup = plot_warmup(df, epsilon=epsilon, ytype=ytype)
            speedups += speedup
    del df
    # Table 1
if 'table_1' in pipeline:
    speedups = pd.DataFrame(speedups)
    speedups.to_pickle('speedups.pkl')
    speedups = pd.read_pickle('speedups.pkl')
    df = speedups.set_index(['ytype', 'data_source', 'epsilon']).unstack('epsilon')
    print(df.loc['err'].round(1).to_latex())


# Figure gaussian
if 'figure_gaussian' in pipeline:
    # output_dirs = [join(get_output_dir(), 'online_grid_gaussian_final')]
    # df = gather(output_dirs)
    # df.to_pickle(join(get_output_dir(), 'all_gaussian.pkl'))
    df = pd.read_pickle(join(get_output_dir(), 'all_gaussian.pkl'))
    for ytype in ['test']:
        for epsilon in np.logspace(-4, -1, 4):
            for refit in [False, True]:
                plot_gaussian(df, refit=refit, epsilon=epsilon, ytype=ytype)
    del df

if 'quiver' in pipeline:
    output_dir = join(get_output_dir(), 'online_grid_quiver_2')
    # for exp_dir in [7, 11]:   # Subsampled, online
    # for exp_dir in [15, 16]:   # Subsampled, online
    for exp_dir in range(17, 17 + 8):   # Subsampled, online
        plot_quiver(output_dir, exp_dir)