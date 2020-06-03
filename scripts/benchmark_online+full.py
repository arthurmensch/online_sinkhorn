import os
from functools import partial
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, Memory, delayed

from onlikhorn.data import Subsampler, make_gmm_1d, make_random_5d, get_cloud_3d, make_gmm
from onlikhorn.dataset import get_output_dir
from onlikhorn.solver import sinkhorn, online_sinkhorn


def run_OT(source):
    np.random.seed(0)

    if source == 'gmm_1d':
        (x, _), (y, _) = make_gmm_1d(1000, 1000)
    elif source == 'gmm':
        (x, _), (y, _) = make_gmm(1000, 1000, 4, 2)
    elif source == 'random_5d':
        (x, _), (y, _) = make_random_5d(1000, 2000)
    elif source == 'cloud_3d':
        (x, _), (y, _) = get_cloud_3d()
        x = x[:11000]  # we do not handle gracefully batches that are not full FIXME
        y = y[:10000]

    epsilon = 1e-2

    n = x.shape[0]
    m = y.shape[0]
    ref_updates = 400
    refine_updates = 100

    mem = Memory(location=None)

    batch_size = int(0.01 * len(x))
    avg_step_size_exp = 0

    ot = mem.cache(sinkhorn)(x, y, ref_updates, epsilon=epsilon)
    f, g = ot.evaluate_potential(x=x, y=y)
    w = ot.compute_ot()
    ref = dict(f=f, g=g, x=x, y=y, w=w)
    x_sampler = Subsampler(x)
    y_sampler = Subsampler(y)

    jobs = []
    grid = []
    # Test other
    # FC Online Sinkhorn
    grid.append(dict(batch_size_exp=0.6, batch_size=batch_size, step_size_exp=0., avg_step_size_exp=avg_step_size_exp, full_update=True))
    grid.append(dict(batch_size_exp=0.1, batch_size=batch_size, step_size_exp=0.5, avg_step_size_exp=avg_step_size_exp, full_update=True))
    grid.append(dict(batch_size_exp=0., batch_size=batch_size, step_size_exp=0.6, avg_step_size_exp=avg_step_size_exp, full_update=True))

    # Online Sinkhorn
    grid.append(dict(batch_size_exp=1.1, batch_size=batch_size, step_size_exp=0., avg_step_size_exp=avg_step_size_exp,
                     full_update=False))
    grid.append(dict(batch_size_exp=0.6, batch_size=batch_size, step_size_exp=0.5, avg_step_size_exp=avg_step_size_exp, full_update=False))
    grid.append(dict(batch_size_exp=0.1, batch_size=batch_size, step_size_exp=1, avg_step_size_exp=avg_step_size_exp, full_update=True))

    # Random Sinkhorn
    grid.append(dict(batch_size_exp=0, batch_size=batch_size, step_size_exp=0., avg_step_size_exp=avg_step_size_exp, full_update=False,))

    # Does it converge
    # grid.append(dict(batch_size_exp=0, batch_size=batch_size, step_size_exp=0., avg_step_size_exp=1., full_update=True,))

    # Baseline
    grid.append(dict(batch_size_exp=0, batch_size=len(x), step_size_exp=0, avg_step_size_exp=0., full_update=True))

    for p in grid:
        if p["batch_size_exp"] == 'auto':
            p["batch_size_exp"] = 1 - p["step_size_exp"] + 0.1
        jobs.append(
            [p,
             delayed(mem.cache(online_sinkhorn))(x_sampler, y_sampler, max_size=(n, m),
              epsilon=epsilon,
              refine_updates=refine_updates,
              ref=ref,
              full_update=p["full_update"], step_size=1,
              step_size_exp=p["step_size_exp"],
              avg_step_size_exp=p["avg_step_size_exp"], averaging=False,
              batch_size=p["batch_size"], batch_size_exp=p["batch_size_exp"],
              )])

    traces = Parallel(n_jobs=9)(job for (_, job) in jobs)
    dfs = []
    for ot, (params, job) in zip(traces, jobs):
        trace = ot.callback.trace
        df = pd.DataFrame(trace)
        for k, v in params.items():
            df[k] = v
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    output_dir = get_output_dir()
    dfs.to_pickle(join(output_dir, f'results_finite_{source}_rates_3.pkl'))


def plot_results(source):
    output_dir = get_output_dir()
    df = pd.read_pickle(join(output_dir, f'results_finite_{source}_rates_3.pkl'))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex='col', sharey='row')
    # df = df.loc[df['batch_size_exp'] == 0.1]
    for index, sub_df in df.groupby(
            by=['batch_size', 'batch_size_exp', 'step_size_exp', "full_update", "avg_step_size_exp"]):
        name = index
        sub_df = sub_df.iloc[:-3]
        axes[0][0].plot(sub_df['computations'], sub_df['w_err'], label=name)
        axes[1][0].plot(sub_df['computations'], sub_df['var_err'], label=name)
        axes[0][2].plot(sub_df['iter'], sub_df['w_err'], label=name)
        axes[1][2].plot(sub_df['iter'], sub_df['var_err'], label=name)
        axes[0][1].plot(sub_df['samples'], sub_df['w_err'], label=name)
        axes[1][1].plot(sub_df['samples'], sub_df['var_err'], label=name)
    for ax in axes.ravel():
        ax.set_yscale('log')
        ax.set_xscale('log')
    axes[1][0].set_xlabel('Computations')
    axes[1][1].set_xlabel('Samples')
    axes[1][2].set_xlabel('Iteration')
    axes[0][0].set_ylabel('W err')
    axes[1][0].set_ylabel('Var err')
    axes[1][1].legend()
    fig.savefig(join(output_dir, f'results_{source}.pdf'))
    plt.show()


def make_prec_table(source):
    output_dir = get_output_dir()
    df = pd.read_pickle(join(output_dir, f'results_warmstart_big_{source}.pkl'))[['computations', 'w_rel', 'name']]

    def find(df, precision):
        index = np.where(df['w_rel'] < precision)[0]
        if len(index) > 0:
            index = index[0]
        else:
            print('Precision not reached')
            index = -1
        return df['computations'].iloc[index]

    precision = {precision: df.groupby(by='name').apply(partial(find, precision=precision))
                 for precision in [1e-3, 5e-4]}
    precision = pd.concat(precision.values(), keys=precision.keys(), names=['precision'])
    speed_up = precision.groupby('precision').apply(lambda x: x.loc[pd.IndexSlice[:, 'Sinkhorn s=1.0/t^0.0']] / x)
    speed_up.rename(index={'Online Sinhkorn b=100, full_update=False': 'Online Sinkhorn 1%',
                           'Online Sinhkorn b=1000, full_update=False': 'Online Sinkhorn 10%',
                           'Sinkhorn s=1.0/t^0.0': 'Full Sinkhorn'}, level='name', inplace=True)
    speed_up.name = 'speed_up'
    print(speed_up.round(2))


run_OT('gmm')
plot_results('gmm')
# make_prec_table('cloud_3d')
