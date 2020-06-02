from os.path import join, expanduser
import pandas as pd

import numpy as np
from joblib import Memory, delayed, Parallel
from sklearn.model_selection import ParameterGrid

from onlikhorn.dataset import get_output_dir

from onlikhorn.data import Subsampler, make_gmm_1d, make_random_5d, get_cloud_3d, make_gmm
from onlikhorn.finite import online_sinkhorn_finite
from onlikhorn.solver import sinkhorn

import matplotlib.pyplot as plt


def run_OT(source):
    np.random.seed(0)

    if source == 'gmm_1d':
        (x, _), (y, _) = make_gmm_1d(1000, 1000)
    elif source == 'gmm':
        (x, _), (y, _) = make_gmm(1000, 1000, 10, 10)
    elif source == 'random_5d':
        (x, _), (y, _) = make_random_5d(1000, 2000)
    elif source == 'cloud_3d':
        (x, _), (y, _) = get_cloud_3d()
        x = x[:11000]  # we do not handle gracefully batches that are not full FIXME
        y = y[:10000]

    epsilon = 1e-1

    n = x.shape[0]
    m = y.shape[0]
    ref_updates = 200
    max_updates = 50
    mem = Memory(location=expanduser('~/cache'))

    ot = mem.cache(sinkhorn)(x, y, ref_updates, epsilon=epsilon, simultaneous=True)
    f, g = ot.evaluate_potential(x=x, y=y)
    w = ot.compute_ot()
    ref = dict(f=f, g=g, x=x, y=y, w=w)
    jobs = []
    # for batch_size in [10, 100, 1000]:
    #     for step_size_exp in [0, 1/2, 1]:
    #         for batch_size_exp in [0, 1/2, 1]:
    #             for full_update in [True, False]:
    #                 jobs.append(
    #                     [f'Online Sinhkorn b={batch_size}, step_size_exp={step_size_exp} full_update={full_update},'
    #                      f'batch_size_exp={batch_size_exp}',
    #                      delayed(online_sinkhorn_finite)
    #                      (x, y,
    #                       epsilon=epsilon,
    #                       max_updates=max_updates,
    #                       ref=ref,
    #                       full_update=full_update, step_size=1,
    #                       step_size_exp=step_size_exp,
    #                       batch_size=batch_size, batch_size_exp=batch_size_exp,
    #                       )])

    grid = ParameterGrid(dict(batch_size=[100], step_size_exp=[0., 1 / 2, 1.], batch_size_exp=[0, 'auto'],
                              full_update=[False, True], averaging=[False, True]))
    grid = list(grid)

    grid.append(dict(batch_size_exp=0, batch_size=len(x), step_size_exp=0, averaging=False, full_update=True))

    for p in grid:
        if p["batch_size_exp"] == 'auto':
            p["batch_size_exp"] = 1 - p["step_size_exp"] + 0.1
        jobs.append(
            [p,
             delayed(mem.cache(online_sinkhorn_finite))
             (x, y,
              epsilon=epsilon,
              max_updates=max_updates,
              ref=ref,
              full_update=p["full_update"], step_size=1,
              step_size_exp=p["step_size_exp"],
              averaging=p['averaging'],
              batch_size=p["batch_size"], batch_size_exp=p["batch_size_exp"],
              )])

    traces = Parallel(n_jobs=6)(job for (_, job) in jobs)
    dfs = []
    for ot, (params, job) in zip(traces, jobs):
        trace = ot.callback.trace
        df = pd.DataFrame(trace)
        for k, v in params.items():
            df[k] = v
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    output_dir = get_output_dir()
    dfs.to_pickle(join(output_dir, f'results_finite_{source}_small.pkl'))


def plot_results(source):
    output_dir = get_output_dir()
    df = pd.read_pickle(join(output_dir, f'results_finite_{source}_small.pkl'))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex='col', sharey='row')
    df = df.loc[(~df['full_update'] | (df['batch_size'] == 1000)) & df['averaging']]
    for index, sub_df in df.groupby(by=['batch_size', 'batch_size_exp', 'step_size_exp', "full_update", "averaging"]):
        name = index
        # computations = [1] + sub_df['computations'].tolist()
        # axes[0][0].plot(computations, [sub_df['w_rel'].iloc[0]] + sub_df['w_rel'].tolist(), label=name)
        # axes[1][0].plot(computations, [sub_df['var_err'].iloc[0]] + sub_df['var_err'].tolist(), label=name)
        # iterations = sub_df['iter'].copy()
        # iterations.iloc[0] = 0.1
        # axes[0][2].plot(iterations, sub_df['w_rel'], label=name)
        # axes[1][2].plot(iterations, sub_df['var_err'], label=name)
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
    # axes[0][0].set_ylim([1e-3, 1e2])
    axes[1][1].legend()
    fig.savefig(join(output_dir, f'results_{source}.pdf'))
    plt.show()


# run_OT('gmm_1d')
plot_results('gmm_1d')
