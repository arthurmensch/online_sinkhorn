import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, Memory, delayed

from onlikhorn.data import Subsampler, make_gmm_1d, make_random_5d, get_cloud_3d
from onlikhorn.dataset import get_output_dir
from onlikhorn.solver import sinkhorn, online_sinkhorn


def run_OT(source):
    np.random.seed(0)

    if source == 'gmm_1d':
        (x, _), (y, _) = make_gmm_1d(1000, 2000)
    elif source == 'random_5d':
        (x, _), (y, _) = make_random_5d(1000, 2000)
    elif source == 'cloud_3d':
        (x, _), (y, _) = get_cloud_3d()
        x = x[::10][:1100]  # we do not handle gracefully batches that are not full FIXME
        y = y[::10][:1000]

    epsilon = 1e-2

    n = x.shape[0]
    m = y.shape[0]
    ref_updates = 200
    refine_updates = 100

    mem = Memory(location=os.path.expanduser('~/cache'))
    mem = Memory(location=None)

    ot = mem.cache(sinkhorn)(x, y, ref_updates, epsilon=epsilon)
    f, g = ot.evaluate_potential(x=x, y=y)
    w = ot.compute_ot()
    ref = dict(f=f, g=g, x=x, y=y, w=w)
    x_sampler = Subsampler(x)
    y_sampler = Subsampler(y)

    jobs = []
    for step_size, step_size_exp in [(1., 0.)]:
        jobs.append((f'Sinkhorn s={step_size}/t^{step_size_exp}',
                     delayed(mem.cache(sinkhorn))(x, y, ref=ref, step_size=step_size, step_size_exp=step_size_exp,
                                                  epsilon=epsilon,
                                                  max_updates=refine_updates)))
    # jobs.append(
    #     ('Random Sinkhorn', delayed(mem.cache(onlikhorn))(x_sampler, y_sampler, ref=ref, max_size=10,
    #                                                             full_update=False, step_size=1., step_size_exp=0.,
    #                                                             max_updates=n * 10, batch_size=10, no_memory=True)))
    for full_update in [False]:
        for batch_size in [100, 1000]:
            jobs.append(
                (f'Online Sinhkorn b={batch_size}, full_update={full_update}',
                 delayed(mem.cache(online_sinkhorn))(x_sampler, y_sampler, max_size=(n, m),
                                                     epsilon=epsilon,
                                                     refine_updates=refine_updates,
                                                     ref=ref,
                                                     full_update=full_update, step_size=1,
                                                     step_size_exp=0 if full_update else 1/2,
                                                     batch_size=batch_size, batch_size_exp=0,
                                                     )))
    # for (step_size_exp, batch_size_exp) in ([1, 0.], [.5, .5], [0., 1.], [.5, 1], [.5, 1]):
    #     jobs.append(
    #         (f'Online Sinhkorn s=1/t^{step_size_exp} b=10 t^{2 * batch_size_exp}',
    #          delayed(mem.cache(onlikhorn))(x_sampler, y_sampler, max_size=n * 10,
    #                                              ref=ref, max_updates='auto',
    #                                              full_update=False, step_size=1.,
    #                                              step_size_exp=step_size_exp,
    #                                              batch_size=10, batch_size_exp=batch_size_exp,
    #                                              no_memory=False)))
    traces = Parallel(n_jobs=3)(job for (name, job) in jobs)
    dfs = []
    for ot, (name, job) in zip(traces, jobs):
        trace = ot.callback.trace
        df = pd.DataFrame(trace)
        df['name'] = name
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    output_dir = get_output_dir()
    dfs.to_pickle(join(output_dir, f'results_warmstart_big_{source}.pkl'))


def plot_results(source):
    output_dir = get_output_dir()
    df = pd.read_pickle(join(output_dir, f'results_warmstart_big_{source}.pkl'))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex='col', sharey='row')
    for name, sub_df in df.groupby(by='name'):
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
    fig.savefig(join(output_dir, 'results.pdf'))
    # fig, ax = plt.subplots(1, 1)
    # for name, sub_df in df.groupby(by='name'):
    #     ax.plot(sub_df['iter'], sub_df['computations'], label=name)
    # ax.set_yscale('log')
    plt.show()

run_OT('cloud_3d')
plot_results('cloud_3d')


