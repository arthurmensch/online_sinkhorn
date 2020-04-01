from os.path import expanduser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, Memory, delayed
from scipy.special import logsumexp

from online_sinkhorn.data import Subsampler


def compute_distance(x, y):
    x2 = np.sum(x ** 2, axis=1)
    y2 = np.sum(y ** 2, axis=1)
    return x2[:, None] + y2[None, :] - 2 * x @ y.T


# noinspection PyUnresolvedReferences
class OT:
    def __init__(self, max_size, dimension, averaging=False):
        self.distance = np.zeros((max_size, max_size))
        self.x = np.empty((max_size, dimension))
        self.p = np.full((max_size, 1), fill_value=-np.float('inf'))
        self.y = np.empty((max_size, dimension))
        self.q = np.full((1, max_size), fill_value=-np.float('inf'))

        self.averaging = False
        self.avg_p = np.full((max_size, 1), fill_value=-np.float('inf'))
        self.avg_q = np.full((1, max_size), fill_value=-np.float('inf'))

        self.max_size = max_size
        self.x_cursor = 0
        self.y_cursor = 0

        self.computations = 0

    def partial_fit(self, *, x=None, y=None, step_size=1., full=False, avg_step_size=None):
        if x is None and y is None:
            raise ValueError

        if x is not None:
            n = x.shape[0]
            new_x_cursor = min(self.x_cursor + n, self.max_size)
            n = new_x_cursor - self.x_cursor
            x = x[:n]
            if x.shape[0] > 0:
                self.x[self.x_cursor:new_x_cursor] = x
                self.distance[self.x_cursor:new_x_cursor, :self.y_cursor] = compute_distance(x, self.y[:self.y_cursor])
                self.computations += self.y_cursor * n
                if full:
                    distance = self.distance[:new_x_cursor, :self.y_cursor]
                else:
                    distance = self.distance[self.x_cursor:new_x_cursor, :self.y_cursor]
                if self.y_cursor == 0:
                    if full:
                        f = np.zeros((new_x_cursor, 1))
                    else:
                        f = np.zeros((n, 1))
                else:
                    f = - logsumexp(self.q[:, :self.y_cursor] - distance, axis=1,
                                    keepdims=True)
                    self.computations += distance.shape[0] * distance.shape[1]
            else:
                f = None
        else:
            f = None
            new_x_cursor = self.x_cursor
            n = None
        if y is not None:
            m = y.shape[0]
            new_y_cursor = min(self.y_cursor + m, self.max_size)
            m = new_y_cursor - self.y_cursor
            y = y[:m]
            if y.shape[0] > 0:
                self.y[self.y_cursor:new_y_cursor] = y
                self.distance[:self.x_cursor, self.y_cursor:new_y_cursor] = compute_distance(self.x[:self.x_cursor], y)
                self.computations += self.x_cursor * m
                if full:
                    distance = self.distance[:self.x_cursor, :new_y_cursor]
                else:
                    distance = self.distance[:self.x_cursor, self.y_cursor:new_y_cursor]
                if self.x_cursor == 0:
                    if full:
                        g = np.zeros((1, new_y_cursor))
                    else:
                        g = np.zeros((1, m))
                else:
                    g = - logsumexp(self.p[:self.x_cursor, :] - distance, axis=0,
                                    keepdims=True)
                    self.computations += distance.shape[0] * distance.shape[1]
            else:
                g = None
        else:
            g = None
            new_y_cursor = self.y_cursor
            m = None
        if x is not None and y is not None:
            self.distance[self.x_cursor:new_x_cursor, self.y_cursor:new_y_cursor] = compute_distance(x, y)
            self.computations += n * m
        if f is not None:
            if full:
                self.p[:new_x_cursor, :] = f - np.log(new_x_cursor)
            else:
                if step_size == 1.:
                    self.p[:self.x_cursor, :] = - float('inf')
                    self.p[self.x_cursor:new_x_cursor, :] = f - np.log(n)
                else:
                    self.p[:self.x_cursor, :] += np.log(1 - step_size)
                    self.p[self.x_cursor:new_x_cursor, :] = np.log(step_size) + f - np.log(n)
                if self.averaging:
                    if avg_step_size == 1.:
                        self.avg_p[:new_x_cursor] = self.p[:new_x_cursor]
                    else:
                        self.avg_p[:new_x_cursor] = np.logaddexp(self.avg_p[:new_x_cursor] + math.log(1 - avg_step_size),
                                                                 self.p[:new_x_cursor] + math.log(avg_step_size))
            self.x_cursor = new_x_cursor
        if g is not None:
            if full:
                self.q[:, :new_y_cursor] = g - np.log(new_y_cursor)
            else:
                if step_size == 1.:
                    self.q[:, :self.y_cursor] = - float('inf')
                    self.q[:, self.y_cursor:new_y_cursor] = g - np.log(m)
                else:
                    self.q[:, :self.y_cursor] += np.log(1 - step_size)
                    self.q[:, self.y_cursor:new_y_cursor] = np.log(step_size) + g - np.log(m)
                if self.averaging:
                    if avg_step_size == 1.:
                        self.avg_q[:, :new_y_cursor] = self.q[:, :new_y_cursor]
                    else:
                        self.avg_q[:, :new_y_cursor] = np.logaddexp(self.avg_q[:, :new_y_cursor] + math.log(1 - avg_step_size),
                                                                    self.q[:, :new_y_cursor] + math.log(avg_step_size))
            self.y_cursor = new_y_cursor

    def refit(self, *, refit_f=True, refit_g=True, step_size=1.):
        self.computations += self.x_cursor * self.y_cursor
        if self.x_cursor == 0 or self.y_cursor == 0:
            return
        if refit_f:
            # shape (:self.x_cursor, 1)
            f = - logsumexp(self.q[:, :self.y_cursor]
                            - self.distance[:self.x_cursor, :self.y_cursor],
                            axis=1, keepdims=True)
            self.computations += self.x_cursor * self.y_cursor
        else:
            f = None
        if refit_g:
            # shape (x_idx, 1)
            g = - logsumexp(self.p[:self.x_cursor, :]
                            - self.distance[:self.x_cursor, :self.y_cursor],
                            axis=0, keepdims=True)
            self.computations += self.x_cursor * self.y_cursor
        else:
            g = None

        if refit_f:
            if step_size == 1:
                self.p[:self.x_cursor, :] = f - np.log(self.x_cursor)
            else:
                self.p[:self.x_cursor, :] = np.logaddexp(f - np.log(self.x_cursor) + np.log(step_size),
                                                         self.p[:self.x_cursor] + np.log(1 - step_size))
        if refit_g:
            if step_size == 1:
                self.q[:, :self.y_cursor] = g - np.log(self.y_cursor)
            else:
                self.q[:self.x_cursor, :] = np.logaddexp(g - np.log(self.x_cursor) + np.log(step_size),
                                                         self.q[:self.x_cursor] + np.log(1 - step_size))

    def compute_ot(self):
        if self.averaging:
            q, p = self.avg_q, self.avg_p
        else:
            q, p = self.q, self.p
        distance = self.distance[:self.x_cursor, :self.y_cursor]
        f = - logsumexp(q[:, :self.y_cursor] - distance, axis=1, keepdims=True)
        g = - logsumexp(p[:self.x_cursor] - distance, axis=0, keepdims=True)
        ff = - logsumexp(g - distance, axis=1, keepdims=True) + np.log(self.y_cursor)
        gg = - logsumexp(f - distance, axis=0, keepdims=True) + np.log(self.x_cursor)
        return (f.mean() + ff.mean() + g.mean() + gg.mean()) / 2

    def evaluate_potential(self, *, x=None, y=None):
        if self.averaging:
            q, p = self.avg_q, self.avg_p
        else:
            q, p = self.q, self.p
        if x is not None:
            distance = compute_distance(x, self.y[:self.y_cursor])
            f = - logsumexp(q[:, :self.y_cursor] - distance, axis=1, keepdims=True)
        else:
            f = None
        if y is not None:
            distance = compute_distance(self.x[:self.x_cursor], y)
            g = - logsumexp(p[:self.x_cursor, :] - distance, axis=0, keepdims=True)
        else:
            g = None
        if f is not None and g is not None:
            return f, g
        elif f is not None:
            return f
        elif g is not None:
            return g
        else:
            raise ValueError

    @property
    def full(self):
        return self.x_cursor == self.max_size or self.y_cursor == self.max_size


def online_sinkhorn(x_sampler, y_sampler, A=1., B=10, a=1 / 2, b=1 / 2, full=False, max_size=1000, n_scale_iter=0,
                    max_iter=None, full_A=1., full_a=0.,
                    averaging=False, ref=None, name=None):
    ot = OT(max_size=max_size, dimension=x_sampler.dim, averaging=averaging)
    n_iter = 1.

    if ref is not None:
        trace = []

    def eval_callback():
        if ref is not None:
            f, g = ot.evaluate_potential(x=ref['x'], y=ref['y'])
            w = ot.compute_ot()
            print(w)
            var_err = var_norm(f - ref['f']) + var_norm(g - ref['g'])
            w_err = np.abs(w - ref['w'])
            trace.append(dict(computations=ot.computations, samples=ot.x_cursor + ot.y_cursor,
                              iter=n_iter,
                              var_err=var_err, w_err=w_err, n_iter=n_iter))

    # growing = (a > 0. or b > 0. or full)
    # if not growing:
    #     assert max_iter is not None
    # if max_iter is None:
    #     max_iter = float('inf')

    while not ot.full:
        if a != 0:
            step_size = A / np.float_power(n_iter, a)
        else:
            step_size = A
        avg_step_size = 1. / n_iter
        if b != 0:
            batch_size = np.ceil(B * np.float_power(n_iter, b * 2)).astype(int)
        else:
            batch_size = B

        x, loga = x_sampler(batch_size)
        y, logb = y_sampler(batch_size)
        ot.partial_fit(x=x, y=y, step_size=step_size, full=full, avg_step_size=avg_step_size)
        if n_iter % 10 == 0:
            ot.refit(refit_f=True, refit_g=True, step_size=1.)
        eval_callback()
        n_iter += 1

    for i in range(n_scale_iter):
        if full_a != 0:
            step_size = full_A / np.float_power(n_iter, full_a)
        else:
            step_size = full_A
        ot.refit(refit_f=True, refit_g=True, step_size=step_size)
        eval_callback()
        n_iter += 1


    if trace is not None:
        return ot, trace
    else:
        return ot


def sinkhorn(x, y, n_iter=100):
    ot = OT(max_size=10000, dimension=x.shape[1])
    ot.partial_fit(x=x, y=y, step_size=1.)  # Fill the cost matrix
    # print('sinkhorn', self.distance)
    for i in range(n_iter):
        ot.refit(refit_f=True, refit_g=True)
        w = ot.compute_ot()
    f, g = ot.evaluate_potential(x=x, y=y)
    return (f, g), w


def var_norm(x):
    return np.max(x) - np.min(x)


def run_OT():
    np.random.seed(0)

    n = 50
    n_ref_iter = 100

    x = np.random.randn(n, 5) / 10
    y = np.random.randn(n, 5) / 10 + 10

    mem = Memory(location=None)

    (f, g), w = mem.cache(sinkhorn)(x, y, n_ref_iter)
    ref = dict(f=f, g=g, x=x, y=y, w=w)

    x_sampler = Subsampler(x)
    y_sampler = Subsampler(y)
    n_scale_iter = 0
    max_size = n * 1000
    configs = [
               # dict(a=0.0, b=0., B=n, A=1., full=False, max_size=n * 100, n_scale_iter=0, name="Sinkhorn slowed"),
               # dict(a=0.0, b=0., B=n, A=1, full=False, max_size=n * 100, n_scale_iter=0, name="Sinkhorn slowed 0.9"),
               # dict(a=0.0, b=0., B=n, A=1, full=False, max_size=n, n_scale_iter=n_ref_iter, name="Sinkhorn"),
               # dict(a=0.2, b=0., B=n, A=1, full=False, max_size=n * 100, n_scale_iter=0, name="Sinkhorn 0.2"),
               # dict(a=0.8, b=0., B=n, A=1, full=False, max_size=n * 100, n_scale_iter=0, name="Sinkhorn 0.8"),
               dict(a=0.0, b=0., B=10, A=1, full=False, max_size=2 * n, max_iter=100, n_scale_iter=0, name="Randomized Sinkhorn"),
               dict(a=0.0, b=0., B=n, A=1, full=False, max_size=2 * n, max_iter=100, n_scale_iter=0, name="Full Randomized Sinkhorn"),
               # dict(a=0.5, b=0., B=10, A=1, full=False, max_size=10 * n, n_scale_iter=0, name="Online Sinkhorn"),
               # dict(a=0.0, b=0., B=10, A=1, full=True, max_size=10 * n, max_iter=100, n_scale_iter=0, name="Growing Sinkhorn"),
               # dict(a=0.0, b=0., B=n, A=1, full=True, max_size=n, max_iter=100, n_scale_iter=100, name="Sinkhorn"),
               # dict(a=0.0, b=0., B=n, A=1, full=False, max_size=n, n_scale_iter=100, full_a=0.0, full_A=1.,
               #      name="Sinkhorn 0.0 (fast)"),
               # dict(a=0.0, b=0., B=n, A=1, full=False, max_size=n, n_scale_iter=100, full_a=0.01, full_A=1.,
               #      name="Sinkhorn 0.01 (fast)"),
               ]
    # configs = [
    #            dict(a=0., b=0., B=10, A=1, full=False, max_size=10, n_scale_iter=n_ref_iter, name="Partial Sinkhorn"),
    #            # dict(a=0., b=0., B=10, A=1, full=False, max_size=max_size, n_scale_iter=n_scale_iter,
    #            #      name="Randomized Sinkhorn"),
    #            # dict(a=0., b=1., B=10, A=1, full=False, max_size=max_size, n_scale_iter=n_scale_iter,
    #            #      name="Randomized Sinkhorn + growing batch"),
    #            # dict(a=0., b=0., B=10, A=1, full=False, max_size=max_size, n_scale_iter=n_scale_iter,
    #            #      averaging=True,
    #            #      name="Randomized Sinkhorn + averaging"),
    #            dict(a=0., b=1., B=10, A=1, full=True, max_size=max_size, n_scale_iter=n_scale_iter,
    #                 name="Full-refit randomized Sinkhorn"),
    #            # dict(a=1., b=0.1, B=10, A=1, full=False, max_size=max_size, n_scale_iter=n_scale_iter,
    #            #      name="Online Sinkhorn 1/n"),
    #            # dict(a=1. / 2, b=1/2, B=10, A=1, full=False, max_size=max_size, n_scale_iter=n_scale_iter,
    #            #      name="Online Sinkhorn 1/sqrt(n)"),
    #            dict(a=1/2, b=1, B=10, A=1, full=False, max_size=max_size, n_scale_iter=n_scale_iter,
    #                 name="Online Sinkhorn 1/sqrt(n) + averaging", averaging=False),
    #            dict(a=1. / 2, b=1 / 2, B=10, A=1, full=True, max_size=max_size, n_scale_iter=n_scale_iter,
    #                 name="Online full-refit Sinkhorn")
    #                      ]

    traces = Parallel(n_jobs=5)(delayed(mem.cache(online_sinkhorn, ignore=['name']))
                                (x_sampler, y_sampler, ref=ref, **config)
                                for config in configs)
    dfs = []
    for (ot, trace), config in zip(traces, configs):
        df = pd.DataFrame(trace)
        for key, value in config.items():
            df[key] = value
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)
    dfs.to_pickle('results.pkl')


def plot_results():
    df = pd.read_pickle('results.pkl')
    fig, axes = plt.subplots(2, 3, sharex='col', sharey='row')
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
    # axes[1][0].set_ylim([1, 1e2])
    axes[1][1].legend()
    plt.show()


if __name__ == '__main__':
    run_OT()
    plot_results()
