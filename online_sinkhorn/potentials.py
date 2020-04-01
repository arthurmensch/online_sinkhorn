from os.path import expanduser
from queue import Full

import numpy as np
from joblib import Parallel, Memory, delayed
from scipy.special import logsumexp
from sklearn.model_selection import ParameterGrid

from online_sinkhorn.data import Sampler, Subsampler

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_distance(x, y):
    x2 = np.sum(x ** 2, axis=1)
    y2 = np.sum(y ** 2, axis=1)
    return x2[:, None] + y2[None, :] - 2 * x @ y.T


# noinspection PyUnresolvedReferences
class OT:
    def __init__(self, max_size, dimension):
        self.distance = np.zeros((max_size, max_size))
        self.x = np.empty((max_size, dimension))
        self.y = np.empty((max_size, dimension))
        self.f = np.full((max_size, 1), fill_value=-np.float('inf'))
        self.loga = np.full((max_size, 1), fill_value=-np.float('inf'))
        self.g = np.full((1, max_size), fill_value=-np.float('inf'))
        self.logb = np.full((1, max_size), fill_value=-np.float('inf'))

        self.max_size = max_size
        self.x_cursor = 0
        self.y_cursor = 0

    def enrich(self, *, x=None, y=None, step_size=1., full=False):
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
                    f = - logsumexp(self.g[:, :self.y_cursor] + self.logb[:, :self.y_cursor] - distance, axis=1,
                                    keepdims=True)
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
                    g = - logsumexp(self.f[:self.x_cursor, :] + self.loga[:self.x_cursor, :] - distance, axis=0,
                                    keepdims=True)
            else:
                g = None
        else:
            g = None
            new_y_cursor = self.y_cursor
            m = None
        if x is not None and y is not None:
            self.distance[self.x_cursor:new_x_cursor, self.y_cursor:new_y_cursor] = compute_distance(x, y)
        if f is not None:
            if full:
                self.f[:new_x_cursor, :] = f
                self.loga[:new_x_cursor, :] = - np.log(new_x_cursor)
            else:
                self.f[self.x_cursor:new_x_cursor, :] = f
                if step_size == 1.:
                    self.loga[:self.x_cursor, :] = - float('inf')
                    self.loga[self.x_cursor:new_x_cursor, :] = - np.log(n)
                else:
                    self.loga[:self.x_cursor, :] += np.log(1 - step_size)
                    self.loga[self.x_cursor:new_x_cursor, :] = np.log(step_size) - np.log(n)
            self.x_cursor = new_x_cursor
        if g is not None:
            if full:
                self.g[:, :new_y_cursor] = g
                self.logb[:, :new_y_cursor] = - np.log(new_y_cursor)
            else:
                self.g[:, self.y_cursor:new_y_cursor] = g
                if step_size == 1.:
                    self.logb[:, :self.y_cursor] = - float('inf')
                    self.logb[:, self.y_cursor:new_y_cursor] = - np.log(m)
                else:
                    self.logb[:, :self.y_cursor] += np.log(1 - step_size)
                    self.logb[:, self.y_cursor:new_y_cursor] = np.log(step_size) - np.log(m)
                    print(self.logb[:, :self.y_cursor])
            self.y_cursor = new_y_cursor

    def scale(self, *, scale_x=True, scale_y=True):
        if self.x_cursor == 0 or self.y_cursor == 0:
            return
        if scale_x:
            # shape (:self.x_cursor, 1)
            f = - logsumexp(self.g[:, :self.y_cursor] + self.logb[:, :self.y_cursor]
                            - self.distance[:self.x_cursor, :self.y_cursor],
                            axis=1, keepdims=True)
        else:
            f = None
        if scale_y:
            # shape (x_idx, 1)
            g = - logsumexp(self.f[:self.x_cursor, :] + self.loga[:self.x_cursor, :]
                            - self.distance[:self.x_cursor, :self.y_cursor],
                            axis=0, keepdims=True)
        else:
            g = None

        if scale_x:
            self.f[:self.x_cursor, :] = f
            self.loga[:self.x_cursor, :] = - np.log(self.x_cursor)
        if scale_y:
            self.g[:, :self.y_cursor] = g
            self.logb[:, :self.y_cursor] = - np.log(self.y_cursor)

    def compute_distance(self):
        distance = self.distance[:self.x_cursor, :self.y_cursor]
        f = - logsumexp(self.g[:, :self.y_cursor] + self.logb[:, :self.y_cursor] - distance, axis=1, keepdims=True)
        g = - logsumexp(self.f[:self.x_cursor] + self.loga[:self.x_cursor, :] - distance, axis=0, keepdims=True)
        ff = - logsumexp(g - distance, axis=1, keepdims=True) + np.log(self.y_cursor)
        gg = - logsumexp(f - distance, axis=0, keepdims=True) + np.log(self.x_cursor)
        return (f.mean() + ff.mean() + g.mean() + gg.mean()) / 2
        # return (((self.f[:self.x_cursor] + f) * np.exp(self.loga[:self.x_cursor])).sum()
        #         + ((self.g[:, :self.y_cursor] + g) * np.exp(self.logb[:, :self.y_cursor])).sum()) / 2

    def online_sinkhorn(self, x_sampler, y_sampler, A=1., B=10, a=1/2, b=1/2, full=False):
        iter = 1
        while self.x_cursor < self.max_size or self.y_cursor < self.max_size:
            if a != 0:
                step_size = A * np.float_power(iter, - a)
            else:
                step_size = A
            if b != 0:
                batch_size = np.ceil(B * np.float_power(iter, b * 2)).astype(int)
            else:
                batch_size = B
            x, loga = x_sampler(batch_size)
            y, logb = y_sampler(batch_size)
            self.enrich(x=x, y=y, step_size=step_size, full=full)
            w = self.compute_distance()
            print('w', w)
            # if iter % 10 == 0:
            #     w = self.compute_distance()
            #     self.scale(scale_x=True, scale_y=True)
            #     wnew = self.compute_distance()
            #     print(f'rescale {w}->{wnew}')
            iter += 1
        w = self.compute_distance()
        for i in range(10):
            self.scale(scale_x=True, scale_y=True)
        wnew = self.compute_distance()
        print(f'last rescale {w}->{wnew}')

    def sinkhorn(self, x, y, n_iter=100):
        self.enrich(x=x, y=y, step_size=1.)
        # print('sinkhorn', self.distance)
        for i in range(n_iter):
            self.scale(scale_x=True, scale_y=True)
            w = self.compute_distance()
            print('w', w)




class Potential:
    def __init__(self, max_size, dimension):
        self.coefs = np.full(max_size, fill_value=-np.float('inf'))
        self.avg_coefs = np.full(max_size, fill_value=-np.float('inf'))
        self.positions = np.zeros((max_size, dimension))
        self.cursor = 0
        self.max_size = max_size
        self.computations = 0
        self.iter = 0

    def evaluate(self, new_position, average=False):
        if self.cursor == 0:
            return np.zeros(len(new_position))
        distance = compute_distance(self.positions[:self.cursor], new_position)
        self.computations += self.cursor * len(new_position)
        if average:
            return - logsumexp(self.avg_coefs[:self.cursor][:, None] - distance, axis=0)
        else:
            return - logsumexp(self.coefs[:self.cursor][:, None] - distance, axis=0)

    def enrich(self, new_positions, new_weights, new_potentials, step_size=1.):
        self.iter += 1
        assert 0 < step_size <= 1.
        if step_size < 1:
            new_cursor = self.cursor + len(new_positions)
            if new_cursor > self.max_size:
                raise Full

            self.coefs[self.cursor:new_cursor] = new_weights + new_potentials + np.log(step_size)
            self.positions[self.cursor:new_cursor] = new_positions
            self.coefs[:self.cursor] += np.log(1 - step_size)
        else:
            new_cursor = len(new_positions)
            if new_cursor > self.max_size:
                raise Full
            self.coefs[:new_cursor] = new_weights[:new_cursor] + new_potentials[:new_cursor]
            self.positions[:new_cursor] = new_positions[:new_cursor]
        self.avg_coefs[:self.cursor] = np.logaddexp(self.avg_coefs[:self.cursor] + np.log(1 - self.iter),
                                                    self.coefs[:self.cursor] + np.log(self.iter))
        self.cursor = new_cursor


def sinkhorn(x, loga, y, logb, n_iter=100, simultaneous=False, fref=None, gref=None, xref=None, yref=None):
    d = x.shape[1]
    n = x.shape[0]
    m = y.shape[0]
    f = Potential(m, d)
    g = Potential(n, d)
    vs = []
    computations = []
    mems = []
    for i in range(n_iter):
        if simultaneous:
            ff, gg = f.evaluate(x), g.evaluate(y)
            f.enrich(y, logb, gg)
            g.enrich(x, loga, ff)
        else:
            gg = g.evaluate(y)
            f.enrich(y, logb, gg)
            ff = f.evaluate(x)
            g.enrich(x, loga, ff)
        if fref is not None:
            feval = f.evaluate(xref)
            geval = g.evaluate(yref)
            v = var_norm(fref - feval) + var_norm(gref - geval)
            vs.append(v)
            computations.append(f.computations + g.computations)
            mems.append(f.cursor + g.cursor)
    return f, g, vs, mems, computations


def online_sinkhorn(x_sampler, y_sampler, A=1., B=10, a=0., b=1., max_size=1000, fref=None, gref=None, xref=None,
                    yref=None):
    d = x_sampler.dim
    f = Potential(max_size, d)
    g = Potential(max_size, d)
    iter = 1
    vs = []
    computations = []
    mems = []
    while f.cursor < max_size or g.cursor < max_size:
        if a != 0:
            step_size = A * np.float_power(iter, - a)
        else:
            step_size = A
        if b != 0:
            batch_size = np.floor(B * np.float_power(iter, b * 2)).astype(int)
        else:
            batch_size = B
        x, loga = x_sampler(batch_size)
        y, logb = y_sampler(batch_size)
        ff, gg = f.evaluate(x), g.evaluate(y)
        if (g.computations + g.computations) > max_size ** 2:
            break
        try:
            f.enrich(y, logb, gg, step_size=step_size)
            g.enrich(x, loga, ff, step_size=step_size)
        except Full:
            break
        iter += 1

        if fref is not None:
            feval = f.evaluate(xref)
            geval = g.evaluate(yref)
            v = var_norm(fref - feval) + var_norm(gref - geval)
            vs.append(v)
            computations.append(f.computations + g.computations)
            mems.append(f.cursor + g.cursor)
    return f, g, vs, mems, computations


def var_norm(x):
    return np.max(x) - np.min(x)

def run_OT():
    np.random.seed(0)

    n = 100

    yref = np.random.randn(n, 1)
    xref = np.random.randn(n, 1) + 10

    x_sampler = Subsampler(xref)
    y_sampler = Subsampler(yref)

    ot = OT(max_size=100, dimension=1)
    ot.online_sinkhorn(x_sampler, y_sampler, B=10, full=False)
    ot.sinkhorn(xref, yref, 100)

def main():
    np.random.seed(0)

    n = 1000

    x_sampler = Sampler(mean=np.array([[1.], [2], [3]]), cov=np.array([[[.1]], [[.1]], [[.1]]]),
                        p=np.ones(3) / 3)
    y_sampler = Sampler(mean=np.array([[0.], [3], [5]]), cov=np.array([[[.1]], [[.1]], [[.4]]]),
                        p=np.ones(3) / 3)
    xref, loga = x_sampler(n)
    yref, logb = y_sampler(n)

    yref = np.random.randn(n, 5)
    xref = np.random.randn(n, 5) + 2
    loga = np.full((n, ), fill_value=-np.log(n))
    logb = np.full((n, ), fill_value=-np.log(n))

    f, g, _, _, _ = sinkhorn(xref, loga, yref, logb)
    fref = f.evaluate(xref)
    gref = g.evaluate(yref)

    x_sampler = Subsampler(xref)
    y_sampler = Subsampler(yref)

    mem = Memory(location=expanduser('~/cache'))

    params = list(iter(ParameterGrid(dict(a=np.linspace(0, 1, 5), b=np.linspace(0, 1, 5)))))
    res = Parallel(n_jobs=5)(
        delayed(mem.cache(online_sinkhorn))(x_sampler, y_sampler, max_size=n * 10, B=10, A=.5,
                                            a=p['a'], b=p['b'], fref=fref, gref=gref,
                                            xref=xref, yref=yref) for p in params)

    df = []
    for p, (_, _, vs, mems, computations) in zip(params, res):
        for i, (v, mem, computation) in enumerate(zip(vs, mems, computations)):
            df.append(dict(mem=mem, v=v, computation=computation, a=p["a"], b=p["b"], iter=i))
    df = pd.DataFrame(df)
    df.to_pickle('results.pkl')


def plot():
    df = pd.read_pickle('results.pkl')
    print(df)
    facet = sns.FacetGrid(data=df, hue="a", col="b")
    facet.map(plt.plot, 'mem', 'v')
    facet.add_legend()
    for ax in facet.axes.ravel():
        ax.set_yscale('log')
        ax.set_xscale('log')

    facet = sns.FacetGrid(data=df, hue="a", col="b")
    facet.map(plt.plot, 'computation', 'v')
    facet.add_legend()
    for ax in facet.axes.ravel():
        ax.set_yscale('log')
        ax.set_xscale('log')
    plt.show()


if __name__ == '__main__':
    # main()
    # plot()
    run_OT()