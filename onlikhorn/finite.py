from typing import Tuple

import numpy as np
from scipy.special import logsumexp

from onlikhorn.data import Subsampler
from onlikhorn.solver import compute_distance, Callback


class FiniteOT:
    def __init__(self, size: Tuple[int, int], dimension,
                 callback=None, step_size=1., step_size_exp=0., max_updates=100,
                 batch_size=1, batch_size_exp=0., avg_step_size=1., avg_step_size_exp=0., epsilon=1,
                 simultaneous=False, compute_distance=True,
                 full_update=False, averaging=False):
        self.size = size
        self.callback = callback
        self.dimension = dimension
        self.step_size = step_size
        self.step_size_exp = step_size_exp
        self.batch_size = batch_size
        self.batch_size_exp = batch_size_exp
        self.avg_step_size = avg_step_size
        self.avg_step_size_exp = avg_step_size_exp
        self.full_update = full_update

        self.compute_distance = compute_distance

        self.epsilon = epsilon

        self.averaging = averaging

        self.simultaneous = simultaneous

        self.max_updates = max_updates

        # Attributes
        self.x_ = np.empty((size[0], dimension))
        self.p_ = np.full((size[0], 1), fill_value=-np.float('inf'))
        self.y_ = np.empty((size[1], dimension))
        self.q_ = np.full((1, size[1]), fill_value=-np.float('inf'))

        if self.compute_distance:
            self.distance_ = np.empty(size)

        if self.averaging:
            self.avg_p_ = np.full((size[0], 1), fill_value=-np.float('inf'))
            self.avg_q_ = np.full((1, size[1]), fill_value=-np.float('inf'))

        self.computations_ = 0
        self.n_updates_ = 0

        self.x_seen_indices_ = set()
        self.y_seen_indices_ = set()

        self.x_sampler = Subsampler(size=size[0], return_idx=True)
        self.y_sampler = Subsampler(size=size[1], return_idx=True)

    def scaled_logsumexp(self, x, axis=None, keepdims=False):
        return self.epsilon * logsumexp(x / self.epsilon, axis=axis, keepdims=keepdims)

    @property
    def n_samples_(self):
        return len(self.y_seen_indices_) + len(self.x_seen_indices_)

    def partial_fit(self, *, x_indices=None, y_indices=None, step_size=1., full_update=False, avg_step_size=1.):
        if x_indices is not None:
            if full_update:
                x_indices = list (self.x_seen_indices_.union(set(x_indices.tolist())))
            y_seen_indices = list(self.y_seen_indices_)

            if len(y_seen_indices) == 0:
                f = np.zeros((len(x_indices), 1))
            else:
                if self.compute_distance:
                    distance = self.distance_[x_indices][:, y_seen_indices]
                else:
                    distance = compute_distance(self.x_[x_indices], self.y_[y_seen_indices])
                    self.computations_ += len(x_indices) * len(y_seen_indices) * self.x_.shape[1]

                f = - self.scaled_logsumexp(self.q_[:, y_seen_indices] - distance, axis=1,
                                          keepdims=True)  # Broadcast
                self.computations_ += distance.shape[0] * distance.shape[1]
        else:
            f = None

        if y_indices is not None:
            if full_update:
                y_indices = list(self.y_seen_indices_.union(set(y_indices.tolist())))
            x_seen_indices = list(self.x_seen_indices_)

            if len(y_seen_indices) == 0:
                g = np.zeros((1, len(y_indices)))
            else:
                if self.compute_distance:
                    distance = self.distance_[x_seen_indices][:, y_indices]

                else:
                    distance = compute_distance(self.x_[x_seen_indices], self.y_[y_indices])
                    self.computations_ += len(y_indices) * len(x_seen_indices) * self.x_.shape[1]

                g = - self.scaled_logsumexp(self.p_[x_seen_indices, :] - distance, axis=0,
                                          keepdims=True)  # Broadcast
                self.computations_ += distance.shape[0] * distance.shape[1]
        else:
            g = None

        if f is not None:
            if full_update:
                self.x_seen_indices_ = set(x_indices)
                x_seen_indices = x_indices
                self.p_[x_seen_indices, :] = f - np.log(len(x_seen_indices))
            else:
                self.x_seen_indices_.update(set(x_indices))
                x_seen_indices = list(self.x_seen_indices_)
                batch_size = len(x_indices)
                if step_size < 1:
                    self.p_[x_seen_indices, :] += np.log(1 - step_size)
                    self.p_[x_indices, :] = np.logaddexp(self.p_[x_indices],
                                                         np.log(step_size) - np.log(batch_size) + f,
                                                         )
                else:
                    self.p_[x_seen_indices, :] = -float('inf')
                    self.p_[x_indices, :] = f - np.log(batch_size)
            if self.averaging:
                self.avg_p_[x_seen_indices] += np.log(1 - avg_step_size)
                self.avg_p_[x_indices] = np.logaddexp(self.avg_p_[x_indices],
                                                      self.p_[x_indices] + np.log(avg_step_size),
                                                      )
            self.n_updates_ += 1
        if g is not None:
            if full_update:
                self.y_seen_indices_ = set(y_indices)
                y_seen_indices = y_indices
                self.q_[:, y_seen_indices] = g - np.log(len(y_seen_indices))
            else:
                self.y_seen_indices_.update(set(y_indices))
                y_seen_indices = list(self.y_seen_indices_)
                batch_size = len(y_indices)
                if step_size < 1:
                    self.q_[:, y_seen_indices] += np.log(1 - step_size)
                    self.q_[:, y_indices] = np.logaddexp(self.q_[:, y_indices],
                                                         np.log(step_size) - np.log(batch_size) + g,
                                                         )
                else:
                    self.q_[:, y_seen_indices] = -float('inf')
                    self.q_[:, y_indices] = g - np.log(batch_size)

            if self.averaging:
                self.avg_q_[:, y_seen_indices] += np.log(1 - avg_step_size)
                self.avg_q_[:, y_indices] = np.logaddexp(self.avg_q_[:, y_indices],
                                                         self.q_[:, y_indices] + np.log(avg_step_size),
                                                         )
            self.n_updates_ += 1

    def _callback(self):
        if self.callback is not None:
            self.callback(self)

    def online_sinkhorn_loop(self, x, y):
        self.x_ = x
        self.y_ = y
        if self.compute_distance:
            self.distance_ = compute_distance(x, y)
            self.computations_ += len(x) * len(y) * y.shape[1]
        while self.n_updates_ < self.max_updates:
            if self.step_size_exp != 0:
                step_size = self.step_size / np.float_power((self.n_updates_ + 1), self.step_size_exp)
            else:
                step_size = self.step_size
            if self.avg_step_size != 0:
                avg_step_size = self.avg_step_size / np.float_power((self.n_updates_ + 1), self.avg_step_size_exp)
            else:
                avg_step_size = self.avg_step_size
            if self.batch_size_exp != 0:
                batch_size = np.ceil(
                    self.batch_size * np.float_power((self.n_updates_ + 1), self.batch_size_exp * 2)).astype(
                    int)
                batch_size_x = min(len(self.x_), batch_size)
                batch_size_y = min(len(self.y_), batch_size)
            else:
                batch_size_x = batch_size_y = self.batch_size
            if batch_size_x == len(self.x_) and batch_size_y == len(self.y_):
                step_size = 1

            x_indices, _ = self.x_sampler(batch_size_x)
            y_indices, _ = self.y_sampler(batch_size_y)
            if self.simultaneous:
                self.partial_fit(x_indices=x_indices,
                                 step_size=step_size, full_update=self.full_update, avg_step_size=avg_step_size)
                self.partial_fit(y_indices=y_indices,
                                 step_size=step_size, full_update=self.full_update, avg_step_size=avg_step_size)
            else:
                self.partial_fit(x_indices=x_indices, y_indices=y_indices,
                                 step_size=step_size, full_update=self.full_update, avg_step_size=avg_step_size)
            self._callback()

    def compute_ot(self):
        x_seen_indices = list(self.x_seen_indices_)
        y_seen_indices = list(self.y_seen_indices_)
        if len(x_seen_indices) == 0 or len(y_seen_indices) == 0:
            return 0
        if self.averaging:
            q, p = self.avg_q_, self.avg_p_
        else:
            q, p = self.q_, self.p_
        if self.compute_distance:
            distance = self.distance_[x_seen_indices][:, y_seen_indices]
        else:
            distance = compute_distance(self.x_[x_seen_indices], self.y_[y_seen_indices])

        f = - self.scaled_logsumexp(q[:, y_seen_indices] - distance, axis=1, keepdims=True)
        g = - self.scaled_logsumexp(p[x_seen_indices] - distance, axis=0, keepdims=True)
        ff = - self.scaled_logsumexp(g - distance, axis=1, keepdims=True) + np.log(len(y_seen_indices))
        gg = - self.scaled_logsumexp(f - distance, axis=0, keepdims=True) + np.log(len(x_seen_indices))
        return (f.mean() + ff.mean() + g.mean() + gg.mean()) / 2

    def evaluate_potential(self, *, x=None, y=None):
        x_seen_indices = list(self.x_seen_indices_)
        y_seen_indices = list(self.y_seen_indices_)
        if self.averaging:
            q, p = self.avg_q_, self.avg_p_
        else:
            q, p = self.q_, self.p_
        if x is not None:
            if len(x_seen_indices) == 0:
                f = np.zeros((x.shape[0], 1))
            else:
                distance = compute_distance(x, self.y_[y_seen_indices])
                f = - self.scaled_logsumexp(q[:, y_seen_indices] - distance, axis=1, keepdims=True)
        else:
            f = None
        if y is not None:
            if len(y_seen_indices) == 0:
                g = np.zeros((1, y.shape[0]))
            else:
                distance = compute_distance(self.x_[x_seen_indices], y)
                g = - self.scaled_logsumexp(p[x_seen_indices, :] - distance, axis=0, keepdims=True)
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


def online_sinkhorn_finite(x, y, ref=None, step_size=1., step_size_exp=0., epsilon=1,
                           batch_size=1, batch_size_exp=0., avg_step_size=1., avg_step_size_exp=0., averaging=False,
                           max_updates=100, full_update=False):
    callback = Callback(ref)
    ot = FiniteOT(dimension=x.shape[1], max_updates=max_updates, callback=callback,
                  epsilon=epsilon, size=(x.shape[0], y.shape[0]),
                  step_size=step_size, step_size_exp=step_size_exp,
                  batch_size=batch_size, batch_size_exp=batch_size_exp, avg_step_size=avg_step_size,
                  avg_step_size_exp=avg_step_size_exp, averaging=averaging,
                  full_update=full_update,
                  )
    ot.online_sinkhorn_loop(x, y)
    return ot
