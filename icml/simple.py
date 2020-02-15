import math
from collections import defaultdict

import torch
from matplotlib import ticker
from sklearn.utils import check_random_state, gen_batches
from joblib import Memory
import numpy as np

import matplotlib.pyplot as plt


def compute_distance(x, y):
    return (torch.sum((x[:, None, :] ** 2 + y[None, :, :] ** 2 - 2 * x[:, None, :] * y[None, :, :]), dim=2)) / 2


def sinkhorn(x, y, eps, n_iter=1000):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    n = x.shape[0]
    m = y.shape[0]

    loga = - math.log(n)
    logb = - math.log(m)

    distance = compute_distance(x, y)

    g = torch.zeros((m,), dtype=torch.float64)
    f = evaluate_potential_finite(g + eps * logb, slice(None), distance, eps)
    errors = defaultdict(list)
    for i in range(n_iter):
        g = evaluate_potential_finite(f + eps * loga, slice(None), distance.transpose(0, 1), eps)
        f = evaluate_potential_finite(g + eps * logb, slice(None), distance, eps)
        w = (f.mean() + g.mean())
        errors['ff'].append(f.numpy())
        errors['gg'].append(g.numpy())
        errors['w'].append(w.item())
        errors['iter'].append(i)
    return (y, g + eps * logb), (x, f + eps * loga), errors


def sample_from_finite(x, m, random_state=None, full_after_one=False, replacement=False):
    random_state = check_random_state(random_state)
    n = x.shape[0]
    first_iter = True

    while True:
        if not first_iter and full_after_one:
            yield np.arange(x.shape[0]), torch.full((n,), fill_value=-math.log(n))
        else:
            if replacement:
                indices = random_state.permutation(n)[:m]
                loga = torch.full((m,), fill_value=-math.log(m))
                yield indices, loga
            indices = random_state.permutation(n)
            for batches in gen_batches(x.shape[0], m):
                these_indices = indices[batches]
                this_m = len(these_indices)
                loga = torch.full((this_m,), fill_value=-math.log(this_m))
                yield these_indices, loga
            first_iter = False


def evaluate_potential_finite(log_pot: torch.tensor, idx: torch.tensor, distance, eps):
    return - eps * torch.logsumexp((- distance[idx] + log_pot[None, :]) / eps, dim=1)


def online_sinkhorn(x, y, eps, samplingx, samplingy, fref, gref, n_iter=100, random_state=None, resample=True, ieta=1.,
                    eta=1., alternate=False, replacement=True):
    random_state = check_random_state(random_state)
    torch.manual_seed(random_state.randint(100000))

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    fref = torch.from_numpy(fref)
    gref = torch.from_numpy(gref)

    n = x.shape[0]
    m = y.shape[0]
    p = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)
    q = torch.full((m,), fill_value=-float('inf'), dtype=x.dtype)

    avg_p = p.clone()
    avg_q = q.clone()

    distance = compute_distance(x, y)
    errors = defaultdict(list)
    u_eta = not isinstance(eta, float)
    u_ieta = not isinstance(ieta, float)

    y_sampler = sample_from_finite(y, samplingy, random_state=random_state,
                                   replacement=replacement)
    x_sampler = sample_from_finite(x, samplingx, random_state=random_state,
                                   replacement=replacement)
    y_idx, logb = next(y_sampler)
    x_idx, loga = next(x_sampler)

    computations = 1
    true_computations = 1
    for i in range(0, n_iter):
        if i % 1 == 0:
            if i == 0:
                ff = torch.zeros(n, dtype=x.dtype)
                gg = torch.zeros(m, dtype=x.dtype)
            else:
                ff = evaluate_potential_finite(avg_q, slice(None), distance, eps)
                gg = evaluate_potential_finite(avg_p, slice(None), distance.transpose(0, 1), eps)
            fff = evaluate_potential_finite(gg + math.log(1 / m) * eps, slice(None), distance, eps)
            ggg = evaluate_potential_finite(ff + math.log(1 / n) * eps, slice(None), distance.transpose(0, 1),
                                            eps)
            ff = (ff + fff) / 2
            gg = (gg + ggg) / 2
            lya = (-ff.mean() - gg.mean() + fref.mean() + gref.mean()
                   + eps * torch.logsumexp(((ff[:, None] + math.log(1 / m) * eps +
                                             math.log(1 / n) * eps + gref[None, :] - distance) / eps).view(-1), dim=0)
                   + eps * torch.logsumexp(((fref[:, None] + math.log(1 / m) * eps +
                                             math.log(1 / n) * eps + gg[None, :] - distance) / eps).view(-1), dim=0))
            w = (ff.mean() + gg.mean())
            errors['ff'].append(ff.numpy())
            errors['gg'].append(gg.numpy())
            errors['lya'].append(lya.numpy())
            errors['w'].append(w.item())
            errors['computation'].append(computations)
            errors['true_computation'].append(true_computations)
            errors['iter'].append(i)

        # Update f
        if resample:
            y_idx, logb = next(y_sampler)
            x_idx, loga = next(x_sampler)
        if i > 0:
            g = evaluate_potential_finite(p, y_idx, distance.transpose(0, 1), eps)
            if not alternate:
                f = evaluate_potential_finite(q, x_idx, distance, eps)
        else:
            g = torch.zeros(samplingy, dtype=x.dtype)
            if not alternate:
                f = torch.zeros(samplingx, dtype=x.dtype)

        if u_ieta:
            if ieta == '1/t':
                ieta_ = torch.tensor(1 / (i + 1))
            elif ieta == '1/sqrt(t)':
                ieta_ = torch.tensor(1 / math.sqrt(i + 1))
            else:
                raise ValueError
        else:
            ieta_ = torch.tensor(ieta)
        if u_eta:
            if eta == '1/t':
                eta_ = torch.tensor(1 / (i + 1))
            elif eta == '1/sqrt(t)':
                eta_ = torch.tensor(1 / math.sqrt(i + 1))
            else:
                raise ValueError
        else:
            eta_ = torch.tensor(eta)

        q += eps * torch.log(- ieta_ + 1)
        update = eps * torch.log(ieta_) + logb * eps + g
        q[y_idx] = eps * torch.logsumexp(torch.cat([q[y_idx][None, :], update[None, :]], dim=0) / eps, dim=0)

        if alternate:
            f = evaluate_potential_finite(q, x_idx, distance, eps)
        p += eps * torch.log(- ieta_ + 1)
        update = eps * torch.log(ieta_) + loga * eps + f
        p[x_idx] = eps * torch.logsumexp(torch.cat([p[x_idx][None, :], update[None, :]], dim=0) / eps, dim=0)

        avg_q += eps * torch.log(- eta_ + 1)
        update = eps * torch.log(eta_) + q[y_idx]
        avg_q[y_idx] = eps * torch.logsumexp(torch.cat([update[None, :], avg_q[y_idx][None, :]]) / eps, dim=0)

        avg_p += eps * torch.log(- eta_ + 1)
        update = eps * torch.log(eta_) + p[x_idx]
        avg_p[x_idx] = eps * torch.logsumexp(torch.cat([update[None, :], avg_p[x_idx][None, :]]) / eps, dim=0)

        computations += len(y_idx) * min(len(y_idx) * (i + 1), n)
        true_computations += len(y_idx) * len(y_idx) * (i + 1)
    return (y, avg_q), (x, avg_p), errors


def compute_value(f, g, x, y, eps):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    f = torch.from_numpy(f)
    g = torch.from_numpy(g)
    C = compute_distance(x, y)
    return (eps * torch.mean(torch.exp((f[:, :, None] + g[:, None, :] - C[None, :, :]) / eps) - 1,
                             dim=(1, 2)) - torch.mean(f, dim=1) - torch.mean(g, dim=1)).numpy()


def simple():
    eps = 10

    x = np.array([[0., 1.], [2., 0.]])
    y = np.array([[0., 0.]])

    mem = Memory(location=None)
    _, _, errors = sinkhorn(x, y, eps=eps, n_iter=100)
    fref = errors['ff'][-1]
    gref = errors['gg'][-1]
    _, _, errors = mem.cache(online_sinkhorn)(x, y, eps=eps, fref=fref, gref=gref, samplingx=1, samplingy=1, eta=1.,
                                              ieta='1/sqrt(t)', n_iter=1000,
                                              alternate=True)
    ff = np.concatenate([f[None, :] for f in errors['ff']])
    gg = np.concatenate([g[None, :] for g in errors['gg']])
    traj = ff + gg
    print(errors['lya'])
    X = np.linspace(traj[:, 0].min() - 1, traj[:, 0].max() + 1, 200)
    Y = np.linspace(traj[:, 1].min() - 1, traj[:, 1].max() + 1, 200)
    Z = np.meshgrid(X, Y)
    f = np.concatenate([z[:, :, None] for z in Z], axis=2).reshape((-1, 2))
    g = np.zeros((f.shape[0], 1))
    values = compute_value(f, g, x, y, eps).reshape((200, 200))
    values = np.minimum(values, 1.5)

    fig, ax = plt.subplots(1, 1)
    # noinspection PyTypeChecker
    cs = ax.contourf(X, Y, values,
                     levels=50,
                     # locator=ticker.LogLocator(subs=tuple(np.linspace(0.1, 1, 10, endpoint=False).tolist()))
                     )
    ax.plot(traj[:, 0], traj[:, 1], color='red', marker='.')
    fig.colorbar(cs)
    plt.show()

class Sampler():
    def __init__(self, mean: torch.tensor, cov: torch.tensor, p: torch.tensor):
        k, d = mean.shape
        k, d, d = cov.shape
        k = p.shape
        self.mean = mean
        self.cov = cov
        self.icov = torch.cat([torch.inverse(cov)[None, :, :] for cov in self.cov], dim=0)
        det = torch.tensor([torch.det(cov) for cov in self.cov])
        self.norm = torch.sqrt((2 * math.pi) ** d * det)
        self.p = p

    def _call(self, n):
        k, d = self.mean.shape
        indices = np.random.choice(k, n, p=self.p.numpy())
        pos = np.zeros((n, d), dtype=np.float32)
        for i in range(k):
            mask = indices == i
            size = mask.sum()
            pos[mask] = np.random.multivariate_normal(self.mean[i], self.cov[i], size=size)
        logweight = np.full_like(pos[:, 0], fill_value=-math.log(n))
        return torch.from_numpy(pos), torch.from_numpy(logweight)

    def __call__(self, n, fake=False):
        if fake:
            if not hasattr(self, 'pos_'):
                self.pos_, self.logweight_ = self._call(n)
            return self.pos_, self.logweight_
        else:
            return self._call(n)

    def log_prob(self, x):
        # b, d = x.shape
        diff = x[:, None, :] - self.mean[None, :]  # b, k, d
        return torch.sum(self.p[None, :] * torch.exp(-torch.einsum('bkd,kde,bke->bk',
                                                                   [diff, self.icov, diff]) / 2) / self.norm, dim=1)


def simple2():
    eps = 1e-1

    x = np.random.randn(20, 2)
    y = np.random.randn(20, 2) + 10

    x_sampler = Sampler(mean=torch.tensor([[1.], [2], [3]]), cov=torch.tensor([[[.1]], [[.1]], [[.1]]]),
                        p=torch.ones(3) / 3)
    y_sampler = Sampler(mean=torch.tensor([[0.], [3], [5]]), cov=torch.tensor([[[.1]], [[.1]], [[.4]]]),
                        p=torch.ones(3) / 3)

    x, loga = x_sampler(50)
    y, logb = y_sampler(100)
    x = x.numpy()
    y = y.numpy()

    mem = Memory(location=None)
    _, _, errors = sinkhorn(x, y, eps=eps, n_iter=100)
    fref = errors['ff'][-1]
    gref = errors['gg'][-1]
    _, _, errors = mem.cache(online_sinkhorn)(x, y, eps=eps, fref=fref, gref=gref, samplingx=50, samplingy=99,
                                              eta=1.,
                                              ieta=1., n_iter=int(1e3),
                                              resample=True,
                                              alternate=False)
    ff = np.concatenate([f[None, :] for f in errors['ff']])
    gg = np.concatenate([g[None, :] for g in errors['gg']])

    diff = ff - fref[None, :]
    var = diff.max(axis=1) - diff.min(axis=1)

    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(errors['lya'])), errors['lya'])
    ax.set_yscale('log')
    ax.set_title('lya')
    plt.show()
    fig, ax = plt.subplots(1, 1)
    ax.plot(range(len(var)), var)
    ax.set_yscale('log')
    ax.set_title('var')
    plt.show()



if __name__ == '__main__':
    simple2()
