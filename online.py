import math

import numpy as np
import torch

import matplotlib.pyplot as plt


def sample_from(x, m):
    n = x.shape[0]
    indices = torch.from_numpy(np.random.permutation(n)[:m])
    loga = torch.full((m,), fill_value=-math.log(m))
    return x[indices], loga


def sample_from_finite(x, m, ):
    n = x.shape[0]
    indices = torch.from_numpy(np.random.permutation(n)[:m])
    loga = torch.full((m,), fill_value=-math.log(m))
    return indices, loga


def compute_distance(x, y):
    return (torch.sum((x[:, None, :] ** 2 + y[None, :, :] ** 2 - 2 * x[:, None, :] * y[None, :, :]), dim=2)) / 2


def evaluate_potential(log_pot: torch.tensor, pos: torch.tensor, x: torch.tensor, eps):
    distance = compute_distance(x, pos)
    return - eps * torch.logsumexp((- distance + log_pot[None, :]) / eps, dim=1)


def stochasic_sinkhorn(x, y, eps, m, n_iter=100, step_size='sqrt'):
    hatf = torch.full((m * n_iter,), fill_value=-float('inf'))
    ahatf = torch.full((m * n_iter,), fill_value=-float('inf'))
    posy = torch.zeros(m * n_iter, 2)
    hatg = torch.full((m * n_iter,), fill_value=-float('inf'))
    ahatg = torch.full((m * n_iter,), fill_value=-float('inf'))
    posx = torch.zeros(m * n_iter, 2)
    sum_eta = 0
    for i in range(0, n_iter):
        if step_size == 'sqrt':
            eta = torch.tensor(1. / math.sqrt(i + 1))
        elif step_size == 'constant':
            eta = torch.tensor(0.01)

        # Update f
        y_, logb = sample_from(y, m)
        if i > 0:
            g = evaluate_potential(hatf[:i * m], posx[:i * m], y_, eps)
        else:
            g = torch.zeros(m)
        hatg[:i * m] += eps * torch.log(1 - eta)
        hatg[i * m:(i + 1) * m] = eps * math.log(eta) + logb * eps + g
        posy[i * m:(i + 1) * m] = y_

        update = eps * math.log(eta) + hatg[:(i + 1) * m]
        ahatg[:(i + 1) * m] = eps * torch.logsumexp(torch.cat([update[None, :], ahatg[:(i + 1) * m][None, :]]) / eps,
                                                    dim=0)

        # Update g
        x_, loga = sample_from(x, m)
        f = evaluate_potential(hatg[:(i + 1) * m], posy[:(i + 1) * m], x_, eps)
        hatf[:i * m] += eps * torch.log(1 - eta)
        hatf[i * m:(i + 1) * m] = eps * math.log(eta) + loga * eps + f
        posx[i * m:(i + 1) * m] = x_

        update = eps * math.log(eta) + hatf[:(i + 1) * m]
        ahatf[:(i + 1) * m] = eps * torch.logsumexp(torch.cat([update[None, :], ahatf[:(i + 1) * m][None, :]]) / eps,
                                                    dim=0)

        sum_eta += eta

    ahatg -= torch.log(sum_eta) * eps
    ahatf -= torch.log(sum_eta) * eps
    f = evaluate_potential(ahatg, posy, x, eps)
    return evaluate_potential(ahatg, posy, x, eps), evaluate_potential(ahatf, posx, y, eps)


def evaluate_potential_finite(log_pot: torch.tensor, idx: torch.tensor, distance, eps):
    return - eps * torch.logsumexp((- distance[idx] + log_pot[None, :]) / eps, dim=1)


def stochasic_sinkhorn_finite(x, y, fref, gref, wref, eps, m, n_iter=100, step_size='constant'):
    n = x.shape[0]
    hatf = torch.full((n,), fill_value=-float('inf'))
    hatg = torch.full((n,), fill_value=-float('inf'))
    distance = compute_distance(x, y)
    for i in range(0, n_iter):
        # eta = torch.tensor(m / n)  # if i < 100 else torch.tensor(1. / (1. + math.pow(i + 1, 0.75)))
        eta = torch.tensor(1. / math.pow(i + 1, 1))
        # eta = torch.tensor(.9)
        # Update f
        y_idx, logb = sample_from_finite(y, m)
        x_idx, loga = sample_from_finite(x, m)
        if i > 0:
            g = evaluate_potential_finite(hatf, y_idx, distance.transpose(0, 1), eps)
            f = evaluate_potential_finite(hatg, x_idx, distance, eps)
        else:
            g = gref[y_idx]
            f = fref[x_idx]
        hatg += eps * torch.log(1 - eta)
        update = eps * math.log(eta) + logb * eps + g
        hatg[y_idx] = eps * torch.logsumexp(torch.cat([hatg[y_idx][None, :], update[None, :]], dim=0) / eps, dim=0)
        hatf += eps * torch.log(1 - eta)
        update = eps * math.log(eta) + loga * eps + f
        hatf[x_idx] = eps * torch.logsumexp(torch.cat([hatf[x_idx][None, :], update[None, :]], dim=0) / eps, dim=0)

        if i % 1000 == 0:
            ff = evaluate_potential_finite(hatg, slice(None), distance, eps)
            gg = evaluate_potential_finite(hatf, slice(None), distance.transpose(0, 1), eps)
            w = ff.mean() + gg.mean()

            loga = torch.full((n,), fill_value=-math.log(n))
            logb = torch.full((n,), fill_value=-math.log(n))
            ff2 = evaluate_potential_finite(gg + logb * eps, slice(None), distance, eps)
            gg2 = evaluate_potential_finite(ff + loga * eps, slice(None), distance.transpose(0, 1), eps)
            fref2 = evaluate_potential_finite(gref + logb * eps, slice(None), distance, eps)
            gref2 = evaluate_potential_finite(fref + loga * eps, slice(None), distance.transpose(0, 1), eps)
            plan = torch.exp((loga[:, None] * eps + ff[:, None] + logb[:, None] * eps + gg[None, :] - distance) / eps)
            planref = torch.exp(
                (loga[:, None] * eps + fref[:, None] + logb[:, None] * eps + gref[None, :] - distance) / eps)
            # print(plan, planref)
            errors = {'f - T(g, b)': var_norm(ff - ff2),
                      'g - T(f, a)': var_norm(gg - gg2),
                      'f - f_ref': var_norm(ff - fref),
                      'g - g_ref': var_norm(gg - gref),
                      'w - wref': (w - wref).item(),
                      'plan_diff': torch.max(torch.abs(plan - planref)),
                      'marg1': torch.max(torch.abs(plan.sum(1) - torch.ones(n) / n)),
                      'marg2': torch.max(torch.abs(plan.sum(0) - torch.ones(n) / n)),
                      # 'fref - T(gref, b)': var_norm(fref - fref2),
                      # 'gref - T(fref, a)': var_norm(gref - gref2),
                      }
            string = f"iter:{i} "
            for k, v in errors.items():
                string += f'[{k}]:{v:.4f} '
            print(string)
    return ff, gg


def var_norm(x):
    return x.max() - x.min()


def sinkhorn(x, y, eps, n_iter=1000):
    n = x.shape[0]
    loga = torch.full((n,), fill_value=-math.log(n))
    logb = torch.full((n,), fill_value=-math.log(n))
    distance = compute_distance(x, y)
    g = torch.zeros((n,))
    f = torch.zeros((n,))
    for i in range(n_iter):
        ff = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
        f_diff = f - ff

        gg = - eps * torch.logsumexp((- distance.transpose(0, 1) + ff[None, :]) / eps + loga[None, :], dim=1)
        g_diff = g - gg
        tol = f_diff.max() - f_diff.min() + g_diff.max() - g_diff.min()
        f, g = ff, gg
    print('tol', tol)
    return f, g


def main():
    n = 100
    m = 5
    eps = 1

    torch.manual_seed(100)
    np.random.seed(100)

    y = torch.randn(n, 2) * 0.1
    x = torch.randn((n, 2)) * 0.1 + 2
    print('===========================================True=====================================')
    f, g = sinkhorn(x, y, eps=eps, n_iter=4000)
    w = f.mean() + g.mean()
    print('w', w)
    # print('===========================================Stochastic=====================================')
    # torch.manual_seed(100)
    # np.random.seed(100)
    # smd_f, smd_g = stochasic_sinkhorn(x, y, eps=eps, m=m, n_iter=1000, step_size='sqrt')
    # print(smd_f.mean() + smd_g.mean())
    # print((f - smd_f).max() - (f - smd_f).min() + (g - smd_g).max() - (g - smd_g).min())
    print('===========================================Finite stochastic=====================================')
    torch.manual_seed(100)
    np.random.seed(100)
    smd_f, smd_g = stochasic_sinkhorn_finite(x, y, fref=f, gref=g, eps=eps, wref=w, m=m, n_iter=100000,
                                             step_size='sqrt')
    print(smd_f.mean() + smd_g.mean())


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


def sampling_sinkhorn(x_sampler, y_sampler, eps, m, grid, n_iter=100, step_size='sqrt'):
    posx = torch.zeros(m * n_iter, 1)
    hatg = torch.full((m * n_iter,), fill_value=-float('inf'))
    hatf = torch.full((m * n_iter,), fill_value=-float('inf'))

    posy = torch.zeros(m * n_iter, 1)
    w = 0
    fevals = []
    gevals = []
    for i in range(0, n_iter):
        if step_size == 'sqrt':
            eta = torch.tensor(i + 1.).pow(torch.tensor(-0.51))
        elif step_size == 'linear':
            eta = torch.tensor(1. / (i + 1))
        elif step_size == 'constant':
            eta = torch.tensor(.1)

        # Update f
        y_, logb = y_sampler(m)
        x_, loga = x_sampler(m)
        if i > 0:
            f = evaluate_potential(hatg[:i * m], posy[:i * m], x_, eps)
            g = evaluate_potential(hatf[:i * m], posx[:i * m], y_, eps)
        else:
            g = torch.zeros(m)
            f = torch.zeros(m)
        hatg[:i * m] += eps * torch.log(1 - eta)
        hatg[i * m:(i + 1) * m] = eps * math.log(eta) + logb * eps + g
        posy[i * m:(i + 1) * m] = y_
        # hatg[:(i + 1) * m] -= eps * torch.logsumexp(hatg[:(i + 1) * m] / eps, dim=0)

        # Update g
        hatf[:i * m] += eps * torch.log(1 - eta)
        hatf[i * m:(i + 1) * m] = eps * math.log(eta) + loga * eps + f
        posx[i * m:(i + 1) * m] = x_
        # hatf[:(i + 1) * m] -= eps * torch.logsumexp(hatf[:(i + 1) * m] / eps, dim=0)
        w *= 1 - eta
        w += eta * (hatf[i * m:(i + 1) * m].mean() + hatg[i * m:(i + 1) * m].mean())

        if i % 10 == 0:
            fevals.append(evaluate_potential(hatg[:(i + 1) * m], posy[:(i + 1) * m], grid, eps))
            gevals.append(evaluate_potential(hatf[:(i + 1) * m], posx[:(i + 1) * m], grid, eps))
    return hatf, posx, hatg, posy, w, fevals, gevals


def one_dimensional_exp():
    eps = 1e-2

    grid = torch.linspace(-1, 7, 500)[:, None]
    C = compute_distance(grid, grid)

    x_sampler = Sampler(mean=torch.tensor([[1.], [2], [3]]), cov=torch.tensor([[[.1]], [[.1]], [[.1]]]),
                        p=torch.ones(3) / 3)
    y_sampler = Sampler(mean=torch.tensor([[0.], [3], [5]]), cov=torch.tensor([[[.1]], [[.1]], [[.4]]]),
                        p=torch.ones(3) / 3)
    #
    x_sampler = Sampler(mean=torch.tensor([[0.]]), cov=torch.tensor([[[.1]]]), p=torch.ones(1))
    y_sampler = Sampler(mean=torch.tensor([[2.]]), cov=torch.tensor([[[.1]]]), p=torch.ones(1))

    px = torch.exp(x_sampler.log_prob(grid))
    py = torch.exp(y_sampler.log_prob(grid))

    fevals = []
    gevals = []
    labels = []
    plans = []

    n_samples = 2000
    x, loga = x_sampler(n_samples)
    y, logb = y_sampler(n_samples)
    # x_sampler.pos_, x_sampler.logweight_ = x, loga
    # y_sampler.pos_, y_sampler.logweight_ = y, logb
    f, g = sinkhorn(x, y, eps=eps, n_iter=10)
    distance = compute_distance(grid, y)
    feval = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
    distance = compute_distance(grid, x)
    geval = - eps * torch.logsumexp((- distance + f[None, :]) / eps + loga[None, :], dim=1)
    distance = compute_distance(x, y)
    plan = loga[:, None] + f[:, None] / eps + logb[None, :] + g[None, :] / eps - distance / eps
    print(torch.logsumexp(plan.view(-1), dim=0))

    plan = (x_sampler.log_prob(grid)[:, None] + feval[:, None] / eps + y_sampler.log_prob(grid)[None, :]
            + geval[None, :] / eps - C / eps)
    print(torch.logsumexp(plan.view(-1), dim=0))

    plans.append((plan, grid, grid))

    fevals.append(feval)
    gevals.append(geval)
    labels.append(f'Sinkhorn n={n_samples}')
    hatf, posx, hatg, posy, w, sto_fevals, sto_gevals = sampling_sinkhorn(x_sampler, y_sampler, m=100, eps=eps,
                                                                          n_iter=100,
                                                                          step_size='sqrt', grid=grid)
    feval = evaluate_potential(hatg, posy, grid, eps)
    geval = evaluate_potential(hatf, posx, grid, eps)
    plan = (x_sampler.log_prob(grid)[:, None] + feval[:, None] / eps + y_sampler.log_prob(grid)[None, :]
            + geval[None, :] / eps - C / eps)
    plans.append((plan, grid, grid))
    print(torch.logsumexp(plan.view(-1), dim=0))

    fevals.append(feval)
    gevals.append(geval)
    labels.append(f'Online Sinkhorn n={10}')
    fevals = torch.cat([feval[None, :] for feval in fevals], dim=0)
    gevals = torch.cat([geval[None, :] for geval in gevals], dim=0)

    fig, axes = plt.subplots(4, 1, figsize=(8, 12))
    fig_plan, axes_plan = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].plot(grid, px, label='alpha')
    axes[0].plot(grid, py, label='beta')
    axes[0].legend()
    for i, (label, feval, geval, (plan, x, y)) in enumerate(zip(labels, fevals, gevals, plans)):
        axes[1].plot(grid, feval, label=label)
        axes[2].plot(grid, geval, label=label)
        plan = plan.numpy()
        axes_plan[i].contourf(y[:, 0], x[:, 0], plan, levels=30)
    # axes_plan[1].add_colorbar()
    colors = plt.cm.get_cmap('Blues')(np.linspace(0.2, 1, len(sto_fevals)))
    axes[3].set_prop_cycle('color', colors)
    for eval in sto_fevals:
        axes[3].plot(grid, eval, label=label)
    axes[2].legend()
    axes[0].set_title('Distributions')
    axes[1].set_title('Potential f')
    axes[2].set_title('Potential g')
    plt.show()


if __name__ == '__main__':
    # main()
    one_dimensional_exp()
