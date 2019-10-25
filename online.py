import math

import numpy as np
import torch


def sample_from(x, m):
    n = x.shape[0]
    indices = torch.from_numpy(np.random.permutation(n)[:m])
    loga = torch.full((m,), fill_value=-math.log(m))
    return x[indices], loga, indices


def compute_distance(x, y):
    return (torch.sum((x[:, None, :] ** 2 + y[None, :, :] ** 2 - 2 * x[:, None, :] * y[None, :, :]), dim=2)) / 2


def evaluate_potential(log_pot: torch.tensor, pos: torch.tensor, x: torch.tensor, eps):
    distance = compute_distance(x, pos)
    return - eps * torch.logsumexp((- distance + log_pot[None, :]) / eps, dim=1)


def stochasic_sinkhorn(x, y, eps, m, n_iter=100, step_size='sqrt'):
    hatf = torch.zeros(m * n_iter)
    posy = torch.zeros(m * n_iter, 2)
    hatg = torch.zeros(m * n_iter)
    posx = torch.zeros(m * n_iter, 2)

    for i in range(0, n_iter):
        if step_size == 'sqrt':
            eta = torch.tensor(1. / math.sqrt(i + 1))
        elif step_size == 'constant':
            eta = torch.tensor(1.)

        # Update f
        y_, logb, y_idx = sample_from(y, m)
        if i > 0:
            g = evaluate_potential(hatf[:i * m], posx[:i * m], y_, eps)
        else:
            g = torch.zeros(m)
        hatg[:i * m] += eps * torch.log(1 - eta)
        hatg[i * m:(i + 1) * m] = eps * math.log(eta) + logb * eps + g
        posy[i * m:(i + 1) * m] = y_

        # Update g
        x_, loga, x_idx = sample_from(x, m)
        f = evaluate_potential(hatg[:(i + 1) * m], posy[:(i + 1) * m], x_, eps)
        hatf[:i * m] += eps * torch.log(1 - eta)
        hatf[i * m:(i + 1) * m] = eps * math.log(eta) + loga * eps + f
        posx[i * m:(i + 1) * m] = x_
    return evaluate_potential(hatg, posy, x, eps), evaluate_potential(hatf, posx, y, eps)


def evaluate_potential_finite(log_pot: torch.tensor, idx: torch.tensor, distance, eps):
    return - eps * torch.logsumexp((- distance[idx] + log_pot[None, :]) / eps, dim=1)


def stochasic_sinkhorn_finite(x, y, eps, m, n_iter=100, step_size='sqrt'):
    n = x.shape[0]
    hatf = torch.full((n,),  fill_value=-float('inf'))
    hatg = torch.full((n,), fill_value=-float('inf'))
    distance = compute_distance(x, y)

    for i in range(0, n_iter):
        if step_size == 'sqrt':
            eta = torch.tensor(1. / math.sqrt(i + 1))
        elif step_size == 'constant':
            eta = torch.tensor(1.)

        # Update f
        _, logb, y_idx = sample_from(y, m)
        if i > 0:
            g = evaluate_potential_finite(hatf, y_idx, distance.transpose(0, 1), eps)
        else:
            g = torch.zeros(m)
        hatg[y_idx] += eps * torch.log(1 - eta)
        update = eps * math.log(eta) + logb * eps + g
        hatg[y_idx] = eps * torch.logsumexp(torch.cat([hatg[y_idx][None, :], update[None, :]], dim=0) / eps, dim=0)
        # Update g
        x_, loga, x_idx = sample_from(x, m)
        f = evaluate_potential_finite(hatg, x_idx, distance, eps)
        hatf[x_idx] += eps * torch.log(1 - eta)
        update = eps * math.log(eta) + loga * eps + f
        hatf[x_idx] = eps * torch.logsumexp(torch.cat([hatf[x_idx][None, :], update[None, :]], dim=0) / eps, dim=0)
    return evaluate_potential_finite(hatg, None, distance, eps), evaluate_potential_finite(hatf, None,
                                                                                           distance.transpose(0, 1),
                                                                                           eps)


def sinkhorn(x, y, eps, n_iter=100):
    n = x.shape[0]
    loga = torch.full((n,), fill_value=-math.log(n))
    logb = torch.full((n,), fill_value=-math.log(n))
    distance = compute_distance(x, y)
    g = torch.zeros((n,))
    f = torch.zeros((n,))
    for i in range(n_iter):
        # print(f, g)
        fn = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
        gn = - eps * torch.logsumexp((- distance.transpose(0, 1) + fn[None, :]) / eps + loga[None, :], dim=1)
        f_diff = f - fn
        g_diff = g - gn
        f = fn
        g = gn
        tol = f_diff.max() - f_diff.min() + g_diff.max() - g_diff.min()

    return f, g


def main():
    n = 5
    m = 5
    eps = 1e-1

    torch.manual_seed(10)
    np.random.seed(10)

    y = torch.randn(n, 2)
    x = torch.randn((n, 2))
    print('===========================================True=====================================')
    f, g = sinkhorn(x, y, eps=eps, n_iter=100)
    print('===========================================Stochastic=====================================')
    smd_f, smd_g = stochasic_sinkhorn(x, y, eps=eps, m=m, n_iter=1000, step_size='constant')
    smd_f_finite, smd_g_finite = stochasic_sinkhorn_finite(x, y, eps=eps, m=m, n_iter=1000)
    print(smd_f_finite, smd_g_finite)
    print(smd_f.mean() + smd_g.mean())
    print(smd_f_finite.mean() + smd_g_finite.mean())
    print(f.mean() + g.mean())


if __name__ == '__main__':
    main()
