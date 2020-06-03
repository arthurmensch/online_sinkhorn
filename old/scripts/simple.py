import matplotlib.pyplot as plt
import numpy as np
import torch

from onlikhorn.dataset import make_gmm_1d, make_gmm_2d
from onlikhorn.algorithm import var_norm, sinkhorn, online_sinkhorn


def simple_test():
    n_iter = 200
    ref_iter = 500
    epsilon = 1e-2
    ref_grid = 400

    x_sampler, y_sampler = make_gmm_1d()
    x, la, _ = x_sampler(ref_grid)
    y, lb, _ = y_sampler(ref_grid)
    F, G = sinkhorn(x, la, y, lb, n_iter=ref_iter, epsilon=epsilon)

    batch_sizes = np.ceil(10 * np.float_power(np.arange(n_iter, dtype=float) + 1, 1)).astype(int)
    lrs = (1 * np.float_power(np.arange(n_iter, dtype=float) + 1, 0))
    lim = np.where(batch_sizes >= ref_grid)[0][0]
    lrs[lim:] = 1
    batch_sizes[lim:] = ref_grid
    batch_sizes = batch_sizes.tolist()
    lrs = lrs.tolist()

    aF, aG = sinkhorn(x, la, y, lb, n_iter=n_iter - lim, epsilon=epsilon)

    oF, oG = online_sinkhorn(x=x, y=y, la=la, lb=lb, x_sampler=x_sampler, y_sampler=y_sampler, use_finite=True,
                             batch_sizes=batch_sizes, lrs=lrs, refit=True, epsilon=epsilon)
    print(var_norm(oF, F, x) + var_norm(oG, G, y))
    print(var_norm(aF, F, x) + var_norm(aG, G, y))

    import matplotlib.pyplot as plt
    x, _ = torch.sort(x[:, 0])
    y, _ = torch.sort(y[:, 0])
    f1 = oF(x[:, None])
    f2 = F(x[:, None])
    g1 = oG(y[:, None])
    g2 = G(y[:, None])
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    ax1.plot(x, f1)
    ax1.plot(x, f2)
    ax2.plot(x, g1)
    ax2.plot(x, g2)
    X = torch.linspace(-4, 4, 100)
    llX = x_sampler.log_prob(X[:, None])
    Y = torch.linspace(-4, 4, 100)
    llY = y_sampler.log_prob(Y[:, None])
    ax0.plot(X, llX)
    ax0.plot(Y, llY)
    plt.show()


def make_2d_grid():
    X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    Z = np.concatenate([X[:, :, None], Y[:, :, None]], axis=2).reshape(-1, 2)
    Z = torch.from_numpy(Z).float()
    return Z


def compute_grad(potential, z):
    z = z.clone()
    z.requires_grad = True
    grad, = torch.autograd.grad(potential(z).sum(), (z,))
    return - grad.detach()

def plot_2d():
    n_iter = 200
    ref_iter = 100
    epsilon = 1e-4
    ref_grid = 1000
    lr_exp = .5
    batch_exp = 0  # > 2*(1 - lr_exp) for convergence if not refit,
    batch_size = 100
    lr = 1
    refit = False
    use_finite = False

    x_sampler, y_sampler = make_gmm_2d()

    x, la, _ = x_sampler(ref_grid)
    y, lb, _ = y_sampler(ref_grid)
    F, G = sinkhorn(x, la, y, lb, n_iter=ref_iter, epsilon=epsilon)

    batch_sizes = np.ceil(batch_size * np.float_power(np.arange(n_iter, dtype=float) + 1, batch_exp)).astype(int)
    lrs = (lr * np.float_power(np.arange(n_iter, dtype=float) + 1, -lr_exp))
    batch_sizes = batch_sizes.tolist()
    lrs = lrs.tolist()

    oF, oG = online_sinkhorn(x=x, y=y, la=la, lb=lb, x_sampler=x_sampler, y_sampler=y_sampler, use_finite=use_finite,
                             batch_sizes=batch_sizes, lrs=lrs, refit=refit, epsilon=epsilon)

    fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(12, 18))

    z = make_2d_grid()
    llx = x_sampler.log_prob(z)
    lly = y_sampler.log_prob(z)

    grad_f = compute_grad(F, x)
    grad_g = compute_grad(G, y)

    grad_fgrid = compute_grad(F, z)
    grad_ggrid = compute_grad(G, z)

    shape = (100, 100)
    axes[0, 0].contour(z[:, 0].view(shape), z[:, 1].view(shape), llx.view(shape), zorder=0, levels=30)
    axes[0, 0].scatter(x[:, 0], x[:, 1], 2, zorder=10)
    axes[0, 0].quiver(x[:, 0], x[:, 1], grad_f[:, 0], grad_f[:, 1], zorder=20)

    axes[1, 0].contour(z[:, 0].view(shape), z[:, 1].view(shape), llx.view(shape), zorder=0, levels=30)
    axes[1, 0].quiver(z[:, 0], z[:, 1], grad_fgrid[:, 0] * llx.exp(), grad_fgrid[:, 1] * llx.exp(), zorder=20)

    axes[0, 1].contour(z[:, 0].view(shape), z[:, 1].view(shape), lly.view(shape), zorder=0, levels=30)
    axes[0, 1].scatter(y[:, 0], y[:, 1], 2, zorder=10)
    axes[0, 1].quiver(y[:, 0], y[:, 1], grad_g[:, 0], grad_g[:, 1], zorder=20)

    axes[1, 1].contour(z[:, 0].view(shape), z[:, 1].view(shape), lly.view(shape), zorder=0, levels=30)
    axes[1, 1].quiver(z[:, 0], z[:, 1], grad_ggrid[:, 0] * lly.exp(), grad_ggrid[:, 1] * lly.exp(), zorder=20)


    plt.show()

plot_2d()