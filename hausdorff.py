import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from geomloss import SamplesLoss


def sample(n_samples):
    mean = np.array([[0, 3], [3, 0], [-3, 0]])
    cov = np.zeros((3, 2, 2))
    eig = np.array([[1, 0], [2, 1]])
    cov[0] = eig @ eig.T
    eig = np.array([[1, 0], [0, 1]])
    cov[1] = eig @ eig.T
    eig = np.array([[1, 0], [0, 1]])
    cov[2] = eig @ eig.T
    cov /= 4
    p = np.array([0.3, 0.5, 0.2])
    indices = np.random.choice(3, n_samples, p=p)
    pos = np.zeros((n_samples, 2))
    for i in range(3):
        mask = indices == i
        size = mask.sum()
        pos[mask] = np.random.multivariate_normal(mean[i], cov[i], size=size)
    logweight = np.full((n_samples, ), fill_value=- math.log(n_samples))
    return pos, logweight, indices


def get_distance(x, y):
    return torch.sum((x[:, None, :] ** 2 + y[None, :, :] ** 2 - 2 * x[:, None, :] * y[None, :, :]), dim=2)


ref, loga, indices = sample(100)
eps = 1e-2
ref = torch.from_numpy(ref)
loga = torch.from_numpy(loga)
distance = get_distance(ref, ref)

f = torch.zeros_like(loga)
for n_iter in range(50):
    fn = - eps * torch.logsumexp((- distance + f[None, :]) / eps + loga[None, :], dim=1)
    f = (f + fn) / 2

xmin, xmax = ref[:, 0].min(), ref[:, 0].max()
ymin, ymax = ref[:, 1].min(), ref[:, 1].max()

X, Y = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
grid = np.concatenate([X[:, :, None], Y[:, :, None]], axis=2).reshape((-1, 2))
grid = torch.from_numpy(grid)
distance = get_distance(grid, ref)

g = - eps * torch.logsumexp((- distance + f[None, :]) / eps + loga[None, :], dim=1)
ll = - eps * torch.logsumexp((- distance) / eps + loga[None, :], dim=1)
g = g.view(*X.shape)
ll = ll.view(*X.shape)


loss_fn = SamplesLoss()
grid = grid[:, None, :]
weights = torch.ones_like(grid[:, :, 0])
ref_weights = torch.exp(loga)[None, :].repeat(grid.shape[0], 1)
refs = ref[None, :, :].repeat(grid.shape[0], 1, 1)
print(weights.shape)
print(ref_weights.shape)
print(grid.shape)
print(refs.shape)
wass = loss_fn(weights, grid, ref_weights, refs).reshape(*X.shape)

g = g.numpy()
ref = ref.numpy()
ll.numpy()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.ravel()
for ax, pot in zip(axes, (g, ll, wass)):
    m = ax.contour(X, Y, pot, levels=30)
    ax.scatter(ref[:, 0], ref[:, 1], s=5, color='red')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    fig.colorbar(m)
fig.show()
