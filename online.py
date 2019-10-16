import math

import torch

n = 1000
eps = 1e-3

def sample_x(n):
    return torch.randn((n, 10)) / 10, torch.full((n, ), fill_value=-math.log(n))

def sample_y(n):
    shift = torch.zeros((10, ))
    shift[0] = 1
    return shift[None, :] + torch.randn((n, 10)) / 10, torch.full((n, ), fill_value=-math.log(n))

def get_distance(x, y):
    return torch.sum((x[:, None, :] ** 2 + y[None, :, :] ** 2 - 2 * x[:, None, :] * y[None, :, :]), dim=2)

g = torch.zeros((n, ))
y, logb = sample_y(n)
x, loga = sample_x(n)

mean_wass = 0

resample = True

if not resample:
    y, logb = sample_y(n)
    x, loga = sample_x(n)
    distance = get_distance(x, y)
else:
    y, logb = sample_y(n)


for n_iter in range(1000):
    if resample:
        x, loga = sample_x(n)
        distance = get_distance(x, y)
    f = - eps * torch.logsumexp((- distance + g[None, :]) / eps + logb[None, :], dim=1)
    if resample:
        y, logb = sample_y(n)
        distance = get_distance(x, y)
    g = - eps * torch.logsumexp((- distance.transpose(0, 1) + f[None, :]) / eps + loga[None, :], dim=1)
    wass = torch.sum(torch.exp(loga) * f) + torch.sum(torch.exp(logb) * g)
    mean_wass *= (1 - 1 / (n_iter + 1))
    mean_wass += wass / (n_iter + 1)
    print(torch.sqrt(wass).item(), torch.sqrt(mean_wass).item())

