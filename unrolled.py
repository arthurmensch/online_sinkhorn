import math

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam


def sample_gmm(n_samples):
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
    return torch.from_numpy(pos.astype(np.float32)) / 10, torch.from_numpy(logweight.astype(np.float32)),\
           torch.from_numpy(indices)


def sample(n_samples):
    return torch.randn(n_samples, 32), torch.full((n_samples, ), fill_value=- math.log(n_samples))


def get_distance(x, y):
    return torch.sum((x[:, :, None, :] ** 2 + y[:, None, :, :] ** 2 - 2 * x[:, :, None, :] * y[:, None, :, :]), dim=3)


class UnrolledSinkhorn(nn.Module):
    def __init__(self):
        super(UnrolledSinkhorn, self).__init__()
        self.epsilon = 1e-4

    def step(self, new_x, new_y, new_loga, new_logb, f, g, x, y, loga, logb):
        distance = get_distance(new_x, y)
        new_f = - self.epsilon * torch.logsumexp((- distance + g[:, None, :]) / self.epsilon + logb[:, None, :], dim=2)
        distance = get_distance(new_y, x)
        new_g = - self.epsilon * torch.logsumexp((- distance + f[:, None, :]) / self.epsilon + loga[:, None, :], dim=2)
        loss = torch.sum(torch.exp(new_loga) * new_f) + torch.sum(torch.exp(new_logb) * new_g)
        return loss, new_f, new_g

    def forward(self, x, y, loga, logb):
        b, t, n, d = x.shape
        b, t, m, d = y.shape
        assert(loga.shape == (b, t, n))
        assert(logb.shape == (b, t, m))
        losses = torch.zeros(b, t)
        f = torch.zeros(b, t, n)
        g = torch.zeros(b, t, m)
        for i in range(1, t):
            losses[:, i], f[:, i], g[:, i] = self.step(x[:, i], y[:, i], loga[:, i], loga[:, i],
                                                       f[:, i - 1], g[:, i - 1], x[:, i - 1], y[:, i - 1],
                                                       loga[:, i - 1], logb[:, i - 1])
        return losses, f, g

def sample_both(sample_length, device='cpu'):
    y, logb, _ = sample_gmm(sample_length)
    y = y[None, :, :].to(device)
    logb = logb[None, :].to(device)
    noise, loga = sample(sample_length)
    noise = noise.to(device)
    x = generator(noise)
    x = x[None, :]
    loga = loga[None, :].to(device)
    return x, y, loga, logb


device = 'cuda:0'

generator = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 128),
                          nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))
generator.to(device)
sinkhorn = UnrolledSinkhorn()


optimizer = Adam(generator.parameters(), lr=1e-3)

sample_length = 100
n_iter = 100
unroll = 100
resample = False

for i in range(n_iter):
    x, y, loga, logb = sample_both(sample_length, device=device)
    f = torch.zeros_like(loga)
    g = torch.zeros_like(logb)
    if resample:
        mean_loss = torch.zeros(1, device=device)
    for j in range(unroll):
        if resample:
            new_x, new_y, new_loga, new_logb = sample_both(sample_length, device=device)
        else:
            new_x, new_y, new_loga, new_logb = x, y, loga, logb
        loss, f, g = sinkhorn.step(new_x, new_y, new_loga, new_logb, f, g, x, y, loga, logb)
        if resample:
            mean_loss += loss
            x, y, loga, logb = new_x, new_y, new_loga, new_logb
    if resample:
        loss = mean_loss / unroll
    optimizer.zero_grad()
    print(loss.item())
    loss.backward()
    optimizer.step()

noise, loga = sample(1000)
with torch.no_grad():
    x = generator(noise.to(device))
x = x.to('cpu')
y, logb, _ = sample_gmm(1000)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
ax.scatter(x[:, 0], x[:, 1], color='blue')
ax.scatter(y[:, 0], y[:, 1], color='red')
plt.show()


