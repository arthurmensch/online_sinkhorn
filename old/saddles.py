import torch

import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.Q = torch.tensor([
            [[3, 8.],
             [0, 3]],
            [[0, 1.],
             [0, 0]],
        ])
        self.offset = torch.tensor([
            [0., 1],
            [-1, 0]
        ])
        self.weight = torch.tensor([1, 1.])

    def forward(self, pos: torch.tensor):
        centered = pos[:, None, :] - self.offset[None, :, :]
        outer = centered[:, :, None, :] * centered[:, :, :, None]
        print(outer.shape)
        S = torch.sum(outer * self.Q[None, :, :, :], dim=(2, 3))
        return torch.sum(S * self.weight[None, :], dim=1)


class Mazumdar(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos):
        return (torch.exp(- .01 * (pos[:, 0] ** 2 + pos[:, 1] ** 2)) *
                ((.3 * pos[:, 0] ** 2 + pos[:, 1]) ** 2
                + (0.5 * pos[:, 1] ** 2 + pos[:, 0]) ** 2))


loss = Mazumdar()

n = 100
X = torch.linspace(-20, 20, n)
Y = torch.linspace(-20, 20, n)
pos = torch.meshgrid(X, Y)
pos = torch.cat([pos[0][:, :, None], pos[1][:, :, None]], dim=2).view(-1, 2)
Z = loss(pos).view(n, n)

import matplotlib.pyplot as plt

plt.contourf(X.numpy(), Y.numpy(), Z.numpy(), levels=50)
plt.show()
