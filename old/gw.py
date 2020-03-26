import torch
import numpy as np


def distance(x, y):
    return torch.sqrt(torch.sum(x ** 2, dim=1)[:, None] + torch.sum(y ** 2, dim=1)[None :]
                      - 2 * x @ y.transpose(0, 1))

def convolution(A, B, pi):
    return - torch.sum(A[:, :, None, None] * B[None, None, :, :] * pi[None, :, None, :], dim=(1,3))

n = 3
alpha = torch.full((n, ), fill_value=1/n)
beta = torch.full((n, ), fill_value=1/n)
x = torch.randn((n, 2))
permutation = np.random.permutation(n)
y = x[permutation]
print(y)
A = distance(x, x)
B = distance(y, y)
print(A, B)

def convolution(A, B, pi):
    return - torch.sum(A[:, :, None, None] * B[None, None, :, :] * pi[None, :, None, :], dim=(1,3))

pi = torch.eye(n)
print(torch.sum(convolution(A, B, pi) * pi))
