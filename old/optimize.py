import numpy as np
import torch
from torch.nn import Parameter

dim = 100
f, g, C = torch.randn(dim).double(), torch.randn(dim).double(), torch.randn(dim, dim).double()


def constraint(f, g, C):
    return f[:, None] + g[None, :] + C


def func(pi: torch.tensor):
    pi = pi.view(dim, dim)
    return .5 * (torch.sum((f - torch.sum(pi, 1)) ** 2) + torch.sum((g - torch.sum(pi, 0)) ** 2) + torch.sum((C - pi) ** 2))


def grad(pi: torch.tensor):
    pi = Parameter(pi)
    f = func(pi)
    grad, = torch.autograd.grad(f, (pi, ))
    return grad


def wrapped_func(pi: np.ndarray):
    pi = pi.reshape((dim, dim))
    return func(torch.from_numpy(pi)).item()


def wrapped_grad(pi: np.ndarray):
    pi = pi.reshape((dim, dim))
    return grad(torch.from_numpy(pi)).view(-1).numpy()


from scipy.optimize import fmin_l_bfgs_b

pi0 = torch.zeros((dim, dim), dtype=torch.double).view(-1).numpy()
dual, value, d = fmin_l_bfgs_b(wrapped_func, pi0, wrapped_grad, bounds=[(0, None)] * (dim ** 2), disp=100)
dual = torch.from_numpy(dual.reshape((dim, dim)))
cons = constraint(f, g, C)
fp, gp, Cp = f - dual.sum(1), g - dual.sum(0), C - dual
print(func(dual))
# print(constraint(fp, gp, Cp).max())
dual = torch.clamp(cons, min=0)
print(func(dual))
fp, gp, Cp = f - dual.sum(1), g - dual.sum(0), C - dual
# print(constraint(fp, gp, Cp).max())
