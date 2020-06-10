import torch

from onlikhorn.dataset import GaussianSampler


def symsqrt(matrix):
    """Compute the square root of a positive definite matrix."""
    # perform the decomposition
    # s, v = matrix.symeig(eigenvectors=True)
    _, s, v = matrix.svd()  # passes torch.autograd.gradcheck()
    # truncate small components
    # above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
    # s = s[..., above_cutoff]
    # v = v[..., above_cutoff]
    # compose the square root matrix
    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)


class GaussianPotential():
    def __init__(self, mean, cov, epsilon):
        self.mean = mean
        self.cov = cov
        self.epsilon = epsilon
        self.dimension = self.mean.shape[0]
        self.const = 0

    def refit(self, G):
        Id = torch.eye(self.dimension, device=self.mean.device)
        C = symsqrt(self.cov @ G.cov + (self.epsilon / 2) ** 2 * Id)
        self.U = G.cov @ torch.inverse(C + self.epsilon / 2 * Id) - Id
        self.other_mean = G.mean

    def __call__(self, x):
        centered = x - self.mean[None, :]
        return (- .5 * torch.sum((centered @ self.U) * centered, dim=1)
                + torch.sum(x * (self.mean - self.other_mean), dim=1))

    def add_weight(self, weight):
        self.const += weight


def sinkhorn_gaussian(x_sampler: GaussianSampler, y_sampler: GaussianSampler, epsilon=1.):
    F = GaussianPotential(x_sampler.mean, x_sampler.cov, epsilon=epsilon)
    G = GaussianPotential(y_sampler.mean, y_sampler.cov, epsilon=epsilon)
    F.refit(G)
    G.refit(F)
    anchor = F(torch.zeros((1, x_sampler.dimension), device=x_sampler.device))
    F.add_weight(anchor)
    G.add_weight(-anchor)
    return F, G
