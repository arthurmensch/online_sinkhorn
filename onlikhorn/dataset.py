import os
import shutil
import urllib.request
from os.path import join

import numpy as np
import torch
from plyfile import PlyData

from onlikhorn.data import Subsampler


class GMMSampler:
    def __init__(self, mean: torch.tensor, cov: torch.tensor, p: torch.tensor):
        k, d = mean.shape
        k, d, d = cov.shape
        k = p.shape
        self.dimension = d
        self.mean = mean
        self.cov = cov
        self.icov = torch.cat([torch.inverse(cov)[None, :, :] for cov in self.cov], dim=0)
        det = torch.tensor([torch.det(cov) for cov in self.cov])
        self.norm = torch.sqrt((2 * np.pi) ** d * det)
        self.p = p
        self.device = 'cpu'

    def to(self, device):
        self.device = device
        return self

    def __call__(self, n):
        k, d = self.mean.shape
        indices = np.random.choice(k, n, p=self.p.numpy())
        pos = np.zeros((n, d), dtype=np.float32)
        for i in range(k):
            mask = indices == i
            size = mask.sum()
            pos[mask] = np.random.multivariate_normal(self.mean[i], self.cov[i], size=size)
        logweight = np.full_like(pos[:, 0], fill_value=-np.log(n))
        return torch.from_numpy(pos).to(self.device), torch.from_numpy(logweight).to(self.device), None

    def log_prob(self, x):
        # b, d = x.shape
        diff = x[:, None, :] - self.mean[None, :]  # b, k, d
        return torch.log(torch.sum(self.p[None, :] * torch.exp(-torch.einsum('bkd,kde,bke->bk',
                                                                             [diff, self.icov, diff]) / 2) / self.norm,
                                   dim=1))


class GaussianSampler:
    def __init__(self, mean: torch.tensor, cov: torch.tensor):
        self.mean = mean
        self.cov = cov
        self.gmm = GMMSampler(mean[None, :], cov[None, :], p=torch.ones_like(mean[[0]]))

    def to(self, device):
        self.gmm.to(device)

    def __call__(self, n):
        return self.gmm(n)

    @property
    def dimension(self):
        return self.gmm.dimension

    def log_prob(self, x):
        return self.gmm.log_prob(x)



def load_ply_file(fname, offset=[-0.011, 0.109, -0.008], scale=.04):
    """Loads a .ply mesh to return a collection of weighted Dirac atoms: one per triangle face."""

    # Load the data, and read the connectivity information:
    plydata = PlyData.read(fname)
    triangles = np.vstack(plydata['face'].data['vertex_indices'])

    # Normalize the point cloud, as specified by the user:
    points = np.vstack([[x, y, z] for (x, y, z) in plydata['vertex']])
    points -= offset
    points /= 2 * scale

    # Our mesh is given as a collection of ABC triangles:
    A, B, C = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]

    # Locations and weights of our Dirac atoms:
    X = (A + B + C) / 3  # centers of the faces
    S = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, 1)) / 2  # areas of the faces

    print("File loaded, and encoded as the weighted sum of {:,} atoms in 3D.".format(len(X)))

    # We return a (normalized) vector of weights + a "list" of points
    X = torch.from_numpy(X).float()
    S = torch.from_numpy(S).float().log()
    S -= torch.logsumexp(S, dim=0)

    return X, S


def make_sphere(n_samples=10000):
    """Creates a uniform sample on the unit sphere."""
    n_samples = int(n_samples)

    indices = np.arange(0, n_samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    points = torch.from_numpy(np.vstack((x, y, z)).T).float()
    weights = torch.full((n_samples, ), fill_value=-np.log(n_samples))

    return points, weights


def get_data_dir():
    data_dir = os.path.expanduser('~/data/online_sinkhorn')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def get_output_dir():
    output_dir = os.path.expanduser('~/output/online_sinkhorn')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def make_dragon(data_dir=None):
    if data_dir is None:
        data_dir = get_data_dir()
    filename = join(data_dir, 'dragon_recon/dragon_vrip_res4.ply')
    if not os.path.exists(filename):
        urllib.request.urlretrieve(
            'http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz',
            join(data_dir, 'dragon.tar.gz'))
        shutil.unpack_archive(join(data_dir, 'dragon.tar.gz'), data_dir)
    x, la =  load_ply_file(filename, offset=[-0.011, 0.109, -0.008], scale=.04)
    return x, torch.full_like(la, fill_value=-np.log(len(x)))


def make_gmm_1d():
    x_sampler = GMMSampler(mean=torch.tensor([[1.], [2], [3]]), cov=torch.tensor([[[.1]], [[.1]], [[.1]]]),
                           p=torch.ones(3) / 3)
    y_sampler = GMMSampler(mean=torch.tensor([[0.], [3], [5]]), cov=torch.tensor([[[.1]], [[.1]], [[.4]]]),
                           p=torch.ones(3) / 3)

    return x_sampler, y_sampler


def make_gmm_2d():
    cov_x = torch.eye(2) * .1, torch.eye(2) * .1, torch.eye(2) * .4
    cov_y = torch.eye(2) * .1, torch.eye(2) * .1, torch.eye(2) * .1
    cov_x = torch.cat([cov[None, :, :, ] for cov in cov_x], dim=0)
    cov_y = torch.cat([cov[None, :, :] for cov in cov_y], dim=0)
    x_sampler = GMMSampler(mean=torch.tensor([[1., 0], [2, 1.], [0., 1.]]), cov=cov_x,
                           p=torch.ones(3) / 3)
    y_sampler = GMMSampler(mean=torch.tensor([[0., -2], [2, -1], [3, 0]]), cov=cov_y,
                           p=torch.ones(3) / 3)

    return x_sampler, y_sampler


def make_gmm(d, modes):
    means_x = np.random.rand(modes, d)
    means_y = np.random.rand(modes, d)
    cov = np.repeat(np.eye(d)[None, :, :], modes, axis=0) * 1e-1
    means_x = torch.from_numpy(means_x).float()
    means_y = torch.from_numpy(means_y).float()
    cov = torch.from_numpy(cov).float()
    p = torch.full((modes, ), fill_value=1 / modes)
    x_sampler = GMMSampler(mean=means_x, cov=cov, p=p)
    y_sampler = GMMSampler(mean=means_y, cov=cov, p=p)

    return x_sampler, y_sampler



def make_gmm_2d_simple():
    cov_x = [torch.eye(2) * .1]
    cov_y = [torch.eye(2) * .1]
    cov_x = torch.cat([cov[None, :, :, ] for cov in cov_x], dim=0)
    cov_y = torch.cat([cov[None, :, :] for cov in cov_y], dim=0)
    x_sampler = GMMSampler(mean=torch.tensor([[1., 0]]), cov=cov_x,
                           p=torch.ones(1))
    y_sampler = GMMSampler(mean=torch.tensor([[0., -2]]), cov=cov_y,
                           p=torch.ones(1))

    return x_sampler, y_sampler


def make_data(data_source, n_samples):
    if data_source == 'dragon':
        x, la = make_sphere()
        y, lb = make_dragon()
        x *= 2
        y *= 2
        x_sampler = Subsampler(x, la)
        y_sampler = Subsampler(y, lb)
    else:
        if data_source == 'gmm_1d':
            x_sampler, y_sampler = make_gmm_1d()
        elif data_source == 'gmm_2d':
            x_sampler, y_sampler = make_gmm_2d()
        elif data_source == 'gmm_10d':
            x_sampler, y_sampler = make_gmm(10, 5)
        else:
            raise ValueError
        x, la, _ = x_sampler(n_samples)
        y, lb, _ = y_sampler(n_samples)
    return x, la, y, lb, x_sampler, y_sampler