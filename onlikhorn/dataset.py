import os
from os.path import join

import numpy as np
import urllib.request
import shutil

from plyfile import PlyData


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
    return (S / np.sum(S)), X


def create_sphere(n_samples=1000):
    """Creates a uniform sample on the unit sphere."""
    n_samples = int(n_samples)

    indices = np.arange(0, n_samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi);
    points = np.vstack((x, y, z)).T
    weights = np.ones(n_samples) / n_samples

    return weights, points


def get_data_dir():
    data_dir = os.path.expanduser('~/data/online_sinkorn')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir


def get_output_dir():
    output_dir = os.path.expanduser('~/output/online_sinkorn')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def get_dragon(data_dir=None):
    if data_dir is None:
        data_dir = get_data_dir()
    filename = join(data_dir, 'dragon_recon/dragon_vrip_res4.ply')
    if not os.path.exists(filename):
        urllib.request.urlretrieve(
            'http://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz',
            join(data_dir, 'dragon.tar.gz'))
        shutil.unpack_archive(join(data_dir, 'dragon.tar.gz'), data_dir)
    return load_ply_file(filename, offset=[-0.011, 0.109, -0.008], scale=.04)

