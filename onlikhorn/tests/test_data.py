import pytest
import torch

from onlikhorn.data import Subsampler
from onlikhorn.dataset import make_data


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_gmm_sampler(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('No cuda')
    x, la, y, lb, x_sampler, y_sampler = make_data('gmm_1d', 100)
    x_sampler = x_sampler.to(device)
    x, la, xidx = x_sampler(10)
    assert(x.device.type == device)
    assert(la.device.type == device)


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_subsampler(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('No cuda')
    x, la, y, lb, x_sampler, y_sampler = make_data('gmm_1d', 100)
    x_sampler = Subsampler(x, la).to(device)
    x, la, xidx = x_sampler(10)
    assert(x.device.type == device)
    assert(la.device.type == device)
    assert isinstance(xidx, list)
