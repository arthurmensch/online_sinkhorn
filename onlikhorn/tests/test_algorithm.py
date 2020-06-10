import pytest
import torch

from onlikhorn.algorithm import sinkhorn, subsampled_sinkhorn, online_sinkhorn, random_sinkhorn
from onlikhorn.dataset import make_data

import numpy as np

from torch.testing import assert_allclose

from onlikhorn.gaussian import sinkhorn_gaussian


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
@pytest.mark.parametrize("save_trace", [True, False, 'ref'])
@pytest.mark.parametrize("algorithm", ['sinkhorn', 'subsampled_sinkhorn', 'random_sinkhorn'])
def test_algorithm(save_trace, algorithm, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('No cuda')
    x, la, y, lb, x_sampler, y_sampler = make_data('gmm_1d', 100)
    x, la, y, lb, x_sampler, y_sampler = (x.to(device), la.to(device),
                                          y.to(device), lb.to(device), x_sampler.to(device), y_sampler.to(device))
    if save_trace == 'ref':
        F, G = sinkhorn(x=x, la=la, y=y, lb=lb, n_iter=10)
        ref = {'train': (F(x), x, G(y), y), 'test': (F(x), x, G(y), y)}
        save_trace = True
    else:
        if save_trace and algorithm == 'random_sinkhorn':
            pytest.skip('Unsupported configuration')
        ref = None
    funcs = {'sinkhorn': sinkhorn, 'subsampled_sinkhorn': subsampled_sinkhorn,
             'random_sinkhorn': random_sinkhorn}
    func = funcs[algorithm]
    res = func(x=x, la=la, y=y, lb=lb, save_trace=save_trace, ref=ref, n_iter=10)
    if save_trace:
        F, G, trace = res
        assert (len(trace) == 10)
    else:
        F, G = res
    assert not np.isnan(F(x).sum().item())
    assert not np.isnan(G(y).sum().item())


def test_precompute_C():
    x, la, y, lb, x_sampler, y_sampler = make_data('gmm_1d', 100)
    F, G = sinkhorn(x=x, la=la, y=y, lb=lb, n_iter=10, precompute_C=False)
    Fp, Gp = sinkhorn(x=x, la=la, y=y, lb=lb, n_iter=10, precompute_C=True)
    assert_allclose(Fp(x), F(x))
    assert_allclose(Gp(y), G(y))


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
@pytest.mark.parametrize("save_trace", [True, False])
@pytest.mark.parametrize("force_full", [True, False])
@pytest.mark.parametrize("batch_size", ['constant', 'growing'])
@pytest.mark.parametrize("lr", ['constant', 'decreasing'])
@pytest.mark.parametrize("refit", [True, False])
@pytest.mark.parametrize("use_finite", [True, False])
def test_online_sinkhorn(save_trace, refit, force_full, use_finite, lr, batch_size, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('No cuda')
    x, la, y, lb, x_sampler, y_sampler = make_data('gmm_1d', 100)
    x, la, y, lb, x_sampler, y_sampler = (x.to(device), la.to(device),
                                          y.to(device), lb.to(device), x_sampler.to(device), y_sampler.to(device))
    if save_trace:
        F, G = sinkhorn(x=x, la=la, y=y, lb=lb, n_iter=10)
        ref = {'train': (F(x), x, G(y), y), 'test': (F(x), x, G(y), y)}
    else:
        ref = None
    if use_finite:
        input = dict(x=x, la=la, y=y, lb=lb)
    else:
        input = dict(x_sampler=x_sampler, y_sampler=y_sampler)
    n_iter = 15
    if batch_size == 'constant':
        batch_sizes = 10
    else:
        batch_sizes = [10 * (i + 1) for i in range(n_iter)]

    if lr == 'constant':
        lrs = 1
    else:
        lrs = [1 / np.sqrt(i + 1) for i in range(n_iter)]
    res = online_sinkhorn(**input,
                          save_trace=save_trace, force_full=force_full,
                          use_finite=use_finite, ref=ref, n_iter=n_iter,
                          batch_sizes=batch_sizes, refit=refit,
                          lrs=lrs)
    if save_trace:
        F, G, trace = res
        assert (len(trace) == 15)
    else:
        F, G = res
    assert not np.isnan(F(x).sum().item())
    assert not np.isnan(G(y).sum().item())


@pytest.mark.parametrize("device", ['cpu', 'cuda'])
def test_gaussian_algorithm(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('No cuda')
    x, la, y, lb, x_sampler, y_sampler = make_data('gaussian_2d', 100)
    x, la, y, lb, x_sampler, y_sampler = (x.to(device), la.to(device),
                                          y.to(device), lb.to(device), x_sampler.to(device), y_sampler.to(device))
    F, G = sinkhorn_gaussian(x_sampler=x_sampler, y_sampler=y_sampler)
    assert not np.isnan(F(x).sum().item())
    assert not np.isnan(G(y).sum().item())