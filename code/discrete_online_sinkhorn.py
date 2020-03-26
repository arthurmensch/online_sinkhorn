import itertools
import math
import os
from collections import defaultdict
from os.path import expanduser

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from joblib import Memory, delayed, Parallel
from matplotlib import rc

# matplotlib.rcParams['backend'] = 'pdf'
# rc('text', usetex=True)
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state, gen_batches


class Sampler():
    def __init__(self, mean: torch.tensor, cov: torch.tensor, p: torch.tensor):
        k, d = mean.shape
        k, d, d = cov.shape
        k = p.shape
        self.mean = mean
        self.cov = cov
        self.icov = torch.cat([torch.inverse(cov)[None, :, :] for cov in self.cov], dim=0)
        det = torch.tensor([torch.det(cov) for cov in self.cov])
        self.norm = torch.sqrt((2 * math.pi) ** d * det)
        self.p = p

    def __call__(self, n):
        k, d = self.mean.shape
        indices = np.random.choice(k, n, p=self.p.numpy())
        pos = np.zeros((n, d), dtype=np.float32)
        for i in range(k):
            mask = indices == i
            size = mask.sum()
            pos[mask] = np.random.multivariate_normal(self.mean[i], self.cov[i], size=size)
        logweight = np.full_like(pos[:, 0], fill_value=-math.log(n))
        return torch.from_numpy(pos), torch.from_numpy(logweight)

    def log_prob(self, x):
        # b, d = x.shape
        diff = x[:, None, :] - self.mean[None, :]  # b, k, d
        return torch.sum(self.p[None, :] * torch.exp(-torch.einsum('bkd,kde,bke->bk',
                                                                   [diff, self.icov, diff]) / 2) / self.norm, dim=1)


def sample_from_finite(x, m, random_state=None, full_after_one=False, replacement=False):
    random_state = check_random_state(random_state)
    n = x.shape[0]
    first_iter = True

    while True:
        if not first_iter and full_after_one:
            yield np.arange(x.shape[0]), torch.full((n,), fill_value=-math.log(n))
        else:
            if replacement:
                if n == m:
                    indices = np.arange(n)
                else:
                    indices = random_state.permutation(n)[:m]
                loga = torch.full((m,), fill_value=-math.log(m))
                yield indices, loga
            else:
                indices = random_state.permutation(n)
                for batches in gen_batches(x.shape[0], m):
                    these_indices = indices[batches]
                    this_m = len(these_indices)
                    loga = torch.full((this_m,), fill_value=-math.log(this_m))
                    yield these_indices, loga
                first_iter = False


def var_norm(x):
    return x.max() - x.min()


def compute_distance(x, y):
    return (torch.sum((x[:, None, :] ** 2 + y[None, :, :] ** 2 - 2 * x[:, None, :] * y[None, :, :]), dim=2)) / 2


def evaluate_potential(log_pot: torch.tensor, pos: torch.tensor, x: torch.tensor, eps):
    distance = compute_distance(x, pos)
    return - eps * torch.logsumexp((- distance + log_pot[None, :]) / eps, dim=1)


def evaluate_potential_finite(log_pot: torch.tensor, idx: torch.tensor, distance, eps):
    return - eps * torch.logsumexp((- distance[idx] + log_pot[None, :]) / eps, dim=1)


def online_sinkhorn(x, y, eps, n_iter=100, random_state=None, resample=True, ieta=1.,
                    samplingx=None, samplingy=None,
                    eta=1., alternate=False, replacement=True):
    random_state = check_random_state(random_state)
    torch.manual_seed(random_state.randint(100000))

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    n = x.shape[0]
    m = y.shape[0]
    p = torch.full((n,), fill_value=-float('inf'), dtype=x.dtype)
    q = torch.full((m,), fill_value=-float('inf'), dtype=x.dtype)

    avg_p = p.clone()
    avg_q = q.clone()

    distance = compute_distance(x, y)
    errors = defaultdict(list)
    u_eta = not isinstance(eta, float)
    u_ieta = not isinstance(ieta, float)

    if samplingy is None:
        if samplingx is not None:
            samplingy = samplingx
        else:
            samplingy = y.shape[0]
    if samplingx is None:
        samplingx = x.shape[0]

    y_sampler = sample_from_finite(y, samplingy, random_state=random_state,
                                   replacement=replacement)
    x_sampler = sample_from_finite(x, samplingx, random_state=random_state,
                                   replacement=replacement)
    y_idx, logb = next(y_sampler)
    x_idx, loga = next(x_sampler)

    computations = 1
    true_computations = 1
    for i in range(0, n_iter):
        if i % 1 == 0:
            if i == 0:
                ff = torch.zeros(n, dtype=x.dtype)
                gg = torch.zeros(m, dtype=x.dtype)
            else:
                ff = evaluate_potential_finite(avg_q, slice(None), distance, eps)
                gg = evaluate_potential_finite(avg_p, slice(None), distance.transpose(0, 1), eps)
            fff = evaluate_potential_finite(gg + math.log(1 / m) * eps, slice(None), distance, eps)
            ggg = evaluate_potential_finite(ff + math.log(1 / n) * eps, slice(None), distance.transpose(0, 1),
                                            eps)
            ff = (ff + fff) / 2
            gg = (gg + ggg) / 2
            w = (ff.mean() + gg.mean())
            errors['ff'].append(ff.numpy())
            errors['gg'].append(gg.numpy())
            errors['w'].append(w.item())
            errors['computation'].append(computations)
            errors['true_computation'].append(true_computations)
            errors['iter'].append(i)

        # Update f
        if resample:
            y_idx, logb = next(y_sampler)
            x_idx, loga = next(x_sampler)
        if i > 0:
            g = evaluate_potential_finite(p, y_idx, distance.transpose(0, 1), eps)
            if not alternate:
                f = evaluate_potential_finite(q, x_idx, distance, eps)
        else:
            g = torch.zeros(samplingy, dtype=x.dtype)
            if not alternate:
                f = torch.zeros(samplingx, dtype=x.dtype)

        if u_ieta:
            if ieta == '1/t':
                ieta_ = torch.tensor(1 / (i + 1))
            elif ieta == '1/sqrt(t)':
                ieta_ = torch.tensor(1 / math.sqrt(i + 1))
            else:
                raise ValueError
        else:
            ieta_ = torch.tensor(ieta)
        if u_eta:
            if eta == '1/t':
                eta_ = torch.tensor(1 / (i + 1))
            elif eta == '1/sqrt(t)':
                eta_ = torch.tensor(1 / math.sqrt(i + 1))
            else:
                raise ValueError
        else:
            eta_ = torch.tensor(eta)

        q += eps * torch.log(- ieta_ + 1)
        update = eps * torch.log(ieta_) + logb * eps + g
        q[y_idx] = eps * torch.logsumexp(torch.cat([q[y_idx][None, :], update[None, :]], dim=0) / eps, dim=0)
        q += evaluate_potential_finite(q, [0], distance, eps)
        if alternate:
            f = evaluate_potential_finite(q, x_idx, distance, eps)
        p += eps * torch.log(- ieta_ + 1)
        update = eps * torch.log(ieta_) + loga * eps + f
        p[x_idx] = eps * torch.logsumexp(torch.cat([p[x_idx][None, :], update[None, :]], dim=0) / eps, dim=0)
        p += evaluate_potential_finite(p, [0], distance.transpose(0, 1), eps)
        avg_q += eps * torch.log(- eta_ + 1)
        update = eps * torch.log(eta_) + q[y_idx]
        avg_q[y_idx] = eps * torch.logsumexp(torch.cat([update[None, :], avg_q[y_idx][None, :]]) / eps, dim=0)

        avg_p += eps * torch.log(- eta_ + 1)
        update = eps * torch.log(eta_) + p[x_idx]
        avg_p[x_idx] = eps * torch.logsumexp(torch.cat([update[None, :], avg_p[x_idx][None, :]]) / eps, dim=0)

        computations += len(y_idx) * min(len(y_idx) * (i + 1), n)
        true_computations += len(y_idx) * len(y_idx) * (i + 1)
    return (y, avg_q), (x, avg_p), errors


def sinkhorn(x, y, eps, n_iter=1000, samplingx=None, samplingy=None):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    loga = -math.log(len(x))
    logb = -math.log(len(y))

    m = y.shape[0]

    distance = compute_distance(x, y)

    g = torch.zeros((m,), dtype=torch.float64)
    f = evaluate_potential_finite(g + eps * logb, slice(None), distance, eps)
    errors = defaultdict(list)
    for i in range(n_iter):
        gg = evaluate_potential_finite(f + eps * loga, slice(None), distance.transpose(0, 1), eps)
        g_diff = g - gg
        g = gg
        f = evaluate_potential_finite(g + eps * logb, slice(None), distance, eps)
        w = (f.mean() + g.mean())
        tol = g_diff.max() - g_diff.min()
        errors['ff'].append(f.numpy())
        errors['gg'].append(g.numpy())
        errors['w'].append(w.item())
        errors['iter'].append(i)
    return (y, g + eps * logb), (x, f + eps * loga), errors


def main():
    n = 1000
    eps = 1e-1

    x_sampler = Sampler(mean=torch.tensor([[1.], [2], [3]]), cov=torch.tensor([[[.1]], [[.1]], [[.1]]]),
                        p=torch.ones(3) / 3)
    y_sampler = Sampler(mean=torch.tensor([[0.], [3], [5]]), cov=torch.tensor([[[.1]], [[.1]], [[.4]]]),
                        p=torch.ones(3) / 3)

    y, logb = y_sampler(n)
    x, loga = x_sampler(n)
    x = np.random.randn(n, 8) + 3
    y = np.random.randn(n, 8)

    mem = Memory(location=expanduser('~/cache'))
    mem = Memory(location=None)
    parameters = ParameterGrid(
        dict(samplingx=[20, 50, 100], eta=['1/sqrt(t)', 1., '1/t'], ieta=['1/sqrt(t)', 1., '1/t'], resample=[True],
             replacement=[True, False], random_state=[1]))
    baseline_parameters = ParameterGrid(dict(m=[50, 100, 1000], eta=[1.], ieta=[1.],
                                             resample=[False], alternate=[True], replacement=[False],
                                             ))
    parameters = list(itertools.chain(parameters, baseline_parameters))
    results = Parallel(n_jobs=18)(
        delayed(mem.cache(online_sinkhorn))(x, y, eps=eps, n_iter=int(1e3), **parameter) for parameter in parameters)
    if not os.path.exists(expanduser('~/output/online_sinkhorn')):
        os.makedirs(expanduser('~/output/online_sinkhorn'))
    joblib.dump((results, parameters), expanduser(f'~/output/online_sinkhorn/smally.pkl'))


def plot():
    results, parameters = joblib.load(expanduser('~/output/online_sinkhorn/smally.pkl'))
    ((x, f), (y, g), errorref) = results[-1]
    res = []
    for ((x, f), (y, g), errors), parameter in zip(results, parameters):
        errors['potential'] = [0 for _ in range(len(errors['ff']))]
        for i in range(len(errors['ff'])):
            errors['potential'][i] = var_norm(errors['ff'][i] - errorref['ff'][-1]) + var_norm(
                errors['gg'][i] - errorref['gg'][-1])
            errors['w'][i] = abs(errors['w'][i] - errorref['w'][-1])
            if parameter['resample']:
                index = f'online_{parameter["m"]}_{parameter["replacement"]}'
                if parameter["m"] == 50:
                    index = '0' + index
            else:
                index = f'sinkhorn_{parameter["m"]}_{parameter["replacement"]}'
            res.append(dict(potential=errors['potential'][i], w=errors['w'][i],
                            index=index,
                            iter=errors['computation'][i], alg='online', **parameter))
    res = pd.DataFrame(res)
    res['potential'] = np.log(res['potential'])
    res['w'] = np.log(res['w'])
    res = res.groupby(by=['index', 'iter', 'ieta', 'eta']).aggregate(['mean', 'std'])
    for metric in ['potential', 'w']:
        res[f"{metric}+std"] = np.exp(res[(metric, "mean")] + res[(metric, "std")])
        res[f"{metric}-std"] = np.exp(res[(metric, "mean")] - res[(metric, "std")])
        res[f'{metric}_mean'] = np.exp(res[(metric, 'mean')])

    res = res.reset_index()
    for metric in ['potential', 'w']:
        grid = sns.FacetGrid(data=res, row='ieta', col='eta', hue='index')
        grid.map(plt.plot, "iter", f"{metric}_mean")
        # grid.map(plt.fill_between, "iter", f"{metric}-std", f"{metric}+std")
        for ax in grid.axes.ravel():
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.set_ylim([1e-6, 1])
        grid.add_legend()
        plt.show()


def plot_early_compute():
    results, parameters = joblib.load(expanduser('~/output/online_sinkhorn/full_after_one_1.pkl'))
    ((x, f), (y, g), errorref) = results[-1]
    res = []
    for ((x, f), (y, g), errors), parameter in zip(results, parameters):
        errors['potential'] = [0 for _ in range(len(errors['ff']))]
        for i in range(len(errors['ff'])):
            errors['potential'][i] = var_norm(errors['ff'][i] - errorref['ff'][-1]) + var_norm(
                errors['gg'][i] - errorref['gg'][-1])
            errors['w'][i] = abs(errors['w'][i] - errorref['w'][-1])
            if parameter['resample']:
                index = f'online_{parameter["m"]}_{parameter["replacement"]}'
                if parameter["m"] == 50:
                    index = '0' + index
            else:
                index = f'sinkhorn_{parameter["m"]}_{parameter["replacement"]}'
            if errors['computation'][i] == 1:
                errors['computation'][i] = 100
            res.append(dict(potential=errors['potential'][i], w=errors['w'][i],
                            index=index,
                            iter=errors['computation'][i], alg='online', **parameter))
    res = pd.DataFrame(res)
    res = res.query("replacement == False")
    res = res.copy()
    res['potential'] = np.log(res['potential'])
    res['w'] = np.log(res['w'])
    res = res.groupby(by=['index', 'iter']).aggregate(['mean', 'std'])

    for metric in ['potential', 'w']:
        res[f"{metric}+std"] = np.exp(res[(metric, "mean")] + res[(metric, "std")])
        res[f"{metric}-std"] = np.exp(res[(metric, "mean")] - res[(metric, "std")])
        res[f'{metric}_mean'] = np.exp(res[(metric, 'mean')])

    res = res.reset_index(['iter'])
    fig, axes = plt.subplots(1, 2, figsize=(4, 2.4), constrained_layout=False)
    fig.subplots_adjust(right=0.97, left=0.17, top=0.95, bottom=0.4, wspace=0.5)
    labels = {'0online_50_False': 'Online Sinkhorn (5\% sampling)',
              'online_100_False': 'Online Sinkhorn (10\% sampling)',
              'sinkhorn_1000_False': 'Sinkhorn'}
    colors = sns.color_palette("Paired")
    colors = {'0online_50_False': colors[5], 'online_100_False': colors[7], 'sinkhorn_1000_False': colors[3]}
    for ax, metric in zip(axes, ['potential', 'w']):
        for index, data in res.groupby('index'):
            ax.plot(data['iter'], data[f"{metric}_mean"], label=labels[index],
                    linestyle='--' if 'sinkhorn' in index else '-',
                    color=colors[index]
                    )
            ax.fill_between(data['iter'], data[f"{metric}-std"], data[f"{metric}+std"],
                            alpha=0.2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([1e2, 1e4, 1e6, 1e8])
        ax.set_xticklabels(['$1$', '$10^4$', 'Full $\\hat C$', '$10^8$'])
        ax.vlines([1e6], 1e-10, 1e3, color='0.3', linewidth=1)
    axes[0].set_ylim([1e-2, 1e1])
    axes[0].set_ylabel(r'$||f_t,g_t - f^\star,g^\star||_{\textrm{var}}$')
    axes[1].set_ylabel('$| \mathcal{W}_t - \mathcal{W} |$')
    axes[1].set_ylim([1e-6, 1])
    axes[0].annotate('Computat°', xy=(0, 0), xycoords="axes fraction", xytext=(-45, -7),
                     textcoords="offset points", ha='left', va='top')
    axes[0].legend(frameon=False, loc='upper left', bbox_to_anchor=(-0.5, -0.13), ncol=2, columnspacing=0.5,
                   )
    sns.despine(fig)
    plt.savefig('early_compute.pdf')
    plt.show()


def plot_comparison():
    results, parameters = joblib.load(expanduser('~/output/online_sinkhorn/very_big.pkl'))
    ((x, f), (y, g), errorref) = results[-1]
    res = []
    for ((x, f), (y, g), errors), parameter in zip(results, parameters):
        if parameter['m'] in [50, 1000]:
            if parameter['ieta'] == 1 and parameter['eta'] == 1 and parameter['resample'] and not parameter[
                'replacement']:
                index = f'random{parameter["m"]}'
                errors['computation'] = np.arange(len(errors['computation'])) * parameter["m"] ** 2
            elif parameter['ieta'] == '1/sqrt(t)' and parameter['eta'] == '1/sqrt(t)' and parameter['resample'] and \
                    parameter['replacement']:
                index = f'avgonline{parameter["m"]}'
            elif parameter['ieta'] == '1/sqrt(t)' and parameter['eta'] == 1 and parameter['resample'] and parameter[
                'replacement']:
                index = f'online{parameter["m"]}'
            elif parameter['ieta'] == 1 and parameter['eta'] == '1/t' and parameter['resample'] \
                    and not parameter['replacement']:
                index = f'avgrandom{parameter["m"]}'
                errors['computation'] = np.arange(len(errors['computation'])) * parameter["m"] ** 2
            elif not parameter['resample']:
                index = f'sinkhorn{parameter["m"]}'
            else:
                continue
        else:
            continue
        errors['potential'] = [0 for _ in range(len(errors['ff']))]
        for i in range(len(errors['ff'])):
            errors['potential'][i] = var_norm(errors['ff'][i] - errorref['ff'][-1]) + var_norm(
                errors['gg'][i] - errorref['gg'][-1])
            errors['w'][i] = abs(errors['w'][i] - errorref['w'][-1])
            if errors['computation'][i] == 1:
                errors['computation'][i] = 100
            res.append(dict(potential=errors['potential'][i], w=errors['w'][i],
                            index=index,
                            iter=errors['computation'][i], alg='online', **parameter))
    res = pd.DataFrame(res)
    res = res.copy()
    res['potential'] = np.log(res['potential'])
    res['w'] = np.log(res['w'])
    res = res.groupby(by=['index', 'iter']).aggregate(['mean', 'std'])

    for metric in ['potential', 'w']:
        res[f"{metric}+std"] = np.exp(res[(metric, "mean")] + res[(metric, "std")])
        res[f"{metric}-std"] = np.exp(res[(metric, "mean")] - res[(metric, "std")])
        res[f'{metric}_mean'] = np.exp(res[(metric, 'mean')])

    res = res.reset_index(['iter'])
    fig, axes = plt.subplots(1, 2, figsize=(4, 2.4), constrained_layout=False)
    fig.subplots_adjust(right=0.97, left=0.17, top=0.95, bottom=0.4, wspace=0.5)
    labels = {'sinkhorn1000': 'Sinkhorn', 'random50': 'Random Sinkhorn (5\%)',
              'online50': 'Online Sinkhorn (5\%)', 'sinkhorn50': 'Sinkhorn (5\%)',
              'avgrandom50': 'Avg random Sinkhorn  (5\%)',
              'avgonline50': 'Avg online Sinkhorn  (5\%)'}
    zindexs = {'sinkhorn1000': 0, 'random50': 2, 'avgonline50': 5,
               'online50': 3, 'sinkhorn50': 1, 'avgrandom50': 4}
    linestyles = {'sinkhorn1000': '--', 'random50': '-',
                  'online50': '-', 'sinkhorn50': '--', 'avgrandom50': '-',
                  'avgonline50': '-'}
    colors = sns.color_palette("Paired")
    indices = ['avgonline50', 'online50', 'avgrandom50', 'random50', 'sinkhorn50', 'sinkhorn1000']
    colors = {'sinkhorn1000': colors[3], 'sinkhorn50': colors[9], 'random50': colors[0], 'avgrandom50': colors[1],
              'online50': colors[4], 'avgonline50': colors[5]}
    for ax, metric in zip(axes, ['potential', 'w']):
        for index in indices:
            data = res.loc[index]
            ax.plot(data['iter'], data[f"{metric}_mean"], label=labels[index],
                    color=colors[index], zorder=zindexs[index],
                    linestyle=linestyles[index])
            ax.fill_between(data['iter'], data[f"{metric}-std"], data[f"{metric}+std"],
                            color=colors[index],
                            alpha=0.2)
        ax.set_xscale('log')
        ax.set_yscale('log')
    axes[0].set_ylim([1e-2, 1e1])
    axes[0].set_ylabel(r'$||f_t,g_t - f^\star,g^\star||_{\textrm{var}}$')
    axes[1].set_ylabel('$| \mathcal{W}_t - \mathcal{W} |$')
    axes[1].set_ylim([1e-6, 1])
    axes[0].annotate('Computat°', xy=(0, 0), xycoords="axes fraction", xytext=(-45, -7),
                     textcoords="offset points", ha='left', va='top')
    axes[0].legend(frameon=False, loc='upper left', bbox_to_anchor=(-0.55, -0.13), ncol=2, columnspacing=0.5,
                   )
    sns.despine(fig)
    plt.savefig('comparison.pdf')
    plt.show()


if __name__ == '__main__':
    main()
    # plot()
    # plot_comparison()
    # plot_early_compute()
