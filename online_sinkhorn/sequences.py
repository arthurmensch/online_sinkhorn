import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from numba import jit
from sklearn.model_selection import ParameterGrid


@jit
def f(mu, alpha, beta, max_iter):
    records = []
    delta = 1.
    n = 0
    computation = 0.
    mem = 0
    for i in range(1, max_iter):
        i = float(i)
        delta = (1 - mu * np.exp(- np.log(i) * alpha)) * delta + np.exp(- np.log(i) * (alpha + beta))
        b = np.exp(np.log(i) * 2 * beta)
        mem += b
        computation += b * n
        n += b
        records.append((delta, mem, computation))
    return records


def grid():
    max_iter = int(1e5)

    mu = 1e-3
    params = ParameterGrid(dict(alpha=[0.], beta=np.linspace(0.0, 2, 9)))
    records = Parallel(n_jobs=4)(delayed(f)(mu, p["alpha"], p["beta"], max_iter) for p in params)

    df = []
    for record, p in zip(records, params):
        for (delta, mem, computation) in record:
            df.append(dict(alpha=p['alpha'], beta=p['beta'], delta=delta, mem=mem, computation=computation))

    records = pd.DataFrame(df)

    import seaborn as sns
    facet = sns.FacetGrid(col='alpha', data=records, hue='beta', height=2)
    facet.map(plt.plot, 'computation', 'delta')
    facet.axes[0, 0].set_yscale('log')
    facet.axes[0, 0].set_xscale('log')
    facet.add_legend()

    # facet = sns.FacetGrid(col='beta', data=records, hue='alpha', height=2)
    # facet.map(plt.plot, 'computation', 'delta')
    # facet.axes[0, 0].set_yscale('log')
    # facet.axes[0, 0].set_xscale('log')
    # facet.add_legend()
    plt.show()


if __name__ == '__main__':
    grid()