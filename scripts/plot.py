from os.path import expanduser, join
output_dir = expanduser('~/output/online_sinkhorn/online_grid9')

import pandas as pd

df_ = pd.read_pickle(join(output_dir, 'all.pkl'))

df_['data_source'].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns
df = df_.copy()
df = df.query('method != "online"')
df  = df.query('method != "random"')
df  = df.query('method != "sinkhorn"')
# df = df.query('lr_exp == "auto" | method != "online_as_warmup"')
df = df.query('(refit == False & batch_exp == .5 & lr_exp == 0) | method != "online_as_warmup"')
df = df.query('method != "subsampled"')


df = df.query('data_source in ["gmm_2d", "gmm_10d", "dragon", ]')
df = df.query('epsilon == 1e-3')

pk = ['data_source', 'epsilon', 'method', 'refit', 'batch_exp', 'lr_exp', 'batch_size', 'n_iter']
df = df.groupby(by=pk).agg(['mean', 'std']).reset_index('n_iter')

plot_err = True
NAMES = {'dragon': 'Stanford 3D', 'gmm_10d': '10D GMM', 'gmm_2d': '2D GMM'}
fig, axes = plt.subplots(1, 3, figsize=(6, 1.8))
for i, ((data_source, epsilon), df2) in enumerate(df.groupby(['data_source', 'epsilon'])):
    for index, df3 in df2.groupby(['method', 'refit', 'batch_exp', 'lr_exp', 'batch_size']):
        n_calls = df3['n_calls']
        train = df3['ref_err_train']
        test = df3['ref_err_test']
        err = df3['fixed_err']
        if plot_err:
            train = err
        if index[0] == 'sinkhorn_precompute':
            label = 'Standard\nSinkhorn'
        else:
            label = 'Online Sinkhorn\nwarmup'
        axes[i].plot(n_calls['mean'], train['mean'], label=label)
        if index[0] != 'sinkhorn_precompute':
            axes[i].fill_between(n_calls['mean'], train['mean'] - train['std'], train['mean'] + train['std'],
                               alpha=0.2)
    axes[i].annotate(NAMES[data_source], xy=(.5, .8), xycoords="axes fraction",
                     ha='center', va='bottom')
axes[0].annotate('Computat.', xy=(-.3, -.13), xycoords="axes fraction",
                 ha='center', va='bottom')
axes[2].legend(loc='center left', frameon=False, bbox_to_anchor=(.7, 0.5), ncol=1)
for ax in axes:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
if plot_err:
    axes[0].set_ylabel('||T(f)-g|| +|| T(g)-f||')
else:
    axes[0].set_ylabel('|| f - f*|| + || g -g*||')
sns.despine(fig)
fig.subplots_adjust(right=0.75)
fig.savefig('online+full.pdf')
plt.show()