import pandas as pd

speedups = pd.read_pickle('speedups.pkl')
print(speedups)
df = speedups.set_index(['ytype', 'data_source', 'epsilon']).unstack('epsilon')
print(df.loc['err'].round(1).to_latex())