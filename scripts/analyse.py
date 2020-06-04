import json
from os.path import join

import torch

import pandas as pd

from onlikhorn.dataset import get_output_dir

exp_dir = join(get_output_dir(), 'online', '32')

with open(join(exp_dir, 'config.json'), 'r') as f:
    config = json.load(f)

result = torch.load(join(exp_dir, 'artifacts', 'results.pkl'))
df = pd.DataFrame(result['trace'])
print(df)