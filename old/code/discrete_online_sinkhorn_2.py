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

results, parameters = joblib.load(expanduser('~/output/online_sinkhorn/high_dim.pkl'))
print(parameters)