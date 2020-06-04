#!/bin/env python
import os
import subprocess
from os.path import join

from sklearn.model_selection import ParameterGrid

from onlikhorn.dataset import get_output_dir
import numpy as np

SLURM_TEMPLATE = """#!/bin/env bash
#SBATCH --job-name=online
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --hint=nomultithread

#SBATCH --time=2:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out

#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH -A glp@gpu
#SBATCH --array=0-{nb_jobs}

source load_modules
cfg=$(cat {config_file} | head -${{SLURM_ARRAY_TASK_ID}} | tail -1) 

python {project_root}/online.py with ${{cfg}}

"""


def create_one(config, index=0):
    job_folder = join(get_output_dir(), 'online', 'jobs')
    project_root = os.path.abspath(os.getcwd())
    if not os.path.exists(job_folder):
        os.makedirs(job_folder)
    filename = join(job_folder, f'run_{index}.slurm')
    config_str = ' '.join(f'{key}={value}' for key, value in config.items())
    with open(filename, 'w+') as f:
        f.write(SLURM_TEMPLATE.format(project_root=project_root,
                                      filename=filename, config_str=config_str))
    return filename


n_seeds = 1
seeds = list(range(n_seeds))
epsilons = [1e-4, 1]
data_sources = ['dragon', 'gmm_1d', 'gmm_2d', 'gmm_10d']
reference = ParameterGrid({'data_source': data_sources,
                           'seed': seeds,
                           'epsilon': epsilons,
                           'method': ['sinkhorn', 'sinkhorn_precompute'],
                           })
subsampled = ParameterGrid({'data_source': data_sources,
                            'batch_size': [100, 1000],
                            'seed': seeds,
                            'epsilon': epsilons,
                            'method': ['subsampled'],
                            })
random = ParameterGrid({'data_source': data_sources,
                        'batch_size': [100, 1000],
                        'seed': seeds,
                        'epsilon': epsilons,
                        'method': ['random'],
                        })
online_non_convergent = ParameterGrid({'data_source': data_sources,
                                       'batch_size': [100],
                                       'seed': seeds,
                                       'epsilon': epsilons,
                                       'method': ['online', 'online_as_warmup'],
                                       'refit': [False, True],
                                       'batch_exp': [0],
                                       'lr_exp': [0, .5, 1]})
online = ParameterGrid({'data_source': data_sources,
                        'batch_size': [100],
                        'seed': seeds,
                        'epsilon': epsilons,
                        'method': ['online', 'online_as_warmup'],
                        'refit': [True, False],
                        'batch_exp': [0, .5, 1],
                        'lr_exp': ['auto']})

grids = [reference, subsampled, random, online_non_convergent, online]

job_folder = join(get_output_dir(), 'online', 'jobs')
project_root = os.path.abspath(os.getcwd())
if not os.path.exists(job_folder):
    os.makedirs(job_folder)

config_str = ''
nb_jobs = 0
for grid in grids:
    for index, config in enumerate(grid):
        config_str += ' '.join(f'{key}={value}' for key, value in config.items()) + '\n'
        nb_jobs += 1

print(nb_jobs)
config_file = join(job_folder, f'config.txt')
with open(config_file, 'w+') as f:
    f.write(config_str)

filename = join(job_folder, f'run.slurm')
with open(filename, 'w+') as f:
    f.write(SLURM_TEMPLATE.format(project_root=project_root,
                                  config_str=config_str, nb_jobs=nb_jobs, config_file=config_file))
subprocess.check_output("sbatch {}".format(filename), shell=True)