from os.path import expanduser

import torch
from joblib import Memory


def torch_cached(func):
    mem = Memory(expanduser('~/cache'))

    def cached_func(*args, **kwargs):
        def false_func(*processed_args, **processed_kwargs):
            return func(*args, **kwargs)
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg = arg.cpu().numpy()
            processed_args.append(arg)
        processed_kwargs = {}
        for k, kwarg in kwargs.items():
            if isinstance(kwarg, torch.Tensor):
                kwarg = kwarg.cpu().numpy()
            processed_kwargs[k] = kwarg
        return mem.cache(false_func)(*processed_args, **processed_kwargs)

    return cached_func