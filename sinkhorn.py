import torch


def c_transform(target_pos, src_pos, src_weight, src_potential):
    # target_pos.shape = (B, T, D)
    # str.shape = (B, T, D)