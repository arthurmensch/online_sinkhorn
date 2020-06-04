import numpy as np
import torch
from sklearn.utils import shuffle


class Subsampler:
    def __init__(self, positions: torch.tensor, weights: torch.tensor, cycle=True):
        self.positions = positions
        self.cycle = cycle
        self.weights = weights

        if self.cycle:
            self.idx = np.arange(len(self.positions), dtype=np.long)
            self.positions, self.weights, self.idx = shuffle(self.positions, self.weights, self.idx)
        self.cursor = 0

    @property
    def device(self):
        return self.positions.device

    @property
    def dimension(self):
        return self.positions.shape[1]

    def to(self, device):
        self.positions = self.positions.to(device)
        self.weights = self.positions.to(device)
        return self

    def __call__(self, n):
        if n >= len(self.positions):
            return self.positions, self.weights, self.idx.tolist()
        if not self.cycle:
            idx = np.random.permutation(len(self.positions))[:n].tolist()
            weights = self.weights[idx]
            positions = self.positions[idx]
        else:
            new_cursor = self.cursor + n
            if new_cursor >= len(self.positions):
                idx = self.idx[self.cursor:].copy()
                positions = self.positions[self.cursor:].clone()
                weights = self.weights[self.cursor:].clone()
                self.positions, self.weights, self.idx = shuffle(self.positions, self.weights, self.idx)
                reset_cursor = new_cursor - len(self.positions)
                idx = np.concatenate([idx, self.idx[:reset_cursor]], axis=0)
                positions = torch.cat([positions, self.positions[:reset_cursor]], dim=0)
                weights = torch.cat([weights, self.weights[:reset_cursor]], dim=0)
                self.cursor = reset_cursor
            else:
                idx = self.idx[self.cursor:new_cursor].copy()
                positions = self.positions[self.cursor:new_cursor].clone()
                weights = self.weights[self.cursor:new_cursor].clone()
                self.cursor = new_cursor
        weights -= torch.logsumexp(weights, dim=0)
        return positions, weights, idx.tolist()

