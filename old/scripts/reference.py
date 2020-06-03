import torch

from onlikhorn.data import make_gmm_1d
from geomloss import SamplesLoss

def from_numpy(*vectors, device='cpu'):
    return [torch.from_numpy(vector).to(device) for vector in vectors]

(x, loga), (y, logb) = make_gmm_1d(5000, 5000)
x, loga, y, logb = from_numpy(x, loga, y, logb, device='cpu')
loss = SamplesLoss(potentials=True, scaling=0.99)
f, g = loss.forward(torch.exp(loga), x, torch.exp(logb), y)
print(f, g)