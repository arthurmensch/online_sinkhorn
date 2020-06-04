from onlikhorn.algorithm import compute_distance
from onlikhorn.dataset import make_data
import numpy as np

x, la, y, lb, x_sampler, y_sampler = make_data('dragon', 10000)
x *= 2
y *= 2
D = compute_distance(x, y)
m = D.max()
# x /= np.sqrt(m)
# y /= np.sqrt(m)
# D = compute_distance(x, y)
# m = D.max()
print(m)