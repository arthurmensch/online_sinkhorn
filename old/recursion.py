import math

def f(a, eta):
    return 1 / (eta + (1 - eta) / (a + 1)) - 1


a = 10
mean = 0
sum_eta = 0
for i in range(2, 10000):
    eta = 1 / math.sqrt(i)
    a = f(a, eta)
    mean += a * eta
    sum_eta += eta
    print(a, mean / sum_eta)
