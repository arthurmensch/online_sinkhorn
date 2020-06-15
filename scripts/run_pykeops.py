import torch

for size in [1000, 10000, 20000]:
    x = torch.randn(size, 3, requires_grad=True).cuda()
    y = torch.randn(size, 3).cuda()

    # Turn our Tensors into KeOps symbolic variables:
    from pykeops.torch import LazyTensor

    x_i = LazyTensor(x[:, None, :])  # x_i.shape = (1e6, 1, 3)
    y_j = LazyTensor(y[None, :, :])  # y_j.shape = ( 1, 2e6,3)

    # We can now perform large-scale computations, without memory overflows:
    D_ij = ((x_i - y_j) ** 2).sum(dim=2)  # Symbolic (1e6,2e6,1) matrix of squared distances
    K_ij = (- D_ij).exp()  # Symbolic (1e6,2e6,1) Gaussian kernel matrix

    # And come back to vanilla PyTorch Tensors or NumPy arrays using
    # reduction operations such as .sum(), .logsumexp() or .argmin().
    # Here, the kernel density estimation   a_i = sum_j exp(-|x_i-y_j|^2)
    # is computed using a CUDA online map-reduce routine that has a linear
    # memory footprint and outperforms standard PyTorch implementations
    # by two orders of magnitude.
    a_i = K_ij.sum(dim=1)  # Genuine torch.cuda.FloatTensor, a_i.shape = (1e6, 1),
    g_x = torch.autograd.grad((a_i ** 2).sum(), [x])  # KeOps supports autograd!
    print(a_i.sum().item())
