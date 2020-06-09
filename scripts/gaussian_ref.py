import numpy as np
from scipy.special import logsumexp
from scipy.linalg import sqrtm, inv
from sklearn.linear_model import LinearRegression


def sinkhorn(a, b, C, sigma, gamma=None, maxiter=1000, tol=1e-5):
    dim_a, dim_b = C.shape
    epsilon = 2 * sigma ** 2
    C = C / epsilon
    v = np.zeros(dim_b)
    logb = np.log(b)
    loga = np.log(a)
    if gamma is None:
        tau = 1
    else:
        gamma = epsilon / (gamma + epsilon)
    # dual potentials are considered divided by eps
    for ii in range(maxiter):
        vold = v.copy()
        u = - tau * epsilon * logsumexp(- C + v[None, :] / epsilon +
                                        logb[None, :], axis=1)
        v = - tau * epsilon * logsumexp(- C + u[:, None] / epsilon +
                                        loga[:, None], axis=0)
        err = abs(v - vold).max()
        err /= max(1., abs(v).max())
        if err < tol and ii > 1:
            # print("Converged after %s iterations." % ii)
            break
    if ii == maxiter - 1:
        print("Sinkhorn did not converge. Last err: %s" % err)
    loss = u.mean() + v.mean()
    loss *= epsilon
    return u, v, loss


def closed_forms(cov_a, cov_b, sigma):
    """Returns the matrices of the potentials
    f(x) = x^t U x
    g(x) = x^t V x
    """
    n = len(cov_a)
    Id = np.eye(n)
    cov_a_s = sqrtm(cov_a)
    icov_a_s = inv(cov_a_s)
    D = sqrtm(cov_a_s.dot(cov_b).dot(cov_a_s) + sigma ** 4 / 4 * Id)
    loss = np.trace(cov_a + cov_b - 2 * D) + sigma ** 2 * n
    loss += - n * sigma ** 2 * np.log(2 * sigma ** 2)
    loss += sigma ** 2 * np.log(np.linalg.det(2 * D + sigma ** 2 * Id))
    C = cov_a_s.dot(D).dot(icov_a_s) - sigma ** 2 / 2 * Id
    E = inv(C + sigma ** 2 * Id)
    U = Id - cov_b.dot(E)
    V = Id - E.dot(cov_a)
    return U, V, loss


def quadratic_linreg(samples, sinkhorn_pot):
    """Quadratic function fit

    Params:
    -------
    samples: (n, d)
    sinkhorn_pot: (n,)

    Returns:
    --------
    mat: (d, d), best fit quadratic function matrix
    """
    n, d = samples.shape
    X = samples[:, :, None] * samples[:, None, :]
    X = X.reshape(n, -1)
    coef = LinearRegression(fit_intercept=True).fit(X, sinkhorn_pot).coef_
    mat = coef.reshape(d, d)
    mat = 0.5 * (mat + mat.T)
    return mat


def sinkhorn_mat_potentials(samples_a, samples_b, sigma):
    """Infer the best quadratic fit of Sinkhorn potentials."""
    n_samples, dim = samples_a.shape
    a = np.ones(n_samples) / n_samples
    b = np.ones(n_samples) / n_samples
    C = ((samples_a[:, None, :] - samples_b[None, :, :]) ** 2).sum(-1)
    u, v, _ = sinkhorn(a, b, C, sigma)
    U_sinkhorn = quadratic_linreg(samples_a, u)
    V_sinkhorn = quadratic_linreg(samples_b, v)
    return U_sinkhorn, V_sinkhorn, u, v


if __name__ == "__main__":
    n_samples = 2000
    seed = 42
    n_trials = 1
    rng = np.random.RandomState(seed)

    dim_title = "dimension"
    epsilon_title = "epsilon"
    left_title = "rel diff in U"
    right_title = "rel diff in V"
    print("%20s" % dim_title, " | ", "%20s" % epsilon_title, " | ",
          "%20s" % left_title, " | ", "%20s" % right_title)
    for _ in range(n_trials):
        dim = 1
        cov_a = rng.randn(dim, dim)
        cov_a = cov_a.dot(cov_a.T)
        cov_b = rng.randn(dim, dim)
        cov_b = cov_b.dot(cov_b.T)

        cov_a = np.eye(1)
        cov_b = np.eye(1)

        mean_a, mean_b = np.zeros((2, dim))
        samples_a = rng.multivariate_normal(mean_a, cov_a, size=n_samples)
        samples_b = rng.multivariate_normal(mean_b, cov_b, size=n_samples)
        # epsilon is entropy regularization; we write eps = 2 sigma^2
        epsilon = 2
        sigma = (epsilon / 2) ** 0.5
        # closed form potentials
        U, V, loss = closed_forms(cov_a, cov_b, sigma)
        print(U, V)

        # best quadratic fit of sinkhorn potentials
        U_sinkhorn, V_sinkhorn, u, v = sinkhorn_mat_potentials(samples_a,
                                                               samples_b,
                                                               sigma)
        diff_u = abs(U - U_sinkhorn).max()
        diff_u /= max(abs(U).max(), abs(U_sinkhorn).max(), 1.)
        diff_v = abs(V - V_sinkhorn).max()
        diff_v /= max(abs(V).max(), abs(V_sinkhorn).max(), 1.)
        print("%20s" % dim, "%20s" % np.round(epsilon, 6),
              "%20s" % np.round(diff_u, 6), " | ",
              "%20s" % np.round(diff_v, 6))