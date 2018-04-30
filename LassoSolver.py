import numpy as np
from sklearn.linear_model import coordinate_descent


class LassoSolver(object):

    def __init__(self, alpha):

        self.l1_lambda = alpha
        self.coef_ = None

    def fit_gram_scikit(self, W, s):

        size = W.shape[0]

        w = np.random.normal(size=size)
        w_c = w.copy(order='C')
        W_c = W.copy(order='C')
        s_c = s.copy(order='C')

        # W is XtX, and S is Xy
        coordinate_descent.cd_fast.enet_coordinate_descent_gram(w_c, self.l1_lambda, 0,  W_c, s_c, s_c, 1000, 1e-8, np.random.mtrand._rand, False, False)
        # coef_, l1_reg, l2_reg, precompute, Xy, y, max_iter, tol, rng, random, positive

        self.coef_ = w_c

    def fit_gram(self, W, s):

        p = W.shape[0]
        assert p == s.shape[0]

        beta = np.random.normal(0, 1, [p])
        prev_beta = np.zeros_like(beta)
        i = 0

        while True:
            j = i % p
            w_col_j = (W[:, j])
            tmp1 = s[j]
            tmp2 = np.matmul(w_col_j, beta) - w_col_j[j] * beta[j]
            x = (tmp1 - tmp2) / p
            numerator = np.sign(x) * max(0, np.abs(x) - self.l1_lambda)
            denom = W[j][j] / p

            beta_j = numerator / denom

            prev_beta[:] = beta
            beta[j] = beta_j

            i += 1

            if np.isclose(prev_beta, beta, rtol=1.e-8).all():
                break
            if i >= 5000:
                print(np.isclose(prev_beta, beta))
                raise Exception(f"Lasso convergence more than 5000... current: {beta}, previous: {prev_beta}")

        self.coef_ = beta


