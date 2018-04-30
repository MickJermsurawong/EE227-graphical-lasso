import numpy as np


class GraphicalLasso(object):

    def __init__(self, l1_solver_f):

        self.l1_solver_f = l1_solver_f

    @staticmethod
    def _partition_blockwise(X, j):
        """
            Assume X is symmetric
        """
        X_j = X[:, j]

        x_22 = X_j[j]

        selector = np.ones_like(X_j, dtype=np.bool)
        selector[j] = False
        x_12 = X_j[selector]

        X_11 = X[selector][:, selector]

        return X_11, x_12, x_22

    @staticmethod
    def _update_partition(X, x, j):
        """
            Symmetric X [nxn]
            Vector x [n-1]
            Update x to X to column/row tiling each element except the jth position
        """
        X[j, :j] = x[:j]
        X[j, j + 1:] = x[j:]

        X[:j, j] = x[:j]
        X[j + 1:, j] = x[j:]

    @staticmethod
    def _is_psd(W):
        return np.all(np.linalg.eigvals(W) > 0)

    def fit(self, S, l1_lambda=0.01, verbose=False):

        assert np.array_equal(S, S.T), "Initial empirical covariance should be symmetric"
        feature_dim = S.shape[0]
        W = np.copy(S)

        # Coordinate descent solver
        l1_solver = self.l1_solver_f(alpha=l1_lambda)

        # Pre-partition S
        s_12_all = []
        for j in range(feature_dim):
            _, S_12, _ = GraphicalLasso._partition_blockwise(S, j)
            s_12_all.append(S_12)

        def solve_inner(W, j):
            """
                Return beta, and estimated w_12
            """
            W_11, _, _ = GraphicalLasso._partition_blockwise(W, j)
            s_12 = s_12_all[j]

            assert GraphicalLasso._is_psd(W_11), f"Each submatrix should be psd.., but given {W}"

            l1_solver.fit_gram_scikit(W_11, s_12)
            beta = l1_solver.coef_

            w_12_estimated = np.matmul(W_11, beta)

            return beta, w_12_estimated

        # Check convergence
        all_betas = [np.ones(feature_dim -1) for _ in range(feature_dim)]
        all_w_12s = [np.ones(feature_dim -1) for _ in range(feature_dim)]

        count_converge = 0
        iter_num = 0
        while True:

            if iter_num % 10 == 0 and verbose is True:
                print("Iter #", iter_num)

            j = iter_num % feature_dim

            beta, w_12_estimated = solve_inner(W, j)
            GraphicalLasso._update_partition(W, w_12_estimated, j)  # diagonal is unchanged

            if np.isclose(all_betas[j], beta).all():
                count_converge += 1
            else:
                count_converge = 0

            all_betas[j] = beta
            all_w_12s[j] = w_12_estimated
            iter_num += 1

            if count_converge == feature_dim:
                break

        # Make each column precision matrix
        Theta = np.zeros_like(W)

        for j in range(feature_dim):
            beta, w_12_estimated = all_betas[j], all_w_12s[j]
            w_22 = W[j][j]

            theta_22 = 1.0 / (w_22 - np.matmul(w_12_estimated.T, beta))
            theta_12 = beta * -1 * theta_22
            self._update_partition(Theta, theta_12, j)
            Theta[j][j] = theta_22

        return Theta

