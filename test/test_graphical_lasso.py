from unittest import TestCase

import numpy as np

from GraphicalLasso import GraphicalLasso as GL
from LassoSolver import LassoSolver


class TestGraphicalLasso(TestCase):

    def __init__(self, *args, **kwargs):
        super(TestGraphicalLasso, self).__init__(*args, **kwargs)
        self.standard_a = np.array([
            [0, 1, 2],
            [1, 4, 3],
            [2, 3, 5]
        ])

    def test_partitioning(self):
        # Partition at last block
        A = np.copy(self.standard_a)
        X_11, x_12, x_22 = GL._partition_blockwise(A, 2)
        self.assertTrue(np.array_equal(X_11, np.array([[0, 1], [1, 4]])))
        self.assertTrue(np.array_equal(x_12, np.array([2, 3])))
        self.assertTrue(x_22 == 5)

        # Partition middle
        X_11, x_12, x_22 = GL._partition_blockwise(A, 1)
        self.assertTrue(np.array_equal(X_11, np.array([[0, 2], [2, 5]])))
        self.assertTrue(np.array_equal(x_12, np.array([1, 3])))
        self.assertTrue(x_22 == 4)

    def test_update_partition(self):

        A = np.copy(self.standard_a)
        GL._update_partition(A, np.array([9, 7]), 1)
        self.assertTrue(np.array_equal(A, np.array([
            [0, 9, 2],
            [9, 4, 7],
            [2, 7, 5]
        ])))

    def test_fit(self):

        Theta_true = np.array([
            [2.0, 0.6, 0.0, 0.0, 0.5],
            [0.6, 2.0, -0.4, 0.3, 0.0],
            [0.0, -0.4, 2.0, -0.2, 0.0],
            [0.0, 0.3, -0.2, 2.0, -0.2],
            [0.5, 0.0, 0.0, -0.2, 2.0]
        ])
        feature_dim = Theta_true.shape[0]

        assert np.array_equal(Theta_true.T, Theta_true), "True precision should be symmetric"

        cov = np.linalg.inv(Theta_true)
        data_set = np.random.multivariate_normal(np.zeros(feature_dim), cov, (20))
        S = np.cov(data_set.T)
        S_inv = np.linalg.inv(S)

        gl = GL(l1_solver_f=LassoSolver)

        theta_estimated = gl.fit(S, l1_lambda=0.0001)

        self.assertEqual(theta_estimated.shape[0], feature_dim)

        print(f"Estimated: \n{np.round(theta_estimated, decimals=2)}")
        print(f"S-inverse: \n{np.round(S_inv, decimals=2)}")

        for i in range(feature_dim):
            for j in range(feature_dim):
                self.assertAlmostEqual(theta_estimated[i][j], S_inv[i][j], delta=0.1)





