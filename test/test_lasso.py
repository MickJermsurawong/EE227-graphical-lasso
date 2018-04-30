from unittest import TestCase

import numpy as np
from sklearn import linear_model

from LassoSolver import LassoSolver


class TestLasso(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLasso, self).__init__(*args, **kwargs)

    def test_fit_gram(self):

        X = np.random.normal(size=(5000, 6))
        w = np.random.normal(size=6) # sparse model
        w[0] = 0
        w[5] = 0
        y = np.matmul(X, w) + np.random.normal(size=5000)

        lasso = linear_model.Lasso(alpha=0.1)
        lasso.fit(X, y)
        lasso_w = lasso.coef_

        XTX = np.matmul(X.T, X)
        XTy = np.matmul(X.T, y)

        gram_lasso = LassoSolver(alpha=0.1)

        gram_lasso.fit_gram_scikit(XTX, XTy)
        g_lasso_w_scikit = gram_lasso.coef_

        gram_lasso.fit_gram(XTX, XTy)
        g_lasso_w = gram_lasso.coef_

        print(f"Lasso weight: \n{np.around(lasso_w, decimals=2)}")
        print(f"Using scikit lasso: \n{np.around(g_lasso_w_scikit, decimals=2)}")
        print(f"Our implementation: \n{np.around(g_lasso_w, decimals=2)}")
        for i in range(6):
            self.assertAlmostEqual(g_lasso_w_scikit[i], g_lasso_w[i], delta=0.1)





