import numpy as np


class NGraphSelection(object):

    def __init__(self, l1_solver_f):

        self.l1_solver_f = l1_solver_f

    def fit(self, X, l1_lambda, mode='AND'):

        p = X.shape[1]
        neigbors = np.zeros((p, p))

        for s in range(p):

            selector = np.ones(p, dtype=np.bool)
            selector[s] = False

            X_s = X[:, s]
            X_rest = X[:, selector]

            l1_solver = self.l1_solver_f(alpha=l1_lambda)
            l1_solver.fit(X_rest, X_s)

            beta_s = l1_solver.coef_
            print(beta_s)

            for i in range(p - 1):
                t = i + 1 if i >= s else i
                if beta_s[i] != 0.0:
                    neigbors[s][t] = 1


        # Make symmetric
        for i in range(p):
            for j in range(p):
                if i > j:

                    if mode == 'AND':
                        if neigbors[i][j] != neigbors[j][i]:
                            neigbors[i][j] = 0
                            neigbors[j][i] = 0
                    elif mode == 'OR':
                        if neigbors[i][j] or neigbors[j][i]:
                            neigbors[i][j] = 1
                            neigbors[j][i] = 1
                    else:
                        raise Exception(f"Unknown mode {mode}")

        return neigbors






