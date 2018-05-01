import numpy as np
from numpy import linalg as LA 

class Lasso(object):
    
    def __init__(self, alpha):
        self.alpha = alpha
        self.coef_ = None
        self.n_iter = None
        
    def soft_tresh(self,x):
        return np.sign(x)*(np.maximum(0,np.abs(x)- self.alpha))

    def fit(self, X, y):
        p = X.shape[1]
        n = X.shape[0]
        assert n == y.shape[0]
        convergence= False
        beta = np.ones(p)
        iteration = 0
        
        
            
        while not convergence:
            j = iteration % p
                
            #store prev beta
            beta_prev = np.copy(beta)

            #calulating residual vector
            res = np.ones(n)
            X_bar = np.delete(X, j, 1)
            beta_bar = np.delete(beta, j)
            assert X_bar.shape[1] == beta_bar.shape[0]
            
            res = y -  np.dot(X_bar, beta_bar)

            #updating current beta
            update = (1/n) * np.dot(X[:,j], res)
            norm_xj = (1/n)*np.square(LA.norm(X[:,j]))
            beta[j] = self.soft_tresh(update)/norm_xj

            #checking for convergence
            
            obj = (1/(2*n)) * np.square(LA.norm(y-np.dot(X,beta))) + self.alpha*LA.norm(beta, 1)
            obj_prev = (1/(2*n)) * np.square(LA.norm(y-np.dot(X,beta_prev))) + self.alpha*LA.norm(beta_prev, 1)
            
            iteration += 1
            if np.isclose(obj, obj_prev, rtol = 0.00001):
                convergence = True  
                self.n_iter = iteration
            
            def intercept(X,y,beta):
                beta_0 = np.mean(y)
                for j in range(0,p):
                    beta_0 -= np.mean(X[:,j]*beta[j])
                return beta_0
                    
                    
            
        self.coef_ = beta 