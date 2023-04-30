"""
Reduced rank regression class.
Requires scipy to be installed.
Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com
Optimal linear 'bottlenecking' or 'multitask learning'.
"""
import numpy as np
from scipy import sparse


def ideal_data(num, dimX, dimY, rrank, noise=1):
    """Low rank data"""
    X = np.random.randn(num, dimX)
    W = np.dot(np.random.randn(dimX, rrank), np.random.randn(rrank, dimY))
    Y = np.dot(X, W) + np.random.randn(num, dimY) * noise
    return X, Y


class ReducedRankRegressor(object):
    """
    Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
    - X is an n-by-p matrix of features.
    - Y is an n-by-q matrix of targets.
    - rrank is a rank constraint.
    - reg is a regularization parameter (optional).
    """
    def __init__(self, rank, reg=None):
        self.rank = rank
        self.reg = reg

    def fit(self, X, Y):
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        if np.size(np.shape(Y)) == 1:
            Y = np.reshape(Y, (-1, 1))
        if self.reg is None:
            self.reg = 0

        CXX = np.dot(X.T, X) + self.reg * sparse.eye(np.size(X, 1))
        CXY = np.dot(X.T, Y)
        _U, _S, V = np.linalg.svd(np.dot(CXY.T, np.dot(np.linalg.pinv(CXX), CXY)))
        self.W = np.asarray(V[0:self.rank, :].T)
        self.A = np.asarray(np.dot(np.linalg.pinv(CXX), np.dot(CXY, self.W)).T)

        return self

    def __str__(self):
        return 'Reduced Rank Regressor (rank = {})'.format(self.rank)

    def predict(self, X):
        """Predict Y from X."""
        if np.size(np.shape(X)) == 1:
            X = np.reshape(X, (-1, 1))
        return np.dot(X, np.dot(self.A.T, self.W.T))

    def get_params(self, deep=True):
        """Return estimator parameter names and values."""
        return {'rank': self.rank, 'reg': self.reg}

    def set_params(self, **params):
        """Set the value of one or more estimator parameters."""
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
