"""
Reduced rank regression class.
Requires scipy to be installed.
Implemented by Chris Rayner (2015)
dchrisrayner AT gmail DOT com
Optimal linear 'bottlenecking' or 'multitask learning'.
"""
import os
import numpy as np
import joblib
from scipy import sparse
from sklearn.preprocessing import StandardScaler
import catrace.exp_collection as ecl


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

        N = X.shape[0]
        CXX = np.dot(X.T, X) + self.reg * N * sparse.eye(np.size(X, 1))
        CXY = np.dot(X.T, Y)
        # Here we use CXX and CXY, not CovXX, CovXY, because
        # CXX = N * CovXX and CXY = N * CovXY
        # but in singular value decomposition, the V is always unitary
        # Hence the scaling by N is only visible in _S
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

    def compute_latent(self, X):
        return np.matmul(X, self.A.T)


def estimate_rrr_experiment(exp_name, rrr_param, trace_dir, out_base_dir, db_dir):
    df_ob = ecl.read_df(trace_dir, exp_name, 'OB', db_dir)
    df_dp = ecl.read_df(trace_dir, exp_name, 'Dp', db_dir)

    scaler_x = StandardScaler()
    x = scaler_x.fit_transform(df_ob)

    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(df_dp)

    estimator = ReducedRankRegressor(**rrr_param)
    estimator.fit(x, y)

    out_dir = os.path.join(db_dir, out_base_dir, exp_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    joblib.dump(estimator, os.path.join(out_dir, 'rrr_model.pkl'))

    y_predict = estimator.predict(x)
    x_latent = estimator.compute_latent(x)
    np.save(os.path.join(out_dir, 'y_predict.npy'), y_predict)
    np.save(os.path.join(out_dir, 'x_latent.npy'), x_latent)
