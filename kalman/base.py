from cmath import isfinite
from typing import Dict, Union
import numpy as np
from scipy.stats import multivariate_normal
from icecream import ic
import copy


class KalmanParams():

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            self.init_from_dict(args[0])
        else:
            self.init_from_kwargs(**kwargs)

    def init_from_dict(self, init_dict):
        self.init_from_kwargs(**init_dict)

    def init_from_kwargs(self, **kwargs):
        self.mu = kwargs['mu']
        self.Sigma = kwargs['Sigma']
        self.B = kwargs['B']
        self.R = kwargs['R']
        self.A = kwargs['A']
        self.Q = kwargs['Q']

    @property
    def n_params(self):
        return self.mu.size + self.Sigma.size + self.B.size + self.R.size + self.A.size + self.Q.size


def filter_step(x, y_est_prev, P_est_prev, A, Q, B, R, estimate_covs=True, y_pred_cov=None, y_est_cov=None):
    """
    Performs one full step of Kalman filtering.
    """
    y_pred = A @ y_est_prev

    if estimate_covs:
        y_pred_cov = A @ P_est_prev @ A.T + Q

    kalman_gain = y_pred_cov @ B.T @ np.linalg.inv(R + B @ y_pred_cov @ B.T)
    T = (np.eye(kalman_gain.shape[0]) - kalman_gain @ B)

    if estimate_covs:
        y_est_cov = T @ y_pred_cov @ T.T + kalman_gain @ R @ kalman_gain.T

    y_est = y_pred + kalman_gain @ (x - B @ y_pred)

    if estimate_covs:
        return y_pred, y_pred_cov, kalman_gain, y_est, y_est_cov

    return y_pred, kalman_gain, y_est


def predict_step(y_est_prev, A, B):
    """
    Predicts the next state and observation based on the last filtered estimate.
    """
    y_pred = A @ y_est_prev
    x_pred = B @ y_pred

    return y_pred, x_pred


class KalmanModel():
    # Using Max Welling's notation
    def __init__(self, verbose=False) -> None:
        self.verbose = verbose

    def set_params(self, params: Union[KalmanParams, Dict]):
        if isinstance(params, KalmanParams):
            self.params = copy.deepcopy(params)
        else:
            self.params = KalmanParams(**params)

    def fit(
        self,
        X: np.ndarray,
        d: int,
        init_params,
        t_end=None,
        predict_params=False,
        n_it=10,
    ):
        self.X = X
        self.tau = X.shape[0]
        self.d = d
        self.k = X.shape[-1]

        self.calculate_filter_cov = True
        self.P_t_t1_estimated = False

        self.P_t_tau_estimated = False

        if t_end is None:
            t_end = self.tau

        self.t_end = t_end
        self.set_params(init_params)

        if t_end > self.tau:
            raise NotImplementedError("Prediction not implemented")

        for _ in range(n_it):
            if t_end <= self.tau:
                self.filter()
                self.smooth()
            if predict_params:
                self.lag_one_covar_smoother()
                self.e_step()
                if self.verbose:
                    print(self.loglikelihood())
            else:
                break
        else:
            self.filter()
            self.smooth()

    def filter(self, t_end_filter=None):
        # Initialize the filter & perform 0-th step.

        if t_end_filter is None:
            t_end_filter = self.tau

        K = np.zeros(shape=(t_end_filter, self.d, self.k))

        y_t_t = np.zeros(shape=(t_end_filter, self.d))

        if self.calculate_filter_cov:
            P_t_t = np.zeros(shape=(t_end_filter, self.d, self.d))
        else:
            P_t_t = self.P_t_t

        # y_hat_t^t-1 -- has offset of 2, i.e. K[t] = K_(t+1)
        y_t_t1 = np.zeros(shape=(t_end_filter, self.d))
        
        try:
            y_t_t1[0] = np.linalg.solve(self.params.B, self.X[0])
            ic("solve worked")
        except np.linalg.LinAlgError:
            try:
                y_t_t1[0], _, _, _ = np.linalg.lstsq(self.params.B, self.X[0], rcond=None)
                ic("lstsq worked")
                ic(y_t_t1[0])
            except np.linalg.LinAlgError:
                y_t_t1[0] = self.params.mu
                ic("Nothing worked, using mu.")


        if self.P_t_t1_estimated:
            P_t_t1 = self.P_t_t1
        else:
            # P_t^t-1 -- has offset of 2, i.e. K[t] = K_(t+1)
            P_t_t1 = np.zeros(shape=(t_end_filter, self.d, self.d))
            P_t_t1[0] = self.params.Sigma

        K[0] = P_t_t1[0] @ self.params.B.T @ np.linalg.inv(
            self.params.R + self.params.B @ P_t_t1[0] @ self.params.B.T)

        # P_t^t (numerically stable version)
        T = (np.eye(K[0].shape[0]) - K[0] @ self.params.B)

        if self.calculate_filter_cov:
            P_t_t[0] = T @ P_t_t1[0] @ T.T + K[0] @ self.params.R @ K[0].T

        # y_hat_t^t
        y_t_t[0] = y_t_t1[0] + K[0] @ (self.X[0] - self.params.B @ y_t_t1[0])

        # incrementally advance filter.
        for t in range(1, t_end_filter):
            if self.calculate_filter_cov:
                y_t_t1[t], P_t_t1[t], K[t], y_t_t[t], P_t_t[t] = filter_step(
                    self.X[t],
                    y_t_t[t - 1],
                    P_t_t[t - 1],
                    self.params.A,
                    self.params.Q,
                    self.params.B,
                    self.params.R)
            else:
                y_t_t1[t], K[t], y_t_t[t] = filter_step(
                    self.X[t], 
                    y_t_t[t - 1],
                    P_t_t[t - 1],
                    self.params.A,
                    self.params.Q,
                    self.params.B,
                    self.params.R,
                    estimate_covs=False,
                    y_pred_cov=P_t_t1[t],
                    y_est_cov=P_t_t[t],)

        # store results
        if self.calculate_filter_cov:
            self.calculate_filter_cov = False
            self.P_t_t1 = P_t_t1
            self.P_t_t = P_t_t

        self.y_t_t = y_t_t
        self.K = K
        self.y_t_t1 = y_t_t1

        return y_t_t, P_t_t, y_t_t1, P_t_t1, K

    def smooth(self, t_end_smooth=None):
        if t_end_smooth is None:
            t_end_smooth = 0

        if self.P_t_tau_estimated:
            P_t_tau = self.P_t_tau
        else:
            # P_t-1^tau
            P_t_tau = np.zeros((self.tau - t_end_smooth, self.d, self.d))
            P_t_tau[-1] = self.P_t_t[-1]

        # y_hat_t^tau
        y_t_tau = np.zeros((self.tau - t_end_smooth, self.d))
        y_t_tau[-1] = self.y_t_t[-1]

        # J_t
        J = np.zeros((self.tau - t_end_smooth - 1, self.d, self.d))

        for t in range(1, self.tau - t_end_smooth):
            P_inv = np.linalg.inv(self.P_t_t1[-t])
            J[-t] = self.P_t_t[-t - 1] @ self.params.A.T @ P_inv
            y_t_tau[-t - 1] = (self.y_t_t[-t - 1] +
                               J[-t] @ (y_t_tau[-t] - self.y_t_t1[-t]))
            if not self.P_t_tau_estimated:
                P_t_tau[-t - 1] = (
                    self.P_t_t[-t - 1] +
                    J[-t] @ (P_t_tau[-t] - self.P_t_t1[-t]) @ J[-t].T)
        self.y_t_tau = y_t_tau

        if not self.P_t_tau_estimated:
            self.P_t_tau_estimated = True
            self.P_t_tau = P_t_tau
        self.J = J

        return y_t_tau, P_t_tau, J

    def lag_one_covar_smoother(self, t_end_smooth=None):
        if t_end_smooth is None:
            t_end_smooth = 0
        # P_(t)(t-1)^tau
        P_tt1_tau = np.zeros((self.tau - t_end_smooth - 1, self.d, self.d))
        P_tt1_tau[-1] = (np.eye(self.d) - self.K[-1] @ self.params.B
                         ) @ self.params.A @ self.P_t_t[-1]

        for t in range(1, self.tau - t_end_smooth - 1):
            P_tt1_tau[-t -
                      1] = (self.P_t_tau[-t] @ self.J[-t - 1].T +
                            self.J[-t] @ (P_tt1_tau[-t] -
                                          self.params.A @ self.P_t_t[-t - 1])
                            @ self.J[-t - 1].T)

        self.P_tt1_tau = P_tt1_tau

        return P_tt1_tau

    def e_step(self):
        M_0 = np.zeros(self.P_t_tau.shape)
        M_1 = np.zeros(self.P_tt1_tau.shape)

        if self.y_t_tau.shape[0] != self.tau:
            raise RuntimeError

        for i in range(M_0.shape[0]):
            M_0[i] = self.P_t_tau[i] + np.outer(self.y_t_tau[i],
                                                self.y_t_tau[i])
            if i >= 1:
                M_1[i - 1] = self.P_tt1_tau[i - 1] + np.outer(
                    self.y_t_tau[i], self.y_t_tau[i - 1])

        mu_new = self.y_t_tau[0]

        Sigma_new = self.P_t_tau[0]  # + cov y_1 if considering multiple runs
        A_new = np.sum(M_1, axis=0) @ np.linalg.inv(np.sum(M_0[:-1], axis=0))

        Q_new = np.mean(M_0[1:] - np.einsum('ij, ...kj->...ik', A_new, M_1),
                        axis=0)

        B_new = (np.sum(np.einsum('...i,...j->...ij', self.X, self.y_t_tau),
                        axis=0) @ np.linalg.inv(np.sum(M_0, axis=0)))

        R_new = (
            np.einsum('...i,...j->...ij', self.X, self.X) -
            np.einsum('ij,...jk->...ik', B_new,
                      np.einsum('...i,...j->...ij', self.y_t_tau, self.X)))
        R_new = np.mean(R_new, axis=0)

        self.params.mu = mu_new
        self.params.Sigma = Sigma_new
        ic(A_new)
        self.params.A = A_new
        self.params.Q = Q_new
        self.params.B = B_new
        self.params.R = R_new

        self.P_t_t_estimated = False
        self.P_t_t1_estimated = False
        self.P_t_tau_estimated = False

        return mu_new, Sigma_new, A_new, Q_new, B_new, R_new

    def loglikelihood(self):
        x_hat_1 = np.einsum('ij,...j->...i', self.params.B, self.y_t_t1)
        x_hat_01 = self.params.B @ self.params.mu

        H_1 = self.params.R + self.params.B @ np.einsum(
            '...ij,jk->...ik', self.P_t_t1[1:], self.params.B.T)

        H_01 = self.params.R + self.params.B @ self.params.Sigma @ self.params.B.T

        res = np.log(
            multivariate_normal(mean=x_hat_01, cov=H_01).pdf(self.X[0]))
        for t in range(1, self.X.shape[0]):
            res += np.log(
                multivariate_normal(mean=x_hat_1[t - 1],
                                    cov=H_1[t - 1]).pdf(self.X[t]))

        return res
