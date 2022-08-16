import numpy as np
from scipy.stats import multivariate_normal
import copy


class KalmanParams():
    """
    
    This just allows a more convenient syntax of params.A instead of
        params['A'].
    """
    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], dict):
            self.init_from_dict(args[0])
        else:
            self.init_from_kwargs(**kwargs)

    def __repr__(self) -> str:
        return str(self.to_dict())

    def init_from_dict(self, init_dict):
        self.init_from_kwargs(**init_dict)

    def init_from_kwargs(self, **kwargs):
        self.mu = kwargs['mu']
        self.Sigma = kwargs['Sigma']
        self.B = kwargs['B']
        self.R = kwargs['R']
        self.A = kwargs['A']
        self.Q = kwargs['Q']

    def to_dict(self):
        _dict = {
            'mu': self.mu,
            'Sigma': self.Sigma,
            'B': self.B,
            'R': self.R,
            'A': self.A,
            'Q': self.Q
        }
        return _dict

    @property
    def n_params(self):
        return self.mu.size + self.Sigma.size + self.B.size + self.R.size + self.A.size + self.Q.size


def filter_step(x, y_est_prev, P_est_prev, A, Q, B, R, estimate_covs=True, y_pred_cov=None, y_est_cov=None):
    """
    Performs one full step of Kalman filtering.

    Estimates the value of y(t) and P(t) based on measurements from t=1...t.

    Args:
        x: The measurement at time t.

        y_est_prev: The estimated value of y at time t-1 based on t-1...t-1
            measurements.

        P_est_prev: The associated covariance.

        A: Matrix that guides the evolution of the state, i.e.
            y(t) = A @ y(t-1) + w_t.

        Q: Covariance matrix of the noise term w_t, w_t ~ Norm(0, Q).

        B: Matrix that relates the state to the hidden measurement, i.e.
            x(t) = B @ y(t) + v.

        R: Covariance matrix of the noise term v_t, v_t ~ Norm(0, R).

        estimate_covs (bool): Whether to estimate the covariance (see Returns).

        y_pred_cov: If the covariances are not estimated, they need to be
            provided.

        y_est_cov: If the covariances are not estimated, they need to be
            provided.

        Returns:
            y_pred: Predicted value of y at time t based on t-1 measurements.

            y_pred_cov: If estimate_covs, also returns the associated covariance.

            kalman_gain: The kalman gain K at time t.

            y_est: Predicted value of y at time t based on t measurements.

            y_est_cov: If estimate_covs, also returns the associated covariance.
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


def smooth_step(y_t_tau_prev, P_t_tau_prev, y_pred, y_pred_cov, y_est_prev, P_est_prev, A, estimate_covs=True):
    """
    Performs one Kalman smoothing step.

    Specifically, this adjusts the estimate of y(t-1) and P(t-1) based on
    later measurements.

    Args:
        y_t_tau_prev: The estimated value of y at time t based on tau
            measurements, where tau > t.

        P_t_tau_prev: The corresponding covariance.

        y_pred: The predicted value of y at time t based on the measurements
            up-to and including t-1. 

        y_pred_cov: The corresponding covariance.

        y_est_prev: The predicted value of y at time t based on the
            measurements up-to and including t.

        y_est_cov: The corresponding covariance.

        A: Matrix that guides the evolution of the state, i.e.
                y(t) = A @ y(t-1) + w_t
            with a noise term w_t.

        estimate_cov (bool): Whether to estimate the covariance y_t_tau or not.

    Returns:
        y_t_tau: The estimated value of y at time t-1 based on tau
            measurements, where tau > t.

        P_t_tau_prev: If estimate_cov, the corresponding covariance.
    """
    P_inv = np.linalg.inv(y_pred_cov)
    J = P_est_prev @ A.T @ P_inv
    y_t_tau = (y_est_prev + J @ (y_t_tau_prev - y_pred))
    if estimate_covs:
        y_t_tau_cov = (
            P_est_prev +
            J @ (P_t_tau_prev - y_pred_cov) @ J.T)

        return y_t_tau, y_t_tau_cov, J

    return y_t_tau, J


class KalmanModel():
    # Using Max Welling's notation
    def __init__(self, verbose=False) -> None:
        self.verbose = verbose

    def set_params(self, X: np.ndarray, params: KalmanParams):
        self.params = copy.deepcopy(params)

        if len(X.shape) != 2:
            raise RuntimeError

        self.X = X
        self.tau = X.shape[-2]
        self.d = self.params.mu.size
        self.k = X.shape[-1]

        self.calculate_filter_cov = True
        self.calculate_smooth_cov = True

        return self

    def fit(
        self,
        mode='estimate',
        n_it=10,
    ):
        self.P_t_tau_estimated = False

        if mode == 'filter':
            self.filter()
            return self.y_t_t, self.P_t_t
        elif mode == 'smooth':
            self.filter()
            self.smooth()
            return self.y_t_tau, self.P_t_tau
        elif mode == 'estimate':
            for _ in range(n_it):
                # E step
                self.filter()
                self.smooth()
                self.lag_one_covar_smoother()

                # M step
                self.m_step()
                if self.verbose:
                    print(self.loglikelihood())
            self.filter()
            self.smooth()
            return self.y_t_tau, self.P_t_tau

    def filter(self):
        """Filters the data using the Kalman Filter.

        Combines a set of measurements x_t with information about the systems
        evolution to produce of the internal state of the
        system.
        """
        # Initialize the filter & perform 0-th step.
        K = np.zeros(shape=(self.tau, self.d, self.k))

        y_t_t = np.zeros(shape=(self.tau, self.d))

        if self.calculate_filter_cov:
            P_t_t = np.zeros(shape=(self.tau, self.d, self.d))
            # P_t^t-1 -- has offset of 2, i.e. K[t] = K_(t+1)
            P_t_t1 = np.zeros(shape=(self.tau, self.d, self.d))
            P_t_t1[0] = self.params.Sigma
        else:
            P_t_t = self.P_t_t
            P_t_t1 = self.P_t_t1

        # y_hat_t^t-1 -- has offset of 2, i.e. K[t] = K_(t+1)
        y_t_t1 = np.zeros(shape=(self.tau, self.d))

        y_t_t1[0] = self.params.mu

        K[0] = P_t_t1[0] @ self.params.B.T @ np.linalg.inv(
            self.params.R + self.params.B @ P_t_t1[0] @ self.params.B.T)

        # P_t^t (numerically stable version)
        T = (np.eye(K[0].shape[0]) - K[0] @ self.params.B)

        if self.calculate_filter_cov:
            P_t_t[0] = T @ P_t_t1[0] @ T.T + K[0] @ self.params.R @ K[0].T

        # y_hat_t^t
        y_t_t[0] = y_t_t1[0] + K[0] @ (self.X[0] - self.params.B @ y_t_t1[0])

        # incrementally advance filter.
        for t in range(1, self.tau):
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

    def smooth(self):
        """Smooths the data using Rauch-Tung-Striebel smoother.

        This generates a new set of estimates y, P which incorporate later
        measurements to increase accuracy. Smoothing is essentially a
        backward complement to the filtering forward pass.
        """
        if self.calculate_smooth_cov:
            # P_t-1^tau
            P_t_tau = np.zeros((self.tau, self.d, self.d))
            P_t_tau[-1] = self.P_t_t[-1]
        else:
            P_t_tau = self.P_t_tau

        # y_hat_t^tau
        y_t_tau = np.zeros((self.tau, self.d))
        y_t_tau[-1] = self.y_t_t[-1]

        # J_t
        J = np.zeros((self.tau - 1, self.d, self.d))

        for t in range(1, self.tau):
            if self.calculate_smooth_cov:
                y_t_tau[-t - 1], P_t_tau[-t - 1], J[-t] = smooth_step(
                    y_t_tau[-t],
                    P_t_tau[-t],
                    self.y_t_t1[-t],
                    self.P_t_t1[-t],
                    self.y_t_t[-t - 1],
                    self.P_t_t[-t - 1],
                    self.params.A
                )
            else:
                y_t_tau[-t - 1], J[-t] = smooth_step(
                    y_t_tau[-t],
                    P_t_tau[-t],
                    self.y_t_t1[-t],
                    self.P_t_t1[-t],
                    self.y_t_t[-t - 1],
                    self.P_t_t[-t - 1],
                    self.params.A,
                    False
                )

        self.y_t_tau = y_t_tau

        if self.calculate_smooth_cov:
            self.calculate_smooth_cov = False
            self.P_t_tau = P_t_tau
        self.J = J

        return y_t_tau, P_t_tau, J

    def lag_one_covar_smoother(self):
        """Estimate the lag one covariance smoother.

        For more information, see Max Welling's notes on the Kalman Filter,
        Appendix C.
        """
        # P_(t)(t-1)^tau
        P_tt1_tau = np.zeros((self.tau - 1, self.d, self.d))
        P_tt1_tau[-1] = (np.eye(self.d) - self.K[-1] @ self.params.B
                         ) @ self.params.A @ self.P_t_t[-2]

        for t in range(1, self.tau - 1):
            P_tt1_tau[-t - 1] = (self.P_t_t[-t - 1] @ self.J[-t - 1].T +
                            self.J[-t] @ (P_tt1_tau[-t] - self.params.A @ self.P_t_t[-t - 1])
                            @ self.J[-t - 1].T)

        self.P_tt1_tau = P_tt1_tau

        return P_tt1_tau

    def m_step(self):
        """Estimate the parameters of the Filter using the EM algorithm.

        Parameters estimated are mu, Sigma, B, R, A and Q.

        Args:
            None

        Returns:
            mu_new, Sigma_new, A_new, Q_new, B_new, R_new: The update parameter
                estimates.
        """
        M_0 = np.zeros(self.P_t_tau.shape)
        M_1 = np.zeros(self.P_tt1_tau.shape)

        if self.y_t_tau.shape[0] != self.tau:
            raise RuntimeError

        for i in range(M_0.shape[0]):
            M_0[i] = self.P_t_tau[i] + np.outer(self.y_t_tau[i],
                                                self.y_t_tau[i])
            if i >= 1:
                M_1[i - 1] = self.P_tt1_tau[i - 1] + np.outer(self.y_t_tau[i], self.y_t_tau[i - 1])

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
        self.params.A = A_new
        self.params.Q = Q_new
        self.params.B = B_new
        self.params.R = R_new

        self.calculate_filter_cov = False
        self.P_t_tau_estimated = False

        return mu_new, Sigma_new, A_new, Q_new, B_new, R_new

    def likelihood(self):
        return np.exp(self.loglikelihood())

    def measurements(self):
        x_hat_01 = self.params.B @ self.params.mu
        x_hat_1 = np.einsum('ij,...j->...i', self.params.B, self.y_t_t1)
        return np.vstack((np.array([x_hat_01]), x_hat_1))

    def loglikelihood(self):
        x_hat_1 = np.einsum('ij,...j->...i', self.params.B, self.y_t_t1)
        x_hat_01 = self.params.B @ self.params.mu

        H_1 = self.params.R + self.params.B @ np.einsum('...ij,jk->...ik', self.P_t_t1[1:], self.params.B.T)

        H_01 = self.params.R + self.params.B @ self.params.Sigma @ self.params.B.T

        res = np.log(multivariate_normal(mean=x_hat_01, cov=H_01).pdf(self.X[0]))
        for t in range(1, self.X.shape[0]):
            res *= np.log(multivariate_normal(mean=x_hat_1[t - 1],
                                    cov=H_1[t - 1]).pdf(self.X[t]))
        return res
