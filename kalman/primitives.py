import numpy as np
from scipy.stats import multivariate_normal


is_sym = lambda a: np.allclose(a, np.swapaxes(a, -1, -2))

def multivar_normal_loglikelihood(X, X_est, X_est_cov):
    res = 0
    inf_counter = 0
    for t in range(0, X.shape[0]):
        # Sometimes the covariance matrix is not symmetric due to numerical
        # instabilities
        X_est_cov_ = (X_est_cov[t] + X_est_cov[t].T) / 2
        p_i = np.log(multivariate_normal(mean=X_est[t], cov=X_est_cov_).pdf(X[t]))
        if np.isinf(p_i):
            inf_counter += 1
            continue
        res += np.log(multivariate_normal(mean=X_est[t], cov=X_est_cov_).pdf(X[t]))
    if inf_counter:
        print(f"Encountered {inf_counter} inf values " +
            f"(that is {(inf_counter/X.shape[0]*100):.2f}%). " +
            "Loglikelihood will be overestimated.")
    return res

def matmul_inv(P, Q):
    """ Calculates P @ Q^(-1) in a numerically stable manner.
    """
    r = np.linalg.solve(Q.T, P.T)
    return r.T


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
        self.check_params(**    kwargs)
        self.mu = kwargs['mu']
        self.Sigma = kwargs['Sigma']
        self.B = kwargs['B']
        self.R = kwargs['R']
        self.A = kwargs['A']
        self.Q = kwargs['Q']

    def check_params(self, **kwargs):
        if len(kwargs['mu'].shape) != 1:
            raise ValueError(f"Expected mu to be of dimension 1, not {len(kwargs['mu'].shape)}")
        self._latent_dim = kwargs['mu'].size

        if kwargs['Sigma'].shape != (self._latent_dim, self._latent_dim):
            raise ValueError(f"Dimension mismatch between mu and Sigma." +
                f"Expected Sigma to be of shape {(self._latent_dim, self._latent_dim)}, " +
                f"not {kwargs['Sigma'].shape}")

        if kwargs['A'].shape != (self._latent_dim, self._latent_dim):
            raise ValueError(f"Dimension mismatch between mu and A." +
                f"Expected A to be of shape {(self._latent_dim, self._latent_dim)}, " +
                f"not {kwargs['A'].shape}")

        if kwargs['Q'].shape != (self._latent_dim, self._latent_dim):
            raise ValueError(f"Dimension mismatch between mu and Q." +
                f"Expected Q to be of shape {(self._latent_dim, self._latent_dim)}, " +
                f"not {kwargs['Q'].shape}")

        if kwargs['B'].shape[1] != self._latent_dim :
            raise ValueError(f"Dimension mismatch between mu and B." +
                f"Expected B to have length {self._latent_dim} " +
                f"along axis 1, not {kwargs['B'].shape[1]}")

        self._out_dim = kwargs['B'].shape[0]

        if kwargs['R'].shape != (self._out_dim, self._out_dim):
            raise ValueError(f"Dimension mismatch between B and R." +
                f"Expected R to be of shape {(self._out_dim, self._out_dim)}, " +
                f"not {kwargs['R'].shape}")

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

    def __eq__(self, other):
        if isinstance(other, KalmanParams):
            return self.to_dict() == other.to_dict()
        elif isinstance(other, dict):
            return self.to_dict() == other
        else:
            raise TypeError

    @property
    def latent_dim(self):
        return self._latent_dim

    @property
    def out_dim(self):
        return self._out_dim

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

    kalman_gain = y_pred_cov @ matmul_inv(B.T, R + B @ y_pred_cov @ B.T)
    T = (np.eye(kalman_gain.shape[0]) - kalman_gain @ B)

    if estimate_covs:
        y_est_cov = T @ y_pred_cov @ T.T + kalman_gain @ R @ kalman_gain.T

    y_est = y_pred + kalman_gain @ (x - B @ y_pred)

    if estimate_covs:
        return y_pred, y_pred_cov, kalman_gain, y_est, y_est_cov

    return y_pred, kalman_gain, y_est


def predict_step(y_est_prev, A, B, estimate_covs=False, P_est_prev=None, Q=None, R=None):
    """
    Predicts the next state and observation based on the last filtered estimate.
    """
    y_pred = A @ y_est_prev
    x_pred = B @ y_pred

    if estimate_covs:
        y_pred_cov = A @ P_est_prev @ A.T + Q
        x_pred_cov = B @ y_pred_cov @ B.T + R
        return y_pred, y_pred_cov, x_pred, x_pred_cov
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

    J = P_est_prev @ matmul_inv(A.T, y_pred_cov)

    y_t_tau = (y_est_prev + J @ (y_t_tau_prev - y_pred))
    if estimate_covs:
        y_t_tau_cov = (
            P_est_prev +
            J @ (P_t_tau_prev - y_pred_cov) @ J.T)
        
        # Sometimes y_t_tau_cov is asymmetric for numerical reasons
        y_t_tau_cov = (y_t_tau_cov + y_t_tau_cov.T) / 2.
        return y_t_tau, y_t_tau_cov, J

    return y_t_tau, J
