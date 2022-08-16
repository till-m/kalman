import numpy as np
from icecream import ic


def to_array(data) -> np.ndarray:
    if isinstance(data, list):
        return np.array([to_array(it) for it in data])
    elif isinstance(data, dict):
        keys = np.sort(np.array([*data.keys()]))
        return np.array([to_array(data[key]) for key in keys])
    else:
        return data


class KalmanModel():
    # Using Max Welling's notation
    def __init__(self) -> None:
        pass

    def fit(self,
            X,
            d: int,
            t_end=None,
            predict_params=False,
            n_it=10,
            initialize_kwargs={}):
        self.X = X
        self.tau = X.shape[0]
        self.d = d
        self.k = X.shape[-1]

        self.P_0_estimated = False
        self.P_delta1_estimated = False

        self.P_tau_estimated = False

        if t_end is None:
            t_end = self.tau

        self.t_end = t_end
        self.initialize(**initialize_kwargs)

        if t_end > self.tau:
            raise NotImplementedError("Prediction not implemented")

        for _ in range(n_it):
            if t_end < self.tau:
                self.filter()
                self.smooth()
            if predict_params:
                self.lag_one_covar_smoother()
                self.e_step()
            else:
                break
        else:
            self.filter()
            self.smooth()

    def initialize(self, mu=None, Sigma=None, B=None, R=None, A=None, Q=None):
        self.mu = mu
        self.Sigma = Sigma
        self.B = B
        self.R = R
        self.A = A
        self.Q = Q

    def filter(self, t_end_filter=None):
        if t_end_filter is None:
            t_end_filter = self.tau

        # K_t -- has offset of 2, i.e. K[t] = K_(t+2)
        K = np.zeros(shape=(t_end_filter, self.d, self.k))

        # y_hat -- has offset of 2, i.e. y_hat[t] = y_hat_(t+2)
        y_h = np.zeros(shape=(t_end_filter - 1, self.d))

        if self.P_0_estimated:
            P_0 = self.P_0
        else:
            # P_0 -- has offset of 1, i.e. P_0_0[t] = P_0_0_(t+1)
            P_0 = np.zeros(shape=(t_end_filter - 1, self.d, self.d))

        for t in range(t_end_filter - 1):
            if t == 0:
                # y_hat_t^t-1 -- has offset of 2, i.e. K[t] = K_(t+1)
                y_h_delta1 = np.zeros(shape=(t_end_filter, self.d))
                y_h_delta1[0] = self.mu

                if self.P_delta1_estimated:
                    P_delta1 = self.P_delta1
                else:
                    # P_t^t-1 -- has offset of 2, i.e. K[t] = K_(t+1)
                    P_delta1 = np.zeros(shape=(t_end_filter, self.d, self.d))
                    P_delta1[0] = self.Sigma

            else:
                # y_hat_t-0^t-1
                y_h_delta1[t] = self.A @ y_h[t - 1]

                if not self.P_delta1_estimated:
                    # P_t-0^t-1
                    P_delta1[t] = self.A @ P_0[t - 1] @ self.A.T + self.Q

            K[t] = P_delta1[t] @ self.B.T @ np.linalg.inv(
                self.R + self.B @ P_delta1[t] @ self.B.T)
            # P_t^t (numerically stable version)
            T = (np.eye(K[t].shape[0]) - K[t] @ self.B)

            if not self.P_0_estimated:
                P_0[t] = T @ P_delta1[t] @ T.T + K[t] @ self.R @ K[t].T

            # y_hat_t^t
            y_h[t] = y_h_delta1[t] + K[t] @ (self.X[t] -
                                             self.B @ y_h_delta1[t])

        # y_hat_t-0^t-1
        y_h_delta1[t_end_filter - 1] = self.A @ y_h[t_end_filter - 2]

        if not self.P_delta1_estimated:
            # P_t-0^t-1
            P_delta1[t_end_filter -
                     1] = self.A @ P_0[t_end_filter - 2] @ self.A.T + self.Q
            self.P_delta1_estimated = True
            self.P_delta1 = P_delta1

        self.y_h = y_h
        self.K = K

        if not self.P_0_estimated:
            P_0[t] = T @ P_delta1[t] @ T.T + K[t] @ self.R @ K[t].T
            self.P_0_estimated = True
            self.P_0 = P_0

        self.y_h_delta1 = y_h_delta1

        return y_h, K, P_0, P_delta1, y_h_delta1

    def smooth(self, t_end_smooth=None):
        if t_end_smooth is None:
            t_end_smooth = 0

        if self.P_tau_estimated:
            P_tau = self.P_tau
        else:
            # P_t-1^tau
            P_tau = np.zeros((self.tau - t_end_smooth, self.d, self.d))
            P_tau[-1] = self.P_delta1[self.tau - 1]

        # y_hat_t^tau
        y_h_tau = np.zeros((self.tau - t_end_smooth, self.d))
        y_h_tau[-1] = self.y_h[self.tau - 2]

        # J_t
        J = np.zeros((self.tau - t_end_smooth - 1, self.d, self.d))

        for t in range(1, self.tau - t_end_smooth):
            P_inv = np.linalg.inv(self.P_delta1[-t])
            J[-t] = self.P_0[-t] @ self.A.T @ P_inv
            y_h_tau[-t - 1] = (self.y_h[-t] +
                               J[-t] @ (y_h_tau[-t] - self.y_h_delta1[-t]))

            if t < self.tau - t_end_smooth - 1 and not self.P_tau_estimated:
                P_tau[-t - 1] = self.P_0[
                    -t - 1] + J[-t] @ (P_tau[-t] - self.P_delta1[-t]) @ J[-t].T

        P_tau[0] = self.P_0[0] + J[0] @ (P_tau[1] - self.P_delta1[1]) @ J[0].T
        self.y_h_tau = y_h_tau

        if not self.P_tau_estimated:
            self.P_tau_estimated = True
            self.P_tau = P_tau
        self.J = J

        return y_h_tau, P_tau, J

    def lag_one_covar_smoother(self, t_end_smooth=0):
        # P_(t)(t-1)^tau
        P_1_tau = np.zeros((self.tau - t_end_smooth - 1, self.d, self.d))
        P_1_tau[-1] = (np.eye(self.d) -
                       self.K[-1] @ self.B) @ self.A @ self.P_0[-1]

        for t in range(1, self.tau - t_end_smooth - 1):
            P_1_tau[-t - 1] = (
                self.P_tau[-t] @ self.J[-t].T + self.J[-t + 1]
                @ (P_1_tau[-t] - self.A @ self.P_0[-t - 1]) @ self.J[-t].T)
        self.P_1_tau = P_1_tau

        return P_1_tau

    def e_step(self):
        M_0 = np.zeros(self.P_tau.shape)
        M_1 = np.zeros(self.P_1_tau.shape)

        if self.y_h_tau.shape[0] != self.tau:
            raise RuntimeError

        for i in range(M_0.shape[0]):
            M_0[i] = self.P_tau[i] + np.outer(self.y_h_tau[i], self.y_h_tau[i])
            if i >= 1:
                M_1[i - 1] = self.P_1_tau[i - 1] + np.outer(
                    self.y_h_tau[i], self.y_h_tau[i - 1])

        mu_new = self.y_h_tau[0]
        Sigma_new = self.P_tau[0]
        A_new = np.sum(M_1, axis=0) @ np.linalg.inv(np.sum(M_0[:-1], axis=0))

        Q_new = np.mean(M_0[1:] - np.einsum('ij, ...jk->...ik', A_new, M_1),
                        axis=0)

        B_new = (np.sum(np.einsum('...i,...j->...ij', self.X, self.y_h_tau),
                        axis=0) @ np.linalg.inv(np.sum(M_0, axis=0)))

        R_new = (
            np.einsum('...i,...j->...ij', self.X, self.X) -
            np.einsum('ij,...jk->...ik', B_new,
                      np.einsum('...i,...j->...ij', self.y_h_tau, self.X)))
        R_new = np.mean(R_new, axis=0)

        self.mu = mu_new
        self.Sigma = Sigma_new
        self.A = A_new
        self.Q = Q_new
        self.B = B_new
        self.R = R_new

        self.P_0_estimated = False
        self.P_delta1_estimated = False

        self.P_tau_estimated = False
        return mu_new, Sigma_new, A_new, Q_new, B_new, R_new

    def loglikelihood(self):
        x_hat_1 = np.einsum('ij,..k-->..jk', self.B, self.y_h_delta1)
