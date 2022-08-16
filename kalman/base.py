import numpy as np
from scipy.stats import multivariate_normal

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

        self.P_t_t_estimated = False
        self.P_t_t1_estimated = False

        self.P_t_tau_estimated = False

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
                #print(self.loglikelihood())
            if predict_params:
                self.lag_one_covar_smoother()
                self.e_step()
                #print(self.loglikelihood())
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

        K = np.zeros(shape=(t_end_filter, self.d, self.k))

        y_t_t = np.zeros(shape=(t_end_filter, self.d))

        if self.P_t_t_estimated:
            P_t_t = self.P_t_t
        else:
            P_t_t = np.zeros(shape=(t_end_filter, self.d, self.d))

        for t in range(t_end_filter):
            if t == 0:
                # y_hat_t^t-1 -- has offset of 2, i.e. K[t] = K_(t+1)
                y_t_t1 = np.zeros(shape=(t_end_filter, self.d))
                y_t_t1[0] = self.mu

                if self.P_t_t1_estimated:
                    P_t_t1 = self.P_t_t1
                else:
                    # P_t^t-1 -- has offset of 2, i.e. K[t] = K_(t+1)
                    P_t_t1 = np.zeros(shape=(t_end_filter, self.d, self.d))
                    P_t_t1[0] = self.Sigma

            else:
                # y_hat_t-0^t-1
                y_t_t1[t] = self.A @ y_t_t[t - 1]

                if not self.P_t_t1_estimated:
                    # P_t-0^t-1
                    P_t_t1[t] = self.A @ P_t_t[t - 1] @ self.A.T + self.Q

            K[t] = P_t_t1[t] @ self.B.T @ np.linalg.inv(
                self.R + self.B @ P_t_t1[t] @ self.B.T)
            # P_t^t (numerically stable version)
            T = (np.eye(K[t].shape[0]) - K[t] @ self.B)

            if not self.P_t_t_estimated:
                P_t_t[t] = T @ P_t_t1[t] @ T.T + K[t] @ self.R @ K[t].T

            # y_hat_t^t
            y_t_t[t] = y_t_t1[t] + K[t] @ (self.X[t] -
                                             self.B @ y_t_t1[t])

        if not self.P_t_t1_estimated:
            self.P_t_t1_estimated = True
            self.P_t_t1 = P_t_t1

        self.y_t_t = y_t_t
        self.K = K

        if not self.P_t_t_estimated:
            self.P_t_t_estimated = True
            self.P_t_t = P_t_t

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
            J[-t] = self.P_t_t[-t-1] @ self.A.T @ P_inv
            y_t_tau[-t - 1] = (self.y_t_t[-t - 1] +
                               J[-t] @ (y_t_tau[-t] - self.y_t_t1[-t]))
            if not self.P_t_tau_estimated:
                P_t_tau[-t - 1] = (self.P_t_t[-t-1] +
                            J[-t] @ (P_t_tau[-t] - self.P_t_t1[-t]) @ J[-t].T)
        self.y_t_tau = y_t_tau

        if not self.P_t_tau_estimated:
            self.P_t_tau_estimated = True
            self.P_t_tau = P_t_tau
        self.J = J

        return y_t_tau, P_t_tau, J

    def lag_one_covar_smoother(self, t_end_smooth=0):
        # P_(t)(t-1)^tau
        P_tt1_tau = np.zeros((self.tau - t_end_smooth - 1, self.d, self.d))
        P_tt1_tau[-1] = (np.eye(self.d) -
                       self.K[-1] @ self.B) @ self.A @ self.P_t_t[-1]

        for t in range(1, self.tau - t_end_smooth - 1):
            P_tt1_tau[-t - 1] = (
                self.P_t_tau[-t] @ self.J[-t - 1].T + self.J[-t]
                @ (P_tt1_tau[-t] - self.A @ self.P_t_t[-t - 1]) @ self.J[-t - 1].T)

        self.P_tt1_tau = P_tt1_tau

        return P_tt1_tau

    def e_step(self):
        M_0 = np.zeros(self.P_t_tau.shape)
        M_1 = np.zeros(self.P_tt1_tau.shape)

        if self.y_t_tau.shape[0] != self.tau:
            raise RuntimeError

        for i in range(M_0.shape[0]):
            M_0[i] = self.P_t_tau[i] + np.outer(self.y_t_tau[i], self.y_t_tau[i])
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

        self.mu = mu_new
        self.Sigma = Sigma_new
        self.A = A_new
        self.Q = Q_new
        self.B = B_new
        self.R = R_new

        self.P_t_t_estimated = False
        self.P_t_t1_estimated = False
        self.P_t_tau_estimated = False

        return mu_new, Sigma_new, A_new, Q_new, B_new, R_new

    def loglikelihood(self):
        x_hat_1 = np.einsum('ij,...j->...i', self.B, self.y_t_t1)
        x_hat_01 = self.B @ self.mu

        H_1 = self.R + self.B @ np.einsum('...ij,jk->...ik', self.P_t_t1[1:],
                                          self.B.T)
        H_01 = self.R + self.B @ self.Sigma @ self.B.T

        res = np.log(
            multivariate_normal(mean=x_hat_01, cov=H_01).pdf(self.X[0]))
        for t in range(1, self.X.shape[0]):
            res += np.log(
                multivariate_normal(mean=x_hat_1[t - 1],
                                    cov=H_1[t - 1]).pdf(self.X[t]))

        return res