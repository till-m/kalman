from warnings import warn
import numpy as np
import copy
from .primitives import multivar_normal_loglikelihood, KalmanParams, filter_step, predict_step, smooth_step, matmul_inv
from icecream import ic

is_sym = lambda a: np.allclose(a, np.swapaxes(a, -1, -2))

class KalmanModel():
    # Using Max Welling's notation
    def __init__(self, verbose=False) -> None:
        self.verbose = verbose

    def set_params(self, X: np.ndarray, params: KalmanParams, U=None):
        self.verify_params(X, U, params)
        self.params = copy.deepcopy(params)

        self.X = X
        self.U = U
        self.tau = X.shape[-2]
        self.d = self.params.mu.size
        self.k = X.shape[-1]

        self._has_control = params.C is not None and U is not None

        self.calculate_filter_cov = True
        self.calculate_smooth_cov = True

        return self

    def verify_params(self, X, U, params):
        #TODO: This needs to go once/if we want to fit multiple runs 
        if len(X.shape) != 2:
            raise RuntimeError

        if X.shape[1] != params.out_dim:
            raise ValueError(f"Dimension mismatch between X and params. " +
                f"Expected X to have length {params.out_dim} " +
                f"along axis 1, not {X.shape[1]}")

        if U is not None:
            if params.C is None:
                raise RuntimeError("Control U provided, but no C in params. ")
            if X.shape[-2] - 1 > U.shape[-2]:
                raise ValueError(f"Dimension mismatch between X and U." +
                    f"Expected U to have at least length {X.shape[-2] - 1} " +
                    f"along axis 0, not {U.shape[-2]}")
        else:
            if params.C is not None:
                msg = "C in params but no U provided, control will be ignored."
                warn(msg, RuntimeWarning)

    @property
    def has_control(self):
        return self._has_control

    def fit(
        self,
        mode='estimate',
        n_it=10,
    ):

        if mode == 'filter':
            self.filter()
            return self.y_t_t, self.P_t_t
        elif mode == 'smooth':
            self.filter()
            self.smooth()
            return self.y_t_tau, self.P_t_tau
        elif mode == 'estimate':
            for i in range(n_it):
                if self.verbose:
                    print(f"\n++++++++++++++  {i}  ++++++++++++++")
                # E step
                self.filter()
                self.smooth()
                self.lag_one_covar_smoother()
                if self.verbose:
                    try:
                        print(self.loglikelihood())
                    except ValueError:
                        print("Calculation of loglikelihood failed.")
                        print("This is likely due to numerical inaccuracies.")

                # M step
                self.m_step()
            self.filter()
            self.smooth()
            if self.verbose:
                print(f"\n++++++++  Final estimate  +++++++++")
                try:
                    print(self.loglikelihood())
                except ValueError:
                    print("Calculation of loglikelihood failed.")
                    print("This is likely due to numerical inaccuracies.")
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

        K[0] = P_t_t1[0] @ matmul_inv(self.params.B.T, (self.params.R + self.params.B @ P_t_t1[0] @ self.params.B.T))

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
                    self.params.R,
                    u=self.U[t] if self._has_control else None,
                    C=self.params.C)
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
                    y_est_cov=P_t_t[t],
                    u=self.U[t] if self._has_control else None,
                    C=self.params.C)

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
                    self.params.A,
                    u=self.U[t] if self._has_control else None,
                    C=self.params.C
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
                    estimate_covs=False,
                    u=self.U[t] if self._has_control else None,
                    C=self.params.C
                )

        self.y_t_tau = y_t_tau

        if self.calculate_smooth_cov:
            self.calculate_smooth_cov = False
            self.P_t_tau = P_t_tau
        self.J = J

        return y_t_tau, P_t_tau, J

    def forecast(self, t_end, estimate_covs=False):
        y_t_tau_pr = np.empty((t_end, self.params.latent_dim))
        P_t_tau_pr = np.empty((
            t_end,
            self.params.latent_dim,
            self.params.latent_dim
        ))

        X_t_tau_pr = np.empty((t_end, self.k))
        X_cov_t_tau_pr = np.empty((t_end, self.k, self.k))

        self.filter()
        self.smooth()
        

        y_t_tau_pr[:self.tau] = self.y_t_tau
        P_t_tau_pr[:self.tau] = self.P_t_tau

        X_t_tau_pr[:self.tau] = self.X

        t_start = len(self.y_t_tau)

        for i in range(t_start, t_end):
            if estimate_covs:
                y_t_tau_pr[i], P_t_tau_pr[i], X_t_tau_pr[i], X_cov_t_tau_pr[i] = predict_step(
                    y_t_tau_pr[i-1],
                    self.params.A,
                    self.params.B,
                    estimate_covs=True,
                    P_est_prev=P_t_tau_pr[i-1],
                    Q=self.params.Q,
                    R=self.params.R,
                    u=self.U[i-1] if self.U is not None else None,
                    C=self.params.C
                    )
            else:
                y_t_tau_pr[i], X_t_tau_pr[i] = predict_step(
                    y_t_tau_pr[i-1],
                    self.params.A,
                    self.params.B,
                    u=self.U[i-1] if self.U is not None else None,
                    estimate_covs=False
                    )

        if estimate_covs:
            return y_t_tau_pr, P_t_tau_pr, X_t_tau_pr, X_cov_t_tau_pr

        return y_t_tau_pr, X_t_tau_pr

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

        if self.has_control:
            N_0 = np.zeros((self.tau-1, *self.params.C.T.shape))
            N_1 = np.zeros((self.tau-1, *self.params.C.T.shape))
        if self.y_t_tau.shape[0] != self.tau:
            raise RuntimeError

        for i in range(M_0.shape[0]):
            M_0[i] = self.P_t_tau[i] + np.outer(self.y_t_tau[i],
                                                self.y_t_tau[i])
            if i >= 1:
                M_1[i - 1] = self.P_tt1_tau[i - 1] + np.outer(self.y_t_tau[i], self.y_t_tau[i - 1])

                if self.has_control:
                    N_0[i - 1] = np.outer(self.U[i], self.y_t_tau[i])
                    N_1[i - 1] = np.outer(self.U[i], self.y_t_tau[i - 1])

        mu_new = self.y_t_tau[0]

        Sigma_new = self.P_t_tau[0]  # + cov y_1 if considering multiple runs
        if self.has_control:
            # Construct C_new
            G = np.linalg.solve(np.mean(M_0[:-1], axis=0), np.mean(N_1, axis=0).T)

            C_new = matmul_inv(
                np.mean(N_0, axis=0).T - np.mean(M_1, axis=0) @ G,
                np.mean(np.einsum('...i,...j->...ij', self.U, self.U), axis=0) + 
                    np.mean(N_1, axis=0) @ G
            )
            A_new = matmul_inv(
                np.mean(M_1, axis=0) - C_new @ np.mean(N_1, axis=0),
                np.mean(M_0[:-1], axis=0))

            # Sum of means is more stable than mean of sums
            #Q_new = (
            #    np.mean(M_0[1:], axis=0)
            #    - (C_new @ np.mean(N_0, axis=0)).T
            #    - A_new @ np.mean(M_1, axis=0).T
            #    + A_new @ (C_new @ np.mean(N_1, axis=0)).T
            #    - C_new @ np.mean(N_0, axis=0)
            #    + C_new @ np.mean(N_1, axis=0) @ A_new.T
            #    + C_new @ np.mean(np.einsum('...i,...j->...ij', self.U, self.U), axis=0) @ C_new.T
            #)
            Q_new = (
                np.mean(M_0[1:], axis=0)
                - 2*(C_new @ np.mean(N_0, axis=0)).T
                - A_new @ np.mean(M_1, axis=0).T
                #+ A_new @ (C_new @ np.mean(N_1, axis=0)).T
                #- C_new @ np.mean(N_0, axis=0)
                + 2* C_new @ np.mean(N_1, axis=0) @ A_new.T
                + C_new @ np.mean(np.einsum('...i,...j->...ij', self.U, self.U), axis=0) @ C_new.T
            )
            Q_new = (Q_new + Q_new.T)/2
            ic(Q_new)
        else:
            A_new = matmul_inv(np.mean(M_1, axis=0), np.mean(M_0[:-1], axis=0))

            # NB: Transpose is handled during einsum
            # Using mean instead of 1/xyz * sum
            Q_new = np.mean(M_0[1:], axis=0) - np.einsum('ij, kj->ik', A_new, np.mean(M_1, axis=0))

        B_new = matmul_inv(np.sum(np.einsum('...i,...j->...ij', self.X, self.y_t_tau),
                        axis=0), np.sum(M_0, axis=0))
        
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
        if self.has_control:
            self.params.C = C_new

        self.calculate_filter_cov = True
        self.calculate_smooth_cov = True

        return mu_new, Sigma_new, A_new, Q_new, B_new, R_new

    def likelihood(self):
        return np.exp(self.loglikelihood())

    def measurements(self, calculate_cov=False):
        x_hat_01 = self.params.B @ self.params.mu
        x_hat_1 = np.einsum('ij,...j->...i', self.params.B, self.y_t_t1[1:])
        if calculate_cov:
            H_1 = self.params.R + self.params.B @ np.einsum('...ij,jk->...ik', self.P_t_t1[1:], self.params.B.T)
            H_01 = self.params.R + self.params.B @ self.params.Sigma @ self.params.B.T
            
            return (np.vstack((np.array([x_hat_01]), x_hat_1)),
                    np.vstack((np.array([H_01]), H_1)))
            
        return np.vstack((np.array([x_hat_01]), x_hat_1))
    
    def loglikelihood(self):
        X_est, X_est_cov = self.measurements(calculate_cov=True)
        return multivar_normal_loglikelihood(self.X, X_est, X_est_cov)
