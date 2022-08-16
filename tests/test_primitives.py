import numpy as np
import kalman
from pytest import approx, raises

from .utils import add_noise, example5_params, example9_params, random_covariance_matrix


def test_KalmanParams():
    _, params = example5_params()
    assert params.n_params == 6

    _dict = params.to_dict()

    assert params == _dict

    assert params == kalman.KalmanParams(_dict)

    assert str(params) == str(params.to_dict())

    with raises(TypeError):
        params == 1.2


def test_filter_step_static_no_noise():
    X, params = example5_params()
    R = params.R

    A = params.A  # Problem is static -- building doesn't change height
    B = params.B  # We're measuring the hidden state
    Q = params.Q  # True building height is noiseless

    y = params.mu  # initial estimate
    P = params.Sigma  # initial variance estimate

    # Taken from https://www.kalmanfilter.net/kalman1d.html
    y_est = np.array(
        [49.69, 48.47, 50.57, 51.68, 51.33, 49.62, 49.21, 49.31, 49.53, 49.57])
    P_est = np.array([[22.5], [11.84], [8.04], [6.08], [4.89], [4.09], [3.52],
                      [3.08], [2.74], [2.47]])

    y_est_hat = []
    P_est_hat = []
    for i in range(len(X)):
        _, _, _, y, P = kalman.filter_step(X[i], y, P, A, Q, B, R)
        y_est_hat.append(y[0])
        P_est_hat.append(P[0])

    y_est_hat = np.array(y_est_hat)
    P_est_hat = np.array(P_est_hat)
    assert y_est_hat == approx(y_est, rel=0.01)
    assert P_est_hat == approx(P_est, rel=0.01)


def test_smooth_step_static_no_noise():
    X, params = example5_params()
    R = params.R

    A = params.A  # Problem is static -- building doesn't change height
    B = params.B  # We're measuring the hidden state
    Q = params.Q  # True building height is noiseless

    y = params.mu  # initial estimate
    P = params.Sigma  # initial variance estimate

    y_t_t = []
    P_t_t = []

    y_t_t1 = []
    P_t_t1 = []

    # filter
    for i in range(len(X)):
        y_pred, y_pred_cov, _, y, P = kalman.filter_step(
            X[i], y, P, A, Q, B, R)
        y_t_t.append(y[0])
        P_t_t.append(P[0])

        y_t_t1.append(y_pred)
        P_t_t1.append(y_pred_cov)

    y_est_hat = []
    P_est_hat = []

    y_t_tau = y_t_t[-1]
    y_t_tau_cov = P_t_t[-1]

    # smooth
    for i in range(len(X) - 1):
        y_t_tau, y_t_tau_cov, _ = kalman.smooth_step(y_t_tau, y_t_tau_cov,
                                                     y_t_t1[-i - 2],
                                                     P_t_t1[-i - 2],
                                                     y_t_t[-i - 1],
                                                     P_t_t[-i - 1], params.A)
        y_est_hat.insert(0, y_t_tau)
        P_est_hat.insert(0, y_t_tau_cov)

    y_est_hat = np.array(y_est_hat)
    P_est_hat = np.array(P_est_hat)

    # 1x1 covariance matrices so no determinant needed
    assert (P_est_hat <= P_t_t[:-1]).all()

def test_loglikelihood():
    n = 100
    d = 10
    X = (np.random.rand(n, d).T / np.random.rand(n)).T
    cov = np.array([random_covariance_matrix(d) for _ in range(n)])

    res = 1/np.sqrt(np.linalg.det(2*np.pi*cov))
    res = np.sum(np.log(res))

    assert approx(res) == kalman.multivar_normal_loglikelihood(X, X, cov)
