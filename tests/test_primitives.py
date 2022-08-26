import numpy as np
import kalman
from pytest import approx, raises, warns

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


def test_KalmanParams_malformed():
    _, params = example5_params()

    with raises(ValueError):
        _dict = params.to_dict()
        _dict['mu'] = np.empty((2, 2))
        kalman.KalmanParams(_dict)

    with raises(ValueError):
        _dict = params.to_dict()
        _dict['Sigma'] = np.empty((params.mu.size + 1, 2))
        kalman.KalmanParams(_dict)
    
    with raises(ValueError):
        _dict = params.to_dict()
        _dict['A'] = np.empty((params.mu.size + 1, 2))
        kalman.KalmanParams(_dict)
    
    with raises(ValueError):
        _dict = params.to_dict()
        _dict['Q'] = np.empty((params.mu.size + 1, 2))
        kalman.KalmanParams(_dict)
    
    with raises(ValueError):
        _dict = params.to_dict()
        _dict['B'] = np.empty((params.mu.size + 1, params.mu.size + 1))
        kalman.KalmanParams(_dict)
    
    with raises(ValueError):
        _dict = params.to_dict()
        _dict['R'] = np.empty((params.B.shape[0] + 1, params.B.shape[0] + 1))
        kalman.KalmanParams(_dict)
    
    assert params.latent_dim == params.mu.size


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


def test_filter_step_dynamic_steps():
    X, params = example9_params()

    y = params.mu  # initial estimate
    P = params.Sigma  # initial variance estimate

    y_est_hat = []
    P_est_hat = []
    for i in range(len(X)):
        _, _, _, y, P = kalman.filter_step(X[i], y, P, params.A, params.Q,
                                           params.B, params.R)
        y_est_hat.append(y)
        P_est_hat.append(P)
    y_est_hat = np.array(y_est_hat)
    P_est_hat = np.array(P_est_hat)

    assert np.array([298.44, 0.25, -1.9, 3.31, -26.2,
                     -0.65]) == approx(y_est_hat[-1], rel=0.1)


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
        y_t_tau, y_t_tau_cov, _ = kalman.smooth_step(y_t_tau,
                                                     y_t_tau_cov,
                                                     y_t_t1[-i - 2],
                                                     P_t_t1[-i - 2],
                                                     y_t_t[-i - 1],
                                                     P_t_t[-i - 1],
                                                     params.A)
        y_est_hat.insert(0, y_t_tau)
        P_est_hat.insert(0, y_t_tau_cov)

    y_est_hat = np.array(y_est_hat)
    P_est_hat = np.array(P_est_hat)

    # 1x1 covariance matrices so no determinant needed
    assert (P_est_hat <= P_t_t[:-1]).all()


def test_predict_step_y():
    X, params = example5_params()
    R = params.R

    A = params.A  # Problem is static -- building doesn't change height
    B = params.B  # We're measuring the hidden state
    Q = params.Q  # True building height is noiseless

    y = params.mu  # initial estimate
    P = params.Sigma  # initial variance estimate

    for i in range(len(X)):
        y_pred1, _ = kalman.predict_step(y, A, B)
        y_pred2, y_pred_cov2, _, _ = kalman.predict_step(y, A, B, True, P, Q, R)

        y_pred_f, y_pred_cov_f, _, y, P = kalman.filter_step(X[i], y, P, A, Q, B, R)

        assert y_pred1 == y_pred_f
        assert y_pred2 == y_pred_f
        assert y_pred_cov2 == y_pred_cov_f


def test_predict_step_x():
    X, params = example5_params()
    R = params.R

    A = params.A  # Problem is static -- building doesn't change height
    B = params.B  # We're measuring the hidden state
    Q = params.Q  # True building height is noiseless

    y = params.mu  # initial estimate
    P = params.Sigma  # initial variance estimate

    kalmod = kalman.KalmanModel()
    kalmod.set_params(X, params)

    kalmod.fit(mode='filter')

    X_est, X_cov_est = kalmod.measurements(True)
    for i in range(len(X)):
        _, _, x_pred, x_pred_cov = kalman.predict_step(
            kalmod.y_t_t1[i],
            A,
            B,
            True,
            kalmod.P_t_t1[i],
            Q,
            R
        )

        assert X_est[i] == x_pred
        assert X_cov_est[i] == x_pred_cov


def test_loglikelihood():
    n = 100
    d = 10
    X = (np.random.rand(n, d).T / np.random.rand(n)).T
    cov = np.array([random_covariance_matrix(d) for _ in range(n)])

    res = 1/np.sqrt(np.linalg.det(2*np.pi*cov))
    res = np.sum(np.log(res))

    assert approx(res) == kalman.multivar_normal_loglikelihood(X, X, cov)


def test_loglikelihood_inf():
    n = 100
    d = 10
    mu = (np.random.rand(n, d).T / np.random.rand(n)).T

    X = (np.random.rand(n, d).T / np.random.rand(n)).T
    X[0,0] = np.inf
    cov = np.array([random_covariance_matrix(d) for _ in range(n)])

    res = 1/np.sqrt(np.linalg.det(2*np.pi*cov))
    res = np.sum(np.log(res))
    
    with warns(RuntimeWarning):
        assert not np.isinf(kalman.multivar_normal_loglikelihood(X, mu, cov))
