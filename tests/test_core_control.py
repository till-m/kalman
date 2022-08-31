import numpy as np
import kalman
from pytest import approx, raises, warns
from kalman.primitives import verify_control

from .utils import add_noise, example5_params, example9_params, example10_params

RANDOM_STATE = 12
np.random.seed(RANDOM_STATE)


def test_no_C():
    X, params = example9_params()
    U = np.random.random((10, 3))
    kalmod = kalman.KalmanModel()
    with raises(RuntimeError):
        kalmod.set_params(X, params, U=U)


def test_malformed_C_axis0():
    _, params = example9_params()
    C = np.random.random((2, 3))

    dict_ = params.to_dict()
    dict_['C'] = C
    with raises(ValueError):
        params = kalman.KalmanParams(dict_)


def test_malformed_C_axis1():
    X, params = example9_params()
    C = np.random.random((6, 4))
    U = np.random.random((X.shape[0]-4, 3))

    dict_ = params.to_dict()
    dict_['C'] = C
    params = kalman.KalmanParams(dict_)
    kalmod = kalman.KalmanModel()
    with raises(ValueError):
        kalmod.set_params(X, params, U=U)


def test_C_no_U():
    X, params = example9_params()
    C = np.random.random((params.latent_dim, 3))
    #U = np.random.random((X.shape[-2], 3))

    dict_ = params.to_dict()
    dict_['C'] = C
    params = kalman.KalmanParams(dict_)

    assert params.has_control

    kalmod = kalman.KalmanModel()
    with warns(RuntimeWarning, match="will be ignored"):
        kalmod.set_params(X, params)


def test_verify_control():
    C = np.array([
        [1., 1.],
        [1., 0.]
    ])

    u = np.array([1., 0.])

    with raises(RuntimeError):
        verify_control(u, None)
    
    with raises(ValueError):
        verify_control(u[0], C)
    
    with warns(RuntimeWarning):
        assert not verify_control(None, C)
    
    assert verify_control(u, C)


def test_filter_step_control():
    X, U, params = example10_params()

    y = params.mu  # initial estimate
    P = params.Sigma  # initial variance estimate

    y_est_hat = []
    P_est_hat = []

    print(U.shape)
    for i in range(len(X)):
        _, _, _, y, P = kalman.filter_step(X[i], y, P, params.A, params.Q,
                                           params.B, params.R, u=U[i], C=params.C)
        y_est_hat.append(y)
        P_est_hat.append(P)
    y_est_hat = np.array(y_est_hat)
    P_est_hat = np.array(P_est_hat)

    assert np.array([-18.35, -1.94]) == approx(y_est_hat[0], rel=0.1)
    assert np.array([
            [228.2, 53.7],
            [53.7, 483.2]
        ]) == approx(P_est_hat[0], rel=0.1)

    assert np.array([-15.1, 7.3]) == approx(y_est_hat[1], rel=0.1)
    assert np.array([
            [166.5, 101.9],
            [101.9, 438.8]
        ]) == approx(P_est_hat[1], rel=0.1)


    assert np.array([776.7, 215.4]) == approx(y_est_hat[-1], rel=0.1)
    assert np.array([
            [49.3, 9.7],
            [9.7, 2.6]
        ]) == approx(P_est_hat[-1], rel=0.1)


def test_filter_step_control():
    X, U, params = example10_params()

    y = params.mu  # initial estimate
    P = params.Sigma  # initial variance estimate

    y_est_hat = []
    P_est_hat = []

    for i in range(len(X)):
        _, _, _, y, P = kalman.filter_step(X[i], y, P, params.A, params.Q,
                                           params.B, params.R, u=U[i], C=params.C)
        y_est_hat.append(y)
        P_est_hat.append(P)
    y_est_hat = np.array(y_est_hat)
    P_est_hat = np.array(P_est_hat)

    assert np.array([-18.35, -1.94]) == approx(y_est_hat[0], rel=0.1)
    assert np.array([
            [228.2, 53.7],
            [53.7, 483.2]
        ]) == approx(P_est_hat[0], rel=0.1)

    assert np.array([-15.1, 7.3]) == approx(y_est_hat[1], rel=0.1)
    assert np.array([
            [166.5, 101.9],
            [101.9, 438.8]
        ]) == approx(P_est_hat[1], rel=0.1)


    assert np.array([776.7, 215.4]) == approx(y_est_hat[-1], rel=0.1)
    assert np.array([
            [49.3, 9.7],
            [9.7, 2.6]
        ]) == approx(P_est_hat[-1], rel=0.1)


def test_filter_control():
    X, U, params = example10_params()

    y = params.mu  # initial estimate
    P = params.Sigma  # initial variance estimate

    y_est_hat = []
    P_est_hat = []

    print(U.shape)
    for i in range(len(X)):
        _, _, _, y, P = kalman.filter_step(X[i], y, P, params.A, params.Q,
                                           params.B, params.R, u=U[i], C=params.C)
        y_est_hat.append(y)
        P_est_hat.append(P)
    y_est_hat = np.array(y_est_hat)
    P_est_hat = np.array(P_est_hat)

    assert np.array([-18.35, -1.94]) == approx(y_est_hat[0], rel=0.1)
    assert np.array([
            [228.2, 53.7],
            [53.7, 483.2]
        ]) == approx(P_est_hat[0], rel=0.1)

    assert np.array([-15.1, 7.3]) == approx(y_est_hat[1], rel=0.1)
    assert np.array([
            [166.5, 101.9],
            [101.9, 438.8]
        ]) == approx(P_est_hat[1], rel=0.1)


    assert np.array([776.7, 215.4]) == approx(y_est_hat[-1], rel=0.1)
    assert np.array([
            [49.3, 9.7],
            [9.7, 2.6]
        ]) == approx(P_est_hat[-1], rel=0.1)


def test_forecast_control():
    X, U, params = example10_params()

    n_train = 10
    kalmod = kalman.KalmanModel()
    X_train = X[:n_train]
    kalmod.set_params(X_train, params, U=U)
    y_t_tau_pr, P_t_tau_pr, X_t_tau_pr, X_cov_t_tau_pr = kalmod.forecast(len(X), estimate_covs=True)

    kalmod2 = kalman.KalmanModel()
    kalmod2.set_params(X, params, U=U)

    y_t_t , P_t_t, _, _, _ = kalmod2.filter()

    assert X[n_train:].flatten() == approx(
        X_t_tau_pr[n_train:].flatten(),
        abs=np.max(np.einsum('...ii->...i', X_cov_t_tau_pr[n_train:]))
    )


    assert y_t_tau_pr[n_train:] == approx(
            y_t_t[n_train:], np.max(np.einsum('...ii->...i', P_t_tau_pr[n_train:])) + 
                np.max(np.einsum('...ii->...i', P_t_t[n_train:]))
        )


def test_parameter_estimation_dynamic_loglikelihood():
    X, U, params = example10_params()

    noisy_params = add_noise(params, random_state=RANDOM_STATE)

    kalmod1 = kalman.KalmanModel(verbose=True)
    kalmod1.set_params(X, noisy_params, U=U)

    kalmod1.fit(n_it=3)

    kalmod2 = kalman.KalmanModel(verbose=True)
    kalmod2.set_params(X, noisy_params, U=U)

    kalmod2.fit(n_it=100)

    assert kalmod1.loglikelihood() <= kalmod2.loglikelihood()
