import numpy as np
import kalman
from pytest import approx, raises

from .utils import add_noise, example5_params, example9_params

RANDOM_STATE = 12
np.random.seed(RANDOM_STATE)


def test_filter_static_no_noise():
    X, params = example5_params()

    # Taken from https://www.kalmanfilter.net/kalman1d.html
    y_est = np.array(
        [49.69, 48.47, 50.57, 51.68, 51.33, 49.62, 49.21, 49.31, 49.53, 49.57])
    P_est = np.array([[22.5], [11.84], [8.04], [6.08], [4.89], [4.09], [3.52],
                      [3.08], [2.74], [2.47]])

    kalmod = kalman.KalmanModel()
    kalmod.set_params(X, params)
    y_est_hat, P_est_hat = kalmod.fit(mode='filter')

    assert np.squeeze(y_est_hat) == approx(y_est, rel=0.01)
    assert np.squeeze(P_est_hat) == approx(np.squeeze(P_est), rel=0.01)


def test_malformed_X():
    X, params = example9_params()

    kalmod = kalman.KalmanModel()
    with raises(ValueError):
        kalmod.set_params(X[:,:1], params)


def test_filter_dynamic():
    X, params = example9_params()

    kalmod = kalman.KalmanModel()
    kalmod.set_params(X, params)
    y_est_hat, P_est_hat = kalmod.fit(mode='filter')

    assert np.array([298.44, 0.25, -1.9, 3.31, -26.2,
                     -0.65]) == approx(y_est_hat[-1], rel=0.1)

    # re-run to check that estimated covariances are not recalculated
    _, _ = kalmod.fit(mode='filter')

    _, P_smoothed = kalmod.fit(mode='smooth')

    assert (np.linalg.det(P_smoothed) <= np.linalg.det(P_est_hat)).all()

    _, P_smoothed2 = kalmod.fit(mode='smooth')

    assert (P_smoothed == P_smoothed2).all()


def test_check_X_dims():
    X, params = example5_params()

    kalmod = kalman.KalmanModel()

    with raises(RuntimeError):
        kalmod.set_params(X.squeeze(), params)


def test_forecast_dynamic_series():
    X, params = example9_params()

    #X = X[:25]
    kalmod = kalman.KalmanModel()
    X_train = X[:20]
    kalmod.set_params(X_train, params)
    y_t_tau_pr, P_t_tau_pr, X_t_tau_pr, X_cov_t_tau_pr = kalmod.forecast(len(X), estimate_covs=True)

    kalmod2 = kalman.KalmanModel()
    kalmod2.set_params(X, params)
    y_t_t , P_t_t, _, _, _ = kalmod2.filter()

    assert X[20:].flatten() == approx(
        X_t_tau_pr[20:].flatten(),
        abs=np.max(np.einsum('...ii->...i', X_cov_t_tau_pr[20:]))
    )

    print(y_t_tau_pr)
    assert y_t_tau_pr[20:] == approx(
            y_t_t[20:], np.max(np.einsum('...ii->...i', P_t_tau_pr[20:])) + 
                np.max(np.einsum('...ii->...i', P_t_t[20:]))
        )


def test_parameter_estimation_static():
    X, params = example5_params()

    noisy_params = add_noise(params, random_state=RANDOM_STATE)

    kalmod = kalman.KalmanModel()
    kalmod.set_params(X, noisy_params)

    kalmod.fit(n_it=5)

    kalmod2 = kalman.KalmanModel().set_params(X, noisy_params)
    kalmod2.fit(mode='filter')

    assert kalmod.loglikelihood() > kalmod2.loglikelihood()


def test_parameter_estimation_dynamic():
    X, params = example9_params()

    R = params.R

    A = params.A
    B = params.B
    Q = params.Q  # True building height is noiseless

    noisy_params = add_noise(params, random_state=RANDOM_STATE)

    kalmod = kalman.KalmanModel()
    kalmod.set_params(X, noisy_params)

    kalmod.fit()

    fitted_params = kalmod.params

    assert (np.linalg.norm(A - noisy_params.A) <
            np.linalg.norm(A - fitted_params.A))
    assert (np.linalg.norm(B - noisy_params.B) <
            np.linalg.norm(B - fitted_params.B))
    assert (np.linalg.norm(Q - noisy_params.Q) <
            np.linalg.norm(Q - fitted_params.Q))
    assert (np.linalg.norm(R - noisy_params.R) <
            np.linalg.norm(R - fitted_params.R))


def test_parameter_estimation_static_loglikelihood():
    X, params = example5_params()

    noisy_params = add_noise(params, random_state=RANDOM_STATE)

    kalmod1 = kalman.KalmanModel()
    kalmod1.set_params(X, noisy_params)

    kalmod1.fit(n_it=3)

    kalmod2 = kalman.KalmanModel()
    kalmod2.set_params(X, noisy_params)

    kalmod2.fit(n_it=100)

    assert kalmod1.loglikelihood() <= kalmod2.loglikelihood()


def test_parameter_estimation_dynamic_loglikelihood():
    X, params = example9_params()

    noisy_params = add_noise(params, random_state=RANDOM_STATE)

    kalmod1 = kalman.KalmanModel(verbose=True)
    kalmod1.set_params(X, noisy_params)

    kalmod1.fit(n_it=3)

    kalmod2 = kalman.KalmanModel(verbose=True)
    kalmod2.set_params(X, noisy_params)

    kalmod2.fit(n_it=10)

    assert kalmod1.loglikelihood() <= kalmod2.loglikelihood()
