from random import random
import simdkalman

import numpy as np
import kalman
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from pytest import approx, raises
from icecream import ic

from .utils import add_noise, example5_params, example9_params

RANDOM_STATE = 12
np.random.seed(RANDOM_STATE)


def test_filter_dynamic():
    X, params = example9_params()

    kalmod = kalman.KalmanModel()
    kalmod.set_params(X, params)
    y_est_hat, P_est_hat = kalmod.fit(mode='filter')

    kf = simdkalman.KalmanFilter(
        state_transition=params.A,
        process_noise=params.Q,
        observation_model=params.B,
        observation_noise=params.R
    )

    res = kf.compute(
        X.reshape((1,) + X.shape),
        0,
        params.mu,
        params.Sigma,
        filtered=True,
        smoothed = False,
        states = True,
        covariances = True,
        observations = True,
        verbose = False)

    assert approx(res.filtered.states.mean.squeeze()) == y_est_hat
    assert approx(res.filtered.states.cov.squeeze()) == P_est_hat


def test_filter_dynamic_noise():
    X, params = example9_params()

    noisy_params = add_noise(params, random_state=RANDOM_STATE)

    kalmod = kalman.KalmanModel()
    kalmod.set_params(X, noisy_params)
    y_est_hat, P_est_hat = kalmod.fit(mode='filter')

    kf = simdkalman.KalmanFilter(
        state_transition=noisy_params.A,
        process_noise=noisy_params.Q,
        observation_model=noisy_params.B,
        observation_noise=noisy_params.R
    )

    res = kf.compute(
        X.reshape((1,) + X.shape),
        0,
        noisy_params.mu,
        noisy_params.Sigma,
        filtered=True,
        smoothed = False,
        states = True,
        covariances = True,
        observations = True,
        verbose = False)

    assert approx(res.filtered.states.mean.squeeze()) == y_est_hat
    assert approx(res.filtered.states.cov.squeeze()) == P_est_hat


def test_smooth_dynamic():
    X, params = example9_params()

    kalmod = kalman.KalmanModel()
    kalmod.set_params(X, params)
    y_est_hat, P_est_hat = kalmod.fit(mode='smooth')

    kf = simdkalman.KalmanFilter(
        state_transition=params.A,
        process_noise=params.Q,
        observation_model=params.B,
        observation_noise=params.R
    )

    res = kf.compute(
        X.reshape((1,) + X.shape),
        0,
        params.mu,
        params.Sigma,
        filtered=True,
        smoothed = True,
        states = True,
        covariances = True,
        observations = True,
        verbose = False)

    assert approx(res.smoothed.states.mean.squeeze()) == y_est_hat
    assert approx(res.smoothed.states.cov.squeeze()) == P_est_hat


def test_smooth_dynamic_noise():
    X, params = example9_params()

    noisy_params = add_noise(params, random_state=RANDOM_STATE)

    kalmod = kalman.KalmanModel()
    kalmod.set_params(X, noisy_params)
    y_est_hat, P_est_hat = kalmod.fit(mode='smooth')

    kf = simdkalman.KalmanFilter(
        state_transition=noisy_params.A,
        process_noise=noisy_params.Q,
        observation_model=noisy_params.B,
        observation_noise=noisy_params.R
    )

    res = kf.compute(
        X.reshape((1,) + X.shape),
        0,
        noisy_params.mu,
        noisy_params.Sigma,
        filtered=True,
        smoothed = True,
        states = True,
        covariances = True,
        observations = True,
        verbose = False)

    assert approx(res.smoothed.states.mean.squeeze()) == y_est_hat
    assert approx(res.smoothed.states.cov.squeeze()) == P_est_hat
    
