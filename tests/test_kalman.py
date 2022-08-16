import numpy as np
import kalman
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from pytest import approx
from icecream import ic
from .utils import example5_params, example9_params
np.random.seed(12)

to_cov = lambda M: M @ M.T

def summarize(kmod):
    print(f"Loss (filtered): {np.mean(all_y - kmod.y_t_t)**2}")
    print(f"Loss (smoothed): {np.mean(all_y - kmod.y_t_tau)**2}")

    plt.plot(range(1, len(all_y) + 1), all_y[:, 0], label='true')
    plt.plot(range(1,
                   len(all_y) + 1),
             kmod.y_t_t[:, 0],
             '.--',
             label='filtered')
    plt.plot(range(1,
                   len(all_y) + 1),
             kmod.y_t_tau[:, 0],
             '.-.',
             label='smoothed')
    plt.legend(loc='best')
    plt.show()


def generate_data(params: kalman.KalmanParams, tau: int):

    all_y = [multivariate_normal(params.mu, params.Sigma).rvs()]
    for t in range(1, tau):
        y_t = params.A @ all_y[-1] + multivariate_normal(cov=params.Q).rvs()
        all_y.append(y_t)

    all_x = []
    for t in range(tau):
        x_t = params.B @ all_y[t] + multivariate_normal(cov=params.R).rvs()
        all_x.append(x_t)

    all_x = np.array(all_x)
    all_y = np.array(all_y)

    return all_x, all_y


def toy_problem():
    # FIXED TOY PROBLEM
    A = np.array([[0.7, 0.8], [0, 0.1]])

    B = np.array([[1., 0.], [0., 0.]])
    Q = 0.5 * np.eye(2)
    R = np.eye(2)

    mu = np.random.rand(state_dim) - 0.5
    Sigma = np.random.rand(state_dim, state_dim)
    Sigma = Sigma @ Sigma.T

    params = {'mu': mu, 'Sigma': Sigma, 'A': A, 'B': B, 'Q': Q, 'R': R}
    return all_x, all_y, params


def test_KalmanParams():
    _, params = example5_params()
    assert params.n_params == 6


def test_filter_step_static_no_noise():
    X, params = example5_params()
    R = params.R

    A = params.A # Problem is static -- building doesn't change height
    B = params.B # We're measuring the hidden state
    Q = params.Q # True building height is noiseless

    y = params.mu # initial estimate
    P = params.Sigma # initial variance estimate

    # Taken from https://www.kalmanfilter.net/kalman1d.html
    y_est = np.array([49.69, 48.47, 50.57, 51.68, 51.33, 49.62, 49.21, 49.31, 49.53, 49.57])
    P_est = np.array([[22.5], [11.84], [8.04], [6.08], [4.89], [4.09], [3.52], [3.08], [2.74], [2.47]])

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

    A = params.A # Problem is static -- building doesn't change height
    B = params.B # We're measuring the hidden state
    Q = params.Q # True building height is noiseless

    y = params.mu # initial estimate
    P = params.Sigma # initial variance estimate

    # Taken from https://www.kalmanfilter.net/kalman1d.html
    y_est = np.array([49.69, 48.47, 50.57, 51.68, 51.33, 49.62, 49.21, 49.31, 49.53, 49.57])
    P_est = np.array([[22.5], [11.84], [8.04], [6.08], [4.89], [4.09], [3.52], [3.08], [2.74], [2.47]])

    y_t_t = []
    P_t_t = []

    y_t_t1 = []
    P_t_t1 = []

    # filter
    for i in range(len(X)):
        y_pred, y_pred_cov, _, y, P = kalman.filter_step(X[i], y, P, A, Q, B, R)
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
        y_t_tau, y_t_tau_cov, _ = kalman.smooth_step(y_t_tau, y_t_tau_cov, y_t_t1[-i -2], P_t_t1[-i -2], y_t_t[-i -1], P_t_t[-i -1], params.A)
        y_est_hat.insert(0, y_t_tau)
        P_est_hat.insert(0, y_t_tau_cov)
    
    y_est_hat = np.array(y_est_hat)
    P_est_hat = np.array(P_est_hat)
    

    assert (P_est_hat <= P_t_t[:-1]).all()


def test_filter_step__dynamic_steps():
    X, params = example9_params()

    y = params.mu # initial estimate
    P = params.Sigma # initial variance estimate

    y_est_hat = []
    P_est_hat = []
    for i in range(len(X)):
        _, _, _, y, P = kalman.filter_step(X[i], y, P, params.A, params.Q, params.B, params.R)
        y_est_hat.append(y)
        P_est_hat.append(P)
    y_est_hat = np.array(y_est_hat)
    P_est_hat = np.array(P_est_hat)

    assert np.array([298.44, -0.51, -2.11, 3.89, -26.2, -1.29]) ==  approx(y_est_hat[-1], rel=0.1)


def test_filter_dynamic():
    X, params = example9_params()

    y = params.mu # initial estimate
    P = params.Sigma # initial variance estimate

    kalmod = kalman.KalmanModel()
    kalmod.set_params(X, params)
    y_est_hat, P_est_hat = kalmod.fit(mode='filter')

    assert np.array([298.44, -0.51, -2.11, 3.89, -26.2, -1.29]) ==  approx(y_est_hat[-1], rel=0.1)

    # re-run to check that estimated covariances are note recalculated
    _, _ = kalmod.fit(mode='filter')
    

    y_smoothed, P_smoothed = kalmod.fit(mode='smooth')

    assert (P_smoothed <= P_est_hat).all()

def _test_simple():
    tau = 1500
    obs_dim = 2
    state_dim = 2
    params = {
        'mu': np.array([2., 0.]),
        'Sigma': np.array([[1000., 0.], [0., 1000.]]),
        'A': np.array([[1., 1.], [0., 1.]]),
        'B': np.array([[1., 0.]]),
        'Q': np.array([[3e-6, 0.], [0., 3e-6]]),
        'R': np.array([[5]])
    }

    params = kalman.KalmanParams(params)

    all_x, all_y = generate_data(params, tau)

    B_bad = np.zeros((obs_dim, state_dim))
    np.fill_diagonal(B_bad,
                     np.diag(np.random.rand(np.min((state_dim, obs_dim)))))

    init_params_bad = {
        'mu': np.array([-0.1]),
        'Sigma': np.array([[0.15]]),
        'A': np.array([[-0.15]]),
        'B': np.array([[1.0]]),
        'Q': np.array([[0.015]]),
        'R': np.array([[0.015]])
    }

    kalmod = kalman.KalmanModel()
    kalmod.fit(all_x,
               init_params=params,
               d=state_dim,
               predict_params=True,
               n_it=50)

    #assert approx(params.A) == kalman.params.A

    for i in [2, 5, 10, 20, 30, 50]:
        kalmod = kalman.KalmanModel()
        kalmod.fit(all_x,
                   init_params=params,
                   d=state_dim,
                   predict_params=True,
                   n_it=i)
        print(kalman.params.A)
    print(params.A)

    plt.plot(all_y)
    plt.plot(kalman.y_t_tau)
    plt.show()

test_smooth_step_static_no_noise()