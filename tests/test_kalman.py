import numpy as np
import kalman
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from pytest import approx
from icecream import ic

np.random.seed(12)

to_cov = lambda M: M @ M.T
#A = np.diag(np.random.rand(state_dim))
#B = np.zeros((obs_dim, state_dim))
#np.fill_diagonal(B, np.diag(np.random.rand(np.min((state_dim, obs_dim)))))

#Q = 0.01 * np.diag(np.random.rand(state_dim)) #0.01 * to_cov(np.random.rand(state_dim, state_dim))

#R = 0.01 * np.diag(np.random.rand(obs_dim)) #0.01 * to_cov(np.random.rand(obs_dim, obs_dim))


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


#def test_basic():
#    kalman = KalmanModel()
#    kalman.fit(all_x, d=state_dim, t_end=6, initialize_kwargs=init_kwargs)
#    kalman.lag_one_covar_smoother()
#
#    kalman = KalmanModel()
#    kalman.fit(all_x, d=state_dim, t_end=6, initialize_kwargs=init_kwargs_bad)
#    kalman.lag_one_covar_smoother()


def __test_integration():
    tau = 1500
    obs_dim = 2
    state_dim = 2
    all_x, all_y, params = toy_problem()

    B_bad = np.zeros((obs_dim, state_dim))
    np.fill_diagonal(B_bad,
                     np.diag(np.random.rand(np.min((state_dim, obs_dim)))))

    init_kwargs_bad = {
        'mu': np.zeros((state_dim)),
        'Sigma': 0.1 * to_cov(np.random.rand(state_dim, state_dim)),
        'A': np.random.rand(state_dim, state_dim),
        'B': B_bad,
        'Q': 0.01 * to_cov(np.random.rand(state_dim, state_dim)),
        'R': 0.01 * to_cov(np.random.rand(obs_dim, obs_dim))
    }

    kalman = kalman.KalmanModel()
    kalman.fit(all_x,
               init_params=init_kwargs_bad,
               d=state_dim,
               t_end=6,
               predict_params=True,
               n_it=50)

    assert approx(A) == kalman.A


def _test_1d():

    def update(mean1, var1, mean2, var2):
        new_mean = (var2 * mean1 + var1 * mean2) / (var2 + var1)
        new_var = 1 / (1 / var2 + 1 / var1)

        return (new_mean, new_var)

    def predict(mean1, var1, mean2, var2):

        return (mean1 + mean2, var1 + var2)

    # measurements
    pos = np.array([5, 6, 7, 9, 10]).astype(float)
    vel = np.array([1, 1, 2, 1, 1]).astype(float)

    pos_var = 4.
    vel_var = 2.

    pos_t = 0.
    var_t = 1e5

    A = np.array([[1., 1.], [0., 1.]])
    B = np.array([[1., 0.]])
    Q = np.array([[0.001, 0], [0, 0]])
    R = np.array([[pos_var]])
    for t in range(len(pos)):
        mu1, sig1 = update(pos_t, var_t, pos[t], pos_var)
        mu1, sig1 = predict(mu1, sig1, vel[t], vel_var)

        mu2, sig2 = kalman.filter_step(pos[t], np.array([pos_t]), np.array([var_t]), A, Q, B, R)
        ic(mu1, sig1)
        ic(mu2, sig2)
    

def test_static_no_noise():
    # Example 5 from https://www.kalmanfilter.net/kalman1d.html
    X = np.array([48.54, 47.11, 55.01, 55.15, 49.89, 40.85, 46.72, 50.05, 51.27, 49.95])
    R = np.array([[25.]])

    A = np.array([[1.]]) # Problem is static -- building doesn't change height
    B = np.array([[1.]]) # We're measuring the hidden state
    Q = np.array([[0.]]) # True building height is noiseless

    y = np.array([60.]) # initial estimate
    P = np.array([[225]]) # initial variance estimate

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


def test_dynamic():
    # Example 9 from https://www.kalmanfilter.net/multiExamples.html
    X = np.array([
        [-393.66,  300.4 ],
       [-375.93,  301.78],
       [-351.04,  295.1 ],
       [-328.96,  305.19],
       [-299.35,  301.06],
       [-273.36,  302.05],
       [-245.89,  300.  ],
       [-222.58,  303.57],
       [-198.03,  296.33],
       [-174.17,  297.65],
       [-146.32,  297.41],
       [-123.72,  299.61],
       [-103.47,  299.6 ],
       [ -78.23,  302.39],
       [ -52.63,  295.04],
       [ -23.34,  300.09],
       [  25.96,  294.72],
       [  49.72,  298.61],
       [  76.94,  294.64],
       [  95.38,  284.88],
       [ 119.83,  272.82],
       [ 144.01,  264.93],
       [ 161.84,  251.46],
       [ 180.56,  241.27],
       [ 201.42,  222.98],
       [ 222.62,  203.73],
       [ 239.4 ,  184.1 ],
       [ 252.51,  166.12],
       [ 266.26,  138.71],
       [ 271.75,  119.71],
       [ 277.4 ,  100.41],
       [ 294.12,   79.76],
       [ 301.23,   50.62],
       [ 291.8 ,   32.99],
       [ 299.89,    2.14]
    ])

    R = np.array([
        [9., 0],
        [0., 9.]
    ])

    A = np.array([
        [1., 1., 0.5, 0., 0., 0.,],
        [0., 1., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.,],
        [0., 0., 0., 1., 1., 0.5],
        [0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 1.]
    ])
    B = np.array([
        [1., 0., 0., 0., 0., 0],
        [0., 0., 0., 1., 0., 0]
    ])

    Q = np.array([
        [0.25, 0.5, 0.5, 0., 0., 0.],
        [0.5, 1., 1., 0., 0., 0.],
        [0.5, 1., 1., 0., 0., 0.],
        [0., 0., 0., 0.25, 0.5, 0.5],
        [0., 0., 0., 0.5, 1., 1.],
        [0., 0., 0., 0.5, 1., 1.,]
    ])

    y = np.array([0., 0., 0., 0., 0., 0.]) # initial estimate
    P = np.diag([500, 500, 500, 500, 500, 500]) # initial variance estimate

    y_est_hat = []
    P_est_hat = []
    for i in range(len(X)):
        _, _, _, y, P = kalman.filter_step(X[i], y, P, A, Q, B, R)
        y_est_hat.append(y)
        P_est_hat.append(P)
    y_est_hat = np.array(y_est_hat)
    P_est_hat = np.array(P_est_hat)

    assert np.array([298.44, -0.51, -2.11, 3.89, -26.2, -1.29]) ==  approx(y_est_hat[-1], rel=0.1)


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
