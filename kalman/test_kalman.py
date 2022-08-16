import numpy as np
from base import KalmanModel, to_array
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

np.random.seed(42)

tau = 10
obs_dim = 5
state_dim = 3

to_random_cov = lambda M: 0.2 * M @ M.T
A = np.random.rand(state_dim, state_dim)
B = np.random.rand(obs_dim, state_dim)
Q = to_random_cov(np.random.rand(state_dim, state_dim))

R = to_random_cov(np.random.rand(obs_dim, obs_dim))

mu = 3*(np.random.rand(state_dim) - 0.5)
Sigma =  np.random.rand(state_dim, state_dim)
Sigma = Sigma @ Sigma.T

all_y = [multivariate_normal(mu, Sigma).rvs()]
for t in range(1, tau):
    y_t = A @ all_y[-1] + multivariate_normal(cov=Q).rvs()
    all_y.append(y_t)

all_x = []
for t in range(tau):
    x_t = B @ all_y[t] + multivariate_normal(cov=R).rvs()
    all_x.append(x_t)

init_kwargs = {
    'mu': mu,
    'Sigma': Sigma,
    'A': A,
    'B': B,
    'Q': Q,
    'R': R
}

init_kwargs_bad = {
    'mu': 3*(np.random.rand(state_dim) - 0.5),
    'Sigma': np.random.rand(state_dim, state_dim),
    'A': np.random.rand(state_dim, state_dim),
    'B': np.random.rand(obs_dim, state_dim),
    'Q': to_random_cov(np.random.rand(state_dim, state_dim)),
    'R': to_random_cov(np.random.rand(obs_dim, obs_dim))
}

kalman = KalmanModel()
kalman.fit(np.array(all_x), d=state_dim, t_end=6, initialize_kwargs=init_kwargs)
kalman.lag_one_covar_smoother()

print(np.array(all_y).shape)
print("+++++++++++++++++")
print(kalman.y_h.shape)
print("+++++++++++++++++")
print(kalman.y_h_tau.shape)
print("+++++++++++++++++")
print("+++++++++++++++++")
print(f"Loss (filtered): {np.mean(np.array(all_y)[1:] - kalman.y_h)**2}")
print(f"Loss (smoothed): {np.mean(np.array(all_y) - kalman.y_h_tau)**2}")

plt.plot(range(1, len(all_y)+1), all_y, label='true')
plt.plot(range(2, len(all_y)+1), kalman.y_h, '--', label='filtered')
plt.plot(range(1, len(all_y)+1), kalman.y_h_tau, '-.', label='smoothed')
plt.legend(loc='best')
plt.show()

kalman = KalmanModel()
kalman.fit(np.array(all_x), d=state_dim, t_end=6, initialize_kwargs=init_kwargs_bad)
kalman.lag_one_covar_smoother()

print(np.array(all_y).shape)
print("+++++++++++++++++")
print(kalman.y_h.shape)
print("+++++++++++++++++")
print(kalman.y_h_tau.shape)
print("+++++++++++++++++")
print(f"Loss (filtered): {np.mean(np.array(all_y)[1:] - kalman.y_h)**2}")
print(f"Loss (smoothed): {np.mean(np.array(all_y) - kalman.y_h_tau)**2}")

plt.plot(range(1, len(all_y)+1), all_y, label='true')
plt.plot(range(2, len(all_y)+1), kalman.y_h, '--', label='filtered')
plt.plot(range(1, len(all_y)+1), kalman.y_h_tau, '-.', label='smoothed')
plt.legend(loc='best')
plt.show()

kalman = KalmanModel()
kalman.fit(np.array(all_x), d=state_dim, t_end=6, initialize_kwargs=init_kwargs_bad, predict_params=True, n_it=2)
kalman.lag_one_covar_smoother()

print(np.array(all_y).shape)
print("+++++++++++++++++")
print(kalman.y_h.shape)
print("+++++++++++++++++")
print(kalman.y_h_tau.shape)
print("+++++++++++++++++")
print(f"Loss (filtered): {np.mean(np.array(all_y)[1:] - kalman.y_h)**2}")
print(f"Loss (smoothed): {np.mean(np.array(all_y) - kalman.y_h_tau)**2}")

plt.plot(range(1, len(all_y)+1), all_y, label='true')
plt.plot(range(2, len(all_y)+1), kalman.y_h, '--', label='filtered')
plt.plot(range(1, len(all_y)+1), kalman.y_h_tau, '-.', label='smoothed')
plt.legend(loc='best')
plt.show()

print(kalman.B)