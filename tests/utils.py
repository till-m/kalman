import numpy as np
import kalman
import copy
from scipy.stats import ortho_group


def random_covariance_matrix(size, random_state=None):
    lam = np.random.rand(size)
    q = ortho_group(dim=size).rvs(random_state=random_state)
    return q @ np.diag(lam) @ q.T


def noisify_covariance_matrix(cov, rel, random_state=None):
    size = cov.shape[0]
    if size == 1 :
        sign = np.sign(np.random.rand() - 0.5) 
        return cov + sign * np.random.rand(size) * np.abs(cov) * rel
    else:
        mag = np.linalg.det(cov)

        rcov = random_covariance_matrix(cov.shape[0])
        scale = (mag/np.linalg.det(rcov) * rel)**(1/size) # size root due to multilinearity of determinant.

        rcov = rcov * scale

        return cov + rcov

def noisify_array(arr, rel, random_state=None):
    scale = (np.max(arr) - np.min(arr)) * rel
    if scale == 0:
        scale = np.max(arr) * rel
    rarr = np.random.normal(size=arr.shape, loc=0, scale=scale)
    return arr + rarr

def add_noise(params: kalman.KalmanParams, random_state=None, rel=1e-2):
    if random_state is not None:
        np.random.seed(random_state)

    noisy_params = copy.deepcopy(params)

    noisy_params.A = noisify_array(noisy_params.A, rel=rel, random_state=random_state)
    noisy_params.B = noisify_array(noisy_params.B, rel=rel,  random_state=random_state)
    noisy_params.mu = noisify_array(noisy_params.mu, rel=rel,  random_state=random_state)

    if params.has_control:
        noisy_params.C = noisify_array(noisy_params.C, rel=rel, random_state=random_state)
    noisy_params.Sigma = noisify_covariance_matrix(
        noisy_params.Sigma,
        rel=rel,
        random_state=random_state
    )
    noisy_params.Q = noisify_covariance_matrix(
        noisy_params.Q,
        rel=rel,
        random_state=random_state
    )
    noisy_params.R = noisify_covariance_matrix(
        noisy_params.R,
        rel=rel,
        random_state=random_state
    )

    return noisy_params


def example5_params():
    # Example 5 from https://www.kalmanfilter.net/kalman1d.html
    X = np.array([48.54, 47.11, 55.01, 55.15, 49.89, 40.85, 46.72, 50.05, 51.27, 49.95]).reshape((-1, 1))

    params = kalman.KalmanParams(
        mu=np.array([60.]),
        Sigma=np.array([[225]]),
        A=np.array([[1.]]),
        B=np.array([[1.]]),
        R=np.array([[25.]]),
        Q=np.array([[0.]])
    )

    return X, params


def example9_params():
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

    params = {
        'mu': np.array([0., 0., 0., 0., 0., 0.]),
        'Sigma': np.diag([500, 500, 500, 500, 500, 500]),
        'B': np.array([
            [1., 0., 0., 0., 0., 0],
            [0., 0., 0., 1., 0., 0]
        ]),
        'R': np.array([
            [9., 0],
            [0., 9.]
        ]),
        'A': np.array([
            [1., 1., 0.5, 0., 0., 0.,],
            [0., 1., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.,],
            [0., 0., 0., 1., 1., 0.5],
            [0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 0., 1.]
        ]),
        'Q': 0.2**2 * np.array([
            [0.25, 0.5, 0.5, 0., 0., 0.],
            [0.5, 1., 1., 0., 0., 0.],
            [0.5, 1., 1., 0., 0., 0.],
            [0., 0., 0., 0.25, 0.5, 0.5],
            [0., 0., 0., 0.5, 1., 1.],
            [0., 0., 0., 0.5, 1., 1.,]
        ])
    }

    return X, kalman.KalmanParams(params)


def example10_params():
    # Example 10 from https://www.kalmanfilter.net/multiExamples.html
    X = np.array([
        -32.4, -11.1, 18., 22.9, 19.5, 28.5, 46.5, 68.9, 48.2, 56.1, 90.5,
        104.9, 140.9, 148., 187.6, 209.2, 244.6, 276.4, 323.5, 357.3, 357.4,
        398.3, 446.7, 465.1, 529.4, 570.4, 636.8, 693.3, 707.3, 748.5
    ])

    u0 = 9.8
    U = np.array([
        u0 + 9.8, 39.72, 40.02, 39.97, 39.81, 39.75, 39.6, 39.77, 39.83, 39.73, 39.87,
        39.81, 39.92, 39.78, 39.98, 39.76, 39.86, 39.61, 39.86, 39.74, 39.87,
        39.63, 39.67, 39.96, 39.8, 39.89, 39.85, 39.9, 39.81, 39.81, 39.68
    ])
    U = U - 9.8

    X = X.reshape((-1, 1))
    U = U.reshape((-1, 1))

    params = {
        'mu': np.array([0., 0.]),
        'Sigma': np.diag([500, 500]),
        'B': np.array([[1., 0.]]),
        'R': np.array([[400]]),
        'A': np.array([
            [1., 0.25],
            [0, 1.]
        ]),
        'Q': 0.1**2 * np.array([
            [1/4**5, 1/2**7],
            [1/2**7, 1/4**2]
        ]),
        'C': np.array([
            [0.0313],
            [0.25]
        ])
    }

    return X, U, kalman.KalmanParams(params)
