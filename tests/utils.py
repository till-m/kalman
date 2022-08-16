import numpy as np
import kalman

def example5_params():
    # Example 5 from https://www.kalmanfilter.net/kalman1d.html
    X = np.array([48.54, 47.11, 55.01, 55.15, 49.89, 40.85, 46.72, 50.05, 51.27, 49.95])

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
        'Q': np.array([
            [0.25, 0.5, 0.5, 0., 0., 0.],
            [0.5, 1., 1., 0., 0., 0.],
            [0.5, 1., 1., 0., 0., 0.],
            [0., 0., 0., 0.25, 0.5, 0.5],
            [0., 0., 0., 0.5, 1., 1.],
            [0., 0., 0., 0.5, 1., 1.,]
        ])
    }

    return X, kalman.KalmanParams(params)