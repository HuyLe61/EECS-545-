"""EECS545 HW5: ICA."""

import numpy as np


def hello():
    print('Hello from ica.py!')


def sigmoid(x: np.ndarray) -> np.ndarray:
    r"""
    A numerically stable sigmoid function for the input x.

    It calculates positive and negative elements with different equations to
    prevent overflow by avoid exponentiation with large positive exponent,
    thus achieving numerical stability.

    For negative elements in x, sigmoid uses this equation

    $$sigmoid(x_i) = \frac{e^{x_i}}{1 + e^{x_i}}$$

    For positive elements, it uses another equation:

    $$sigmoid(x_i) = \frac{1}{e^{-x_i} + 1}$$

    The two equations are equivalent mathematically.

    Parameters
    ----------
    x : np.ndarray (float64)
        The input samples

    Outputs
    -------
    np.ndarray (float64) of the same shape as x
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)

    # specify dtype! otherwise, it may all becomes zero, this could have different
    # behaviors depending on numpy version
    z = np.zeros_like(x, dtype=float)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])

    top = np.ones_like(x, dtype=float)
    top[neg_mask] = z[neg_mask]
    s = top / (1 + z)
    return s


def unmixer(X: np.ndarray) -> np.ndarray:
    '''
    Given mixed sources X, find the filter W by SGD on the maximum likelihood.

    Parameters
    ----------
    X : np.ndarray (float64) of shape (n_timesteps, n_microphones)

    Outputs
    -------
    np.ndarray (float64) of shape (n_microphones, n_microphones)
    '''
    # M: length
    # N: number of microphones
    M, N = X.shape
    W = np.eye(N)
    losses = []

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    for alpha in anneal:
        print('working on alpha = {0}'.format(alpha))
        for xi in X:
            W += alpha * filter_grad(xi, W)
    return W


def filter_grad(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    '''
    Calculate the gradient of the filter W on a data point x.
    Used for SGD in unmixer.

    Parameters
    ----------
    x : np.ndarray (float64) of shape (n_microphones)
    W : np.ndarray (float64) of shape (n_microphones, n_microphones)

    Outputs
    -------
    np.ndarray (float64) of shape (n_microphones, n_microphones)
    '''
    g = sigmoid(W @ x)
    grad = np.outer(1 - 2 * g, x) + np.linalg.inv(W).T
    return grad


def unmix(X: np.ndarray, W: np.ndarray):
    '''
    Unmix the sources X using the filter W.

    Parameters
    ----------
    X : np.ndarray (float64) of shape (n_timesteps, n_microphones)
    W : np.ndarray (float64) of shape (n_microphones, n_microphones)

    Outputs
    -------
    np.ndarray (float64) of shape (n_timesteps, n_microphones)

    '''
    return X @ W.T
