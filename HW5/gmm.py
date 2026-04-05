"""EECS545 HW5 Q1. K-means"""

import numpy as np
import scipy.stats
import scipy.special
from typing import NamedTuple, Union, Literal


def hello():
    print('Hello from gmm.py!')


class GMMState(NamedTuple):
    """Parameters to a GMM Model."""
    pi: np.ndarray  # [K]
    mu: np.ndarray  # [K, d]
    sigma: np.ndarray  # [K, d, d]


def train_gmm(train_data: np.ndarray,
              init_pi: np.ndarray,
              init_mu: np.ndarray,
              init_sigma: np.ndarray,
              *,
              num_iterations: int = 50,
              ) -> GMMState:
    """Fit a GMM model.

    Arguments:
        train_data: A numpy array of shape (N, d), where
            N is the number of data points
            d is the dimension of each data point. Note: you should NOT assume
              d is always 3; rather, try to implement a general K-means.
        init_pi: The initial value of pi. Shape (K, )
        init_mu: The initial value of mu. Shape (K, d)
        init_sigma: The initial value of sigma. Shape (K, d, d)
        num_iterations: Run EM (E-steps and M-steps) for this number of
            iterations.

    Returns:
        A GMM parameter after running `num_iterations` number of EM steps.
    """
    # Sanity check
    N, d = train_data.shape
    K, = init_pi.shape
    assert init_mu.shape == (K, d)
    assert init_sigma.shape == (K, d, d)


    
    pi, mu, sigma = init_pi.copy(), init_mu.copy(), init_sigma.copy()

    for it in range(num_iterations):
        # E-step: compute responsibilities gamma(z_nk)
        # log N(x_n | mu_k, sigma_k) for each n, k
        log_probs = np.zeros((N, K))
        for k in range(K):
            log_probs[:, k] = scipy.stats.multivariate_normal.logpdf(
                train_data, mean=mu[k], cov=sigma[k]
            )

        # log(pi_k) + log N(x_n | mu_k, sigma_k)
        log_weighted = np.log(pi)[np.newaxis, :] + log_probs  # (N, K)

        # Use logsumexp for numerical stability
        log_sum = scipy.special.logsumexp(log_weighted, axis=1, keepdims=True)  # (N, 1)
        log_gamma = log_weighted - log_sum  # (N, K)
        gamma = np.exp(log_gamma)  # (N, K)

        # Log-likelihood
        ll = log_sum.sum()
        print(f'Iteration {it:2d}: log-likelihood = {ll:.2f}')

        # M-step
        N_k = gamma.sum(axis=0)  # (K,)

        pi = N_k / N

        for k in range(K):
            mu[k] = (gamma[:, k:k+1].T @ train_data) / N_k[k]
            diff = train_data - mu[k]  # (N, d)
            sigma[k] = (diff.T * gamma[:, k]) @ diff / N_k[k]

    return GMMState(pi, mu, sigma)




def compress_image(image: np.ndarray, gmm_model: GMMState) -> np.ndarray:
    """Compress image by mapping each pixel to the mean value of a
    Gaussian component (hard assignment).

    Arguments:
        image: A numpy array of shape (H, W, 3) and dtype uint8.
        gmm_model: type GMMState. A GMM model parameters.
    Returns:
        compressed_image: A numpy array of (H, W, 3) and dtype uint8.
            Be sure to round off to the nearest integer.
    """
    H, W, C = image.shape
    K = gmm_model.mu.shape[0]
    pi, mu, sigma = gmm_model

    pixels = image.reshape(-1, C).astype(float)  # (H*W, 3)

    log_probs = np.zeros((pixels.shape[0], K))
    for k in range(K):
        log_probs[:, k] = scipy.stats.multivariate_normal.logpdf(
            pixels, mean=mu[k], cov=sigma[k]
        )

    log_weighted = np.log(pi)[np.newaxis, :] + log_probs
    assignments = np.argmax(log_weighted, axis=1)

    compressed_image = np.round(mu[assignments]).astype(np.uint8).reshape(H, W, C)

    assert compressed_image.dtype == np.uint8
    assert compressed_image.shape == (H, W, C)
    return compressed_image
