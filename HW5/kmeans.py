"""EECS545 HW5 Q1. K-means"""

import numpy as np
import sklearn.metrics


def hello():
    print('Hello from kmeans.py!')


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute the pixel error between the data and compressed data.

    Please do not change this function!

    Arguments:
        x: A numpy array of shape (N*, d), where d is the data dimension.
        y: A numpy array of shape (N*, d), where d is the data dimension.
    Return:
        errors: A numpy array of shape (N*). Euclidean distances.
    """
    assert x.shape == y.shape
    error = np.sqrt(np.sum(np.power(x - y, 2), axis=-1))
    return error


def train_kmeans(train_data: np.ndarray, initial_centroids, *,
                 num_iterations: int = 50):
    """K-means clustering.

    Arguments:
        train_data: A numpy array of shape (N, d), where
            N is the number of data points
            d is the dimension of each data point. Note: you should NOT assume
              d is always 3; rather, try to implement a general K-means.
        initial_centroids: A numpy array of shape (K, d), where
            K is the number of clusters. Each data point means the initial
            centroid of cluster. You should NOT assume K = 16.
        num_iterations: Run K-means algorithm for this number of iterations.

    Returns:
        centroids: A numpy array of (K, d), the centroid of K-means clusters
            after convergence.
    """
    # Sanity check
    N, d = train_data.shape
    K, d2 = initial_centroids.shape
    if d != d2:
        raise ValueError(f"Invalid dimension: {d} != {d2}")

    # We assume train_data contains a real-valued vector, not integers.
    assert train_data.dtype.kind == 'f'

    centroids = initial_centroids.copy()
    for i in range(num_iterations):
        dists = sklearn.metrics.pairwise_distances(train_data, centroids)
        assignments = np.argmin(dists, axis=1)

        compressed = centroids[assignments]
        mean_error = euclidean_distance(train_data, compressed).mean()

        for k in range(K):
            mask = assignments == k
            if np.any(mask):
                centroids[k] = train_data[mask].mean(axis=0)

        print(f'Iteration {i:2d}: mean error = {mean_error:2.2f}')

    # This should contain the centroid points after convergence.
    assert centroids.shape == (K, d)
    return centroids


def compress_image(image: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Compress image by mapping each pixel to the closest centroid.

    Arguments:
        image: A numpy array of shape (H, W, 3) and dtype uint8.
        centroids: A numpy array of shape (K, 3), each row being the centroid
            of a cluster.
    Returns:
        compressed_image: A numpy array of (H, W, 3) and dtype uint8.
            Be sure to round off to the nearest integer.
    """
    H, W, C = image.shape
    K, C2 = centroids.shape
    assert C == C2 == 3, "Invalid number of channels."
    assert image.dtype == np.uint8

    pixels = image.reshape(-1, C).astype(float)
    dists = sklearn.metrics.pairwise_distances(pixels, centroids)
    assignments = np.argmin(dists, axis=1)
    compressed_image = np.round(centroids[assignments]).astype(np.uint8).reshape(H, W, C)

    assert compressed_image.dtype == np.uint8
    assert compressed_image.shape == (H, W, C)
    return compressed_image
