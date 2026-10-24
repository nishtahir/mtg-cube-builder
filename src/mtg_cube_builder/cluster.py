"""Small numerical helpers: spherical k-means and masked top-k means."""

import numpy as np


def spherical_kmeans(
    points: np.ndarray,
    k: int,
    rng: np.random.Generator,
    iters: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Clusters unit vectors by cosine similarity.

    Args:
      points: An (n, d) array of unit vectors.
      k: The number of clusters. Capped at n.
      rng: Source of randomness for choosing the initial centres.
      iters: The number of assignment and update rounds.

    Returns:
      The (k, d) unit-length cluster centres and the size of each cluster.
    """
    k = min(k, len(points))
    centres = points[rng.choice(len(points), size=k, replace=False)].copy()
    for _ in range(iters):
        labels = np.argmax(points @ centres.T, axis=1)
        for j in range(k):
            members = points[labels == j]
            if len(members):
                mean = members.mean(axis=0)
                centres[j] = mean / (np.linalg.norm(mean) + 1e-12)
    labels = np.argmax(points @ centres.T, axis=1)
    return centres, np.bincount(labels, minlength=k)


def topk_mean(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Averages the k largest finite entries of each row.

    Args:
      matrix: An (n, m) array. Entries that are -inf are ignored.
      k: How many entries to average. Capped at m.

    Returns:
      The row means, and how many finite entries each mean used. A row with no
      finite entries has mean 0.
    """
    k = min(k, matrix.shape[1])
    top = -np.sort(-matrix, axis=1)[:, :k]
    valid = np.isfinite(top)
    means = np.where(valid, top, 0).sum(axis=1) / np.maximum(
        valid.sum(axis=1), 1
    )
    return means, valid.sum(axis=1)
