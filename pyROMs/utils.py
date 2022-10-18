import numpy as np

from numpy.linalg import svd

from typing import Union
from collections.abc import Iterable

SVDRank = Union[int, float]
Snapshots = Union[np.ndarray, Iterable]
Shape = tuple[int, ...]
SVDOutput = tuple[np.ndarray, np.ndarray, np.ndarray, int]


def format_2darray(X: Snapshots) -> tuple[np.ndarray, Shape]:
    """
    Private method which formats the training snapshots appropriately
    for an SVD. If the data is already 2D, the original data is returned.
    Otherwise, the data is reshaped into a 2D numpy ndarray with
    column-wise snapshots. When this is done, the reformatted data and
    original snapshot shape is returned.

    Parameters
    ----------
    X : numpy.ndarray or Iterable
        The training data.

    Returns
    -------
    numpy.ndarray (n_features, n_snapshots)
        The formatted 2D training data
    tuple[int, int]
        The original input shape.

    """
    if isinstance(X, np.ndarray) and X.ndim == 2:
        snapshots = X
        snapshots_shape = None
    else:

        input_shapes = [np.asarray(x).shape for x in X]

        if len(set(input_shapes)) != 1:
            raise ValueError("Snapshots have not the same dimension.")

        snapshots_shape = input_shapes[0]
        snapshots = np.transpose([np.asarray(x).flatten() for x in X])
    return snapshots, snapshots_shape


def compute_svd(X: np.ndarray, svd_rank: SVDRank) -> SVDOutput:
    """
    Compute the singular value decomposition of the matrix X.

    Parameters
    ----------
    X : np.ndarray
    svd_rank : int or float
        The rank for mode truncation. If 0, use the optimal rank.
        If a float in (0.0, 0.5], use the rank corresponding to the
        number of singular values whose relative values are greater
        than the argument. If a float (0.5, 1.0), use the minimum
        number of modes such that the energy content is greater than
        the argument. If a positive integer, use that rank.

    Returns
    -------
    U : numpy.ndarray (2-D)
        The left singular vectors
    s : numpy.ndarray (1-D)
        The singular values
    Vstar : numpy.ndarray (2-D)
        The conjugate transpose right singular vectors.
    rank : int
        The rank to use to truncate.
    """

    U, s, Vstar = svd(X, full_matrices=False)

    # Optimal rank
    if svd_rank == 0:
        def omega(x):
            return 0.56 * x ** 3 - 0.95 * x ** 2 + 1.82 * x + 1.43

        beta = np.divide(*sorted(X.shape))
        tau = np.median(s) * omega(beta)
        rank = np.sum(s > tau)

    # Cutoff trucation
    elif 0.0 < svd_rank < 0.5:
        rank = len(s[s / max(s) > svd_rank])

    # Energy truncation
    elif 0.5 <= svd_rank < 1.0:
        cumulative_energy = np.cumsum(s ** 2 / np.sum(s ** 2))
        rank = np.searchsorted(cumulative_energy, svd_rank) + 1

    # Fixed rank
    elif svd_rank >= 1 and isinstance(svd_rank, int):
        rank = min(svd_rank, X.shape[1])

    else:
        rank = X.shape[1]

    return U, s, Vstar, rank

