import numpy as np
from numpy import ndarray

from typing import Tuple, Union, TypeVar

Rank = Union[float, int]
SVD = Tuple[ndarray, ndarray, ndarray, int]


def compute_svd(X: ndarray, svd_rank: Rank = -1) -> SVD:
    """
    Compute the SVD of X and the truncation rank.

    Parameters
    ----------
    X : ndarray
        The matrix to decompose.
    svd_rank : int or float, default is -1
        The SVD truncation rank. If -1, no truncation is used.
        If a positive integer, the truncation rank is the argument.
        If a float between 0 and 1, the minimum number of modes
        needed to obtain an information content greater than the
        argument is used.

    Returns
    -------
    ndarray
        The left singular vector matrix stored column-wise.
    ndarray
        The singular values array
    ndarray
        The right singular vector matrix stored column-wise
    int
        The truncation rank of the SVD
    """
    # Perform the SVD
    U, s, V = np.linalg.svd(X, full_matrices=False)
    V = V.conj().T

    # Compute the rank
    if 0.0 < svd_rank < 1.0:
        cumulative_energy = np.cumsum(s / sum(s))
        rank = np.searchsorted(cumulative_energy, svd_rank) + 1
    elif isinstance(svd_rank, int) and svd_rank >= 1:
        rank = min(svd_rank, X.shape[1])
    else:
        rank = X.shape[1]

    return U, s, V, rank



