"""
Utility functions common to more than one ROM.
"""

import numpy as np
from numpy import ndarray
from typing import Union, Iterable, Tuple

DataType = Union[ndarray, Iterable]


def compute_rank(svd_rank: Union[int, float],
                 X: ndarray, U: ndarray,
                 s: ndarray) -> int:
    """
    Compute the SVD rank based on the input data and
    full SVD result.

    Parameters
    ----------
    svd_rank : int or float, default 0
        The rank for the truncation. If 0, the method computes the
        optimal rank and uses it for truncation. If positive interger, the
        method uses the argument for the truncation. If float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`. If -1, the method does
        not compute truncation.
    X : ndarray
        Input matrix.
    U : ndarray (X.shape[0], min(X.shape))
        The left singular vectors, column-wise.
    s : ndarray (min(X.shape),)
        The singular values vector.
    V : ndarray (X.shape[1], min(X.shape))
        The right singular vectors, column-wise.

    Returns
    -------
    int
        The computed rank
    """
    def omega(x):
        return 0.56*x**3 - 0.95*x**2 + 1.82*x + 1.43

    # Compute optimal SVD rank
    if svd_rank == 0:
        beta = np.divide(*sorted(X.shape))
        tau = np.median(s) * omega(beta)
        rank = np.sum(s > tau)

    # Compute energy based rank
    elif 0 < svd_rank < 1:
        cumulative_energy = np.cumsum(s**2 / sum(s**2))
        rank = np.searchsorted(cumulative_energy, svd_rank) + 1

    # Fixed rank
    elif svd_rank >= 1 and isinstance(svd_rank, int):
        rank = min(svd_rank, U.shape[1])

    # Otherwise, no truncation
    else:
        rank = X.shape[1]

    return rank


def _row_major_2darray(X: DataType) -> Tuple[ndarray, tuple]:
    """
            Private method to format input snapshots as a column-wise
            two-dimensionl array. If reshaping is required, the original
            snapshot shapes are stored nd output.

            Parameters
            ----------
            X : ndarray or iterable
                The input snapshots.

            Returns
            -------
            ndarray (n_snapshots, n_features)
                The formatted input snapshots.
            tuple of int
                The original shape of the input snapshots.
            """
    # Handle already formatted snapshots
    if isinstance(X, ndarray) and X.ndim == 2:
        snapshots = X
        snapshots_shape = None

    # Format the snapshots
    else:
        # Get the shape of each snapshot
        input_shapes = [np.asarray(x).shape for x in X]
        if len(set(input_shapes)) != 1:
            raise ValueError(
                'Snapshots must all have the same shape.')

        snapshots_shape = input_shapes[0]
        snapshots = np.array([np.asarray(x).flatten() for x in X])

    return snapshots, snapshots_shape
