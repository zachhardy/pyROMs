import numpy as np
from numpy import ndarray
from numpy.linalg import norm
import matplotlib.pyplot as plt

from typing import Union, Tuple

Rank = Union[float, int]
Dataset = Tuple[ndarray, ndarray]


class PODBase:
    """
    Principal Orthogonal Decomposition base class.

    Parameters
    ----------
    svd_rank : int or float, default -1
        The SVD truncation rank. If -1, no truncation is used.
        If a positive integer, the truncation rank is the argument.
        If a float between 0 and 1, the minimum number of modes
        needed to obtain an information content greater than the
        argument is used.
    """

    from ._plotting import (plot_singular_values,
                            plot_coefficients,
                            plot_reconstruction_errors)

    def __init__(self, svd_rank: Rank = -1) -> None:
        self.svd_rank: Rank = svd_rank

        self._snapshots: ndarray = None
        self._parameters: ndarray = None

        self._n_modes: int = 0
        self._modes: ndarray = None

        self._singular_values: ndarray = None
        self._b: ndarray = None

    @property
    def snapshots(self) -> ndarray:
        """
        Get the original training data.

        Returns
        -------
        ndarray (n_snapshots, n_features)
        """
        return self._snapshots

    @property
    def n_snapshots(self) -> int:
        """
        Get the number of snapshots provided to the model.

        Returns
        -------
        int
        """
        return self.snapshots.shape[0]

    @property
    def n_features(self) -> int:
        """
        Get the number of features in each snapshot.

        Returns
        -------
        int
        """
        return self.snapshots.shape[1]

    @property
    def parameters(self) -> ndarray:
        """
        Get the original training parameters.

        Returns
        -------
        ndarray (n_snapshots, n_parameters)
        """
        return self._parameters

    @property
    def n_parameters(self) -> int:
        """
        Get the number of parameters that describe a snapshot.

        Returns
        -------
        int
        """
        return self.parameters.shape[1]

    @property
    def modes(self) -> ndarray:
        """
        Get the POD modes, stored column-wise.

        Returns
        -------
        ndarray (n_featurs, n_modes)
        """
        return self._modes[:, :self.n_modes]

    @property
    def n_modes(self) -> int:
        """
        Get the number of modes.

        Returns
        -------
        int
        """
        return self._n_modes

    @property
    def amplitudes(self) -> ndarray:
        """
        Get the POD mode amplitudes that define the training data.

        Returns
        -------
        ndarray (n_snapshots, n_modes)
        """
        return self._b[:, :self.n_modes]

    @property
    def singular_values(self) -> ndarray:
        """
        Get the singular values of the POD modes.

        Returns
        -------
        ndarray (n_snapshots,)
        """
        return self._singular_values

    @property
    def reconstructed_data(self) -> ndarray:
        """
        Get the reconstructed training data using the model.

        Returns
        -------
        ndarray (n_snapshots, n_features)
            The reconstructed training data.
        """
        return self.amplitudes @ self.modes.T

    def fit(self, X: ndarray, Y: ndarray = None) -> None:
        raise NotImplementedError(
            f'Subclasses must implement abstact method '
            f'{self.__class__.__name__}.fit')

    def compute_rank(self, svd_rank: Rank) -> int:
        """
        Compute the SVD rank to use.

        Parameters
        ----------
        svd_rank : int
            The energy content to retain, or the fixed number
            of POD modes to use.
        """
        X, s = self.snapshots, self.singular_values
        if 0.0 < svd_rank < 1.0:
            cumulative_energy = np.cumsum(s / sum(s))
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif isinstance(svd_rank, int) and svd_rank >= 1:
            rank = min(svd_rank, min(X.shape) - 1)
        else:
            rank = min(X.shape) - 1
        return rank

    def reconstruction_error(self) -> float:
        """
        Get the l2 error in the reconstructed training data using
        the truncated model.

        Returns
        -------
        float
        """
        X = self.snapshots
        X_pred = self.reconstructed_data
        return norm(X - X_pred)

    def untruncated_reconstruction_error(self) -> float:
        """
        Get the l2 error in the reconstructed training data using
        an untruncated model.

        Returns
        -------
        float
        """
        X = self.snapshots
        X_pred = X @ self._modes @ self._modes.T
        return norm(X - X_pred)

    def compute_error_decay(self) -> ndarray:
        """
        Compute the decay in the error. This method computes the
        error decay as a function number of modes included in the
        model.

        Returns
        -------
        ndarray (n_modes,)
        """
        errors = []
        X = self.snapshots
        for n in range(self.n_snapshots):
            X_pred = X @ self._modes[:, :n] @ self._modes.T[:n]
            err = norm(X - X_pred) / norm(X)
            errors.append(err)
        return np.array(errors)

    @staticmethod
    def center_data(data: ndarray) -> ndarray:
        """
        Center the data by removing the mean and scaling
        by the standard deviation row-wise.

        Parameters
        ----------
        data : ndarray (n_snapshots, n_modes or n_features)
            The data to center.

        Returns
        -------
        ndarray (n_snapshots, n_modes or n_features)
            The centered data.
        """
        if data.ndim == 1:
            return (data - np.mean(data)) / np.std(data)
        else:
            mean = np.mean(data, axis=1).reshape(-1, 1)
            std = np.std(data, axis=1).reshape(-1, 1)
            return (data - mean) / std

    @staticmethod
    def _validate_data(X: ndarray, Y: ndarray = None) -> Dataset:
        """
        Validate training data.

        Parameters
        ----------
        X : ndarray (n_snapshots, n_features)
            2D matrix containing training snapshots
            stored row-wise.
        Y : ndarray (n_snapshots, n_parameters) or None
            Matrix containing training parameters stored
            row-wise.

        Returns
        -------
       The inputs
        """
        # Check types for X and Y
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError('X must be a numpy.ndarray or list.')
        if Y is not None:
            if not isinstance(Y, (np.ndarray, list)):
                raise TypeError('Y must be a numpy.ndarray, list, or None.')

        # Format X
        X = np.asarray(X)
        if X.ndim != 2:
            raise AssertionError('X must be 2D data.')

        # Format Y
        if Y is not None:
            Y = np.asarray(Y).reshape(len(Y), -1)
            if len(Y) != len(X):
                raise AssertionError('There must be a parameter set for '
                                     'each provided snapshot.')
        return X, Y
