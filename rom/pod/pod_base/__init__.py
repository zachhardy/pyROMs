import numpy as np
from numpy import ndarray
from numpy.linalg import norm

from typing import Union, Tuple, List

Rank = Union[float, int]
Dataset = Tuple[ndarray, ndarray]
SVD = Tuple[ndarray, ndarray, ndarray]


class PODBase:
    """Principal Orthogonal Decomposition base class.
    """

    from ._plotting import (plot_singular_values,
                            plot_1D_modes,
                            plot_coefficients,
                            plot_error_decay)

    def __init__(self, svd_rank: Rank = -1) -> None:
        """
        Parameters
        ----------
        svd_rank : int or float, default -1
            The SVD truncation rank. If -1, no truncation is used.
            If a positive integer, the truncation rank is the argument.
            If a float between 0 and 1, the minimum number of modes
            needed to obtain an information content greater than the
            argument is used.
        """
        self.svd_rank: Rank = svd_rank

        self._snapshots: ndarray = None
        self._parameters: ndarray = None

        self._n_modes: int = 0
        self._modes: ndarray = None

        self._singular_values: ndarray = None
        self._b: ndarray = None

    @property
    def snapshots(self) -> ndarray:
        """Get the original training data.

        Returns
        -------
        ndarray (n_snapshots, n_features)
        """
        return self._snapshots

    @property
    def n_snapshots(self) -> int:
        """Get the number of snapshots.

        Returns
        -------
        int
        """
        return self.snapshots.shape[0]

    @property
    def n_features(self) -> int:
        """Get the number of features in each snapshot.

        Returns
        -------
        int
        """
        return self.snapshots.shape[1]

    @property
    def parameters(self) -> ndarray:
        """Get the original training parameters.

        Returns
        -------
        ndarray (n_snapshots, n_parameters)
        """
        return self._parameters

    @property
    def n_parameters(self) -> int:
        """Get the number of parameters that describe a snapshot.

        Returns
        -------
        int
        """
        return self.parameters.shape[1]

    @property
    def singular_values(self) -> ndarray:
        """Get the singular values.

        Returns
        -------
        ndarray (n_snapshots,)
        """
        return self._singular_values

    @property
    def modes(self) -> ndarray:
        """Get the modes, stored column-wise.

        Returns
        -------
        ndarray (n_features, n_modes)
        """
        return self._modes

    @property
    def n_modes(self) -> int:
        """Get the number of modes.

        Returns
        -------
        int
        """
        return self.modes.shape[1]

    @property
    def amplitudes(self) -> ndarray:
        """Get the mode amplitudes per snapshot.

        Returns
        -------
        ndarray (n_snapshots, n_modes)
        """
        return self._b[:, :self.n_modes]

    @property
    def reconstructed_data(self) -> ndarray:
        """Get the reconstructed training data.

        Returns
        -------
        ndarray (n_snapshots, n_features)
            The reconstructed training data.
        """
        return self.amplitudes @ self.modes.T

    @property
    def reconstruction_error(self) -> float:
        """Compute the training data reconstruction error.

        Returns
        -------
        float
            The relative L2 error between the training snapshots
            and the reconstructed snapshots.
        """
        X = self.snapshots
        X_pred = self.reconstructed_data
        return norm(X - X_pred) / norm(X)

    def fit(self, X: ndarray, Y: ndarray = None) -> None:
        raise NotImplementedError(
            f'Subclasses must implement abstact method '
            f'{self.__class__.__name__}.fit')

    def _compute_svd(self, X: ndarray, svd_rank: Rank = -1) -> SVD:
        """Compute the truncated singular value decomposition of X

        Parameters
        ----------
        X : ndarray
        svd_rank : int, default -1
        """
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        self._singular_values = s
        V = Vh.conj().T

        if svd_rank is None:
            svd_rank = self.svd_rank

        if 0.0 < svd_rank < 1.0:
            cumulative_energy = np.cumsum(s ** 2 / sum(s ** 2))
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif isinstance(svd_rank, int) and svd_rank >= 1:
            rank = min(svd_rank, min(X.shape))
        else:
            rank = X.shape[1]
        return U[:, :rank], s[:rank], V[:, :rank]

    def compute_error_decay(self, skip: int = 1,
                            end: int = None) -> Tuple[List[float], List[int]]:
        """Compute the decay in the error.

        This method computes the  error decay as a function
        of number of modes included in the model.

        Parameters
        ----------
        skip : int, default 1
            Interval to use when varying number of modes.
        end : int, default None
            The largest number of modes to compute the reconstruction
            error for. If None, the last mode will be the last one.

        Returns
        -------
        ndarray
            The reproduction error as a function of n_modes.
        ndarray
            The corresponding number of modes to each entry of
            the error vector.
        """
        X, Y = self.snapshots, self.parameters
        svd_rank_original = self.svd_rank
        if end is None or end > min(X.shape) - 1:
            end = min(X.shape) - 1

        errors: List[float] = []
        n_modes: List[int] = []
        for n in range(0, end, skip):
            self.svd_rank = n + 1
            self.fit(X, Y)
            error = self.reconstruction_error.real
            errors.append(error)
            n_modes.append(n)

        self.svd_rank = svd_rank_original
        self.fit(X, Y)
        return errors, n_modes

    @staticmethod
    def _center_data(data: ndarray) -> ndarray:
        """Center the data.

        This removes the mean and scales by the standard
        deviation row-wise.

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
        """Validate training data.

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
