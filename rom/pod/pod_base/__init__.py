import numpy as np
from numpy import ndarray
from numpy.linalg import norm

from typing import Union, Tuple, List

SVDRankType = Union[int, float]
SVDOutputType = Union[ndarray, ndarray, ndarray]
RankErrorType = Tuple[ndarray, ndarray]
InputType = Union[ndarray, List[ndarray]]
FormattedInputType = Tuple[ndarray, Tuple[int, int], ndarray]


class PODBase:
    """
    Principal Orthogonal Decomposition base class.

    Parameters
    ----------
    svd_rank : int or float, default -1
        The SVD rank to use for truncation. If a positive integer,
        the minimum of this number and the maximum possible rank
        is used. If a float between 0 and 1, the minimum rank to
        achieve the specified energy content is used. If -1, no
        truncation is performed.
    """

    from ._plotting import (plot_singular_values,
                            plot_1D_modes,
                            plot_coefficients,
                            plot_error_decay)

    def __init__(self, svd_rank: SVDRankType = -1) -> None:

        self._svd_rank: SVDRankType = svd_rank

        self._snapshots: ndarray = None
        self._snapshots_shape: Tuple[int, int] = None

        self._parameters: ndarray = None

        self._U: ndarray = None
        self._Sigma: ndarray = None

        self._modes: ndarray = None
        self._b: ndarray = None

    @property
    def svd_rank(self) -> SVDRankType:
        """
        Return the set SVD rank.

        Returns
        -------
        float or int
        """
        return self._svd_rank

    @property
    def snapshots(self) -> ndarray:
        """
        Get the original training data.

        Returns
        -------
        ndarray (n_features, n_snapshots)
        """
        return self._snapshots

    @property
    def n_snapshots(self) -> int:
        """
        Get the number of snapshots.

        Returns
        -------
        int
        """
        return self.snapshots.shape[1]

    @property
    def n_features(self) -> int:
        """
        Get the number of features in each snapshot.

        Returns
        -------
        int
        """
        return self.snapshots.shape[0]

    @property
    def parameters(self) -> ndarray:
        """
        Get the original training parameters.

        Returns
        -------
        ndarray (n_parameters, n_snapshots)
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
        return self.parameters.shape[0]

    @property
    def singular_values(self) -> ndarray:
        """
        Get the singular values.

        Returns
        -------
        ndarray (n_snapshots,)
        """
        return self._Sigma

    @property
    def modes(self) -> ndarray:
        """
        Get the modes, stored column-wise.

        Returns
        -------
        ndarray (n_features, n_modes)
        """
        return self._modes

    @property
    def n_modes(self) -> int:
        """
        Get the number of modes.

        Returns
        -------
        int
        """
        return self.modes.shape[1]

    @property
    def amplitudes(self) -> ndarray:
        """
        Get the mode amplitudes per snapshot.

        Returns
        -------
        ndarray (n_modes, n_snapshots)
        """
        return self._b

    @property
    def reconstructed_data(self) -> ndarray:
        """
        Get the reconstructed training data.

        Returns
        -------
        ndarray (n_features, n_snapshots)
        """
        return self.modes @ self.amplitudes

    @property
    def reconstruction_error(self) -> float:
        """
        Compute the training data reconstruction L^2 error.

        Returns
        -------
        float
        """
        X = self.snapshots
        X_pod = self.reconstructed_data
        return norm(X - X_pod) / norm(X)

    def fit(self, X: ndarray, Y: ndarray = None) -> None:
        raise NotImplementedError(
            f'Subclasses must implement abstact method '
            f'{self.__class__.__name__}.fit')

    def _compute_svd(self) -> SVDOutputType:
        """
        Compute the truncated singular value decomposition
        of the snapshots.
        """
        X = self._snapshots
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        V = Vh.conj().T

        svd_rank = self._svd_rank
        if 0.0 < svd_rank < 1.0:
            cumulative_energy = np.cumsum(s ** 2 / sum(s ** 2))
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif isinstance(svd_rank, int) and svd_rank >= 1:
            rank = min(svd_rank, min(X.shape))
        else:
            rank = X.shape[1]

        self._U = U
        self._Sigma = s
        return U[:, :rank], s[:rank], V[:, :rank]

    def compute_rankwise_errors(self, skip: int = 1,
                                end: int = None) -> RankErrorType:
        """
        Compute the error as a function of rank.

        Parameters
        ----------
        skip : int, default 1
            Interval between ranks to compute errors for. The default
            is to compute errors at all ranks.
        end : int, default -1
            The last rank to compute the error for. The default is to
            end at the maximum rank.

        Returns
        -------
        ndarray (varies,)
            The ranks used to compute errors.
        ndarray (varies,)
            The errors for each rank.
        """
        X, Y = self._snapshots, self._parameters
        orig_rank = self._svd_rank
        if end is None or end > min(X.shape) - 1:
            end = min(X.shape) - 1

        errors, ranks = [], []
        for r in range(0, end, skip):
            self._svd_rank = r + 1
            self.fit(X, Y)

            error = self.reconstruction_error.real
            errors.append(error)
            ranks.append(r + 1)

        self._svd_rank = orig_rank
        self.fit(X, Y)
        return ranks, errors

    @staticmethod
    def _validate_data(X: InputType,
                       Y: ndarray = None) -> FormattedInputType:
        """
        Validate the training data.

        Parameters
        ----------
        X : ndarray or List[ndarray]
            The training snapshots.
        Y : ndarray
            The training parameter labels.

        Returns
        -------
        ndarray (n_features, n_snapshots)
            The formatted snapshots.
        Tuple[int, int]
            The training snapshot shape.
        ndarray (n_parameters, n_snapshots)
            The formatted parameters.
        """
        if isinstance(X, ndarray) and X.ndim == 2:
            snapshots = X
            snapshots_shape = None
        else:
            input_shapes = [np.asarray(x).shape for x in X]

            if len(set(input_shapes)) != 1:
                raise ValueError(
                    'All snapshots must have the same dimension.')

            snapshots_shape = input_shapes[0]
            snapshots = np.transpose([np.array(x).flatten() for x in X])

        if Y is not None:
            if isinstance(Y, ndarray) and Y.ndim == 2:
                if Y.shape[1] != snapshots.shape[1]:
                    raise ValueError(
                        'There must be the same number of parameter sets '
                        'as snapshots in the training data.')
            else:
                Y = np.array(Y).reshape(-1, Y.shape[1])

        return snapshots, snapshots_shape, Y
