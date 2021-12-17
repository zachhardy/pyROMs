import warnings

import numpy as np
from numpy import ndarray
from numpy.linalg import norm

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from os.path import splitext

from typing import Union, Tuple, List

SVDOutputType = Tuple[ndarray, ndarray, ndarray]


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

    from ._plotting1d import plot_modes_1D, plot_snapshots_1D
    from ._plotting2d import plot_modes_2D, plot_snapshots_2D

    def __init__(self, svd_rank: Union[int, float] = -1) -> None:

        self._svd_rank: Union[int, float] = svd_rank

        self._snapshots: ndarray = None
        self._snapshots_shape: Tuple[int, int] = None

        self._parameters: ndarray = None

        self._U: ndarray = None
        self._Sigma: ndarray = None

        self._modes: ndarray = None
        self._b: ndarray = None

    @property
    def svd_rank(self) -> Union[int, float]:
        """
        Return the set SVD rank.

        Returns
        -------
        float or int
        """
        return self._svd_rank

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
    def n_modes(self) -> int:
        """
        Get the number of modes.

        Returns
        -------
        int
        """
        return self.modes.shape[1]

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
    def snapshots(self) -> ndarray:
        """
        Get the original training data.

        Returns
        -------
        ndarray (n_features, n_snapshots)
        """
        return self._snapshots

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

    def _compute_svd(self, X: ndarray,
                     svd_rank: Union[int, float] = None) -> SVDOutputType:
        """
        Compute the truncated singular value decomposition
        of the snapshots.

        Parameters
        ----------
        X : ndarray
            The matrix to decompose.
        svd_rank : int or float

        """
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        V = Vh.conj().T

        if svd_rank is None:
            svd_rank = self._svd_rank

        if svd_rank == 0:
            omega = lambda x: 0.56*x**3 - 0.95*x**2 + 1.82*x + 1.43
            beta = np.divide(*sorted(X.shape))
            tau = np.median(s) * omega(beta)
            rank = np.sum(s > tau)
        elif 0 < svd_rank < 1:
            cumulative_energy = np.cumsum(s ** 2 / np.sum(s ** 2))
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif svd_rank >= 1 and isinstance(svd_rank, int):
            rank = min(svd_rank, U.shape[1])
        else:
            rank = X.shape[1]

        self._U = U
        self._Sigma = s
        return U[:, :rank], s[:rank], V[:, :rank]

    @staticmethod
    def _col_major_2darray(X) -> Tuple[ndarray, Tuple[int, int]]:
        """
        Private method that takes as input the snapshots and stores them into a
        2D matrix, by column. If the input data is already formatted as 2D
        array, the method saves it, otherwise it also saves the original
        snapshots shape and reshapes the snapshots.

        Parameters
        ----------
        X : ndarray or List[ndarray]
            The input snapshots.

        Returns
        -------
        ndarray : 2D matrix containing flattened snapshots
        Tuple[int, int] : The shape of the original snapshots.

        :param X: the input snapshots.
        :type X: int or numpy.ndarray
        :return: the 2D matrix that contains the flatten snapshots, the shape
            of original snapshots.
        :rtype: numpy.ndarray, tuple
        """
        # If the data is already 2D ndarray
        if isinstance(X, np.ndarray) and X.ndim == 2:
            snapshots = X
            snapshots_shape = None
        else:
            input_shapes = [np.asarray(x).shape for x in X]

            if len(set(input_shapes)) != 1:
                raise ValueError('Snapshots have not the same dimension.')

            snapshots_shape = input_shapes[0]
            snapshots = np.transpose([np.asarray(x).flatten() for x in X])

        # check condition number of the data passed in
        cond_number = np.linalg.cond(snapshots)
        if cond_number > 10e4:
            warnings.warn(
                "Input data matrix X has condition number {}. "
                "Consider preprocessing data, passing in augmented data matrix, or regularization methods."
                    .format(cond_number))

        return snapshots, snapshots_shape

    def plot_singular_values(self,
                             normalized: bool = True,
                             logscale: bool = True,
                             show_rank: bool = False,
                             filename: str = None) -> None:
        """
        Plot the singular value spectrum.

        Parameters
        ----------
        normalized : bool, default True
            Flag for normalizing the spectrum to its max value.
        logscale : bool, default True
            Flag for a log scale on the y-axis.
        show_rank : bool, default False
            Flag for showing the truncation location.
        filename : str, default None.
            A location to save the plot to, if specified.
        """
        # Format the singular values
        svals = self.singular_values
        if normalized:
            svals /= sum(svals)

        # Define the plotter
        plotter = plt.semilogy if logscale else plt.plot

        # Make figure
        plt.figure()
        plt.xlabel('n', fontsize=12)
        plt.ylabel('Singular Value' if not normalized
                   else 'Relative Singular Value')
        plotter(svals, '-*b')
        if show_rank:
            plt.axvline(self.n_modes - 1, color='r',
                        ymin=svals.min(), ymax=svals.max())
        plt.tight_layout()
        if filename is not None:
            base, ext = splitext(filename)
            plt.savefig(base + '.pdf')

    def plot_coefficients(self,
                          mode_indices: List[int] = None,
                          one_plot: bool = True,
                          filename: str = None) -> None:
        """
        Plot the POD coefficients as a function of parameter values.

        Parameters
        ----------
        mode_indices : List[int], default None
            The indices of the modes to plot. The default behavior
            is to plot all modes.
        filename : str, default None
            A location to save the plot to, if specified.
        """
        y = self.parameters

        # Handle mode indices input
        if mode_indices is None:
            mode_indices = list(range(self.n_modes))
        elif isinstance(mode_indices, int):
            mode_indices = [mode_indices]
        else:
            for i, idx in enumerate(mode_indices):
                if not -self.n_modes <= idx < self.n_modes:
                    raise AssertionError('Invalid mode index encountered.')
                if idx < 0:
                    mode_indices[i] = self.n_modes + idx

        # One-dimensional parameter spaces
        if y.shape[1] == 1:
            y = y.flatten()

            # Sort by parameter values
            idx = np.argsort(y)
            y, amplitudes = y[idx], self.amplitudes.T[idx]

            # Plot on one axes
            if one_plot:
                fig: Figure = plt.figure()
                ax: Axes = fig.add_subplot(111)
                ax.set_xlabel('Parameter Value', fontsize=12)
                ax.set_ylabel('POD Coefficient Value', fontsize=12)
                for idx in mode_indices:
                    vals = amplitudes[:, idx] / max(abs(amplitudes[:, idx]))
                    ax.plot(y, vals, '-*', label=f'Mode {idx}')
                ax.legend()
                ax.grid(True)

                plt.tight_layout()
                if filename is not None:
                    basename, ext = splitext(filename)
                    plt.savefig(basename + '.pdf')

            # Plot separately
            else:
                for idx in mode_indices:
                    fig: Figure = plt.figure()
                    ax: Axes = fig.add_subplot(111)
                    ax.grid(True)
                    ax.set_xlabel('Parameter Value', fontsize=12)
                    ax.set_ylabel('POD Coefficient Value', fontsize=12)
                    ax.plot(y, amplitudes[:, idx], '-*', label=f'Mode {idx}')
                    ax.legend()

                    plt.tight_layout()
                    if filename is not None:
                        basename, ext = splitext(filename)
                        plt.savefig(basename + f'_{idx}.pdf')
