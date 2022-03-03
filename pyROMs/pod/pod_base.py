import numpy as np
from numpy import ndarray
from numpy.linalg import norm

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import warnings
from os.path import splitext
from typing import Union, Tuple, List

from ..plotting_mixin import PlottingMixin

SVDOutputType = Tuple[ndarray, ndarray, ndarray]


class PODBase(PlottingMixin):
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

    def __init__(self, svd_rank: Union[int, float] = 0) -> None:

        # Model parameters
        self.svd_rank: Union[int, float] = svd_rank

        # Training information
        self._snapshots: ndarray = None
        self._snapshots_shape: Tuple[int, int] = None

        self._parameters: ndarray = None

        # SVD information
        self._U: ndarray = None
        self._s: ndarray = None
        self._Vstar: ndarray = None

        # POD information
        self._modes: ndarray = None
        self._b: ndarray = None

    @property
    def n_snapshots(self) -> int:
        return self._snapshots.shape[0]

    @property
    def n_features(self) -> int:
        return self._snapshots.shape[1]

    @property
    def n_modes(self) -> int:
        return self._modes.shape[1]

    @property
    def n_parameters(self) -> int:
        return self._parameters.shape[1]

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
    def parameters(self) -> ndarray:
        """
        Get the original training parameters.

        Returns
        -------
        ndarray (n_snapshots, n_parameters)
        """
        return self._parameters

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
        Get the reconstructed snapshots.

        Returns
        -------
        ndarray (n_snapshots, n_features)
        """
        return np.transpose(self._modes.dot(self._b))

    @property
    def reconstruction_error(self) -> float:
        """
        Compute the reconstruction error between the snapshots
        and the reconstructed data.

        Returns
        -------
        float
        """
        X: ndarray = self._snapshots
        X_dmd: ndarray = self.reconstructed_data
        return norm(X - X_dmd) / norm(X)

    @property
    def snapshot_errors(self) -> ndarray:
        """
        Compute the reconstruction error for each snapshot.

        Returns
        -------
        ndarray (n_snapshots,)
        """
        X: ndarray = self._snapshots
        X_dmd: ndarray = self.reconstructed_data
        return norm(X - X_dmd, axis=1) / norm(X, axis=1)

    @property
    def left_svd_modes(self) -> ndarray:
        """
        Get the left singular vectors, column-wise.

        Returns
        -------
        ndarray (n_features, n_snapshots)
        """
        return self._U

    @property
    def right_svd_modes(self) -> ndarray:
        """
        Get the right singular vectors.

        Returns
        -------
        ndarray (n_snapshots, n_snapshots)
        """
        return np.transpose(np.conj(self._Vstar))

    @property
    def singular_values(self) -> ndarray:
        """
        Get the singular values.

        Returns
        -------
        ndarray (n_snapshots,)
        """
        return self._s

    def fit(self, X: ndarray, Y: ndarray = None) -> None:
        raise NotImplementedError(
            f'Subclasses must implement abstact method '
            f'{self.__class__.__name__}.fit')

    def print_summary(self, skip_line: bool = False) -> None:
        """
        Print a summary of the DMD model.
        """
        msg = '===== POD Summary ====='
        header = '='*len(msg)
        if skip_line:
            print()
        print('\n'.join([header, msg, header]))
        print(f"{'# of Modes':<20}: {self.n_modes}")
        print(f"{'# of Snapshots':<20}: {self.n_snapshots}")
        print(f"{'Reconstruction Error':<20}: "
              f"{self.reconstruction_error:.3e}")
        print(f"{'Mean Snapshot Error':<20}: "
              f"{np.mean(self.snapshot_errors):.3e}")
        print(f"{'Max Snapshot Error':<20}: "
              f"{np.max(self.snapshot_errors):.3e}")

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
        # Check the inputs
        if self.modes is None:
            raise ValueError(
                'The fit method must be performed first.')

        if mode_indices is None:
            mode_indices = list(range(self.n_modes))
        elif isinstance(mode_indices, int):
            mode_indices = [mode_indices]

        # Get parameter values
        y = self.parameters

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
                    idx = idx if idx >= 0 else self.n_modes + idx
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
