import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from numpy.linalg import svd
from scipy.linalg import pinv

from typing import Union, Optional
from collections.abc import Iterable

from ..rom_base import ROMBase
from ..utils import format_2darray


SVDRank = Union[int, float]
Indices = Components = Union[int, Iterable[int]]
Snapshots = Parameters = Union[np.ndarray, Iterable]


class POD(ROMBase):
    """
    Implementation of the proper orthogonal decomposition.
    """

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    def __init__(self, svd_rank: SVDRank = 0) -> None:
        """
        Parameters
        ----------
        svd_rank : int or float, default 0
            The rank for mode truncation. If 0, use the optimal rank.
            If a float in (0.0, 0.5], use the rank corresponding to the
            number of singular values whose relative values are greater
            than the argument. If a float in (0.5, 1.0), use the minimum
            number of modes such that the energy content is greater than
            the argument. If a positive integer, use that rank.
        """
        super().__init__()
        self._svd_rank = svd_rank

    @property
    def modes(self) -> np.ndarray:
        """
        Return the POD modes.

        Returns
        -------
        numpy.ndarray (n_features, n_modes)
        """
        if self._U is not None:
            return self._U[:, :self._rank]

    @property
    def reconstructed_data(self) -> np.ndarray:
        """
        Return the reconstructed training data.

        Returns
        -------
        numpy.ndarray (n_features, n_snapshots)
        """
        return self.modes @ self._b

    def fit(self, X: Snapshots, Y: Optional[Parameters] = None) -> 'POD':
        """
        Fit the POD model to the specified data.

        Parameters
        ----------
        X : numpy.ndarray or Iterable
            The training snapshots.
        Y : numpy.ndarray or Iterable, default None
            The training parameters

        Returns
        -------
        POD
        """
        X, Xshape = format_2darray(X)

        self._snapshots = X
        self._snapshots_shape = Xshape

        # Perform the SVD
        self._U, self._s, self._Vstar = svd(X, False)
        self._rank = self._compute_rank(self._svd_rank)

        # Compute amplitudes
        Sigma = np.diag(self._s[:self._rank])
        self._b = Sigma @ self._Vstar[:self._rank]

        return self

    def refit(self, svd_rank: SVDRank) -> 'POD':
        """
        Re-fit the POD model to the specified SVD rank.

        Parameters
        ----------
        svd_rank : int or float
            The rank for mode truncation. If 0, use the optimal rank.
            If a float in (0.0, 0.5], use the rank corresponding to the
            number of singular values whose relative values are greater
            than the argument. If a float in (0.5, 1.0), use the minimum
            number of modes such that the energy content is greater than
            the argument. If a positive integer, use that rank.

        Returns
        -------
        POD
        """
        self._svd_rank = svd_rank
        self._rank = self._compute_rank(self._svd_rank)

        # Recompute amplitudes
        Sigma = np.diag(self._s[:self._rank])
        self._b = Sigma @ self._Vstar[:self._rank]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data X to the low-rank space.

        Parameters
        ----------
        X : numpy.ndarray (n_features, varies)
            The snapshot data to transform.

        Returns
        -------
        numpy.ndarray (n_snapshots, n_modes)
            The low-rank representation of X.
        """
        if self.modes is None:
            raise AssertionError("The POD model must be fit.")

        if X.shape[0] != self.n_features:
            msg = "The number of features must match the number " \
                  "of features in the training data."
            raise AssertionError(msg)
        return pinv(self.modes) @ X

    def print_summary(self) -> None:
        """
        Print a summary of the POD model.
        """
        print()
        print("=======================")
        print("===== POD Summary =====")
        print("=======================")
        print(f"{'# of Modes':<20}: {self.n_modes}")
        print(f"{'# of Snapshots':<20}: {self.n_snapshots}")
        print(f"{'Reconstruction Error':<20}: "
              f"{self.reconstruction_error:.3g}")
        print(f"{'Mean Snapshot Error':<20}: "
              f"{np.mean(self.snapshot_errors):.3g}")
        print(f"{'Max Snapshot Error':<20}: "
              f"{np.max(self.snapshot_errors):.3g}\n")

    def plot_coefficients(
            self,
            mode_indices: Optional[Indices] = None,
            one_plot: bool = True,
            filename: Optional[str] = None
    ) -> None:
        """
        Plot the POD coefficients as a function of parameter values.

        Parameters
        ----------
        mode_indices : list of int, default None
            The indices of the modes to plot. The default behavior
            is to plot all modes.
        one_plot : bool, default True
        filename : str, default None
            A location to save the plot to, if specified.
        """
        # Handle mode indices input
        if mode_indices is None:
            mode_indices = list(range(self.n_modes))
        elif isinstance(mode_indices, int):
            mode_indices = [mode_indices]
        else:
            for i, idx in enumerate(mode_indices):
                if idx < 0 or idx >= self.n_modes:
                    raise AssertionError("Invalid mode index encountered.")
        
        # One-dimensional parameter spaces
        if self.n_parameters == 1:
            y = self._parameters.ravel()

            # Sort by parameter values
            ind = np.argsort(y)
            y, amplitudes = y[ind], self.amplitudes[ind].T

            # Plot on one figure
            if one_plot:
                plt.figure()
                plt.xlabel("Parameter Value")
                plt.ylabel("POD Coefficient")
                for idx in mode_indices:
                    vals = amplitudes[idx] / max(abs(amplitudes[idx]))
                    plt.plot(y, vals, '-*', label=f"Mode {idx}")
                plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
                plt.grid(True)
                plt.tight_layout()

                if filename is not None:
                    base, ext = os.path.splitext(filename)
                    plt.savefig(f"{base}.pdf")

            # Plot separately
            else:
                for idx in mode_indices:
                    plt.figure()
                    plt.xlabel("Parameter Value")
                    plt.ylabel("POD Coefficient")
                    plt.plot(y, amplitudes[idx], '-*', label=f"Mode {idx}")
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()

                    if filename is not None:
                        base, ext = os.path.splitext(filename)
                        plt.savefig(f"{base}_{idx}.pdf")
        plt.show()
