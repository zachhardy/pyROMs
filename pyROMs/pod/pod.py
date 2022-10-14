import warnings
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from numpy.linalg import svd
from os.path import splitext

from typing import Union

from ..rom_base import ROMBase


class POD(ROMBase):
    """
    Implementation of the proper orthogonal decomposition.
    """

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 12

    def __init__(self, svd_rank: Union[int, float] = -1) -> None:
        """
        Parameters
        ----------
        svd_rank : int or float, default -1
            The SVD rank to use for truncation. If a positive integer,
            the minimum of this number and the maximum possible rank
            is used. If a float between 0 and 1, the minimum rank to
            achieve the specified energy content is used. If -1, no
            truncation is performed.
        """
        super().__init__()

        self._svd_rank: Union[int, float] = svd_rank

        self._snapshots: np.ndarray = None
        self._snapshots_shape: tuple[int, int] = None

        # SVD information
        self._rank: int = 0
        self._U: np.ndarray = None
        self._Sigma: np.ndarray = None
        self._Vstar: np.ndarray = None

        # Amplitudes
        self._b: np.ndarray = None

    @property
    def svd_rank(self) -> int:
        """
        Return the current SVD rank.

        Returns
        -------
        int or float
        """
        return self._svd_rank

    @property
    def snapshots(self) -> np.ndarray:
        """
        Return the underlying snapshot data.

        Returns
        -------
        numpy.ndarray (n_features, n_snapshots)
        """
        return self._snapshots

    @property
    def n_snapshots(self) -> int:
        """
        Return the number of snapshots.

        Returns
        -------
        int
        """
        return self._snapshots.shape[1]

    @property
    def n_features(self) -> int:
        """
        Return the number of features in each snapshot.

        Returns
        -------
        int
        """
        return self._snapshots.shape[0]

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
    def n_modes(self) -> int:
        """
        Return the number of modes.

        Returns
        -------
        int
        """
        return self._rank

    @property
    def amplitudes(self) -> np.ndarray:
        """
        Return the mode amplitudes per snapshot.

        Returns
        -------
        numpy.ndarray (n_snapshots, n_modes)
        """
        return self._b

    @property
    def reconstructed_data(self) -> np.ndarray:
        """
        Return the reconstructed training data.

        Returns
        -------
        numpy.ndarray (n_features, n_snapshots)
        """
        return self.modes @ self._b.T

    @property
    def reconstruction_error(self) -> float:
        """
        Compute the training data reconstruction \f$ \ell_2 \f$ error.

        Returns
        -------
        float
        """
        X = self.snapshots
        X_pod = self.reconstructed_data
        return norm(X - X_pod) / norm(X)

    @property
    def left_svd_modes(self) -> np.ndarray:
        """
        Return the left singular vectors.

        Returns
        -------
        numpy.ndarray (n_features, n_snapshots)
        """
        return self._U

    @property
    def right_svd_modes(self) -> np.ndarray:
        """
        Return the right singular vectors.

        Returns
        -------
        numpy.ndarray (n_snapshots, n_snapshots)
        """
        return np.conjugate(np.transpose(self._Vstar))

    @property
    def singular_values(self) -> np.ndarray:
        """
        Return the singular values.

        Returns
        -------
        numpy.ndarray (n_snapshots,)
        """
        return self._Sigma

    @property
    def snapshot_errors(self) -> np.ndarray:
        """
        Return the snapshot-wise reconstruction error.

        Returns
        -------
        numpy.ndarray (n_snapshots,)
        """
        X = self.snapshots
        X_dmd = self.reconstructed_data
        return norm(X - X_dmd, axis=0) / norm(X, axis=0)

    def fit(self, X: np.ndarray) -> 'POD':
        """
        Fit the POD model to the specified data.

        Parameters
        ----------
        X : numpy.ndarray
            The training snapshots.
        Y : ndarray (n_snapshots, n_parameters), default None
            The training parameters.

        Returns
        -------
        POD
        """
        X, Xshape = self._format_2darray(X)

        self._snapshots = X
        self._snapshots_shape = Xshape

        # Perform the SVD
        self._U, self._Sigma, self._Vstar = svd(X, full_matrices=False)
        self._rank = self._compute_rank()

        # Compute amplitudes
        s = np.diag(self._Sigma[:self._rank])
        self._b = np.transpose(s @ self._Vstar[:self._rank])

        return self

    def refit(self, svd_rank: Union[int, float]) -> 'POD':
        """
        Re-fit the POD model to the specified SVD rank.

        Parameters
        ----------
        svd_rank : int or float, default -1
            The SVD rank to use for truncation. If a positive integer,
            the minimum of this number and the maximum possible rank
            is used. If a float between 0 and 1, the minimum rank to
            achieve the specified energy content is used. If -1, no
            truncation is performed.

        Returns
        -------
        POD
        """
        self._svd_rank = svd_rank
        self._rank = self._compute_rank()

        # Recompute amplitudes
        s = np.diag(self._Sigma[:self._rank])
        self._b = np.transpose(s @ self._Vstar[:self._rank])

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
        return X.T @ self.modes

    def _compute_rank(self) -> int:
        """
        Return the POD rank given the singular values and SVD rank.

        Returns
        -------
        int
        """
        # Optimal rank
        if self._svd_rank == 0:
            def omega(x):
                return 0.56 * x ** 3 - 0.95 * x ** 2 + 1.82 * x + 1.43

            beta = np.divide(*sorted(self.snapshots))
            tau = np.median(self._Sigma) * omega(beta)
            return np.sum(self._Sigma > tau)

        # Energy truncation
        elif 0 < self._svd_rank < 1:
            s = self._Sigma
            cumulative_energy = np.cumsum(s ** 2 / np.sum(s ** 2))
            return np.searchsorted(cumulative_energy, self._svd_rank) + 1

        # Fixed rank
        elif self._svd_rank >= 1 and isinstance(self._svd_rank, int):
            return min(self._svd_rank, self._U.shape[1])

        # Full rank
        else:
            return self.snapshots.shape[1]

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
            mode_indices: Union[int, list[int]] = None,
            one_plot: bool = True,
            filename: str = None
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
                    base, ext = splitext(filename)
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
                        base, ext = splitext(filename)
                        plt.savefig(f"{base}_{idx}.pdf")
        plt.show()
