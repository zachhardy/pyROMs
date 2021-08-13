import numpy as np
from numpy import ndarray
from numpy.linalg import norm

from .dmd_base import DMDBase


class DMD(DMDBase):
    """
    Dynamic mode decomposition model.
    """

    def fit(self, X: ndarray, verbose: bool = True) -> None:
        """
        Fit the DMD model to the provided data.

        Parameters
        ----------
        X : ndarray (n_snapshots, n_features)
            A matrix of snapshots stored row-wise.
        """
        X, X_shape = self._validate_data(X)

        # ======================================== Save the input data
        self._snapshots: ndarray = np.copy(X)
        self._snapshots_shape: tuple = X_shape

        # ======================================== Split snapshots
        X0 = self._snapshots[:-1].T
        X1 = self._snapshots[1:].T

        # ======================================== Compute the SVD
        U, s, V = self._compute_svd(X0)

        # ======================================== Low-rank operator
        self._a_tilde = self._construct_atilde(U, s, V, X1)
        self._eigs, self._eigvecs = np.linalg.eig(self._a_tilde)

        # ======================================== Compute DMD modes
        self._modes = self._compute_modes(U, s, V, X1)

        # ======================================== Set default time steps
        n = self.n_snapshots
        self.original_time = {'t0': 0, 'tf': n - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tf': n - 1, 'dt': 1}

        # ======================================== Compute amplitudes
        self._b = self._compute_amplitudes()

        # ======================================== Sort and filter modes
        self.sort_modes()
        self.filter_out_unstable_modes()

        # ======================================== Print summary
        if verbose:
            print("\n*** DMD model information ***")

            n = self.n_modes
            print(f"Number of Modes:\t\t{n}")

            s = self._singular_values
            print(f"Smallest Kept Singular Value:\t{s[n - 1] / sum(s):.3e}")

            ic = self.snapshots[0]
            fit = (self.modes @ self.amplitudes).ravel()
            ic_error = norm(ic - fit) / norm(ic)
            print(f"Initial Condition Error:\t{ic_error:.3e}")

            error = self.reconstruction_error
            print(f"Reconstruction Error:\t\t{error:.3e}\n")
