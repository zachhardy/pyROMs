import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from scipy.linalg import eig, pinv2

import matplotlib.pyplot as plt

from typing import Union

from .dmd_base import DMDBase
from ..svd import compute_svd


class DMD(DMDBase):
    """
    Dynamic mode decomposition model.

    Parameters
    ----------
    svd_rank : int or float, default 1.0 - 1.0e-8
        The SVD truncation rank. If -1, no truncation is used.
        If a positive integer, the truncation rank is the argument.
        If a float between 0 and 1, the minimum number of modes
        needed to obtain an information content greater than the
        argument is used.
    exact : bool, default False
        Flag for exact modes. If False, projected modes are used.
    ordering : 'amplitudes' or 'eigenvalues', default 'eiganvalues'
        The sorting method applied to the dynamic modes.
    """

    def fit(self, X: ndarray, verbose: bool = True) -> 'DMD':
        """
        Fit the DMD model to the provided data.

        Parameters
        ----------
        X : ndarray (n_snapshots, n_features)
            A matrix of snapshots stored row-wise.
        """
        X, x_shape = self.validate_data(X)

        # Save the input data
        self._snapshots: ndarray = np.copy(X)
        self._snapshots_shape: tuple = x_shape

        # Split snapshots (n_features, n_snapshots - 1)
        X0 = self._snapshots[:-1].T
        X1 = self._snapshots[1:].T

        # Compute the SVD
        U, s, V = np.linalg.svd(X0, full_matrices=False)
        self._left_svd_modes = U
        self._right_svd_modes = V.conj().T
        self._singular_values = s

        # Determine the number of modes
        self._n_modes = self.compute_rank(self.svd_rank)

        # Compute the reduced-rank evolution operator
        self._A_tilde = self.construct_lowrank_op(X1)

        # Eigendecomposition of Atilde
        self._eigs, _, self._modes = self.eig_from_lowrank_op(X1)

        # Compute amplitudes
        self._b = self.compute_amplitudes()

        # Set default timesteps
        n = self.n_snapshots
        self.original_time = {'t0': 0, 'tf': n - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tf': n - 1, 'dt': 1}

        # Sort the modes
        self.sort_modes()

        # Print summary
        if verbose:
            print('\n*** DMD model information ***')

            n = self.n_modes
            print(f'Number of Modes:\t\t{n}')

            s = self._singular_values
            print(f'Smallest Kept Singular Value:\t{s[n - 1] / sum(s):.3e}')

            ic = self.snapshots[0]
            fit = (self.modes @ self.amplitudes).ravel()
            ic_error = norm(ic - fit)
            print(f'Initial Condition Error:\t{ic_error:.3e}')

            error = self.reconstruction_error
            print(f'Reconstruction Error:\t\t{error:.3e}\n')
        return self
