import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from scipy.linalg import eig, pinv2
import matplotlib.pyplot as plt

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

    def fit(self, X: ndarray, original_time: dict = None,
            verbose: bool = True) -> 'DMD':
        """
        Fit the DMD model to the provided data.

        Parameters
        ----------
        X : ndarray (n_snapshots, n_features)
            A matrix of snapshots stored row-wise.
        original_time : dict
            A dictionary containing the initial and final
            times and the time step size.
        verbose : bool, default True
            Flag for printing model summary.
        """
        # Validate inputs
        X = self._validate_data(X)

        # Save the input data
        self._snapshots = np.copy(X)
        self.n_snapshots = self._snapshots.shape[0]
        self.n_features = self._snapshots.shape[1]

        # Split snapshots (n_features, n_snapshots - 1)
        X0 = self._snapshots[:-1].T
        X1 = self._snapshots[1:].T

        # Compute the SVD
        U, s, V, rank = compute_svd(X0, self.svd_rank)
        self.n_modes = rank
        self._left_svd_modes = U
        self._right_svd_modes = V
        self._singular_values = s

        # Compute the reduced-rank evolution operator
        self._A_tilde = self._construct_lowrank_op(X1)

        # Eigendecomposition of Atilde
        tmp = self._eig_from_lowrank_op(X1)
        self._eigs = tmp[0]
        self._modes = tmp[2]

        # Compute amplitudes
        self._b = self._compute_amplitudes()

        # Sort the modes
        self._sort_modes()

        # Set original and DMD time
        n = self.n_snapshots
        if original_time is None:
            self.original_time = {'t0': 0, 'tf': n - 1, 'dt': 1}
            self.dmd_time = {'t0': 0, 'tf': n - 1, 'dt': 1}
        elif isinstance(original_time, dict):
            # Check the dictionary
            keys = ['t0', 'tf', 'dt']
            for key in keys:
                if key not in list(original_time.keys()):
                    msg = f'{key} not found in original_time.'
                    raise KeyError(msg)

            # Set the dictionary
            self.original_time = original_time
            self.dmd_time = original_time
        else:
            raise TypeError('original_time must be a dict.')

        # Set initialized flag
        self.initialized = True

        # Print summary
        if verbose:
            print('\n*** DMD model information ***')

            n = self.n_modes
            print(f'Number of Modes:\t\t{n}')

            s = self._singular_values
            print(f'Smallest Kept Singular Value:\t{s[n - 1] / sum(s):.3e}')

            ic = self.snapshots[0]
            fit = (self.modes @ self.amplitudes).ravel()
            ic_error = norm(ic - fit, ord=2) / norm(ic, ord=2)
            print(f'Initial Condition Error:\t{ic_error:.3e}')

            error = self.reconstruction_error
            print(f'Reconstruction Error:\t\t{error:.3e}\n')
        return self
