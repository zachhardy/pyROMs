import numpy as np
from numpy.linalg import eig, norm
import matplotlib.pyplot as plt

from .dmd_base import DMDBase
from svd import compute_svd

default = 1.0 - 1.0e-8

class DMD(DMDBase):
    """Dynamic mode decomposition model.

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

    def fit(self, X, timesteps=None):
        """Fit the DMD model to the provided data.

        Parameters
        ----------
        X : ndarray (n_snapshots, n_features)
            A matrix of snapshots stored row-wise.
        timesteps : ndarray (n_snapshots), default None
            Array of timestamps for the snapshots.
        """
        # Validate inputs
        X, timesteps = self._validate_data(X, timesteps)

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
        self._eigs, self._modes = self._eig_from_lowrank_op(X1)

        # Compute amplitudes
        self._b = self._compute_amplitudes()

        # Sort the modes
        self._sort_modes(self.ordering)

        # Set timesteps
        if timesteps is None:
            timesteps = np.arange(0.0, self.n_snapshots, 1.0)
        self.original_timesteps = timesteps
        self.dmd_timesteps = timesteps
        self.dt = timesteps[1] - timesteps[0]
        return self
