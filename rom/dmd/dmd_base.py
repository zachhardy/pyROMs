import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from scipy.linalg import eig, pinv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import Union, Tuple, List

Rank = Union[float, int]
Eig = Tuple[ndarray, ndarray, ndarray]


class DMDBase:
    """
    Dynamic Mode Decomposition base class.

    Parameters
    ----------
    svd_rank : int or float, default -1
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

    from ._plotting import (plot_singular_values,
                            plot_1D_profiles,
                            plot_dynamics,
                            plot_1D_profiles_and_dynamics,
                            plot_mode_evolutions,
                            plot_timestep_errors,
                            plot_error_decay)

    def __init__(self, svd_rank: Rank = -1, exact: bool = False,
                 ordering: str = 'eigenvalues') -> None:
        self.svd_rank: Rank = svd_rank
        self.exact: bool = exact
        self.ordering: str = ordering
        self.initialized: bool = False

        self.original_time: dict = None
        self.dmd_time: dict = None

        self._snapshots: ndarray = None
        self._snapshots_shape: tuple = None

        self._modes: ndarray = None
        self._n_modes: int = None
        self._eigs: ndarray = None
        self._A_tilde: ndarray = None
        self._b: ndarray = None

        self._left_svd_modes: ndarray = None
        self._right_svd_modes: ndarray = None
        self._singular_values: ndarray = None

    def fit(self, X: ndarray,
            original_time: dict = None) -> 'DMDBase':
        """
        Abstract method to fit the model to training data.

        This must be inplemented in subclasses.
        """
        raise NotImplementedError(
            f'Subclasses must implement abstact method '
            f'{self.__class__.__name__}.fit')

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
    def reconstructed_data(self) -> ndarray:
        """
        Get the reconstructed training data.

        Returns
        -------
        ndarray (n_snapshots, n_features)
        """
        return (self.modes @ self.dynamics).T

    @property
    def original_timesteps(self) -> ndarray:
        """
        Get the original time steps.

        Returns
        -------
        ndarray (n_snapshots,)
        """
        times = np.arange(
            self.original_time['t0'],
            self.original_time['tf'] + self.original_time['dt'],
            self.original_time['dt'])
        return times[[t <= self.original_time['tf'] for t in times]]

    @property
    def dmd_timesteps(self) -> ndarray:
        """
        Get the timesteps to sample.

        Returns
        -------
        ndarray
        """
        times = np.arange(self.dmd_time['t0'],
                          self.dmd_time['tf'] + self.dmd_time['dt'],
                          self.dmd_time['dt'])
        return times[[t <= self.dmd_time['tf'] for t in times]]

    @property
    def modes(self) -> ndarray:
        """
        Get the DMD modes, stored column-wise.

        Returns
        -------
        ndarray (n_features, n_modes)
        """
        return self._modes

    @property
    def dynamics(self) -> ndarray:
        """
        Get the dynamics for the dmd timesteps.

        Returns
        -------
        ndarray (n_modes, n_snapshots)
            The dynamics matrix.
        """
        t0 = self.original_time['t0']
        exp_arg = np.outer(self.omegas, self.dmd_timesteps - t0)
        return np.exp(exp_arg) * self._b[:, None]

    @property
    def A_tilde(self) -> ndarray:
        """
        Get the reduced order evolution operator.

        Returns
        -------
        ndarray (n_modes, n_modes)
        """
        return self._Atilde

    @property
    def amplitudes(self) -> ndarray:
        """
        Get the amplitudes of the DMD modes.

        Returns
        -------
        ndarray (n_modes,)
        """
        return self._b

    @property
    def eigs(self) -> ndarray:
        """
        Get the eigenvalues of the evolution operator.

        Returns
        -------
        ndarray (n_modes,)
        """
        return self._eigs

    @property
    def omegas(self) -> ndarray:
        """
        Get the continuous eigenvalues of the evolution operator.

        Returns
        -------
        ndarray (n_modes,)
        """
        return np.log(self.eigs, dtype=complex) / self.original_time['dt']

    @property
    def growth_rate(self) -> ndarray:
        """
        Get the temporal growth rates of the eigenvalues of the
        evolution operator.

        Returns
        -------
        ndarray (n_modes,)
        """
        return self.omegas.real

    @property
    def frequency(self) -> ndarray:
        """
        Get the temporal frequencies of the eigenvalues of the
        evolution operator.

        Returns
        -------
        ndarray (n_modes,)
        """
        return self.omegas.imag / 2.0 / np.pi

    @property
    def operator(self) -> ndarray:
        """
        Construct the approximate full order evolution operator.

        Returns
        -------
        ndarray (n_features, n_features)
        """
        rcond = 10.0 * np.finfo(float).eps
        pinv_modes = pinv(self.modes, rcond=rcond)
        return self.modes @ np.diag(self.eigs) @ pinv_modes

    @property
    def singular_values(self) -> ndarray:
        """
        Get the singular values.

        Returns
        -------
        ndarray (n_modes,)
        """
        return self._singular_values

    @property
    def n_modes(self) -> int:
        """
        Get the number of modes.

        Returns
        -------
        int
        """
        return self._n_modes

    @property
    def n_snapshots(self) -> int:
        """
        Get the number of snapshots.

        Returns
        -------
        int
        """
        return self.snapshots.shape[0]

    @property
    def n_features(self) -> int:
        """
        Get the number of features in a snapshot.

        Returns
        -------
        int
        """
        return self.snapshots.shape[1]

    def construct_lowrank_op(self, X: ndarray) -> ndarray:
        """
        Construct the low-rank operator from the SVD of X
        and the matrix Y.

        Parameters
        ----------
        X : ndarray (n_features, n_snapshots - 1)
            The last n_snapshots - 1 snapshots.

        Returns
        -------
        ndarray (rank, rank)
            The low-rank evolution operator.
        """
        Ur = self._left_svd_modes[:, :self.n_modes]
        Vr = self._right_svd_modes[:, :self.n_modes]
        Sr = self._singular_values[:self.n_modes]
        return Ur.conj().T @ X @ Vr * np.reciprocal(Sr)

    def eig_from_lowrank_op(self, X: ndarray) -> Eig:
        """

        Parameters
        ----------
        X : ndarray (n_features, n_snapshots - 1)
            The last n_snapshots - 1 snapshots.

        Returns
        -------
        ndarray : (n_modes,)
            The eigenvalues of the evolution operator.
        ndarray : (n_features, n_modes)
            The left eigenvectors of the evolution operator.
        ndarray : (n_features, n_modes)
            The right eigenvectors of the evolution operator.

        """
        w, vl, vr = eig(self._A_tilde, left=True)

        # Filter out zero eigenvalues
        non_zero_mask = w != 0.0
        w = w[non_zero_mask]
        vl = vl.T[non_zero_mask].T
        vr = vr.T[non_zero_mask].T

        # Compute the full-rank eigenvectors
        if self.exact:
            Vr = self._right_svd_modes[:, :self.n_modes]
            inv_Sr = np.reciprocal(self._singular_values[:self.n_modes])
            eigvecs_l = (X @ Vr * inv_Sr) @ vl * np.reciprocal(w)
            eigvecs_r = (X @ Vr * inv_Sr) @ vr * np.reciprocal(w)
        else:
            Ur = self._left_svd_modes[:, :self.n_modes]
            eigvecs_l = Ur @ vl
            eigvecs_r = Ur @ vr

        # Full-rank eigenvalues are the low-rank eigenvalues
        eigvals = w

        return eigvals, eigvecs_l, eigvecs_r

    def compute_amplitudes(self) -> ndarray:
        """
        Compute the amplitudes for the dynamic modes. This
        method fits the modes to the first snapshot, which
        is assumed to be the initial condition.

        Returns
        -------
        ndarray (n_modes,)
            The dynamic mode amplitudes.
        """
        x = self.snapshots[0]
        b: ndarray = np.linalg.lstsq(self.modes, x, rcond=None)[0]
        for m in range(self.n_modes):
            if b[m].real < 0.0:
                b[m] *= -1.0
                self._modes[:, m] *= -1.0
        return b

    def sort_modes(self) -> None:
        """
        Sort the dynamic modes based upon the specified criteria.
        This method updates the ordering of the private attributes.
        """
        if self.ordering is None:
            return
        if self.ordering not in ['amplitudes', 'eigenvalues']:
            raise AssertionError('Invalid ordering type.')

        # Determine sorted index mapping
        if self.ordering == 'amplitudes':
            idx = np.argsort(self.amplitudes)[::-1]
        elif self.ordering == 'eigenvalues':
            idx = np.argsort(self.eigs)[::-1]

        # Reset _eigs, _b, and _modes based on this
        self._b = self._b[idx]
        self._eigs = self._eigs[idx]
        self._modes = self._modes[:, idx]

    @property
    def reconstruction_error(self) -> float:
        """
        Compute the training data reconstruction error.

        Returns
        -------
        float
            The relative l2 reconstruction error.
        """
        X = self.snapshots
        X_pred = self.reconstructed_data
        return norm(X - X_pred)

    def compute_timestep_errors(self) -> ndarray:
        """
        Compute the errors as a function time step.

        Returns
        -------
        ndarray (n_samples,)
        """
        X = self.snapshots
        X_pred = self.reconstructed_data
        errors = np.zeros(self.n_snapshots)
        for t in range(self.n_snapshots):
            error = norm(X_pred[t] - X[t])
            errors[t] = error
        return errors

    def compute_error_decay(self) -> ndarray:
        """
        Compute the decay in the error.

        This method computes the error decay as a function
        of truncation level.

        Returns
        -------
        ndarray (n_modes,)
            The reproduction error as a function of n_modes.
        """
        errors = []
        svd_rank_original = self.svd_rank
        X, tinfo = self.snapshots, self.original_time
        for n in range(min(X.shape) - 1):
            params = self.get_params()
            params['svd_rank'] = n + 1
            dmd = self.__class__(**params)
            dmd.fit(self.snapshots, self.original_time, verbose=False)
            X_pred = dmd.reconstructed_data
            error = norm(X - X_pred)
            errors += [error]
        return np.array(errors)

    def get_params(self) -> dict:
        return {'svd_rank': self.svd_rank, 'exact': self.exact,
                'ordering': self.ordering}

    @staticmethod
    def validate_data(X: ndarray) -> Tuple[ndarray, tuple]:
        """
        Parameters
        ----------
        X : ndarray (n_snapshots, n_features)
            A matrix of snapshots stored row-wise.

        Returns
        -------
        The inputs
        """
        # Check for ndarrays
        if isinstance(X, ndarray) and X.ndim == 2:
            snapshots = X
            snapshots_shape = None
        else:
            input_shapes = [np.asarray(x).shape for x in X]
            if len(set(input_shapes)) != 1:
                msg = 'All snapshots do not have the same dimension.'
                raise ValueError(msg)
            snapshots = np.transpose([np.asarray(x).ravel() for x in X])
            snapshots_shape = input_shapes[0]
        return snapshots, snapshots_shape


def compute_error_decay(obj: DMDBase) -> ndarray:
    """
    Compute the decay in the error.

    This method computes the error decay as a function
    of truncation level.

    Parameters
    ----------
    obj : DMDBase

    Returns
    -------
    ndarray (n_modes,)
        The reproduction error as a function of n_modes.
    """
    errors = []
    X = obj.snapshots
    for n in range(min(X.shape) - 1):
        params = obj.get_params()
        params['svd_rank'] = n + 1
        dmd = obj.__class__(**params)
        dmd.fit(X, verbose=False)
        X_pred = dmd.reconstructed_data
        error = norm(X - X_pred)
        errors += [error]
    return np.array(errors)
