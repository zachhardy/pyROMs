import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from scipy.linalg import pinv

from typing import Union, Tuple, List

Rank = Union[float, int]
Eig = Tuple[ndarray, ndarray]
SVD = Tuple[ndarray, ndarray, ndarray]


class DMDBase:
    """Dynamic Mode Decomposition base class.

    Attributes
    ----------
    svd_rank : float or int
        The SVD truncation rank. If -1, no truncation is used.
        If a positive integer, the truncation rank is the argument.
        If a float between 0 and 1, the minimum number of modes
        needed to obtain an information content greater than the
        argument is used.

    exact : bool
        Flag for exact modes. If False, projected modes are used.

    opt : bool, default False
            Flag for optimal mode amplitudes. If False, the modes are
            fit to the initial condition.

    ordering : {"AMPLITUDE", "EIGENVALUE"}
            The sorting method applied to the dynamic modes.

    original_timestamps : ndarray (n_snapshots,)
        The timestamps corresponding to each snapshot in the
        training data.

    dmd_timestamps : ndarray
        The timestamps to sample the model at. This is defaulted
        to the same timestamps as the original ones.
    """

    from ._plotting import (plot_singular_values,
                            plot_1D_modes,
                            plot_dynamics,
                            plot_1D_modes_and_dynamics,
                            plot_1D_mode_evolutions,
                            plot_eigs,
                            plot_timestep_errors,
                            plot_error_decay)

    def __init__(self, svd_rank: Rank = -1, exact: bool = False,
                 opt: bool = False, ordering: str = "AMPLITUDE") -> None:
        """Constructor.

        Parameters
        ----------
        svd_rank : int or float, default -1
        exact : bool, default False
        opt : bool, default False
        ordering : {"AMPLITUDE", "EIGENVALUE"}, default "AMPLITUDE"
        """
        self.svd_rank: Rank = svd_rank
        self.exact: bool = exact
        self.opt: bool = opt
        self.ordering: str = ordering

        self.original_timesteps: ndarray = None
        self.dmd_timesteps: ndarray = None

        self._snapshots: ndarray = None
        self._snapshots_shape: tuple = None

        self._a_tilde: ndarray = None
        self._eigs: ndarray = None
        self._eigvecs: ndarray = None
        self._modes: ndarray = None
        self._b: ndarray = None

        self._singular_values: ndarray = None

    @property
    def snapshots(self) -> ndarray:
        """Get the original training data.

        Returns
        -------
        ndarray (n_samples, n_features)
        """
        return self._snapshots

    @property
    def n_snapshots(self) -> int:
        """Get the number of snapshots in the training data.

        Returns
        -------
        int
        """
        return self.snapshots.shape[0]

    @property
    def n_features(self) -> int:
        """Get the number of features in each snapshot.

        Returns
        -------
        int
        """
        return self.snapshots.shape[1]

    @property
    def singular_values(self) -> ndarray:
        """Get the singular values.

        Returns
        -------
        ndarray (n_snapshots - 1,)
        """
        return self._singular_values

    @property
    def a_tilde(self) -> ndarray:
        """Get the low-rank evolution operator.

        Returns
        -------
        ndarray (n_modes, n_modes)
        """
        return self._a_tilde

    @property
    def modes(self) -> ndarray:
        """Get the modes, stored column-wise.

        Returns
        -------
        ndarray (n_features, n_modes)
        """
        return self._modes

    @property
    def n_modes(self) -> int:
        """Get the number of modes.

        Returns
        -------
        int
        """
        return self.modes.shape[1]

    @property
    def dynamics(self) -> ndarray:
        """Compute the dynamics operator.

        Returns
        -------
        ndarray (n_modes, n_snapshots)
            The dynamics matrix, where each row contains the
            corresponding modes's weighting for the given time step.
        """
        t0 = self.original_timesteps[0]
        exp_arg = np.outer(self.omegas, self.dmd_timesteps - t0)
        return np.exp(exp_arg) * self._b[:, None]

    @property
    def eigs(self) -> ndarray:
        """Get the discrete eigenvalues.

        Returns
        -------
        ndarray (n_modes,)
            These are the raw eigenvalues from the decomposition
            of the low-rank dynamics operator.
        """
        return self._eigs

    @property
    def omegas(self) -> ndarray:
        """Get the continuous eigenvalues.

        Returns
        -------
        ndarray (n_modes,)
            These are the mapped continuous eigenvalues from the
            low-rank operator using the fixed time step size of
            the snapshots.
        """
        dt = np.diff(self.original_timesteps)[0]
        return np.log(self.eigs, dtype=complex) / dt

    @property
    def eigvecs(self) -> ndarray:
        """Get the low-rank eigenvectors.

        Returns
        -------
        ndarray (n_modes, n_modes)
        """
        return self._eigvecs

    @property
    def growth_rate(self) -> ndarray:
        """Get the temporal growth rates of the eigenvalues.

        Returns
        -------
        ndarray (n_modes,)
            The real part of the exponential coefficients that
            govern each mode.
        """
        return self.omegas.real

    @property
    def frequency(self) -> ndarray:
        r"""Get the temporal frequencies of the eigenvalues.

        Returns
        -------
        ndarray (n_modes,)
            The imaginary part of the exponential coefficients
            that govern each mode scaled by 1 / 2 \pi.
        """
        return self.omegas.imag / 2.0 / np.pi

    @property
    def amplitudes(self) -> ndarray:
        """Get the mode amplitudes.

        Returns
        -------
        ndarray (n_modes,)
            The coefficients of each mode that minimize
            the error between the model and the data set.
        """
        return self._b

    @property
    def operator(self) -> ndarray:
        """The approximate full order evolution operator.

        Returns
        -------
        ndarray (n_features, n_features)
            The full-order dynamics operator
        """
        rcond = 10.0 * np.finfo(float).eps
        pinv_modes = pinv(self.modes, rcond=rcond)
        return self.modes @ np.diag(self.eigs) @ pinv_modes

    @property
    def reconstructed_data(self) -> ndarray:
        """Get the reconstructed training data.

        Returns
        -------
        ndarray (n_snapshots, n_features)
            The reconstructed data set from the model.
        """
        return (self.modes @ self.dynamics).T

    @property
    def reconstruction_error(self) -> float:
        """Compute the training data reconstruction error.

        Returns
        -------
        float
            The relative L2 error between the training snapshots
            and the reconstructed snapshots.
        """
        X = self.snapshots
        X_pred = self.reconstructed_data.real
        return norm(X - X_pred) / norm(X)

    def fit(self, X: ndarray, verbose: bool = True) -> None:
        raise NotImplementedError(
            f'Subclasses must implement abstact method '
            f'{self.__class__.__name__}.fit')

    def _compute_svd(self, X: ndarray, svd_rank: Rank = -1) -> SVD:
        """Compute the truncated singular value decomponsition of X

        Parameters
        ----------
        X : ndarray
        svd_rank : int, default -1
        """
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        self._singular_values = s
        V = Vh.conj().T

        if 0.0 < svd_rank < 1.0:
            cumulative_energy = np.cumsum(s ** 2 / sum(s ** 2))
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif isinstance(svd_rank, int) and svd_rank >= 1:
            rank = min(svd_rank, min(X.shape))
        else:
            rank = X.shape[1]
        return U[:, :rank], s[:rank], V[:, :rank]

    @staticmethod
    def _construct_atilde(U: ndarray, s: ndarray,
                          V: ndarray, Y: ndarray) -> ndarray:
        """Construct the low-rank evolution operator.

        Parameters
        ----------
        U : ndarray (n_features, rank)
            The left singular vectors, stored column-wise.
        s : ndarray (rank,)
            The sigular values.
        V : ndarray (n_snapshots - 1, rank)
            The right singular vectors, stored column-wise
        Y : ndarray (n_features, n_snapshots - 1)
                The last n_snapshots - 1 snapshots, stored
                column-wise.

        Returns
        -------
        ndarray (rank, rank)
            The low-rank evolution operator.
        """
        return U.conj().T @ Y @ V * np.reciprocal(s)

    def _compute_modes(self, U: ndarray, s: ndarray,
                       V: ndarray, Y: ndarray) -> Eig:
        """Compute the dynamic modes.

        Compute the dynamic modes by taking the eigendecomposition
        of the low-rank evolution operator and mapping the results
        back to high-dimensional space.

        Parameters
        ----------
        U : ndarray (n_features, rank)
            The left singular vectors, stored column-wise.
        s : ndarray (rank,)
            The sigular values.
        V : ndarray (n_snapshots - 1, rank)
            The right singular vectors, stored column-wise
        Y : ndarray (n_features, n_snapshots - 1)
                The last n_snapshots - 1 snapshots, stored
                column-wise.

        Returns
        -------
        ndarray : (n_modes,)
            The eigenvalues of the evolution operator.
        ndarray : (n_features, n_modes)
            The eigenvectors of the evolution operator.

        """
        if self.exact:
            modes = Y @ V * np.reciprocal(s) \
                    @ self._eigvecs * np.reciprocal(self._eigs)
        else:
            modes = U @ self._eigvecs

        # Normalize the full-order eigenvectors
        for m in range(modes.shape[1]):
            modes[:, m] /= norm(modes[:, m])
        return modes

    def _compute_amplitudes(self) -> ndarray:
        """Compute the amplitudes for the dynamic modes.

        This method either fits the first snapshot, if opt is
        false, or performs an optimal fit over the full data set,
        if true.

        Returns
        -------
        ndarray (n_modes,)
            The dynamic mode amplitudes.
        """
        if self.opt:
            meshgrid = np.meshgrid(self.omegas, self.dmd_timesteps)
            vander = np.exp(np.multiply(*meshgrid)).T

            U, s, Vh = np.linalg.svd(self._snapshots.T, full_matrices=False)
            V = Vh.conj().T

            P = self.eigvecs.conj().T @ self.eigvecs * \
                np.conj(vander @ vander.conj().T)

            S = np.diag(s).conj()
            Y = U.conj().T @ self.modes
            q = np.diag(vander @ V @ S @ Y).conj()
            return np.linalg.solve(P, q)
        else:
            x0 = self.snapshots[0]
            return np.linalg.lstsq(self.modes, x0, rcond=None)[0]

    def sort_modes(self) -> None:
        """Sort the modes based on the specified criteria.
        """
        if self.ordering is None:
            return
        if self.ordering not in ["AMPLITUDE", "EIGENVALUE"]:
            raise AssertionError("Invalid ordering type.")

        # Determine sorted index mapping
        if self.ordering == "AMPLITUDE":
            ind = np.argsort(np.abs(self.amplitudes.real))[::-1]
        else:
            ind = np.argsort(np.abs(self.omegas.real))[::-1]

        # Reset _eigs, _eigvecs, _modes, and _b based on this
        self._eigs = self._eigs[ind]
        self._eigvecs = self._eigvecs[:, ind]
        self._modes = self._modes[:, ind]
        self._b = self._b[ind]

    def remove_unstable_modes(self) -> None:
        """Remove modes whose eigenvalues are unstable.
        """
        stable_mask = np.abs(self.eigs.real) < 1.0
        n_unstable = sum([1 for val in stable_mask if not val])
        if n_unstable > 1:
            print(f"Unstable modes removed:\t{n_unstable}")

        self._eigs = self.eigs[stable_mask]
        self._eigvecs = self.eigvecs.T[stable_mask].T
        self._modes = self.modes.T[stable_mask].T
        self._b = self.amplitudes.T[stable_mask].T

    def compute_timestep_errors(self) -> ndarray:
        """Compute the errors as a function time step.

        Returns
        -------
        ndarray (n_samples,)
        """
        X = self.snapshots
        X_pred = self.reconstructed_data
        errors = np.zeros(self.n_snapshots)
        for t in range(self.n_snapshots):
            error = norm(X_pred[t] - X[t]) / norm(X[t])
            errors[t] = error
        return errors

    def compute_error_decay(self, skip: int = 1,
                            end: int = None) -> ndarray:
        """Compute the decay in the error.

        This method computes the error decay as a function
        of truncation level.

        Parameters
        ----------
        skip : int, default 1
            Interval to use when varying number of modes.
        end : int, default None
            The largest number of modes to compute the reconstruction
            error for. If None, the last mode will be the last one.

        Returns
        -------
        ndarray
            The reproduction error as a function of n_modes.
        ndarray
            The corresponding number of modes to each entry of
            the error vector.
        """
        X, times = self.snapshots, self.original_timesteps
        svd_rank_original = self.svd_rank
        if end is None or end > min(X.shape) - 1:
            end = min(X.shape) - 1

        errors: List[float] = []
        n_modes: List[int] = []
        for n in range(0, end, skip):
            self.svd_rank = n + 1
            self.fit(X, times, verbose=False)
            error = self.reconstruction_error.real
            errors.append(error)
            n_modes.append(n)

        self.svd_rank = svd_rank_original
        self.fit(X, times, verbose=False)
        return errors, n_modes

    def get_params(self) -> dict:
        return {"svd_rank": self.svd_rank, "exact": self.exact,
                "opt": self.opt, "ordering": self.ordering}

    @staticmethod
    def _validate_data(X: ndarray) -> Tuple[ndarray, tuple]:
        """Ensure the input data is formatted correctly.

        Parameters
        ----------
        X : ndarray (n_snapshots, n_features)
            A matrix of snapshots stored row-wise.

        Returns
        -------
        ndarray, Tuple[int, int]
        """
        # Check for ndarrays
        if isinstance(X, ndarray) and X.ndim == 2:
            snapshots = X
            snapshots_shape = None
        else:
            input_shapes = [np.asarray(x).shape for x in X]
            if len(set(input_shapes)) != 1:
                raise ValueError(
                    f"All snapshots do not have the same dimension.")
            snapshots = np.transpose([np.asarray(x).ravel() for x in X])
            snapshots_shape = input_shapes[0]
        return snapshots, snapshots_shape
