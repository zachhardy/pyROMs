import numpy as np
from numpy import ndarray
from numpy.linalg import norm, svd, eig, multi_dot

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from os.path import splitext
from typing import Union, Iterable, List

from ..plotting_mixin import PlottingMixin

InputType = Union[ndarray, Iterable]


class DMDBase(PlottingMixin):
    """
    Dynamic Mode Decomposition base class inherited from PyDMD.

    Parameters
    ----------
    svd_rank : int or float, default 0
        The rank for the truncation. If 0, the method computes the
        optimal rank and uses it for truncation. If positive interger, the
        method uses the argument for the truncation. If float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`. If -1, the method does
        not compute truncation.
    exact : bool, default False
        Flag to compute either exact DMD or projected DMD.
    opt : bool or int, default False
        If True, optimal amplitudes are computes. If False, the
        amplitudes are computed via a fit to the first snapshot. If an int,
        they are computed via a fit to the specifiec snapshot indes.
    sorted_eigs : {'real', 'abs'} or None, default None
         Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`.
    """

    def __init__(self,
                 svd_rank: Union[int, float] = 0,
                 exact: bool = False,
                 opt: Union[bool, int] = False,
                 sorted_eigs: Union[bool, str] = None) -> None:

        # Model parameters
        self.svd_rank: Union[int, float] = svd_rank
        self.exact: bool = exact
        self.opt: Union[bool, int] = opt
        self.sorted_eigs: str = sorted_eigs

        # Training information
        self._snapshots: ndarray = None
        self._snapshots_shape: tuple = None

        self.original_time: dict = {}
        self.dmd_time: dict = {}

        # SVD information
        self._U: ndarray = None
        self._s: ndarray = None
        self._V: ndarray = None

        # DMD information
        self._Atilde: ndarray = None
        self._eigvals: ndarray = None
        self._eigvecs: ndarray = None
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
    def snapshots(self) -> ndarray:
        """
        Get the original training data.

        Returns
        -------
        ndarray (n_snapshots, n_features)
        """
        return self._snapshots

    @property
    def Atilde(self) -> ndarray:
        """
        Get the reduced-order evolution operator.

        Returns
        -------
        ndarray (n_modes, n_modes)
        """
        return self._Atilde

    @property
    def eigvals(self) -> ndarray:
        """
        Get the eigenvalues from the DMD model.

        Returns
        -------
        ndarray (n_modes,)
        """
        return self._eigvals

    @property
    def omegas(self) -> ndarray:
        """
        Get the continuous time-eigenvalues of the modes.

        Returns
        -------
        ndarray (n_modes,)
        """
        return np.log(self.eigvals) / self.original_time['dt']

    @property
    def frequency(self) -> ndarray:
        """
        Get the dynamic frequencies of the modes.

        Returns
        -------
        ndarray (n_modes,)
        """
        return self.omegas.imag / (2.0*np.pi)

    @property
    def eigvecs(self) -> ndarray:
        """
        Get the low-dimensional eigenvectors, column-wise.

        Returns
        -------
        ndarray (n_modes, n_modes)
        """
        return self._eigvecs

    @property
    def modes(self) -> ndarray:
        """
        Get the high-dimensional dynamic modes, column-wise.

        Returns
        -------
        ndarray (n_features, n_modes)
        """
        return self._modes

    @property
    def amplitudes(self) -> ndarray:
        """
        Get the amplitudes of the modes.

        Returns
        -------
        ndarray (n_modes,)
        """
        return self._b

    @property
    def dynamics(self) -> ndarray:
        """
        Get the time evolution of each mode.

        Returns
        -------
        ndarray (n_modes, varies)
        """
        # Compute base matrix of eigvals
        base = np.repeat(self.eigvals[:, None],
                         self.dmd_timesteps.shape[0],
                         axis=1)

        # Compute the powers for each snapshot
        powers = np.divide(self.dmd_timesteps - self.original_time['t0'],
                           self.original_time['dt'])

        # Compute \Lambda^{(t - t0)/dt} * b
        return np.power(base, powers) * self._b[:, None]

    @property
    def original_timesteps(self) -> ndarray:
        """
        Get the timesteps of the original snapshots.

        Returns
        -------
        ndarray (n_snapshots - 1,)
        """
        return np.arange(
            self.original_time['t0'],
            self.original_time['tend'] + self.original_time['dt'],
            self.original_time['dt']
        )

    @property
    def dmd_timesteps(self) -> ndarray:
        """
        Get the timesteps of the reconstructed snapshots.

        Returns
        -------
        ndarray (n_snapshots - 1,)
        """
        return np.arange(
            self.dmd_time['t0'],
            self.dmd_time['tend'] + self.dmd_time['dt'],
            self.dmd_time['dt']
        )

    @property
    def reconstructed_data(self) -> ndarray:
        """
        Get the reconstructed snapshots.

        Returns
        -------
        ndarray (n_snapshots, n_features)
        """
        return np.transpose(self._modes.dot(self.dynamics))

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
        return norm(X - X_dmd, axis=1)/norm(X, axis=1)

    @property
    def left_svd_modes(self) -> ndarray:
        """
        Get the left singular vectors, column-wise.

        Returns
        -------
        ndarray (n_features, n_snapshots - 1)
        """
        return self._U

    @property
    def right_svd_modes(self) -> ndarray:
        """
        Get the right singular vectors, column-wise.

        Returns
        -------
        ndarray (n_snapshots - 1,) * 2
        """
        return self._V

    @property
    def singular_values(self) -> ndarray:
        """
        Get the singular values vector.

        Returns
        -------
        ndarray (n_snapshots - 1,)
        """
        return self._s

    def fit(self, X: InputType) -> 'DMDBase':
        """
        Abstract method to fit the DMD model to inputs X.
        It has to be implemented in subclasses.

        Parameters
        ----------
        X : ndarray or iterable
            The input snapshots.
        """
        raise NotImplementedError(
            f'Subclass must implement the method '
            f'{self.__class__.__name__}.fit')

    def find_optimal_parameters(self) -> None:
        """
        Abstract method to find optimal hyper-parameters.
        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(
            f'Subclass must implement abstract method '
            f'{self.__class__.__name__}.find_optimal_parameters')

    @staticmethod
    def _compute_Atilde(U: ndarray, s: ndarray,
                        V: ndarray, Y: ndarray) -> ndarray:
        """
        Compute the low-rank evolution operator `Atilde`.

        Parameters
        ----------
        U : ndarray (n_features, n_modes)
            The left singular vectors.
        s : ndarray (n_modes,)
            The singular values vector.
        V : ndarray (n_snapshots - 1, n_modes)
            The right singular vectors.
        Y : ndarray (n_features, n_snapshots - 1)
            Matrix containing snapshots x_1, ..., x_n,
            column-wise.

        Returns
        -------
        ndarray (n_modes, n_modes)
        """
        return multi_dot([np.conj(U.T), Y, V]) * np.reciprocal(s)

    def _decompose_Atilde(self) -> None:
        """
        Compute the eigendecomposition of the low-rank
        evolution operator `Atilde`.
        """
        self._eigvals, self._eigvecs = eig(self._Atilde)

        # Determine the sorting routine
        if self.sorted_eigs is not None:
            if self.sorted_eigs == 'abs':
                def k(tp):
                    return abs(tp[0])
            elif self.sorted_eigs == 'real':
                def k(tp):
                    eig = tp[0]
                    if isinstance(eig, complex):
                        return eig.real, eig.imag
                    return eig.real, 0.0
            else:
                raise ValueError(f'Invalid value for sorted_eigs.')

            # Sort based on sorting function k
            eigpairs = zip(self._eigvals, self._eigvecs)
            a, b = zip(*sorted(eigpairs), key=k)
            self._eigvals = np.array([val for val in a])
            self._eigvecs = np.array([vec for vec in b]).T

    def _compute_modes(self, U: ndarray, s: ndarray,
                       V: ndarray, Y: ndarray) -> None:
        """
        Private method to compute the high-dimensional dynamic
        modes after performing an eigendecomposition of `Atilde`.

        Parameters
        ----------
        U : ndarray (n_features, n_modes)
            The left singular vectors, column-wise.
        s : ndarray (n_modes,)
            The singular values vector.
        V : ndarray (n_snapshots - 1, n_modes)
            The right singular vectors, column-wise.
        Y : ndarray (n_features, n_snapshots - 1)
            Matrix containing snapshots x_1, ..., x_n,
            column-wise
        """
        if self.exact:
            self._modes = (Y.dot(V) * np.reciprocal(s)).dot(self._eigvecs)
        else:
            self._modes = U.dot(self._eigvecs)

    def _compute_amplitudes(self) -> ndarray:
        """
        Compute the amplitude coefficients according to the
        `self.opt` parameter.
        """
        # Optimized amplitudes
        if isinstance(self.opt, bool) and self.opt:
            # Compute Vandermonde matrix
            vander = np.vander(self.eigvals,
                               len(self.dmd_timesteps),
                               True)

            # Perform SVD on all snapshots
            U, s, V = svd(self._snapshots.T, False)

            # Compute LHS, RHS
            P = np.multiply(
                np.dot(self.modes.conj().T, self.modes),
                np.conj(np.dot(vander, vander.conj().T))
            )

            tmp = np.linalg.multi_dot([U, np.diag(s), V]).conj().T
            q = np.conj(np.diag(multi_dot([vander, tmp, self.modes])))

            # Solve Pb = q
            b = np.linalg.solve(P, q)

        # Amplitudes fit to a specific snapshot
        else:
            if isinstance(self.opt, bool):
                snapshot_idx = 0
            else:
                snapshot_idx = self.opt

            b = np.linalg.lstsq(
                self._modes,
                self._snapshots[snapshot_idx],
                rcond=None)[0]

        # Ensure positive amplitudes
        for i in range(self.n_modes):
            if b[i] < 0.0:
                b[i] *= -1.0
                self._modes[:, i] *= -1.0

        return b

    def print_summary(self) -> None:
        """
        Print a summary of the DMD model.
        """
        msg = '===== DMD Summary ====='
        header = '='*len(msg)
        print('\n'.join([header, msg, header]))
        print(f"{'# of Modes':<20}: {self.n_modes}")
        print(f"{'# of Snapshots':<20}: {self.n_snapshots}")
        print(f"{'Reconstruction Error':<20}: "
              f"{self.reconstruction_error:.3e}")
        print(f"{'Mean Snapshot Error':<20}: "
              f"{np.mean(self.snapshot_errors):.3e}")
        print(f"{'Max Snapshot Error':<20}: "
              f"{np.max(self.snapshot_errors):.3e}")

    def plot_dynamics(self,
                      mode_indices: List[int] = None,
                      logscale: bool = False,
                      filename: str = None) -> None:
        """
        Plot the dynamics behaviors of the modes at the DMD timesteps.

        Parameters
        ----------
        mode_indices : List[int], default None
            The indices of the modes to plot. The default behavior
            is to plot all modes.
        logscale : bool, default False
            Flag for plotting on a logscale
        filename : str, default None
            A location to save the plot to, if specified.
        """
        # Check the inputs
        if self.modes is None:
            raise ValueError('The fit method must be performed first.')

        t = self.dmd_timesteps

        if mode_indices is None:
            mode_indices = list(range(self.n_modes))
        elif isinstance(mode_indices, int):
            mode_indices = [mode_indices]

        # Plot each mode dynamic specified
        for idx in mode_indices:
            idx += 0 if idx > 0 else self.n_modes
            dynamic: ndarray = self.dynamics[idx] / self._b[idx]
            omega = np.log(self.eigvals[idx]) / self.original_time['dt']

            # Make figure
            fig: Figure = plt.figure()
            fig.suptitle(f'DMD Dynamics {idx}\n$\omega$ = '
                         f'{omega.real:.3e}'
                         f'{omega.imag:+.3g}', fontsize=12)

            # Plot the dynamics
            ax: Axes = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('t', fontsize=12)
            ax.set_ylabel('Real', fontsize=12)
            plotter = ax.semilogy if logscale else ax.plot
            plotter(t, dynamic.real)

            plt.tight_layout()
            if filename is not None:
                base, ext = splitext(filename)
                plt.savefig(base + f'_{idx}.pdf')
