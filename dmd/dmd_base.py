import numpy as np
from numpy.linalg import eig, norm
import matplotlib.pyplot as plt


class DMDBase:
    """Dynamic Mode Decomposition base class.

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
    def __init__(self, svd_rank=-1, exact=False, ordering='eigenvalues'):
        self.svd_rank = svd_rank
        self.exact = exact
        self.ordering = ordering
        self.n_snapshots = 0
        self.n_features = 0
        self.n_modes = 0

        self.original_timesteps = None
        self.dmd_timesteps = None
        self.dt = 0.0

        self._snapshots = None
        self._modes = None
        self._eigs = None
        self._A_tilde = None
        self._b = None

        self._left_svd_modes = None
        self._right_svd_modes = None
        self._singular_values = None

    @property
    def snapshots(self):
        """
        Get the original training data.

        Returns
        -------
        ndarray (n_snapshots, n_features)
        """
        return self._snapshots

    @property
    def modes(self):
        """
        Get the DMD modes, stored column-wise.

        Returns
        -------
        ndarray (n_features, n_modes)
        """
        return self._modes

    @property
    def A_tilde(self):
        """
        Get the reduced order evolution operator.

        Returns
        -------
        ndarray (n_modes, n_modes)
        """
        return self._Atilde

    @property
    def eigs(self):
        """
        Get the eigenvalues of the evolution operator.

        Returns
        -------
        ndarray (n_modes,)
        """
        return self._eigs

    @property
    def omegas(self):
        """
        Get the continuous eigenvalues of the evolution operator.

        Returns
        -------
        ndarray (n_modes,)
        """
        return np.log(self._eigs, dtype=complex) / self.dt

    @property
    def frequency(self):
        """
        Get the temporal frequencies of the eigenvalues of the
        evolution operator.

        Returns
        -------
        ndarray (n_modes,)
        """
        return self.omegas.imag / 2.0 / np.pi

    @property
    def amplitudes(self):
        """
        Get the amplitudes of the DMD modes.

        Returns
        -------
        ndarray (n_modes,)
        """
        return self._b

    @property
    def dynamics(self):
        """
        Get the dynamics of each mode.

        Returns
        -------
        ndarray (n_modes, n_snapshots)
        """
        t0 = self.original_timesteps[0]
        exp_arg = np.outer(self.omegas, self.dmd_timesteps - t0)
        return np.exp(exp_arg) * self._b[:, None]

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed training data.

        Returns
        -------
        ndarray (n_snapshots, n_features)
        """
        return (self.modes @ self.dynamics).T

    def reconstruction_error(self):
        """
        Compute the training data reconstruction error.

        Returns
        -------
        float
            The relative l2 reconstruction error.
        """
        X = self.snapshots.real
        X_pred = self.reconstructed_data.real
        return norm(X - X_pred, ord=2) / norm(X, ord=2)

    def compute_error_decay(self):
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
        X, timesteps = self.snapshots, self.original_timesteps
        for n in range(self.n_snapshots - 1):
            self.svd_rank = n + 1
            self.fit(X, timesteps)
            X_pred = self.reconstructed_data
            error = norm(X - X_pred, ord=2) / norm(X, ord=2)
            errors += [error]
        self.svd_rank = svd_rank_original
        self.fit(X, timesteps)
        return np.array(errors)

    def _construct_lowrank_op(self, X):
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

    def _eig_from_lowrank_op(self, X):
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
            The eigenvectors of the evolution operator.
        """
        lowrank_eigvals, lowrank_eigvecs = eig(self._A_tilde)

        # Filter out zero eigenvalues
        non_zero_mask = lowrank_eigvals != 0.0
        lowrank_eigvals = lowrank_eigvals[non_zero_mask]
        lowrank_eigvecs = lowrank_eigvecs.T[non_zero_mask].T

        # Compute the full-rank eigenvectors
        if self.exact:
            Vr = self._right_svd_modes[:, :self.n_modes]
            Sr = self._singular_values[:self.n_modes]
            eigvecs = (X @ Vr * np.reciprocal(Sr)) @ lowrank_eigvecs
        else:
            Ur = self._left_svd_modes[:, :self.n_modes]
            eigvecs = Ur @ lowrank_eigvecs

        # Normalize eigenvectors and ensure positive max abs
        for m in range(self.n_modes):
            eigvecs[:, m] /= norm(eigvecs[:, m])

        # Full-rank eigenvalues are the low-rank eigenvalues
        eigvals = lowrank_eigvals

        return eigvals, eigvecs

    def _compute_amplitudes(self):
        """
        Compute the amplitudes for the dynamic modes. This
        method fits the modes to the first snapshot, which
        is assumed to be the initial condition.

        :return: The dynamic mode amplitudes.
        :rtype: numpy.ndarray (n_modes,)
        """
        x = self.snapshots[0]
        b = np.linalg.lstsq(self.modes, x, rcond=None)[0]
        for m in range(self.n_modes):
            if b[m].real < 0.0:
                b[m] *= -1.0
                self._modes[:, m] *= -1.0
        return b

    def _sort_modes(self, ordering='amplitudes'):
        """
        Sort the dynamic modes based upon the specified criteria.
        This method updates the ordering of the private attributes.

        Parameters
        ----------
        ordering : 'eigenvalues', 'amplitudes', or None
        """
        if ordering is None:
            return
        if ordering not in ['amplitudes', 'eigenvalues']:
            raise AssertionError('Invalid ordering type.')

        # Determine sorted index mapping
        if ordering == 'amplitudes':
            idx = np.argsort(self.amplitudes)[::-1]
        elif ordering == 'eigenvalues':
            idx = np.argsort(self.eigs)[::-1]

        # Reset _eigs, _b, and _modes based on this
        self._b = self._b[idx]
        self._eigs = self._eigs[idx]
        self._modes = self._modes[:, idx]

    def plot_singular_values(self, logscale=True):
        """Plot the singular value spectrum.

        Parameters
        ----------
        logscale : bool, default False
            Flag for plotting on a linear or log scale y-axis.
        """
        s = self._singular_values
        data = s / sum(s)

        fig, ax = plt.subplots()
        ax.set_xlabel('Singular Value #', fontsize=12)
        ax.set_ylabel(r'$\sigma / \sum{{\sigma}}$', fontsize=12)
        plotter = plt.semilogy if logscale else plt.plot
        plotter(data, 'b-*', label='Singular Values')
        ax.axvline(self.n_modes - 1, color='r',
                   ymin=1e-12, ymax=1.0 - 1.0e-12)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_modes(self, grid, mode_index=None):
        """
        Plot the DMD modes.

        Parameters
        ----------
        grid : ndarray
            The x-axis of the mode plot.
        mode_index : int or list of int
            The mode index, or indices, to plot.
        """
        # Input checks
        if self._modes is None:
            raise AssertionError('No modes found. The `fit` function '
                                 'must be used first.')
        if grid is None:
            raise AssertionError('A grid must be supplied.')

        # Ensure mode index is iterable
        if mode_index is None:
            mode_index = list(range(self.n_modes))
        elif isinstance(mode_index, int):
            mode_index = [mode_index]

        # Number of components
        n_components = self.n_features // len(grid)
        if not isinstance(n_components, int):
            raise AssertionError('Modes incompatible with the '
                                 'supplied grid.')

        # Plot the modes
        for idx in mode_index:
            # Get the mode
            mode = self._modes[:, idx] * self._b[idx]
            argmax = np.argmax(np.abs(mode))
            if mode[argmax] < 0.0:
                mode *= -1.0

            # Setup and plot
            fig, ax = plt.subplots(ncols=2)
            fig.suptitle(f'DMD Mode {idx}\n'
                      f'$\omega$ = {self.omegas[idx].real:.2e}'
                      f'{self.omegas[idx].imag:+.2e}j')
            ax[0].set_ylabel('Real Part')
            ax[1].set_ylabel('Imaginary Part')

            for c in range(n_components):
                vals = mode[c::n_components]
                ax[0].plot(grid, vals.real, label=f'Component {c}')
                ax[1].plot(grid, vals.imag, label=f'Component {c}')
            ax[0].grid(True)
            ax[1].grid(True)
            ax[0].legend()
            ax[1].legend()
            plt.tight_layout()
        plt.show()

    def plot_reconstruction_errors(self, logscale=True):
        """Plot the reconstruction errors.

        Parameters
        ----------
        logscale : bool
            Flag for plotting on a linear or log scale y-axis.
        """
        errors = self.compute_error_decay()
        s = self._singular_values
        spectrum = s / sum(s)

        fig, ax = plt.subplots()
        ax.set_xlabel('# of Modes', fontsize=12)
        ax.set_ylabel(r'Relative $\ell^2$ Error', fontsize=12)
        plotter = plt.semilogy if logscale else plt.plot
        plotter(spectrum, 'b-*', label='Singular Values')
        plotter(errors, 'r-*', label='Reconstruction Errors')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _validate_data(X, timesteps=None):
        """

        Parameters
        ----------
        X : ndarray (n_snapshots, n_features) or iterable
            A matrix of snapshots stored row-wise.
        original_time : dict
            Dictionary containing t_start, t_end, and dt keys.
        Returns
        -------
        The inputs
        """
        # Check types for X and timesteps
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError('X must be a ndarray or list.')
        if timesteps is not None:
            if not isinstance(timesteps, (np.ndarray, list)):
                raise TypeError('timesteps must be a ndarray or list.')

        # Format X
        X = np.asarray(X)
        if X.ndim != 2:
            raise AssertionError('X must be 2D data.')

        # Format timesteps
        if timesteps is not None:
            timesteps = np.asarray(timesteps).ravel()
        return X, timesteps