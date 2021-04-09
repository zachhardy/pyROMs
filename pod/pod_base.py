import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


class PODBase:
    """Principal Orthogonal Decomposition base class.

    Parameters
    ----------
    svd_rank : int or float, default -1
        The SVD truncation rank. If -1, no truncation is used.
        If a positive integer, the truncation rank is the argument.
        If a float between 0 and 1, the minimum number of modes
        needed to obtain an information content greater than the
        argument is used.
    """

    def __init__(self, svd_rank=-1):
        self.svd_rank = svd_rank
        self.n_snapshots = 0
        self.n_features = 0
        self.n_parameters = 0
        self.n_modes = 0

        self._snapshots = None
        self._parameters = None
        self._modes = None
        self._singular_values = None
        self._b = None

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
    def parameters(self):
        """
        Get the original training parameters.

        Returns
        -------
        ndarray (n_snapshots, n_parameters)
        """
        return self._parameters

    @property
    def modes(self):
        """
        Get the POD modes, stored column-wise.

        Returns
        -------
        ndarray (n_featurs, n_modes)
        """
        return self._modes[:, :self.n_modes]

    @property
    def amplitudes(self):
        """
        Get the POD mode amplitudes that define the training data.

        Returns
        -------
        ndarray (n_snapshots, n_modes)
        """
        return self._b[:, :self.n_modes]

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed training data using the model.

        Returns
        -------
        ndarray (n_snapshots, n_features)
            The reconstructed training data.
        """
        return self.amplitudes @ self.modes.T

    def fit(self, X, Y=None):
        """
        Abstract method to fit the model to training data.

        This must be inplemented in subclasses.
        """
        raise NotImplementedError(
            f'Subclasses must implement abstact method '
            f'{self.__class__.__name__}.fit')

    def reconstruction_error(self):
        """
        Get the error in the reconstructed training data using
        the truncated model.

        Returns
        -------
        float
            The relative l2 error or the reconstructed data.
        """
        X = self.snapshots
        X_pred = self.reconstructed_data
        return norm(X - X_pred, ord=2) / norm(X, ord=2)

    def untruncated_reconstruction_error(self):
        """
        Get the error in the reconstructed training data using
        an untruncated model.

        Returns
        -------
        float
            The relative l2 error or the reconstructed data
            from an untruncated model.
        """
        X = self.snapshots
        X_pred = X @ self._modes @ self._modes.T
        return norm(X - X_pred, ord=2) / norm(X, ord=2)

    def compute_error_decay(self):
        """Compute the decay in the error.

        This method computes the error decay as a function
        of truncation level.

        Returns
        -------
        ndarray (n_modes,)
            The reproduction error as a function of n_modes.
        """
        errors = []
        X = self.snapshots
        for n in range(self.n_snapshots):
            X_pred = X @ self._modes[:, :n] @ self._modes.T[:n]
            err = norm(X - X_pred, ord=2) / norm(X, ord=2)
            errors.append(err)
        return np.array(errors)

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

    def plot_coefficients(self, modes=None, normalize=False):
        """Plot the POD coefficients as a function of parameter.

        Parameters
        ----------
        modes : int or list of int, default 0
            The mode indices to plot the coefficients for.
            If an int, only that mode is plotted. If a list,
            all modes with the supplied indices are plotted on the
            same Axes.
        normalize : bool, default False
            Flag to remove the mean and normalize by the standard
            deviation of each mode coefficient function.
        ax : Axes, default None
            The Axes to plot on.

        Returns
        -------
        Axes
            The Axes that was plotted on.
        """
        y = self.parameters

        # One-dimensional parameter spaces
        if y.shape[1] == 1:
            # Sort by parameter values
            ind = np.argsort(y, axis=0).ravel()
            y, amplitudes = y[ind], self.amplitudes[ind]

            # Get modes to plot
            if isinstance(modes, int):
                modes = [modes]
            elif isinstance(modes, list):
                modes = [m for m in modes if m < self.n_modes]
            else:
                modes = [m for m in range(self.n_modes)]

            # Format amplitudes
            if normalize:
                amplitudes = self.center_data(amplitudes)

            # Plot plot modes
            for m in modes:
                fig, ax = plt.subplots()
                ax.set_xlabel('Parameter Value', fontsize=12)
                ax.set_ylabel('POD Coefficient Value', fontsize=12)
                ax.plot(y, amplitudes[:, m], '-*', label=f'Mode {m}')
                plt.grid(True)
                plt.legend()
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
    def center_data(data):
        """
        Center the data by removing the mean and scaling
        by the standard deviation row-wise.

        Parameters
        ----------
        data : ndarray (n_snapshots, n_modes or n_features)
            The data to center.

        Returns
        -------
        ndarray (n_snapshots, n_modes or n_features)
            The centered data.
        """
        if data.ndim == 1:
            return (data - np.mean(data)) / np.std(data)
        else:
            mean = np.mean(data, axis=1).reshape(-1, 1)
            std = np.std(data, axis=1).reshape(-1, 1)
            return (data - mean) / std

    @staticmethod
    def _validate_data(X, Y=None):
        """
        Validate training data.

        Parameters
        ----------
        X : ndarray (n_snapshots, n_features)
            2D matrix containing training snapshots
            stored row-wise.
        Y : ndarray (n_snapshots, n_parameters) or None
            Matrix containing training parameters stored
            row-wise.

        Returns
        -------
       The inputs
        """
        # Check types for X and Y
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError('X must be a numpy.ndarray or list.')
        if Y is not None:
            if not isinstance(Y, (np.ndarray, list)):
                raise TypeError('Y must be a numpy.ndarray, list, or None.')

        # Format X
        X = np.asarray(X)
        if X.ndim != 2:
            raise AssertionError('X must be 2D data.')

        # Format Y
        if Y is not None:
            Y = np.asarray(Y).reshape(len(Y), -1)
            if len(Y) != len(X):
                raise AssertionError('There must be a parameter set for '
                                     'each provided snapshot.')
        return X, Y
