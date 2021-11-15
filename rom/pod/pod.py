import numpy as np

from numpy import ndarray
from numpy.linalg import norm
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from typing import Union

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF

from .pod_base import PODBase

Rank = Union[float, int]
TestData = Union[float, ndarray]


class POD(PODBase):
    """Principal Orthogonal Decomposition class.
    """

    def fit(self, X: ndarray, Y: ndarray = None,
            verbose: bool = True) -> None:
        """Compute the POD of the inupt data.

        Parameters
        ----------
        X : ndarray (n_features, n_snapshots)
            The training snapshots stored row-wise.
        Y : ndarray (n_parameters, n_snapshots), default None
            The training parameters stored row-wise.
        verbose : bool, default False
        """
        X, Xshape, Y = self._validate_data(X, Y)
        self._snapshots = X
        self._snapshots_shape = Xshape
        self._parameters = Y

        # Perform the SVD
        U, s, V = self._compute_svd()
        self._modes = U

        # Compute amplitudes
        self._b = self.transform(X)

        # Print summary
        if verbose:
            msg = '=' * 10 + ' POD Summary ' + '=' * 10
            header = '=' * len(msg)
            print('\n'.join(['', header, msg, header]))
            print(f'Number of Modes:\t{self.n_modes}')
            print(f'Reconstruction Error:\t{self.reconstruction_error:.3e}')
            print()

    def transform(self, X: ndarray) -> ndarray:
        """
        Transform the data X to the low-rank space.

        Parameters
        ----------
        X : ndarray (n_features, n_snapshots)
            The snapshot data to transform.

        Returns
        -------
        ndarray (n_modes, n_snapshots)
            The low-rank representation of X.
        """
        if self.modes is None:
            raise AssertionError('The POD model must be fit.')

        if X.shape[0] != self.n_features:
            raise AssertionError(
                'The number of features must match the number '
                'of features in the training data.')
        return self.modes.T @ X

    def predict(self, Y: ndarray, method: str = 'cubic') -> ndarray:
        """
        Predict a full-order result for a set of parameters.

        Parameters
        ----------
        Y : ndarray (n_snapshots, n_parameters)
            The query parameters.
        method : str {'linear', 'cubic', 'gp'}, default 'cubic'
            The prediction method to use.

        Returns
        -------
        ndarray (n_features, n_snapshots)
        """
        if Y.shape[1] != self.n_parameters:
            raise ValueError(
                'Y must have the same number of parameters as '
                'the training data.')

        amplitudes = self._interpolate(Y, method)
        return self.modes @ amplitudes

    def _interpolate(self, Y: ndarray, method: str = 'cubic') -> ndarray:
        """
        Interpolate POD mode amplitudes.

        Parameters
        ----------
        Y : ndarray (n_snapshots, n_parameters)
            The query parameters.
        method : str {'linear', 'cubic', 'nearest'}, default 'cubic'
            The prediction method to use.

        Returns
        -------
        ndarray (n_modes, n_snapshots)
            The predictions for the amplitudes corresponding to
            the query parameters.
        """
        # Regular interpolation
        if method in ['linear', 'cubic', 'nearest']:
            args = (self.parameters, self.amplitudes.T, Y)
            amplitudes = griddata(*args, method=method.lower())

        # Gaussian Process interpolation
        else:
            # TODO: This needs some work for consistent accuracy
            kernel = C(1.0) * RBF(1.0)
            gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=100,
                                          alpha=1e-4, normalize_y=True)
            gp.fit(self.parameters, self.amplitudes)
            amplitudes = gp.predict(Y)
        return amplitudes.reshape(len(Y), self.n_modes)
