import numpy as np

from numpy import ndarray
from numpy.linalg import norm
from scipy.interpolate import griddata, RBFInterpolator
import matplotlib.pyplot as plt
from typing import Union

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF

from .pod_base import PODBase

Rank = Union[float, int]
TestData = Union[float, ndarray]


class POD(PODBase):
    """
    Proper Orthogonal Decomposition.
    """

    def fit(self, X: ndarray, Y: ndarray = None) -> None:
        """
        Compute the POD of the inupt data.

        Parameters
        ----------
        X : ndarray (n_snapshots, n_features)
            The training snapshots stored row-wise.
        Y : ndarray (n_snapshots, n_parameters), default None
            The training parameters stored row-wise.
        """
        X, Xshape = self._col_major_2darray(X)
        self._snapshots = X
        self._snapshots_shape = Xshape

        if Y.shape[0] != X.shape[0]:
            raise ValueError('The number of parameter sets must '
                             'match the number of snapshots.')
        self._parameters = Y

        # Perform the SVD
        U, s, V = self._compute_svd(X, self._svd_rank)
        self._modes = U

        # Compute amplitudes
        self._b = self.transform(X)

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

    def predict(self, Y: ndarray, method: str = 'linear') -> ndarray:
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
                'The number of parameters per query must match '
                'the number of parameters per snapshot.')

        amplitudes = self._interpolate(Y, method)
        return self._modes @ amplitudes

    def _interpolate(self, Y: ndarray, method: str,
                     eps: float = 1.0) -> ndarray:
        """
        Interpolate POD mode amplitudes.

        Parameters
        ----------
        Y : ndarray (n_snapshots, n_parameters)
            The query parameters.
        method : str {'linear', 'cubic', 'nearest', 'rbf', 'rbf_' + varies}
            The prediction method to use.
        eps : float, default 1.0
            The shape parameter to scale the inputs to the RBF. This is only
            applicable when the RBF is not scale invariant.

        Returns
        -------
        ndarray (n_modes, n_snapshots)
            The predictions for the amplitudes corresponding to
            the query parameters.
        """
        # Regular interpolation
        if method in ['linear', 'cubic', 'nearest']:
            args = (self._parameters, self._b.T, Y)
            amplitudes = griddata(*args, method=method.lower())

        # Radial basis function interpolation
        elif 'rbf' in method:
            if method == 'rbf':
                kernel = 'thin_plate_spline'
            elif '_' in method:
                kernel = '_'.join(method.split('_')[1:])
            else:
                raise ValueError('Specific RBFs must be specified as '
                                 'rbf_<function name>.')

            interp = RBFInterpolator(self.parameters, self._b.T,
                                     kernel=kernel, epsilon=eps)
            amplitudes = interp(Y)

        # Gaussian Process interpolation
        else:
            # TODO: This needs some work for consistent accuracy
            kernel = ConstantKernel(1.0) * RBF(1.0)
            gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=100,
                                          alpha=1e-4, normalize_y=True)
            gp.fit(self._parameters, self._b.T)
            amplitudes = gp.predict(Y)
        return amplitudes.reshape(len(Y), self.n_modes).T
