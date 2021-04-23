import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from scipy.interpolate.ndgriddata import griddata
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF

from .pod_base import PODBase
from ..svd import compute_svd

from typing import Union

TestData = Union[float, ndarray]

class POD(PODBase):
    """Principal Orthogonal Decomposition class.

    Parameters
    ----------
    svd_rank : int or float, default -1
        The SVD truncation rank. If -1, no truncation is used.
        If a positive integer, the truncation rank is the argument.
        If a float between 0 and 1, the minimum number of modes
        needed to obtain an information content greater than the
        argument is used.
    """

    def fit(self, X: ndarray, Y: ndarray = None) -> 'POD':
        """
        Compute the principal orthogonal decomposition
        of the inupt data.

        Parameters
        ----------
        X : ndarray
            The training snapshots stored row-wise.
            The shape should be (n_snapshots, n_features)
        Y : ndarray, default None
            The training parameters stored row-wise.
            The shape should be (n_snapshots, n_parameters)
            if not None.

        """
        X, Y = self._validate_data(X, Y)

        # Save the input data
        self._snapshots = np.copy(X)
        self._parameters = np.copy(Y)
        self.n_snapshots = self._snapshots.shape[0]
        self.n_features = self._snapshots.shape[1]
        self.n_parameters = self._parameters.shape[1]

        # Perform the SVD
        U, s, _, rank = compute_svd(X.T, self.svd_rank)
        self._modes = U  # shape = (n_features, n_snapshots)
        self._singular_values = s
        self.n_modes = rank

        # Compute coefficients
        self._b = self.transform(X)
        return self

    def transform(self, X: ndarray) -> ndarray:
        """
        Transform the data X to the low-rank space.

        Parameters
        ----------
        X : ndarray (n_snapshots, n_features)
            The snapshot data to transform.

        Returns
        -------
        ndarray (n_snapshots, n_modes)
            The low-rank representation of X.
        """
        if self.modes is None:
            raise ValueError('Model must first be fit.')

        if X.shape[1] != self.n_features:
            raise AssertionError('The number of features in X and '
                                 'the training data must agree.')
        rank = self.n_modes
        return X @ self.modes

    def predict(self, Y: ndarray, method: str = 'cubic') -> ndarray:
        """
        Predict a full-order result given a set of parameters
        using the provided interpolation method.

        Parameters
        ----------
        Y : ndarray (n_snapshots, n_parameters)
            The query parameters.
        method : str {'linear', 'cubic', 'gp'}, default 'cubic'
            The interpolation method to use. 'gp' stands for
            Gaussian Processes.

        Returns
        -------
        ndarray (n_snapshots, n_features)
            The predictions for the snapshots corresponding to
            the query parameters.
        """
        if Y.shape[1] != self.n_parameters:
            raise ValueError('Y must have the same number of '
                             'parameters as the training data.')

        amplitudes = self.interpolate(Y, method)
        return amplitudes @ self.modes.T

    def interpolate(self, Y: TestData, method: str) -> ndarray:
        """
        Interpolate POD mode amplitudes using the given method
        for the given query parameters.

        Parameters
        ----------
        Y : ndarray (n_snapshots, n_parameters)
            The query parameters.
        method : str {'linear', 'cubic', 'gp'}, default 'cubic'
            The interpolation method to use. 'gp' stands for
            Gaussian Processes.

        Returns
        -------
        ndarray (n_snapshots, n_modes)
            The predictions for the amplitudes corresponding to
            the query parameters.
        """
        # Regular interpolation
        if method in ['linear', 'cubic']:
            args = (self.parameters, self.amplitudes, Y)
            amplitudes = griddata(*args, method=method)

        # Gaussian Process interpolation
        elif method == 'gp':
            # TODO: This needs some work for consistent accuracy
            kernel = C(1.0) * RBF(1.0)
            gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=100,
                                          alpha=1e-4, normalize_y=True)
            gp.fit(self.parameters, self.amplitudes)
            amplitudes = gp.predict(Y)
        return amplitudes.reshape(len(Y), self.n_modes)
