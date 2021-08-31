import numpy as np

from numpy import ndarray
from numpy.linalg import norm
from scipy.interpolate.ndgriddata import griddata
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
            verbose: bool = False) -> None:
        """Compute the POD of the inupt data.

        Parameters
        ----------
        X : ndarray (n_snapshots, n_features)
            The training snapshots stored row-wise.
        Y : ndarray (n_snapshots, n_parameters), default None
            The training parameters stored row-wise.
        verbose : bool, default False
        """
        X, Y = self._validate_data(X, Y)

        # Save the input data
        self._snapshots = np.copy(X)
        self._parameters = np.copy(Y)

        # Perform the SVD
        U, s, V = self._compute_svd(X.T, self.svd_rank)
        self._modes = U

        # Compute amplitudes
        self._b = self.transform(X)

        # Print summary
        if verbose:
            print("\n*** POD model information ***")

            n = self.n_modes
            print(f"Number of Modes:\t\t{n}")

            s = self._singular_values
            print(f"Smallest Kept Singular Value:\t{s[n - 1] / sum(s):.3e}")

            error = self.reconstruction_error.real
            print(f"Reconstruction Error:\t\t{error:.3e}")

    def transform(self, X: ndarray) -> ndarray:
        """Transform the data X to the low-rank space.

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
            raise ValueError("Model must first be fit.")

        if X.shape[1] != self.n_features:
            raise AssertionError(
                "The number of features in X and the training "
                "data must agree.")
        return X @ self.modes

    def predict(self, Y: ndarray, method: str = "CUBIC") -> ndarray:
        """Predict a full-order result for a set of parameters.

        Parameters
        ----------
        Y : ndarray (n_snapshots, n_parameters)
            The query parameters.
        method : str {"LINEAR", "CUBIC", "GP"}, default "CUBIC"
            The interpolation method to use. "GP" stands for
            Gaussian Processes.

        Returns
        -------
        ndarray (n_snapshots, n_features)
            The predictions for the snapshots corresponding to
            the query parameters.
        """
        if Y.shape[1] != self.n_parameters:
            raise ValueError("Y must have the same number of "
                             "parameters as the training data.")
        amplitudes = self.interpolate(Y, method)
        return amplitudes @ self.modes.T

    def interpolate(self, Y: TestData, method: str) -> ndarray:
        """Interpolate POD mode amplitudes.

        Parameters
        ----------
        Y : ndarray (n_snapshots, n_parameters)
            The query parameters.
        method : str {"LINEAR", "CUBIC", "GP"}, default "CUBIC"
            The interpolation method to use. "GP" stands for
            Gaussian Processes.

        Returns
        -------
        ndarray (n_snapshots, n_modes)
            The predictions for the amplitudes corresponding to
            the query parameters.
        """
        # Regular interpolation
        if method in ["LINEAR", "CUBIC"]:
            args = (self.parameters, self.amplitudes, Y)
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
