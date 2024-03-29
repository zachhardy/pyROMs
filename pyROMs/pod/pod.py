import numpy as np

from numpy import ndarray
from numpy.linalg import svd
from scipy.interpolate import griddata, RBFInterpolator
import matplotlib.pyplot as plt
from typing import Union

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF

from .pod_base import PODBase
from ..utils import compute_rank, _row_major_2darray

Rank = Union[float, int]
TestData = Union[float, ndarray]


class POD(PODBase):
    """
    Proper Orthogonal Decomposition.
    """

    def __init__(self, svd_rank: Union[int, float] = 0,
                 method: str ='rbf',
                 epsilon: float = 1.0) -> None:
        super().__init__(svd_rank)
        self._method: str = method
        self._epsilon = epsilon

        self._interpolant: callable = None

    def fit(self, X: ndarray, Y: ndarray, **kwargs) -> "POD":
        """
        Compute the POD of the inupt data.

        Parameters
        ----------
        X : ndarray (n_snapshots, n_features)
            The training snapshots stored row-wise.
        Y : ndarray (n_snapshots, n_parameters), default None
            The training parameters stored row-wise.
        interpolation_method : str, default 'linear'
            The prediction method to use. Options are 'linear',
            'cubic', 'nearest', 'rbf', 'rbf_<function name>', and
            'gp'.
        """
        if "svd_rank" in kwargs:
            self.svd_rank = kwargs["svd_rank"]
        if "method" in kwargs:
            self._method = kwargs["method"]
        if "epsilon" in kwargs:
            self._epsilon = kwargs["epsilon"]

        # Format training information, define SVD flag
        X, Xshape = _row_major_2darray(X)
        do_svd = not np.array_equal(X, self._snapshots)
        self._snapshots = X
        self._snapshots_shape = Xshape

        Y = np.array(Y)
        if Y.ndim == 1:
            Y = np.atleast_2d(Y).T
        if Y.shape[0] != X.shape[0]:
            raise ValueError("The number of parameter sets must "
                             "match the number of snapshots.")
        self._parameters = Y

        # Perform the SVD
        if do_svd:
            U, s, Vstar = np.linalg.svd(X.T, full_matrices=False)
            self._U, self._s, self._Vstar = U, s, Vstar

        # Compute rank, trucate
        args = (X, self._U, self._s)
        rank = compute_rank(self.svd_rank, *args)
        self._modes = self._U[:, :rank]

        # Compute amplitudes
        s = np.diag(self._s[:self.n_modes])
        Vstar = self._Vstar[:self.n_modes]
        self._b = np.dot(s, Vstar)

        # Create interpolant
        try:
            kwargs = {'epsilon': self._epsilon}
            self._init_interpolant(self._method, **kwargs)
        except:
            pass

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
        ndarray (n_modes, n_snapshots)
            The low-rank representation of X.
        """
        if self.modes is None:
            raise AssertionError("The POD model must be fit.")

        if X.shape[1] != self.n_features:
            raise AssertionError(
                "The number of features must match the number "
                "of features in the training data.")
        return self._modes.T @ X.T

    def predict(self, Y: ndarray) -> ndarray:
        """
        Predict a full-order result for a set of parameters.

        Parameters
        ----------
        Y : ndarray (varies, n_parameters)
            The query parameters.
        method : str {'linear', 'cubic', 'nearest', 'rbf', 'rbf_' + varies}
            The prediction method to use.

        Returns
        -------
        ndarray (varies, n_features)
        """
        # Format test values
        Y = np.array(Y) if isinstance(Y, list) else Y
        if Y.ndim == 1:
            Y = np.atleast_2d(Y)
            Y = Y.T if self.n_parameters == 1 else Y

        # Check test values
        if Y.shape[1] != self.n_parameters:
            raise ValueError(
                "The number of parameters per query must match "
                "the number of parameters per snapshot.")

        # Make prediction
        amplitudes = self._interpolate(Y)
        return np.transpose(self._modes @ amplitudes)

    def _interpolate(self, Y: ndarray) -> ndarray:
        """
        Interpolate POD mode amplitudes.

        Parameters
        ----------
        Y : ndarray (varies, n_parameters)
            The query parameters.

        Returns
        -------
        ndarray (n_modes, varies)
            The predictions for the amplitudes corresponding to
            the query parameters.
        """
        # Gaussian Process query
        if self._method != 'gp':
            if self.n_parameters == 1:
                methods = ['linear', 'cubic', 'nearest']
                if self._method in methods:
                    Y = Y.ravel() if self.n_parameters == 1 else Y
            amplitudes = self._interpolant(Y)

        # Regular interpolation query
        else:
            amplitudes = self._interpolant.predict(Y)
        return amplitudes.reshape(len(Y), self.n_modes).T

    def _init_interpolant(self, method: str, **kwargs) -> None:
        """
        Private method to initialize the interpolant.

        Parameters
        ----------
        method : str
            The interpolation method to use. Options are 'linear',
            'cubic', 'nearest', 'rbf', 'rbf_<function name>', and
            'gp'.
        kwargs
            Key word arguments for interpolators
        """
        pts, vals = self._parameters, self._b.T

        # Standard interpolants
        if method in ['linear', 'cubic', 'nearest']:
            if self.n_parameters == 1:
                from scipy.interpolate import interp1d
                interp = interp1d(pts.ravel(), vals, method, axis=0)
            else:
                if method == 'linear':
                    from scipy.interpolate import LinearNDInterpolator
                    interp = LinearNDInterpolator(pts, vals, rescale=True)
                elif method == 'nearest':
                    from scipy.interpolate import NearestNDInterpolator
                    interp = NearestNDInterpolator(pts, vals, rescale=True)
                elif method == 'cubic':
                    if self.n_parameters > 2:
                        raise AssertionError(
                            "Cubic interpolation is only available in one- "
                            "and two-dimensions.")
                    from scipy.interpolate import CloughTocher2DInterpolator
                    interp = CloughTocher2DInterpolator(pts, vals, rescale=True)

        # Radial basis function interpolators
        if 'rbf' in method:
            # default kernel to thin plate spline
            if method == 'rbf':
                method += "_thin_plate_spline"

            # split to find kernel function
            if '_' in method:
                kernel = '_'.join(method.split('_')[1:])
            else:
                raise ValueError("RBF interpolators must be specified as "
                                 "rbf_<function_name>.")

            from scipy.interpolate import RBFInterpolator
            interp = RBFInterpolator(pts, vals, kernel=kernel, **kwargs)

        # Create Gaussian Process
        elif method == 'gp':
            from sklearn.gaussian_process.kernels import ConstantKernel
            from sklearn.gaussian_process.kernels import RBF
            from sklearn.gaussian_process import GaussianProcessRegressor

            kernel = ConstantKernel()*RBF()
            interp = GaussianProcessRegressor(kernel, **kwargs)
            interp.fit(pts, vals)

        # Set the interpolator
        self._method = method
        self._interpolant = interp
