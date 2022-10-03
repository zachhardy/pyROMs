import numpy as np

from scipy.interpolate import griddata, RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF

from typing import Union

from .pod import POD


class POD_MCI(POD):
    """
    Implementation of the POD Mode Coefficient Interpolation method.
    """

    def __init__(
            self,
            svd_rank: Union[int, float] = -1,
            interpolant: str = "rbf",
            **kwargs
    ) -> None:
        """
        Parameters
        ----------
        svd_rank : int or float, default -1
            The SVD rank to use for truncation. If a positive integer,
            the minimum of this number and the maximum possible rank
            is used. If a float between 0 and 1, the minimum rank to
            achieve the specified energy content is used. If -1, no
            truncation is performed.
        interpolant : {'linear', 'cubic', 'nearest', 'rbf', 'rbf_<kernel>'}
            The interpolation method to use. Default uses a radial basis
            function interpolant with a thin plate spline kernel function.
            If an alternative kernel function is desired for a radial basis
            function interpolant, it should be specified via 'rbf_<kernel>'.
        kwargs : varies
            epsilon : float
                The shape parameter for some types of kernel functions
            neighbors : int
                The number of neighbors to use to query the radial basis
                function interpolant with. Default uses all data points.
            degree : int
                The degree of an added polynomial term. See scipy
                documentation for more.
        """

        # Check interpolant method
        if interpolant not in ["linear", "cubic", "nearest"]:
            if "rbf" not in interpolant:
                msg = f"{interpolant} is not a valid interpolation method."
                raise ValueError(msg)

        super().__init__(svd_rank)

        self._parameters: np.ndarray = None

        self._interp_method: str = interpolant
        self._interp_args: dict = kwargs
        self._interpolant: callable = None

    @property
    def paramaters(self) -> np.ndarray:
        """
        Return the parameters.

        Returns
        -------
        numpy.ndarray (n_snapshots, n_parameters)
        """
        return self._parameters

    @property
    def n_parameters(self) -> int:
        """
        Return the number of parameters that describe a snapshot.

        Returns
        -------
        int
        """
        return self._parameters.shape[1]

    @property
    def interpolation_method(self) -> str:
        """
        Return the interpolation method.

        Returns
        -------
        str
        """
        return self._interp_method

    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'POD_MCI':
        """
        Fit the POD-MCI model to the input data.

        Parameters
        ----------
        X : numpy.ndarray
            The training snapshots.
        Y : numpy.ndarray (n_snapshots, n_parameters), default None
            The training parameters.

        Returns
        -------
        POD_MCI
        """

        ##################################################
        # Format and check inputs
        ##################################################

        X, Xshape = self._format_2darray(X)
        if Y.shape[0] != X.shape[1]:
            msg = "The number of parameter sets does not match " \
                  "the number of training snapshots."
            raise ValueError(msg)

        self._parameters = np.atleast_2d(Y)
        if self._parameters.shape[0] == 1:
            self._parameters = self._parameters.T

        ##################################################
        # Fit the model
        ##################################################

        super().fit(X)
        self._init_interpolant()

    def refit(
            self,
            svd_rank: Union[int, float],
            interpolant: str = "rbf",
            **kwargs
    ) -> 'POD_MCI':
        """
        Re-fit the POD-MCI model to the specified SVD rank with
        the specified interpolant.

        Parameters
        ----------
        svd_rank : int or float, default -1
            The SVD rank to use for truncation. If a positive integer,
            the minimum of this number and the maximum possible rank
            is used. If a float between 0 and 1, the minimum rank to
            achieve the specified energy content is used. If -1, no
            truncation is performed.
        interpolant : {'linear', 'cubic', 'nearest', 'rbf', 'rbf_<kernel>'}
            The interpolation method to use. Default uses a radial basis
            function interpolant with a thin plate spline kernel function.
            If an alternative kernel function is desired for a radial basis
            function interpolant, it should be specified via 'rbf_<kernel>'.
        kwargs : varies
            epsilon : float
                The shape parameter for some types of kernel functions
            neighbors : int
                The number of neighbors to use to query the radial basis
                function interpolant with. Default uses all data points.
            degree : int
                The degree of an added polynomial term. See scipy
                documentation for more.

        Returns
        -------
        POD_MCI
        """
        super().refit(svd_rank)
        self._interp_method = interpolant
        self._interp_args = kwargs
        self._init_interpolant()

    def predict(self, Y: np.ndarray) -> np.ndarray:
        """
        Predict a full-order result for a set of parameters.

        Parameters
        ----------
        Y : numpy.ndarray (varies, n_parameters)
            The query parameters.

        Returns
        -------
        numpy.ndarray (n_features, varies)
        """
        if Y.shape[1] != self.n_parameters:
            msg = "The number of parameters per query must match " \
                  "the number of parameters per snapshot."
            raise ValueError(msg)

        if self._interp_method in ["linear", "cubic", "nearest"]:
            Y = Y.ravel() if self.n_parameters == 1 else Y
        b = self._interpolant(Y)
        return self.modes @ b.T

    def _init_interpolant(self) -> None:
        """
        Private method to initialize the interpolant.
        """
        method = self._interp_method
        kwargs = self._interp_args
        pts, vals = self._parameters, self._b

        # Standard interpolants
        if method in ["linear", "cubic", "nearest"]:
            if self.n_parameters == 1:
                from scipy.interpolate import interp1d
                interp = interp1d(pts.ravel(), vals, method, axis=0)
            else:
                if method == "linear":
                    from scipy.interpolate import LinearNDInterpolator
                    interp = LinearNDInterpolator(pts, vals, rescale=True)

                elif method == "nearest":
                    from scipy.interpolate import NearestNDInterpolator
                    interp = NearestNDInterpolator(pts, vals, rescale=True)

                else:
                    if self.n_parameters > 2:
                        msg = "Only 1D and 2D cases are implemented " \
                              "for cubic interpolants."
                        raise NotImplementedError(msg)

                    from scipy.interpolate import CloughTocher2DInterpolator
                    interp = CloughTocher2DInterpolator(pts, vals, rescale=True)

        # Radial basis function interpolants
        else:

            # Handle default case
            if "_" not in method:
                method = f"{method}_thin_plate_spline"

            # Parse kernel function
            kernel = "_".join(method.split("_")[1:])

            from scipy.interpolate import RBFInterpolator
            interp = RBFInterpolator(pts, vals, kernel=kernel, **kwargs)

        self._interp_method = method
        self._interpolant = interp
