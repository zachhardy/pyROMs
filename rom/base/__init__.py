import warnings

import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from os.path import splitext
from typing import List, Tuple

from ..utils import format_subplots
from pyPDEs.utilities import Vector


class ROMBase:

    from ._plotting1d import plot_modes_1D, plot_snapshots_1D
    from ._plotting2d import plot_modes_2D, plot_snapshots_2D

    def __init__(self) -> None:
        self._snapshots: ndarray = None
        self._snapshots_shape: Tuple[int, int] = None

    @property
    def n_snapshots(self) -> int:
        return self.snapshots.shape[1]

    @property
    def n_features(self) -> int:
        return self.snapshots.shape[0]

    @property
    def n_modes(self) -> int:
        return self.modes.shape[1]

    @property
    def snapshots(self) -> ndarray:
        return self._snapshots

    @property
    def modes(self) -> ndarray:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f'{cls_name}.modes must be implemented.')

    @property
    def singular_values(self) -> ndarray:
        cls_name = self.__class__.__name__
        raise NotImplementedError(
            f'{cls_name}.singular_values must be implemented.')

    @staticmethod
    def _col_major_2darray(X) -> Tuple[ndarray, Tuple[int, int]]:
        """
        Private method that takes as input the snapshots and stores them into a
        2D matrix, by column. If the input data is already formatted as 2D
        array, the method saves it, otherwise it also saves the original
        snapshots shape and reshapes the snapshots.

        Parameters
        ----------
        X : ndarray or List[ndarray]
            The input snapshots.

        Returns
        -------
        ndarray : 2D matrix containing flattened snapshots
        Tuple[int, int] : The shape of the original snapshots.
        """
        # If the data is already 2D ndarray
        if isinstance(X, np.ndarray) and X.ndim == 2:
            snapshots = X
            snapshots_shape = None
        else:
            input_shapes = [np.asarray(x).shape for x in X]

            if len(set(input_shapes)) != 1:
                raise ValueError('Snapshots have not the same dimension.')

            snapshots_shape = input_shapes[0]
            snapshots = np.transpose([np.asarray(x).flatten() for x in X])

        # check condition number of the data passed in
        cond_number = np.linalg.cond(snapshots)
        if cond_number > 10e4:
            warnings.warn(
                "Input data matrix X has condition number {}. "
                "Consider preprocessing data, passing in augmented data matrix, or regularization methods."
                    .format(cond_number))

        return snapshots, snapshots_shape

    def plot_singular_values(self,
                             normalized: bool = True,
                             logscale: bool = True,
                             show_rank: bool = False,
                             filename: str = None) -> None:
        """
        Plot the singular value spectrum.

        Parameters
        ----------
        normalized : bool, default True
            Flag for normalizing the spectrum to its max value.
        logscale : bool, default True
            Flag for a log scale on the y-axis.
        show_rank : bool, default False
            Flag for showing the truncation location.
        filename : str, default None.
            A location to save the plot to, if specified.
        """
        # Format the singular values
        svals = self.singular_values
        if normalized:
            svals /= sum(svals)

        # Define the plotter
        plotter = plt.semilogy if logscale else plt.plot

        # Make figure
        plt.figure()
        plt.xlabel('n', fontsize=12)
        plt.ylabel('Singular Value' if not normalized
                   else 'Relative Singular Value')
        plotter(svals, '-*b')
        if show_rank:
            plt.axvline(self.n_modes - 1, color='r',
                        ymin=svals.min(), ymax=svals.max())
        plt.tight_layout()
        if filename is not None:
            base, ext = splitext(filename)
            plt.savefig(base + '.pdf')

    @staticmethod
    def format_subplots(n_plots: int) -> Tuple[int, int]:
        """
        Determine the number of rows and columns for subplots.

        Parameters
        ----------
        n_plots : int

        Returns
        -------
        int, int : n_rows, n_cols
        """
        if n_plots < 4:
            n_rows, n_cols = 1, n_plots
        elif 4 <= n_plots < 9:
            tmp = int(np.ceil(np.sqrt(n_plots)))
            n_rows = n_cols = tmp
            for n in range(1, n_cols + 1):
                if n * n_cols >= n_plots:
                    n_rows = n
                    break
        else:
            raise AssertionError('Maximum number of subplots is 9. '
                                 'Consider modifying the visualization.')
        return n_rows, n_cols

