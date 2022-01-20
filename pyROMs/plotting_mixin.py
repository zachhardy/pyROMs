import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from os.path import splitext
from typing import List, Tuple

from pyPDEs.utilities import Vector

from typing import Union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyROMs.pod import PODBase
    from pyROMs.dmd import DMDBase


class PlottingMixin:
    """
    Mixin class for plotting with ROMs.
    """

    def plot_singular_values(self: Union['PODBase', 'DMDBase', 'PlottingMixin'],
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
            svals = svals / max(svals)

        # Define the plotter
        plotter = plt.semilogy if logscale else plt.plot

        # Make figure
        plt.figure()
        plt.xlabel('n', fontsize=12)
        plt.ylabel('Singular Value' if not normalized
                   else 'Relative Singular Value')
        plotter(svals, '-*b')
        if show_rank:
            plt.axhline(svals[self.n_modes - 1], color='r',
                        xmin=0, xmax=len(svals) - 1)
        plt.tight_layout()
        if filename is not None:
            base, ext = splitext(filename)
            plt.savefig(base + '.pdf')

    def plot_modes_1D(self: Union['PODBase', 'DMDBase', 'PlottingMixin'],
                      mode_indices: List[int] = None,
                      components: List[int] = None,
                      x: ndarray = None,
                      filename: str = None) -> None:
        """
        Plot 1D modes.

        Parameters
        ----------
        mode_indices : List[int], default None
            The indices of the modes to plot. The default behavior
            is to plot all modes.
        components : List[int], default None
            The components of the modes to plot. The default behavior
            is to plot all components.
        x : ndarray, default None
            The grid the modes are defined on. The default behaviors
            is a grid from 0 to n_features - 1.
        filename : str, default None
            A location to save the plot to, if specified.
        """
        # Check inputs
        if self.modes is None:
            raise ValueError('The fit method must be performed first.')

        if x is None:
            x = np.arange(0, self.n_features, 1)

        n_components = self.n_features // len(x)
        if not isinstance(n_components, int):
            raise AssertionError(
                'The grid must be an integer factor of n_features.')

        if mode_indices is None:
            mode_indices = list(range(self.n_modes))
        elif isinstance(mode_indices, int):
            mode_indices = [mode_indices]

        if components is None:
            components = list(range(n_components))
        elif isinstance(components, int):
            components = [components]

        # Plot each mode specified
        for idx in mode_indices:
            idx += 0 if idx >= 0 else self.n_modes
            mode: ndarray = self.modes[:, idx]

            # Make figure
            fig: Figure = plt.figure()
            fig.suptitle(f'Mode {idx}', fontsize=12)

            # Plot real part
            ax: Axes = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('r', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            for c in components:
                c += 0 if c >= 0 else n_components
                label = f'Component {c}'
                vals = mode.real[c::n_components]
                ax.plot(x, vals, label=label)

            plt.tight_layout()
            if filename is not None:
                base, ext = splitext(filename)
                plt.savefig(base + f'_{idx}.pdf')

    def plot_snapshots_1D(self: Union['PODBase', 'DMDBase', 'PlottingMixin'],
                          snapshot_indices: List[int] = None,
                          components: List[int] = None,
                          x: ndarray = None,
                          filename: str = None) -> None:
        """
        Plot 1D snapshots.

        Parameters
        ----------
        snapshot_indices : List[int], default None
            The indices of the snapshots to plot. The default behavior
            is to plot all modes.
        components : List[int], default None
            The components of the modes to plot. The default behavior
            is to plot all components.
        x : ndarray, default None
            The grid the modes are defined on. The default behaviors
            is a grid from 0 to n_features - 1.
        filename : str, default None
            A location to save the plot to, if specified.
        """
        if self.snapshots is None:
            raise ValueError('No input snapshots found.')

        if x is None:
            x = np.arange(0, self.n_features, 1)

        n_components = self.n_features // len(x)
        if not isinstance(n_components, int):
            raise AssertionError(
                'The grid must be an integer factor of n_features.')

        if snapshot_indices is None:
            snapshot_indices = list(range(self.n_snapshots))
        elif isinstance(snapshot_indices, int):
            snapshot_indices = [snapshot_indices]

        if components is None:
            components = list(range(n_components))
        elif isinstance(components, int):
            components = [components]

        # Plot each snapshot
        for idx in snapshot_indices:
            idx += 0 if idx >= 0 else self.n_snapshots
            snapshot: ndarray = self.snapshots[idx].real

            # Make figure
            fig: Figure = plt.figure()
            fig.suptitle(f'Snapshot {idx}', fontsize=12)

            # Plot real part
            ax: Axes = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('r', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            for c in components:
                c += 0 if c >= 0 else n_components
                label = f'Component {c}'
                vals = snapshot.real[c::n_components]
                ax.plot(x, vals, label=label)

            plt.tight_layout()
            if filename is not None:
                base, ext = splitext(filename)
                plt.savefig(base + f'_{idx}.pdf')

    def plot_modes_2D(self: Union['PODBase', 'DMDBase', 'PlottingMixin'],
                      mode_indices: List[int] = None,
                      components: List[int] = None,
                      x: ndarray = None,
                      y: ndarray = None,
                      filename: str = None) -> None:
        """
        Plot 2D modes.

        Parameters
        ----------
        mode_indices : List[int], default None
            The indices of the modes to plot. The default behavior
            is to plot all modes.
        components : List[int], default None
            The components of the modes to plot. The default behavior
            is to plot all components.
        x, y : ndarray, default None
            The x,y-nodes the grid is defined on. The default behavior uses
            the stored dimensions in `_snapshots_shape`.
        filename : str, default None
            A location to save the plot to, if specified.
        """
        # Check the inputs
        if self.modes is None:
            raise ValueError(
                'The fit method must be performed first.')

        if x is None and y is None:
            if self._snapshots_shape is None:
                raise ValueError('There is no information about the '
                                 'original shape of the snapshots.')
            elif len(self._snapshots_shape) != 2:
                raise ValueError(
                    'The dimension of the snapshots is not 2D.')
            else:
                x = np.arange(self._snapshots_shape[0])
                y = np.arange(self._snapshots_shape[1])
        x, y = np.unique(x), np.unique(y)
        X, Y = np.meshgrid(x, y)

        n_components = self.n_features // (len(x) * len(y))
        if not isinstance(n_components, int):
            raise AssertionError(
                'The grid must be an integer factor of n_features.')

        if mode_indices is None:
            mode_indices = list(range(self.n_modes))
        elif isinstance(mode_indices, int):
            mode_indices = [mode_indices]

        if components is None:
            components = list(range(n_components))
        elif isinstance(components, int):
            components = [components]

        # Determine the subplot dimensions
        n_rows, n_cols = self.format_subplots(len(components))

        # Plot each mode specified
        for idx in mode_indices:
            idx += 0 if idx >= 0 else self.n_modes
            mode: ndarray = self.modes[:, idx].real

            # Make figure
            fig: Figure = plt.figure()
            fig.suptitle(f'Mode {idx}', fontsize=12)

            # Plot each component specified
            for i, c in enumerate(components):
                c += 0 if c >= 0 else n_components
                vals = mode[c::n_components].reshape(X.shape)

                # Create subplot
                ax: Axes = fig.add_subplot(n_rows, n_cols, i + 1)
                ax.set_title(f'Component {c}', fontsize=12)
                if i % n_cols == 0:
                    ax.set_ylabel('y', fontsize=12)
                if i >= (n_rows - 1) * n_cols:
                    ax.set_xlabel('x', fontsize=12)
                im = ax.pcolor(X, Y, vals, cmap='jet', shading='auto',
                               vmin=vals.min(), vmax=vals.max())
                fig.colorbar(im, ax=ax)
                ax.set_aspect('auto')

            plt.tight_layout(pad=2.0)
            if filename is not None:
                base, ext = splitext(filename)
                plt.savefig(base + f'_{idx}.pdf')

    def plot_snapshots_2D(self: Union['PODBase', 'DMDBase', 'PlottingMixin'],
                          snapshot_indices: List[int] = None,
                          components: List[int] = None,
                          x: ndarray = None,
                          y: ndarray = None,
                          filename: str = None) -> None:
        """
        Plot 2D snapshots.

        Parameters
        ----------
        snapshot_indices : List[int], default None
            The indices of the snapshot to plot. The default behavior
            is to plot all snapshots.
        components : List[int], default None
            The components of the snapshots to plot. The default behavior
            is to plot all components.
        x, y : ndarray, default None
            The x,y-nodes the grid is defined on. The default behavior uses
            the stored dimensions in `_snapshots_shape`.
        filename : str, default None
            A location to save the plot to, if specified.
        """
        # Check the inputs
        if self.snapshots is None:
            raise ValueError('No input snapshots found.')

        if x is None and y is None:
            if self._snapshots_shape is None:
                raise ValueError('There is no information about the '
                                 'original shape of the snapshots.')
            elif len(self._snapshots_shape) != 2:
                raise ValueError(
                    'The dimension of the snapshots is not 2D.')
            else:
                x = np.arange(self._snapshots_shape[0])
                y = np.arange(self._snapshots_shape[1])
        x, y = np.unique(x), np.unique(y)
        X, Y = np.meshgrid(x, y)

        n_components = self.n_features // (len(x) * len(y))
        if not isinstance(n_components, int):
            raise AssertionError(
                'The grid must be an integer factor of n_features.')

        if snapshot_indices is None:
            snapshot_indices = list(range(self.n_snapshots))
        elif isinstance(snapshot_indices, int):
            snapshot_indices = [snapshot_indices]

        if components is None:
            components = list(range(n_components))
        elif isinstance(components, int):
            components = [components]

        # Determine the subplot dimensions
        n_rows, n_cols = self.format_subplots(len(components))

        # Plot each mode specified
        for idx in snapshot_indices:
            idx += 0 if idx >= 0 else self.n_snapshots
            snapshot: ndarray = self.snapshots[idx]

            # Make figure
            fig: Figure = plt.figure()
            fig.suptitle(f'Snapshot {idx}', fontsize=12)

            # Plot each component specified
            for i, c in enumerate(components):
                c += 0 if c >= 0 else n_components
                vals = snapshot[c::n_components].reshape(X.shape)

                # Create subplot
                ax: Axes = fig.add_subplot(n_rows, n_cols, i + 1)
                ax.set_title(f'Component {c}', fontsize=12)
                if i % n_cols == 0:
                    ax.set_ylabel('y', fontsize=12)
                if i >= (n_rows - 1) * n_cols:
                    ax.set_xlabel('x', fontsize=12)
                im = ax.pcolor(X, Y, vals, cmap='jet', shading='auto',
                               vmin=0.0, vmax=vals.max())
                fig.colorbar(im, ax=ax)
                ax.set_aspect('auto')

            plt.tight_layout(pad=2.0)
            if filename is not None:
                base, ext = splitext(filename)
                plt.savefig(base + f'_{idx}.pdf')

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
