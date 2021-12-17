import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from os.path import splitext

from pyPDEs.utilities import Vector

from ...utils import format_subplots

from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from . import PODBase


def plot_modes_2D(self: 'PODBase',
                  mode_indices: List[int] = None,
                  components: List[int] = None,
                  grid: List[Vector] = None,
                  filename: str = None) -> None:
    """
    Plot 2D POD modes.

    Parameters
    ----------
    mode_indices : List[int], default None
        The indices of the modes to plot. The default behavior
        is to plot all modes.
    components : List[int], default None
        The components of the modes to plot. The default behavior
        is to plot all components.
    grid : List[Vector], default None
        The grid the modes are defined on. The default behaviors
        is a grid from 0 to n_features - 1.
    filename : str, default None
        A location to save the plot to, if specified.
    """
    # Check the inputs
    if self.modes is None:
        raise ValueError('The fit method must be performed first.')

    if grid is None:
        if self._snapshots_shape is None:
            raise ValueError('There is no information about the '
                             'original shape of the snapshots.')
        elif len(self._snapshots_shape) != 2:
            raise ValueError('The dimension of the snapshots is not 2D.')
        else:
            x = np.arange(self._snapshots_shape[0])
            y = np.arange(self._snapshots_shape[1])
    else:
        x = np.unique([node.x for node in grid])
        y = np.unique([node.y for node in grid])
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
    n_rows, n_cols = format_subplots(len(components))

    # Plot each mode specified
    for idx in mode_indices:
        idx += 0 if idx > 0 else self.n_modes
        mode: ndarray = self.modes[:, idx]

        # Make figure
        fig: Figure = plt.figure()
        fig.suptitle(f'POD Mode {idx}', fontsize=12)

        # Plot each component specified
        for i, c in enumerate(components):
            c += 0 if c > 0 else n_components
            vals = mode[c::n_components].reshape(X.shape)

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


def plot_snapshots_2D(self: 'PODBase',
                      snapshot_indices: List[int] = None,
                      components: List[int] = None,
                      grid: List[Vector] = None,
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
    grid : List[Vector], default None
        The grid the snapshots are defined on. The default behaviors
        is a grid from 0 to n_features - 1.
    filename : str, default None
        A location to save the plot to, if specified.
    """
    # Check the inputs
    if self._snapshots is None:
        raise ValueError('No input snapshots found.')

    if grid is None:
        if self._snapshots_shape is None:
            raise ValueError('There is no information about the '
                             'original shape of the snapshots.')
        elif len(self._snapshots_shape) != 2:
            raise ValueError('The dimension of the snapshots is not 2D.')
        else:
            x = np.arange(self._snapshots_shape[0])
            y = np.arange(self._snapshots_shape[1])
    else:
        x = np.unique([node.x for node in grid])
        y = np.unique([node.y for node in grid])
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
    n_rows, n_cols = format_subplots(len(components))

    # Plot each mode specified
    for idx in snapshot_indices:
        idx += 0 if idx > 0 else self.n_snapshots
        snapshot: ndarray = self.snapshots[:, idx]

        # Make figure
        fig: Figure = plt.figure()
        fig.suptitle(f'Snapshot {idx}', fontsize=12)

        # Plot each component specified
        for i, c in enumerate(components):
            c += 0 if c > 0 else n_components
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
