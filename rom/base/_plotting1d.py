import warnings

import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from os.path import splitext
from typing import List

from pyPDEs.utilities import Vector

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import ROMBase


def plot_modes_1D(self: 'ROMBase',
                  mode_indices: List[int] = None,
                  components: List[int] = None,
                  grid: List[Vector] = None,
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
    grid : List[Vector], default None
        The grid the modes are defined on. The default behaviors
        is a grid from 0 to n_features - 1.
    filename : str, default None
        A location to save the plot to, if specified.
    """
    # Check inputs
    if self.modes is None:
        raise ValueError('The fit method must be performed first.')

    if grid is None:
        x = np.arange(0, self.n_features, 1)
    else:
        if not all([isinstance(node, Vector) for node in grid]):
            raise TypeError('The grid must be a list of Vector objects.')
        x = [node.z for node in grid]

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
        print(idx)
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


def plot_snapshots_1D(self: 'ROMBase',
                      snapshot_indices: List[int] = None,
                      components: List[int] = None,
                      grid: List[Vector] = None,
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
    grid : List[Vector], default None
        The grid the modes are defined on. The default behaviors
        is a grid from 0 to n_features - 1.
    filename : str, default None
        A location to save the plot to, if specified.
    """
    if self.snapshots is None:
        raise ValueError('No input snapshots found.')

    if grid is None:
        x = np.arange(0, self.n_features, 1)
    else:
        if not all([isinstance(node, Vector) for node in grid]):
            raise TypeError('The grid must be a list of Vector objects.')
        x = [node.z for node in grid]

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
        snapshot: ndarray = self.snapshots[:, idx].real

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
