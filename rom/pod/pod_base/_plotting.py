import numpy as np
from numpy import ndarray
from os.path import splitext
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from rom.pod.pod_base import PODBase


def plot_scree(self: 'PODBase',
               normalized: bool = True,
               logscale: bool = True,
               show_cutoff: bool = False,
               filename: str = None) -> None:
    """
    Plot the singular value spectrum.

    Parameters
    ----------
    normalized : bool, default True
        Flag for normalizing the spectrum to its max value.
    logscale : bool, default True
        Flag for a log scale on the y-axis.
    show_cutoff : bool, default False
        Flag for showing the truncation location.
    filename : str, default None.
        A location to save the plot to, if specified.
    """
    # Get the spectrum
    s = self.singular_values
    if normalized:
        s /= sum(s)

    # Determine the correct plotter
    plotter = plt.semilogy if logscale else plt.plot

    # Make figure
    plt.figure()
    plt.xlabel('n', fontsize=12)
    plt.ylabel('Singular Value' if not normalized
               else 'Relative Singular Value')
    plt.grid(True)
    plotter(s, '-*b')
    if show_cutoff:
        plt.axvline(self.n_modes - 1, color='r',
                    ymin=1.0e-12, ymax=1.0-1.0e-12)
    plt.tight_layout()

    # Save figure
    if filename is not None:
        base, ext = splitext(filename)
        plt.savefig(base + '.pdf')


def plot_modes_1D(self: 'PODBase',
                  mode_indices: List[int] = None,
                  components: List[int] = None,
                  x: ndarray = None,
                  imaginary: bool = False,
                  filename: str = None) -> None:
    """
    Plot 1D DMD modes.

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
    imaginary : default False
        Flag for plotting the imaginary part of the dynamics.
    filename : str, default None
        A location to save the plot to, if specified.
    """
    # Check that modes have been computed
    if self.modes is None:
        raise AssertionError('DMD model has not been fit.')

    # Handle grid input
    if x is None:
        x = np.arange(0, self.n_features, 1)

    n_components = self.n_features // len(x)
    if not isinstance(n_components, int):
        raise AssertionError('Invalid grid provided.')

    # Handle mode indices input
    if mode_indices is None:
        mode_indices = list(range(self.n_modes))
    elif isinstance(mode_indices, int):
        mode_indices = [mode_indices]
    else:
        for i, idx in enumerate(mode_indices):
            if not -self.n_modes <= idx < self.n_modes:
                raise AssertionError('Invalid mode index encountered.')
            if idx < 0:
                mode_indices[i] = self.n_modes + idx

    # Handle components input
    if components is None:
        components = list(range(n_components))
    elif isinstance(components, int):
        components = [components]
    else:
        for c, comp in enumerate(components):
            if not -n_components <= comp < n_components:
                raise AssertionError('Invalid component ecountered.')
            if comp < 0:
                components[c] = n_components + comp

    # Plot each mode specified
    for idx in mode_indices:
        mode: ndarray = self.modes.T[idx]

        # Make figure
        fig: Figure = plt.figure()
        fig.suptitle(f'POD Mode {idx}')
        n_plots = 2 if imaginary else 1

        # Plot real part
        real_ax: Axes = fig.add_subplot(1, n_plots, 1)
        real_ax.set_xlabel('r', fontsize=12)
        real_ax.set_ylabel(r'Real', fontsize=12)
        real_ax.grid(True)
        for c in components:
            label = f'Component {c}'
            vals = mode.real[c::n_components]
            real_ax.plot(x, vals, label=label)

        # Plot imaginary part
        if imaginary:
            imag_ax: Axes = fig.add_subplot(1, 2, n_plots)
            imag_ax.set_xlabel('r', fontsize=12)
            imag_ax.set_ylabel(r'Imaginary', fontsize=12)
            imag_ax.grid(True)
            for c in components:
                label = f'Component {c}'
                vals = mode.imag[c::n_components]
                imag_ax.plot(x, vals, label=label)

        plt.tight_layout()
        if filename is not None:
            base, ext = splitext(filename)
            plt.savefig(base + f'_{idx}.pdf')


def plot_coefficients(self: 'PODBase',
                      mode_indices: List[int] = None,
                      filename: str = None) -> None:
    """
    Plot the POD coefficients as a function of parameter values.

    Parameters
    ----------
    mode_indices : List[int], default None
        The indices of the modes to plot. The default behavior
        is to plot all modes.
    filename : str, default None
        A location to save the plot to, if specified.
    """
    y = self.parameters

    # Handle mode indices input
    if mode_indices is None:
        mode_indices = list(range(self.n_modes))
    elif isinstance(mode_indices, int):
        mode_indices = [mode_indices]
    else:
        for i, idx in enumerate(mode_indices):
            if not -self.n_modes <= idx < self.n_modes:
                raise AssertionError('Invalid mode index encountered.')
            if idx < 0:
                mode_indices[i] = self.n_modes + idx

    # One-dimensional parameter spaces
    if y.shape[0] == 1:
        y = y.flatten()

        # Sort by parameter values
        idx = np.argsort(y)
        y, amplitudes = y[idx], self.amplitudes[idx]

        # Plot plot modes
        for idx in mode_indices:
            fig: Figure = plt.figure()
            ax: Axes = fig.add_subplot(111)
            ax.grid(True)
            ax.set_xlabel('Parameter Value', fontsize=12)
            ax.set_ylabel('POD Coefficient Value', fontsize=12)
            ax.plot(y, amplitudes[:, idx], '-*', label=f'Mode {idx}')
            ax.legend()

            plt.tight_layout()
            if filename is not None:
                basename, ext = splitext(filename)
                plt.savefig(basename + f'_{idx}.pdf')


def plot_rankwise_errors(self: 'PODBase', skip: int = 1,
                         end: int = -1,
                         logscale: bool = True,
                         filename: str = None) -> None:
    """
    Plot the reconstruction errors.

    Parameters
    ----------
    skip : int, default 1
        The number of modes to skip in between each data point.
    end : int, default -1
        The most modes to reconstruct a model for. If None, the
        plot ends at n_modes = n_snapshots - 1.
    logscale : bool, default True
        Flag for plotting on a linear or log scale y-axis.
    filename : str, default None
        If specified, the location to save the plot.
    """
    ranks, errors = self.compute_rankwise_errors(skip, end)
    s = self.singular_values
    s /= sum(s)

    # Determine the correct plotter
    plotter = plt.semilogy if logscale else plt.plot

    # Make the figure
    plt.figure()
    plt.xlabel('# of Modes')
    plt.ylabel(r'$L^2$ Error')
    plt.grid(True)
    plotter(s, '-*b', label='Singular Values')
    plotter(ranks, errors, '-or', label='Reconstruction Errors')
    plt.tight_layout()

    # Save the figure
    if filename is not None:
        basename, ext = splitext(filename)
        plt.savefig(basename + '.pdf')
