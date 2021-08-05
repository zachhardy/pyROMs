import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .pod_base import PODBase


def plot_singular_values(self: 'PODBase',
                         logscale: bool = True) -> None:
    """
    Plot the singular value spectrum.

    Parameters
    ----------
    logscale : bool, default False
        Flag for plotting on a linear or log scale y-axis.
    """
    s = self._singular_values
    data = s / sum(s)

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(111)
    ax.set_xlabel('Singular Value #', fontsize=12)
    ax.set_ylabel(r'$\sigma / \sum{{\sigma}}$', fontsize=12)
    plotter = plt.semilogy if logscale else plt.plot
    plotter(data, 'b-*', label='Singular Values')
    ax.axvline(self.n_modes - 1, color='r',
               ymin=1e-12, ymax=1.0 - 1.0e-12)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()


def plot_coefficients(self: 'PODBase', modes: ndarray = None,
                      normalize: bool = False) -> None:
    """
    Plot the POD coefficients as a function of parameter.

    Parameters
    ----------
    modes : int or List[int], default 0
        The mode indices to plot the coefficients for.
        If an int, only that mode is plotted. If a list,
        all modes with the supplied indices are plotted on the
        same Axes.
    normalize : bool, default False
        Flag to remove the mean and normalize by the standard
        deviation of each mode coefficient function.
    """
    y = self.parameters

    # One-dimensional parameter spaces
    if y.shape[1] == 1:
        # Sort by parameter values
        ind = np.argsort(y, axis=0).ravel()
        y, amplitudes = y[ind], self.amplitudes[ind]

        # Get modes to plot
        if isinstance(modes, int):
            modes = [modes]
        elif isinstance(modes, list):
            modes = [m for m in modes if m < self.n_modes]
        else:
            modes = [m for m in range(self.n_modes)]

        # Format amplitudes
        if normalize:
            amplitudes = self.center_data(amplitudes)

        # Plot plot modes
        for m in modes:
            fig: Figure = plt.figure()
            ax: Axes = fig.add_subplot(111)
            ax.set_xlabel('Parameter Value', fontsize=12)
            ax.set_ylabel('POD Coefficient Value', fontsize=12)
            ax.plot(y, amplitudes[:, m], '-*', label=f'Mode {m}')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()


def plot_reconstruction_errors(self: 'PODBase',
                               logscale: bool = True) -> None:
    """
    Plot the reconstruction errors.

    Parameters
    ----------
    logscale : bool
        Flag for plotting on a linear or log scale y-axis.
    """
    errors = self.compute_error_decay()
    s = self._singular_values
    spectrum = s / sum(s)

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(111)
    ax.set_xlabel('# of Modes', fontsize=12)
    ax.set_ylabel(r'Relative $\ell^2$ Error', fontsize=12)
    plotter = plt.semilogy if logscale else plt.plot
    plotter(spectrum, 'b-*', label='Singular Values')
    plotter(errors, 'r-*', label='Reconstruction Errors')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
