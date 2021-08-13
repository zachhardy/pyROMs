import numpy as np
from numpy import ndarray
from os.path import splitext
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import Union, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .pod_base import PODBase


def plot_singular_values(
        self: 'PODBase', normalized: bool = True,
        logscale: bool = True,
        filename: str = None) -> None:
    """
    Plot the singular value spectrum.

    Parameters
    ----------
    normalized : bool, default True
        If True, the singular values are normalized by
        the sum of all singular values.
    logscale : bool, default False
        If True, the plot will have a logarithmic y-axis.
    filename : str, default None
        If specified, the location to save the plot.
    """
    spectrum = self.singular_values
    if normalized:
        spectrum /= sum(spectrum)

    plt.figure()
    plt.xlabel("Singular Value #")
    plt.ylabel(r"$\sigma / \sum{{\sigma}}$")
    plotter = plt.semilogy if logscale else plt.plot
    plotter(spectrum, "b-*", label="Singular Values")
    plt.axvline(self.n_modes - 1, color="r",
                ymin=1e-12, ymax=1.0 - 1.0e-12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if filename is not None:
        basename, ext = splitext(filename)
        plt.savefig(basename + ".pdf")


def plot_coefficients(
        self: 'PODBase', indices: List[int] = None,
        normalize: bool = False,
        filename: str = None) -> None:
    """
    Plot the POD coefficients as a function of parameter values.

    Parameters
    ----------
    indices : int or List[int], default None
        The mode indices to plot the coefficients for.
        If an int, only that mode is plotted. If a list, those
        modes specified are plotted. If None, all modes are
        plotted.
    normalize : bool, default False
        Flag to remove the mean and normalize by the standard
        deviation of each mode coefficient function.
    filename : str, default None
        If specified, the location to save the plot.
    """
    y = self.parameters

    # One-dimensional parameter spaces
    if y.shape[1] == 1:
        # Sort by parameter values
        ind = np.argsort(y, axis=0).ravel()
        y, amplitudes = y[ind], self.amplitudes[ind]

        # Get modes to plot
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, list):
            indices = [i for i in indices if i < self.n_modes]
        else:
            indices = [i for i in range(self.n_modes)]

        # Format amplitudes
        if normalize:
            amplitudes = self._center_data(amplitudes)

        # Plot plot modes
        for ind in indices:
            fig: Figure = plt.figure()
            ax: Axes = fig.add_subplot(111)
            ax.set_xlabel("Parameter Value")
            ax.set_ylabel("POD Coefficient Value")
            ax.plot(y, amplitudes[:, ind], "-*", label=f"Mode {ind}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            if filename is not None:
                basename, ext = splitext(filename)
                plt.savefig(basename + f"_{ind}.pdf")


def plot_error_decay(
        self: 'PODBase', skip: int = 1,
        end: int = None,
        logscale: bool = True,
        filename: str = None) -> None:
    """
    Plot the reconstruction errors.

    Parameters
    ----------
    logscale : bool
        Flag for plotting on a linear or log scale y-axis.
    """
    errors, n_modes = self.compute_error_decay(skip, end)
    s = self._singular_values
    spectrum = s / sum(s)

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(111)
    ax.set_xlabel("# of Modes")
    ax.set_ylabel(r"Relative $\ell^2$ Error")
    plotter = plt.semilogy if logscale else plt.plot
    plotter(spectrum, "b-*", label="Singular Values")
    plotter(n_modes, errors, "r-*", label="Reconstruction Errors")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
