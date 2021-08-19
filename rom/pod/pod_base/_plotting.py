import numpy as np
from numpy import ndarray
from os.path import splitext
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from rom.pod.pod_base import PODBase


def plot_singular_values(
        self: "PODBase", normalized: bool = True,
        logscale: bool = True,
        filename: str = None) -> None:
    """Plot the singular value spectrum.

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


def plot_1D_modes(
        self: "PODBase", indices: List[int] = None,
        x: ndarray = None,
        components: List[int] = None,
        filename: str = None) -> None:
    """Plot the DMD mode profiles.

    Parameters
    ----------
    indices : List[int], default None
        The indices of the modes to plot. If None, all modes are plotted.

    x : ndarray, default None
        The spatial grid the mode is defined on.

    components : list of int, default None
        If the snapshots were multi-component, this input is to define
        which components to plot.  The number of components is defined
        by the integer division of the snapshot length and the
        length of the supplied grid.

    filename : str, default None
        If specified, the location to save the plot.
    """
    # ======================================== Initialization check
    if self.modes is None:
        raise AssertionError(
            f"DMD model is not initialized. To initialize a "
            f"model, run the {self.__class__.__name__}.fit method.")

    # ======================================== Validate grid input
    if x is None:
        x = np.arange(0, self.n_features, 1)

    n_components = self.n_features // len(x)
    if not isinstance(n_components, int):
        raise AssertionError("Incompatible grid provided.")

    # ======================================== Validate indices input
    if indices is None:
        indices = list(range(self.n_modes))
    elif isinstance(indices, int):
        indices = [indices]
    if any([not -self.n_modes <= ind < self.n_modes for ind in indices]):
        raise AssertionError("Invalid mode index encountered.")

    # ======================================== Validate components input
    if components is None:
        components = list(range(n_components))
    elif isinstance(components, int):
        components = [components]
    if any([c >= n_components for c in components]):
        raise AssertionError("Invalid component encountered.")

    # ======================================== Loop over modes to plot
    for ind in indices:
        fig: Figure = plt.figure()
        fig.suptitle(f"POD Mode {ind}")

        real_ax: Axes = fig.add_subplot(1, 2, 1)
        imag_ax: Axes = fig.add_subplot(1, 2, 2)

        real_ax.set_title("Real")
        imag_ax.set_title("Imaginary")

        # ============================== Plot component-wise
        mode = self.modes.T[ind]
        for c in components:
            label = f"Component {c}"
            vals = mode[c::n_components]
            real_ax.plot(x, vals.real, label=label)
            imag_ax.plot(x, vals.imag, label=label)

        real_ax.legend()
        imag_ax.legend()
        real_ax.grid(True)
        imag_ax.grid(True)
        plt.tight_layout()

        # ======================================== Save plot
        if filename is not None:
            basename, ext = splitext(filename)
            plt.savefig(basename + f"_{ind}.pdf")


def plot_coefficients(
        self: "PODBase", indices: List[int] = None,
        normalize: bool = False,
        filename: str = None) -> None:
    """
    Plot the POD coefficients as a function of parameter values.

    Parameters
    ----------
    indices : List[int], default None
        The indices of the modes to plot. If None, all modes are plotted.

    normalize : bool, default False
        Flag to remove the mean and normalize by the standard deviation of
        each mode coefficient function.

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


def plot_error_decay(self: "DMDBase", skip: int = 1,
                     end: int = None,
                     logscale: bool = True,
                     filename: str = None) -> None:
    """Plot the reconstruction errors.

    Parameters
    ----------
    skip : int, default 1
        The number of modes to skip in between each data point.

    end : int, default None
        The most modes to reconstruct a model for. If None, the
        plot ends at n_modes = n_snapshots - 1

    logscale : bool
        Flag for plotting on a linear or log scale y-axis.

    filename : str, default None
        If specified, the location to save the plot.
    """
    errors, n_modes = self.compute_error_decay(skip, end)
    spectrum = self.singular_values
    spectrum /= sum(spectrum)

    plt.figure()
    plt.xlabel("# of Modes")
    plt.ylabel(r"$\ell^2$ Error")
    plotter = plt.semilogy if logscale else plt.plot
    plotter(spectrum, "b-*", label="Singular Values")
    plotter(n_modes, errors, "r-*", label="Reconstruction Errors")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        basename, ext = splitext(filename)
        plt.savefig(basename + ".pdf")
