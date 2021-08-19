import numpy as np
from numpy import ndarray
from os.path import splitext
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from . import DMDBase


def plot_singular_values(
        self: "DMDBase", normalized: bool = True,
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
    # ======================================== Get singular values
    spectrum = self.singular_values
    if normalized:
        spectrum /= sum(spectrum)

    # ======================================== Plot them
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

    # ======================================== Save plot
    if filename is not None:
        basename, ext = splitext(filename)
        plt.savefig(basename + ".pdf")


def plot_1D_modes(
        self: "DMDBase", indices: List[int] = None,
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
        fig.suptitle(f"DMD Mode {ind}\n$\omega$ = "
                     f"{self.omegas[ind].real:.3e}"
                     f"{self.omegas[ind].imag:+.5g}j")

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


def plot_dynamics(
        self: "DMDBase", indices: List[int] = None,
        t: ndarray = None,
        filename: str = None) -> None:
    """Plot DMD mode dynamics.

    Parameters
    ----------
    indices : List[int], default None
        The indices of the modes to plot. If None, all modes are plotted.

    t : ndarray, default None
        The temporal grid the mode is defined on.

    filename : str, default None
        If specified, the location to save the plot.
    """
    # ======================================== Initialization check
    if self.dynamics is None:
        raise AssertionError(
            f"DMD model is not initialized. To initialize a "
            f"model, run the {self.__class__.__name__}.fit method.")

    # ======================================== Validate time input
    if t is None:
        t = np.arange(0, self.n_snapshots, 1)

    # ======================================== Validate indices input
    if indices is None:
        indices = list(range(self.n_modes))
    elif isinstance(indices, int):
        indices = [indices]
    if any([not -self.n_modes <= ind < self.n_modes for ind in indices]):
        raise AssertionError("Invalid mode index encountered.")

    # ======================================== Loop over dynamics to plot
    for ind in indices:
        fig: Figure = plt.figure()
        fig.suptitle(f"DMD Mode {ind}\n$\omega$ = "
                     f"{self.omegas[ind].real:.3e}"
                     f"{self.omegas[ind].imag:+.5g}j")

        real_ax: Axes = fig.add_subplot(1, 2, 1)
        imag_ax: Axes = fig.add_subplot(1, 2, 2)

        real_ax.set_title("Real")
        imag_ax.set_title("Imaginary")

        dynamic = self.dynamics[ind]
        real_ax.plot(t, dynamic.real)
        imag_ax.plot(t, dynamic.imag)

        real_ax.grid(True)
        imag_ax.grid(True)
        plt.tight_layout()

        # ============================== Save plot
        if filename is not None:
            basename, ext = splitext(filename)
            plt.savefig(basename + f"_{ind}.pdf")


def plot_1D_modes_and_dynamics(
        self: "DMDBase", indices: List[int] = None,
        x: ndarray = None, t: ndarray = None,
        components: List[int] = None,
        filename: str = None) -> None:
    """Plot the DMD mode and dynamics.

    Parameters
    ----------
    indices : List[int], default None
        The indices of the modes to plot. If None, all modes are plotted.

    x : ndarray, default None
        The spatial grid the mode is defined on.

    t : ndarray, default None
        The temporal grid the dynamics are defined on.

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

    # ======================================== Validate time input
    if t is None:
        t = np.arange(0, self.n_snapshots, 1)

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
    if any([not -n_components <= c < n_components for c in components]):
        raise AssertionError("Invalid component encountered.")

    # ======================================== Loop over modes to plot
    for ind in indices:
        fig: Figure = plt.figure()
        fig.suptitle(f"DMD Mode {ind}\n$\omega$ = "
                     f"{self.omegas[ind].real:.2e}"
                     f"{self.omegas[ind].imag:+.2e}j")

        real_axs: List[Axes] = [fig.add_subplot(2, 2, 1),
                                fig.add_subplot(2, 2, 2)]
        imag_axs: List[Axes] = [fig.add_subplot(2, 2, 3),
                                fig.add_subplot(2, 2, 4)]

        real_axs[0].set_title("Profile")
        real_axs[1].set_title("Dynamic")
        real_axs[0].set_ylabel("Real")
        imag_axs[0].set_ylabel("Imaginary")

        # ============================== Plot component-wise
        mode = self.modes.T[ind]
        for c in components:
            label = f"C{c}"
            vals = mode[c::n_components]
            real_axs[0].plot(x, vals.real, label=label)
            imag_axs[0].plot(x, vals.imag, label=label)

        # ============================== Plot dynamics
        dynamic = self.dynamics[ind]
        real_axs[1].plot(t, dynamic.real)
        imag_axs[1].plot(t, dynamic.imag)

        iterable = zip(range(2), real_axs, imag_axs)
        for i, real_ax, imag_ax in iterable:
            real_ax.grid(True)
            imag_ax.grid(True)
            if i == 0:
                real_ax.legend()
                imag_ax.legend()
        plt.tight_layout()

        # ============================== Save plot
        if filename is not None:
            basename, ext = splitext(filename)
            plt.savefig(basename + f"_{ind}.pdf")


def plot_1D_mode_evolutions(self, indices: List[int] = None,
                            x: ndarray = None, t: ndarray = None,
                            components: List[int] = None,
                            filename: str = None) -> None:
    """Plot the DMD mode evolution in 2D.

    Parameters
    ----------
    indices : List[int], default None
        The indices of the modes to plot. If None, all modes are plotted.

    x : ndarray, default None
        The spatial grid the mode is defined on.

    t : ndarray, default None
        The temporal grid the dynamics are defined on.

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

    # ======================================== Validate time input
    if t is None:
        t = np.arange(0, self.n_snapshots, 1)

    # ======================================== Create meshgrid
    if x.ndim == t.ndim == 1:
        x, t = np.meshgrid(x, t)
    if x.ndim != 2 or t.ndim != 2:
        raise AssertionError("x, t must be a meshgrid format.")

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
    if any([not -n_components <= c < n_components for c in components]):
        raise AssertionError("Invalid component encountered.")
    n_cols = int(np.ceil(np.sqrt(len(components))))
    n_rows = 1
    for n in range(n_cols):
        if n * n_cols > len(components):
            n_rows = n
            break

    # ======================================== Loop over modes to plot
    for ind in indices:
        fig: Figure = plt.figure()
        fig.suptitle(f"DMD Mode {ind}\n$\omega$ = "
                     f"{self.omegas[ind].real:.2e}"
                     f"{self.omegas[ind].imag:+.5g}j")

        # ============================== Loop over components
        mode = self.modes.T[ind].reshape(-1, 1)
        dynamic = self.dynamics[ind].reshape(1, -1)
        evolution = (mode @ dynamic).T.real
        for n, c in enumerate(components):
            fig.add_subplot(n_rows, n_cols, n + 1)
            vals = evolution[:, c::n_components]
            plt.pcolormesh(x, t, vals, cmap="jet", shading="auto",
                           vmin=vals.min(), vmax=vals.max())
            plt.colorbar()
        plt.tight_layout()

        # ============================== Save plot
        if filename is not None:
            basename, ext = splitext(filename)
            plt.savefig(basename + f"_{ind}.pdf")


def plot_eigs(self: "DMDBase",
              show_unit_circle: bool = True,
              filename: str = None) -> None:
    """Plot the eigenvalues.

    Parameters
    ----------
    show_unit_circle : bool, default True
        If True, the unit circle is plotted for reference.
    filename : str, default None
        If specified, the location to save the plot.
    """
    if self.eigs is None:
        raise AssertionError(
            f"DMD model is not initialized. To initialize a "
            f"model, run the {self.__class__.__name__}.fit method.")

    plt.figure(figsize=(6, 6))
    ax: Axes = plt.gca()
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")

    points, = ax.plot(self.eigs.real, self.eigs.imag, "bo")

    if show_unit_circle:
        unit_circle = plt.Circle(
            (0.0, 0.0), 1.0, fill=False, color="green",
            label="Unit Circle", linestyle="--")
        ax.add_artist(unit_circle)
        ax.add_artist(
            plt.legend([points, unit_circle],
                       ["Eigenvalues", "Unit Circle"],
                       loc="best"))

    limit = 0.75 * np.max(np.ceil(np.absolute(self.eigs)))
    limit = max(limit, 1.25)
    ax.set_xlim((-limit, limit))
    ax.set_ylim((-limit, limit))
    ax.set_aspect("equal")
    ax.grid(True)
    plt.tight_layout()

    if filename is not None:
        basename, ext = splitext(filename)
        plt.savefig(basename + ".pdf")


def plot_timestep_errors(self: "DMDBase",
                         logscale: bool = True,
                         filename: str = None) -> None:
    """Plot the reconstruction error at each time step

    Parameters
    ----------
    logscale : bool

    filename : str, default None
        If specified, the location to save the plot.
    """
    times = self.original_timestamps
    errors = self.compute_timestep_errors()

    # Setup plot
    plt.figure()
    plt.xlabel("Time (s)")
    plt.ylabel(r"Relative $\ell^2$ Error")
    plotter = plt.semilogy if logscale else plt.plot
    plotter(times, errors, "r-*", label="Reconstruction Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        basename, ext = splitext(filename)
        plt.savefig(basename + ".pdf")


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
