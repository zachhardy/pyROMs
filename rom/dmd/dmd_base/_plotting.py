import numpy as np
from numpy import ndarray
from numpy.linalg import norm

import matplotlib.pyplot as plt

from os.path import splitext

from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import TYPE_CHECKING, List, Union
if TYPE_CHECKING:
    from . import DMDBase


def plot_scree(self: 'DMDBase',
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


def plot_modes_1D(self: 'DMDBase',
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
        fig.suptitle(f'DMD Mode {idx}')
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


def plot_dynamics(self: 'DMDBase',
                  mode_indices: List[int] = None,
                  t: ndarray = None,
                  imaginary: bool = False,
                  logscale: bool = False,
                  filename: str = None) -> None:
    """
    Plot the dynamics of the modes.

    Parameters
    ----------
    mode_indices : List[int], default None
        The indices of the modes to plot. The default behavior
        is to plot all modes.
    t : ndarray, default None
        The temporal grid the modes are defined on. The default behaviors
        is a grid from 0 to n_snapshots.
    imaginary : default False
        Flag for plotting the imaginary part of the dynamics.
    logscale : bool, default True
        Flag for plotting on a linear or log scale y-axis.
    filename : str, default None
        A location to save the plot to, if specified.
    """
    # Check that dynamics have been computed
    if self.dynamics is None:
        raise AssertionError('DMD model has not been fit.')

    # Validate the temporal grid input
    if t is None:
        t = np.arange(0, self.n_snapshots, 1)

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

    # Plot each mode specified
    for idx in mode_indices:
        dynamic: ndarray = self.dynamics[idx]

        # Make figure
        fig: Figure = plt.figure()
        fig.suptitle(f'DMD Dynamics {idx}\n$\omega$ = '
                     f'{self.omegas[idx].real:.3e}'
                     f'{self.omegas[idx].imag:+.3g}j')
        n_plots = 2 if imaginary else 1

        # Plot real part
        real_ax: Axes = fig.add_subplot(1, n_plots, 1)
        real_ax.set_xlabel('r', fontsize=12)
        real_ax.set_ylabel('Real', fontsize=12)
        real_ax.grid(True)
        real_plotter = real_ax.semilogy if logscale else real_ax.plot
        real_plotter(t, dynamic.real/norm(dynamic.real))

        # Plot the imaginary part
        if imaginary:
            imag_ax: Axes = fig.add_subplot(1, n_plots, 2)
            imag_ax.set_xlabel('t', fontsize=12)
            imag_ax.set_ylabel('Imaginary', fontsize=12)
            imag_ax.grid(True)
            imag_plotter = imag_ax.semilogy if logscale else imag_ax.plot
            imag_plotter(t, dynamic.imag/norm(dynamic.imag))

        # Save the plot
        plt.tight_layout()
        if filename is not None:
            base, ext = splitext(filename)
            plt.savefig(base + f'_{idx}.pdf')


def plot_eigs(self: 'DMDBase',
              show_unit_circle: bool = True,
              filename: str = None) -> None:
    """Plot the eigenvalues.

    Parameters
    ----------
    show_unit_circle : bool, default True
        If True, the unit circle is plotted for reference.
    filename : str, default None
        Location to save the plot to, if specified.
    """
    if self.eigenvalues is None:
        raise AssertionError('DMD model has not been fit.')

    plt.figure(figsize=(6, 6))
    ax: Axes = plt.gca()
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')

    eigs: ndarray = self.eigenvalues
    points, = ax.plot(eigs.real, eigs.imag, 'bo')

    if show_unit_circle:
        unit_circle = plt.Circle(
            (0.0, 0.0), 1.0, fill=False, color='green',
            label='Unit Circle', linestyle="--")
        ax.add_artist(unit_circle)
        ax.add_artist(
            plt.legend([points, unit_circle],
                       ['Eigenvalues', 'Unit Circles'],
                       loc='best'))

    limit = 0.75 * np.max(np.ceil(np.absolute(eigs)))
    limit = max(limit, 1.25)
    ax.set_xlim((-limit, limit))
    ax.set_ylim((-limit, limit))
    ax.set_aspect('equal')
    ax.grid(True)
    plt.tight_layout()

    if filename is not None:
        base, ext = splitext(filename)
        plt.savefig(base + '.pdf')


def plot_timestep_errors(self: 'DMDBase',
                         logscale: bool = True,
                         filename: str = None) -> None:
    """
    Plot the reconstruction error at each time step

    Parameters
    ----------
    logscale : bool, default True
        Flag for plotting on a linear or log scale y-axis.
    filename : str, default None
        If specified, the location to save the plot.
    """
    # Get the data
    times = self.snapshot_timesteps
    errors = self.timestep_errors

    # Make the figure
    plt.figure()
    plt.xlabel("Time (s)")
    plt.ylabel(r"Relative $\ell^2$ Error")
    plotter = plt.semilogy if logscale else plt.plot
    plotter(times[1:], errors[1:], "r-*", label="Reconstruction Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    if filename is not None:
        basename, ext = splitext(filename)
        plt.savefig(basename + ".pdf")


def plot_error_decay(self: 'DMDBase', skip: int = 1,
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
        plt.savefig(basename + ".pdf")
