import numpy as np
from numpy import ndarray
from numpy.linalg import norm
from os.path import splitext
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from .dmd_base import DMDBase

def plot_singular_values(
        self: 'DMDBase', normalized: bool = True,
        logscale: bool = True,
        filename: str = None) -> None:
    """

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
    plt.xlabel('Singular Value #')
    plt.ylabel(r'$\sigma / \sum{{\sigma}}$')
    plotter = plt.semilogy if logscale else plt.plot
    plotter(spectrum, 'b-*', label='Singular Values')
    plt.axvline(self.n_modes - 1, color='r',
                ymin=1e-12, ymax=1.0 - 1.0e-12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if filename is not None:
        basename, ext = splitext(filename)
        plt.savefig(basename + '.pdf')
    else:
        plt.show()


def plot_1D_profiles(
        self: 'DMDBase', indices: List[int] = None,
        x: ndarray = None,
        components: List[int] = None,
        filename: str = None) -> None:
    """
    Plot the DMD mode profiles.

    Parameters
    ----------
    indices : list of int, default None
        The indices of the modes to plot.
    x : ndarray, default None
        The spatial grid.
    components : list of int, default None
        If the snapshots were multi-component, this
        input is to define which components to plot.
        The number of components is defined by the
        integer division of the snapshot length and the
        length of the supplied grid, x.
    filename : str, default None
        If specified, the location to save the plot.
    """
    # Initialization check
    if self.modes is None:
        raise AssertionError(
            f'DMD model is not initialized. To initialize a '
            f'model, run the {self.__class__.__name__}.fit method.'
        )
    if x is None:
        x = np.arange(0, self.n_features, 1)

    # Check grid
    n_components = self.n_features // len(x)
    if not isinstance(n_components, int):
        raise AssertionError('Incompatible x provided.')

    # Define indices iterable
    if indices is None:
        indices = list(range(self.n_modes))
    elif isinstance(indices, int):
        indices = [indices]
    if any([not 0 <= ind < self.n_modes for ind in indices]):
        raise AssertionError('Invalid mode index encountered.')

    # Define components iterable
    if components is None:
        components = list(range(n_components))
    elif isinstance(components, int):
        components = [components]
    if any([c >= n_components for c in components]):
        raise AssertionError('Invalid component encountered.')

    # Loop over indices of profiles to plot
    for ind in indices:
        fig: Figure = plt.figure()
        fig.suptitle(f'DMD Mode {ind}\n$\omega$ = '
                     f'{self.omegas[ind].real:.2e}'
                     f'{self.omegas[ind].imag:+.2e}')

        real_ax: Axes = fig.add_subplot(1, 2, 1)
        imag_ax: Axes = fig.add_subplot(1, 2, 2)

        real_ax.set_title('Real')
        imag_ax.set_title('Imaginary')

        # Loop over components
        mode = self.modes.T[ind]
        for c in components:
            label = f'Component {c}'
            vals = mode[c::n_components]
            real_ax.plot(x, vals.real, label=label)
            imag_ax.plot(x, vals.imag, label=label)

        real_ax.legend()
        imag_ax.legend()
        real_ax.grid(True)
        imag_ax.grid(True)
        plt.tight_layout()

        if filename is not None:
            basename, ext = splitext(filename)
            plt.savefig(basename + f'_{ind}.pdf')
        else:
            plt.show()


def plot_dynamics(
        self: 'DMDBase', indices: List[int] = None,
        t: ndarray = None,
        filename: str = None) -> None:
    """
    Plot DMD mode dynamics.

    Parameters
    ----------
    indices : list of int, default None
        The mode indices to plot
    t : ndarray, default None
        The temporal grid the mode is defined on.
    filename : str, default None
        If specified, the location to save the plot.
    """
    # Initialization check
    if self.dynamics is None:
        raise AssertionError(
            f'DMD model is not initialized. To initialize a '
            f'model, run the {self.__class__.__name__}.fit method.'
        )
    if t is None:
        t = self.original_timesteps

    # Define indices iterable
    if indices is None:
        indices = list(range(self.n_modes))
    elif isinstance(indices, int):
        indices = [indices]
    if any([not 0 <= ind < self.n_modes for ind in indices]):
        raise AssertionError('Invalid mode index encountered.')

    # Loop over indices
    for ind in indices:
        fig: Figure = plt.figure()
        fig.suptitle(f'DMD Mode {ind}\n$\omega$ = '
                     f'{self.omegas[ind].real:.2e}'
                     f'{self.omegas[ind].imag:+.2e}')

        real_ax: Axes = fig.add_subplot(1, 2, 1)
        imag_ax: Axes = fig.add_subplot(1, 2, 2)

        real_ax.set_title('Real')
        imag_ax.set_title('Imaginary')

        dynamic = self.dynamics[ind]
        real_ax.plot(t, dynamic.real)
        imag_ax.plot(t, dynamic.imag)

        real_ax.grid(True)
        imag_ax.grid(True)
        plt.tight_layout()

        if filename is not None:
            basename, ext = splitext(filename)
            plt.savefig(basename + f'_{ind}.pdf')
        else:
            plt.show()


def plot_1D_profiles_and_dynamics(
        self: 'DMDBase', indices: List[int] = None,
        x: ndarray = None, t: ndarray = None,
        components: List[int] = None,
        filename: str = None) -> None:
    """
    Plot the DMD mode and dynamics.

    Parameters
    ----------
    indices : list of int, default None
        The mode indices to plot.
    x : ndarray, default None
        The spatial grid the mode is defined on.
    t : ndarray, default None
        The temporal grid the dynamics are defined on.
    components : list of int, default None
        The component of the mode to plot.
        The number of components in a mode is defined
        by the integer division of the mode and the
        supplied grid. By default, all components are
        plotted.
    filename : str, default None
        If specified, the location to save the plot.
    """
    # Initialization check
    if self.modes is None:
        raise AssertionError(
            f'DMD model is not initialized. To initialize a '
            f'model, run the {self.__class__.__name__}.fit method.'
        )
    if x is None:
        x = np.arange(0, self.n_features, 1)
    if t is None:
        t = np.arange(0, self.n_snapshots, 1)

    # Check grid
    n_components = self.n_features // len(x)
    if not isinstance(n_components, int):
        raise AssertionError('Incompatible grid encountered.')

    # Define indices iterable
    if indices is None:
        indices = list(range(self.n_modes))
    elif isinstance(indices, int):
        indices = [indices]

    # Filter out bad indices
    indices = [i for i in indices if 0 <= i < self.n_modes]
    if any([not 0 <= ind < self.n_modes for ind in indices]):
        raise AssertionError('Invalid mode index encountered.')

    # Define components iterable
    if components is None:
        components = list(range(n_components))
    elif isinstance(components, int):
        components = [components]
    if any([c >= n_components for c in components]):
        raise AssertionError('Invalid component encountered.')

    # Loop over indices
    for ind in indices:
        fig: Figure = plt.figure()
        fig.suptitle(f'DMD Mode {ind}\n$\omega$ = '
                     f'{self.omegas[ind].real:.2e}'
                     f'{self.omegas[ind].imag:+.2e}j')

        real_axs: List[Axes] = [fig.add_subplot(2, 2, 1),
                                fig.add_subplot(2, 2, 2)]
        imag_axs: List[Axes] = [fig.add_subplot(2, 2, 3),
                                fig.add_subplot(2, 2, 4)]

        real_axs[0].set_title('Profile')
        real_axs[1].set_title('Dynamic')
        real_axs[0].set_ylabel('Real')
        imag_axs[0].set_ylabel('Imaginary')

        # Plot modes
        mode = self.modes.T[ind]
        for c in components:
            label = f'Component {c}'
            vals = mode[c::n_components]
            real_axs[0].plot(x, vals.real, label=label)
            imag_axs[0].plot(x, vals.imag, label=label)

        # Plot dynamics
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

        if filename is not None:
            basename, ext = splitext(filename)
            plt.savefig(basename + f'_{ind}.pdf')
        else:
            plt.show()

def plot_mode_evolutions(
        self, indices: List[int] = None,
        x: ndarray = None, t: ndarray = None,
        components: List[int] = None,
        filename: str = None):
    """
    Plot the DMD mode and dynamics.

    Parameters
    ----------
    indices : list of int, default None
        The mode indices to plot.
    x : ndarray, default None
        The spatial grid the mode is defined on.
    t : ndarray, default None
        The temporal grid the dynamics are defined on.
    components : list of int, default None
        The component of the mode to plot.
        The number of components in a mode is defined
        by the integer division of the mode and the
        supplied grid. By default, all components are
        plotted.
    filename : str, default None
        If specified, the location to save the plot.
    """
    # Initialization check
    if self.modes is None:
        raise AssertionError(
            f'DMD model is not initialized. To initialize a '
            f'model, run the {self.__class__.__name__}.fit method.'
        )
    if x is None:
        x = np.arange(0, self.n_features, 1)
    if t is None:
        t = np.arange(0, self.n_snapshots, 1)

    # Check grid
    n_components = self.n_features // len(x)
    if not isinstance(n_components, int):
        raise AssertionError('Incompatible grid encountered.')

    # Format x and t into meshgrid format, if not.
    if x.ndim == t.ndim == 1:
        x, t = np.meshgrid(x, t)
    if x.ndim != 2 or t.ndim != 2:
        raise AssertionError('x, t must be a meshgrid format.')

    # Define indices iterable
    if indices is None:
        indices = list(range(self.n_modes))
    elif isinstance(indices, int):
        indices = [indices]

    # Filter out bad indices
    indices = [i for i in indices if 0 <= i < self.n_modes]
    if any([not 0 <= ind < self.n_modes for ind in indices]):
        raise AssertionError('Invalid mode index encountered.')

    # Define components iterable
    if components is None:
        components = list(range(n_components))
    elif isinstance(components, int):
        components = [components]
    if any([c >= n_components for c in components]):
        raise AssertionError('Invalid component encountered.')
    dim = int(np.ceil(np.sqrt(len(components))))

    # Loop over indices
    for ind in indices:
        fig: Figure = plt.figure()
        fig.suptitle(f'DMD Mode {ind}\n$\omega$ = '
                     f'{self.omegas[ind].real:.2e}'
                     f'{self.omegas[ind].imag:+.2e}j')

        # Plot the evolution component-wise
        mode = dmd.modes.T[ind].reshape(-1, 1)
        dynamic = dmd.dynamics[ind].reshape(1, -1)
        evolution = (mode @ dynamic).T.real
        for n, c in enumerate(components):
            ax: Axes = fig.add_subplot(dim, dim, n + 1)
            vals = evolution[:, c::n_components]
            plt.pcolormesh(x, t, vals, cmap='jet')
            plt.colorbar()
        plt.tight_layout()

        if filename is not None:
            basename, ext = splitext(basename)
            plt.savefig(basename + f'_{ind}.pdf')
        else:
            plt.show()


def plot_timestep_errors(
        self: 'DMDBase', logscale: bool = True,
        filename: str = None) -> None:
    """
    Plot the reconstruction error as a function of time step.

    Parameters
    ----------
    logscale : bool
    filename : str, default None
        If specified, the location to save the plot.
    """
    times = self.original_timesteps
    errors = self.compute_timestep_errors()

    # Setup plot
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel(r'Relative $\ell^2$ Error')
    plotter = plt.semilogy if logscale else plt.plot
    plotter(times, errors, 'r-*', label='Reconstruction Error')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        basename, ext = splitext(filename)
        plt.savefig(basename + '.pdf')
    else:
        plt.show()

def plot_error_decay(
        self: 'DMDBase', normalized: bool = True,
        logscale: bool = True,
        filename: str = None) -> None:
    """
    Plot the reconstruction errors.

    Parameters
    ----------
    normalized : bool, default True
        If True, the singular values are normalized by
        the sum of all singular values.
    logscale : bool
        Flag for plotting on a linear or log scale y-axis.
    filename : str, default None
        If specified, the location to save the plot.
    """
    from .dmd_base import compute_error_decay
    errors = compute_error_decay(self)
    spectrum = self.singular_values
    if normalized:
        spectrum /= sum(spectrum)

    plt.figure()
    plt.xlabel('# of Modes')
    plt.ylabel(r'$\ell^2$ Error')
    plotter = plt.semilogy if logscale else plt.plot
    plotter(spectrum, 'b-*', label='Singular Values')
    plotter(errors, 'r-*', label='Reconstruction Errors')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        basename, ext = splitext(filename)
        plt.savefig(basename + '.pdf')
    else:
        plt.show()
