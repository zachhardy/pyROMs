import numpy as np
from numpy import ndarray
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from typing import TYPE_CHECKING, List
if TYPE_CHECKING:
    from .dmd_base import DMDBase

def plot_singular_values(
        self: 'DMDBase',
        logscale: bool = True) -> None:
    """

    Parameters
    ----------
    logscale : bool, default False
        Flag for plotting on a linear- or log-scale y-axis.
    """
    s = self._singular_values / sum(self._singular_values)

    # Setup plot
    fig: Figure = plt.figure()

    # Labels
    plt.xlabel('Singular Value #')
    plt.ylabel(r'$\sigma / \sum{{\sigma}}$')

    # Plot data
    plotter = plt.semilogy if logscale else plt.plot
    plotter(s, 'b-*', label='Singular Values')
    plt.axvline(self.n_modes - 1, color='r',
                ymin=1e-12, ymax=1.0 - 1.0e-12)

    # Postprocessing
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_1D_profiles(
        self: 'DMDBase', indices: List[int] = None,
        x: ndarray = None,
        components: List[int] = None,
        imag: bool = False) -> None:
    """
    Plot the DMD mode profiles.

    Parameters
    ----------
    indices : list of int, default None
        The mode indices to plot
    x : ndarray, default None
        The spatial grid the mode is defined on.
    components : list of int, default None
        The component of the mode to plot.
        The number of components in a mode is defined
        by the integer division of the mode and the
        supplied grid. By default, all components are
        plotted.
    imag : bool, default False
        Whether or not to plot the imaginary part.
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

    # Loop over mode indices
    for ind in indices:
        # Get the mode, normalize it
        mode = self.modes[:, ind] / norm(self.modes[:, ind])

        # Initialize the figure
        fig: Figure = plt.figure()
        fig.suptitle(f'DMD Mode {ind}\n$\omega$ = '
                     f'{self.omegas[ind].real:.2e}'
                     f'{self.omegas[ind].imag:+.2e}')

        # Setup Axes
        axs: List[Axes] = []
        if not imag:
            axs.append(fig.add_subplot(1, 1, 1))
        else:
            axs.append(fig.add_subplot(1, 2, 1))
            axs.append(fig.add_subplot(1, 2, 2))
            axs[0].set_title('Real')
            axs[1].set_title('Imaginary')

        # Loop over components
        for c in components:
            label = f'Component {c}'
            vals = mode[c::n_components]

            # Plot the mode
            axs[0].plot(x, vals.real, label=label)
            if imag:
                axs[1].plot(x, vals.imag, label=label)

        # Finalize plot
        for ax in axs:
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.show()


def plot_dynamics(
        self: 'DMDBase', indices: List[int] = None,
        t: ndarray = None,
        imag: bool = False) -> None:
    """
    Plot DMD mode dynamics.

    Parameters
    ----------
    indices : list of int, default None
        The mode indices to plot
    t : ndarray, default None
        The temporal grid the mode is defined on.
    imag : bool, default False
        Whether or not to plot the imaginary part.
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
        # Get the dynamic, normalize it
        dynamic = self.dynamics[ind] / norm(self.dynamics[ind])

        # Initialize figure
        fig: Figure = plt.figure()
        fig.suptitle(f'DMD Mode {ind}\n$\omega$ = '
                     f'{self.omegas[ind].real:.2e}'
                     f'{self.omegas[ind].imag:+.2e}')

        # Setup Axes
        axs: List[Axes] = []
        if not imag:
            axs.append(fig.add_subplot(1, 1, 1))
        else:
            axs.append(fig.add_subplot(1, 2, 1))
            axs.append(fig.add_subplot(1, 2, 2))
            axs[0].set_title('Real')
            axs[1].set_title('Imaginary')

        # Plot the dynamics
        axs[0].plot(t, dynamic.real)
        if imag:
            axs[1].plot(t, dynamic.imag)

        # Finalize plots
        for ax in axs:
            ax.grid(True)
        plt.tight_layout()
        plt.show()


def plot_1D_profiles_and_dynamics(
        self: 'DMDBase', indices: List[int] = None,
        x: ndarray = None, t: ndarray = None,
        components: List[int] = None,
        imag: bool = False) -> None:
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
    imag : bool, default False
        Whether or not to plot the imaginary part.
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
        # Get mode and dynamic, normalize them
        mode = self.modes[:, ind] / norm(self.modes[:, ind])
        dynamic = self.dynamics[ind] / norm(self.dynamics[ind])

        fig: Figure = plt.figure()
        fig.suptitle(f'DMD Mode {ind}\n$\omega$ = '
                     f'{self.omegas[ind].real:.2e}'
                     f'{self.omegas[ind].imag:+.2e}j')

        # Setup Axes
        axs: List[Axes] = []
        if not imag:
            axs.append(fig.add_subplot(1, 2, 1))
            axs.append(fig.add_subplot(1, 2, 2))
            axs[0].set_title('Profile')
            axs[1].set_title('Dynamics')
        else:
            axs.append(fig.add_subplot(2, 2, 1))
            axs.append(fig.add_subplot(2, 2, 2))
            axs.append(fig.add_subplot(2, 2, 3))
            axs.append(fig.add_subplot(2, 2, 4))
            axs[0].set_title('Profile')
            axs[1].set_title('Dynamics')
            axs[0].set_ylabel('Real')
            axs[2].set_ylabel('Imaginary')

        # Loop over components
        for c in components:
            vals = mode[c::n_components]
            label = f'Component {c}'

            # Plot the mode
            axs[0].plot(x, vals.real, label=label)
            if imag:
                axs[2].plot(x, vals.imag, label=label)

        # Plot the dynamics
        axs[1].plot(t, dynamic.real)
        if imag:
            axs[3].plot(t, dynamic.imag)

        # Finalize plots
        for i, ax in enumerate(axs):
            ax.grid(True)
            if i in [0, 2]:
                ax.legend()
        plt.tight_layout()
        plt.show()

def plot_mode_evolutions(
        self, indices: List[int] = None,
        x: ndarray = None, t: ndarray = None,
        components: List[int] = None):
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
        # Get mode and dynamics, compute evolution
        mode = self.modes[:, ind].reshape(-1, 1)
        dynamic = self.dynamics[ind].reshape(1, -1)
        evol = (mode @ dynamic).T.real

        # Initialize figure
        fig: Figure = plt.figure()
        fig.suptitle(f'DMD Mode {ind}\n$\omega$ = '
                     f'{self.omegas[ind].real:.2e}'
                     f'{self.omegas[ind].imag:+.2e}j')

        # Plot the evolution component-wise
        for n, c in enumerate(components):
            ax: Axes = fig.add_subplot(dim, dim, n + 1)
            vals = evol[:, c::n_components]
            plt.pcolormesh(x, t, vals, cmap='jet')
            plt.colorbar()
        plt.tight_layout()
        plt.show()


def plot_timestep_errors(
        self: 'DMDBase', logscale: bool = True) -> None:
    """
    Plot the reconstruction error as a function of time step.

    Parameters
    ----------
    logscale : bool
    """
    times = self.original_timesteps
    errors = self.compute_timestep_errors()

    # Setup plot
    fig: Figure = plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel(r'Relative $\ell^2$ Error')

    # Plot data
    plotter = plt.semilogy if logscale else plt.plot
    plotter(times, errors, 'r-*', label='Reconstruction Error')

    # Postprocess
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_decay(
        self: 'DMDBase', logscale: bool = True) -> None:
    """
    Plot the reconstruction errors.

    Parameters
    ----------
    logscale : bool
        Flag for plotting on a linear or log scale y-axis.
    fname : str, default None
        Filename for saving the plot.
    """
    from .dmd_base import compute_error_decay
    errors = compute_error_decay(self)
    spectrum = self._singular_values / sum(self._singular_values)

    # Setup plot
    fig: Figure = plt.figure()
    plt.xlabel('# of Modes')
    plt.ylabel(r'Relative $\ell^2$ Error')

    # Plot data
    plotter = plt.semilogy if logscale else plt.plot
    plotter(spectrum, 'b-*', label='Singular Values')
    plotter(errors, 'r-*', label='Reconstruction Errors')

    # Postprocess
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
