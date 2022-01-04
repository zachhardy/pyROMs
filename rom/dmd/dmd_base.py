import numpy as np
from numpy import ndarray

from numpy.linalg import svd
from numpy.linalg import norm

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from os.path import splitext
from typing import Union, List, Iterable

from ..plotting_mixin import PlottingMixin
from pydmd.dmdbase import DMDBase as PyDMDBase


class DMDBase(PlottingMixin, PyDMDBase):
    """
    Dynamic Mode Decomposition base class inherited from PyDMD.

    Parameters
    ----------
    svd_rank : int or float, default 0
        The rank for the truncation. If 0, the method computes the
        optimal rank and uses it for truncation. If positive interger, the
        method uses the argument for the truncation. If float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`. If -1, the method does
        not compute truncation.
    tlsq_rank : int, default 0
        Rank truncation computing Total Least Square. Default is 0,
        which means no truncation.
    exact : bool, default False
        Flag to compute either exact DMD or projected DMD.
    opt : bool or int, default False
        If True, amplitudes are computed like in optimized DMD  (see
        :func:`~dmdbase.DMDBase._compute_amplitudes` for reference). If
        False, amplitudes are computed following the standard algorithm. If
        `opt` is an integer, it is used as the (temporal) index of the snapshot
        used to compute DMD modes amplitudes (following the standard algorithm).
        The reconstruction will generally be better in time instants near the
        chosen snapshot; however increasing `opt` may lead to wrong results when
        the system presents small eigenvalues. For this reason a manual
        selection of the number of eigenvalues considered for the analyisis may
        be needed (check `svd_rank`). Also setting `svd_rank` to a value between
        0 and 1 may give better results.
    rescale_mode : {'auto'}, None, or ndarray, default None
        Scale Atilde as shown in 10.1016/j.jneumeth.2015.10.010 (section 2.4)
        before computing its eigendecomposition. None means no rescaling,
        'auto' means automatic rescaling using singular values, otherwise the
        scaling factors.
    forward_backward : bool, default False
        If True, the low-rank operator is computed
        like in fbDMD (reference: https://arxiv.org/abs/1507.02264).
    sorted_eigs : {'real', 'abs'} or False, default False
         Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`.
    """

    def __init__(self,
                 svd_rank: Union[int, float] = 0,
                 tlsq_rank: int = 0,
                 exact: bool = False,
                 opt: Union[bool, int] = False,
                 rescale_mode: Union[str, None, ndarray] = None,
                 forward_backward: bool = False,
                 sorted_eigs: Union[bool, str] = False) -> None:
        PyDMDBase.__init__(self, svd_rank, tlsq_rank, exact, opt,
                           rescale_mode, forward_backward, sorted_eigs)

        self._svd_modes: ndarray = None  # svd modes
        self._svd_vals: ndarray = None  # singular values

    @property
    def opt(self) -> Union[bool, int]:
        return self._opt

    @opt.setter
    def opt(self, value: Union[bool, int]) -> None:
        self._opt = value

    @property
    def tlsq_rank(self) -> int:
        return self._tlsq_rank

    @tlsq_rank.setter
    def tlsq_rank(self, value: int) -> None:
        self._tlsq_rank = value

    @property
    def svd_rank(self) -> Union[int, float]:
        return self.operator._svd_rank

    @svd_rank.setter
    def svd_rank(self, value: Union[int, float]) -> None:
        self.operator._svd_rank = value

    @property
    def rescale_mode(self) -> Union[str, None, ndarray]:
        return self.operator._rescale_mode

    @rescale_mode.setter
    def rescale_mode(self, value: Union[str, None, ndarray]) -> None:
        self.operator._rescale_mode = value

    @property
    def exact(self) -> bool:
        return self.operator._exact

    @exact.setter
    def exact(self, value: bool) -> None:
        self.operator._exact = value

    @property
    def forward_backward(self) -> bool:
        return self.operator._forward_backward

    @forward_backward.setter
    def forward_backward(self, value: bool) -> None:
        self.operator._forward_backward = value

    @property
    def n_snapshots(self) -> int:
        return self.snapshots.shape[1]

    @property
    def n_features(self) -> int:
        return self.snapshots.shape[0]

    @property
    def n_modes(self) -> int:
        return self.modes.shape[1]

    @property
    def singular_values(self) -> ndarray:
        if self._svd_vals is None:
            if self._snapshots is not None:
                _, self._svd_vals, _ = svd(self._snapshots[:, :-1])
        return self._svd_vals

    @property
    def svd_modes(self) -> ndarray:
        return self._svd_modes

    @property
    def reconstruction_error(self) -> float:
        """
        Get the reconstruction error over the snapshots.

        Returns
        -------
        float
        """
        X: ndarray = self.snapshots
        Xdmd: ndarray = self.reconstructed_data
        return norm(X - Xdmd) / norm(X)

    @property
    def snapshot_reconstruction_errors(self) -> ndarray:
        """
        Get the reconstruction error at each snapshot.

        Returns
        -------
        ndarray (n_snapshots,)
        """
        X: ndarray = self.snapshots
        Xdmd: ndarray = self.reconstructed_data
        return norm(X-Xdmd, axis=0) / norm(X, axis=0)

    def fit(self, X: Union[ndarray, Iterable]):
        """
        Abstract method to fit the snapshots matrices.

        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(
            f'Subclass must implement abstract '
            f'method {self.__class__.__name__}.fit')

    def find_optimal_parameters(self) -> None:
        """
        Abstract method to find optimal hyper-parameters

        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(
            f'Subclass must implement abstract method '
            f'{self.__class__.__name__}.find_optimal_parameters')

    def plot_dynamics(self,
                      mode_indices: List[int] = None,
                      logscale: bool = False,
                      filename: str = None) -> None:
        """
        Plot the dynamics behaviors of the modes at the DMD timesteps.

        Parameters
        ----------
        mode_indices : List[int], default None
            The indices of the modes to plot. The default behavior
            is to plot all modes.
        logscale : bool, default False
            Flag for plotting on a logscale
        filename : str, default None
            A location to save the plot to, if specified.
        """
        # Check the inputs
        if self.modes is None:
            raise ValueError('The fit method must be performed first.')

        t = self.dmd_timesteps

        if mode_indices is None:
            mode_indices = list(range(self.n_modes))
        elif isinstance(mode_indices, int):
            mode_indices = [mode_indices]

        # Plot each mode dynamic specified
        for idx in mode_indices:
            idx += 0 if idx > 0 else self.n_modes
            dynamic: ndarray = self.dynamics[idx] / self._b[idx]
            omega = np.log(self.eigs[idx]) / self.original_time['dt']

            # Make figure
            fig: Figure = plt.figure()
            fig.suptitle(f'DMD Dynamics {idx}\n$\omega$ = '
                         f'{omega.real:.3e}'
                         f'{omega.imag:+.3g}', fontsize=12)

            # Plot the dynamics
            ax: Axes = fig.add_subplot(1, 1, 1)
            ax.set_xlabel('t', fontsize=12)
            ax.set_ylabel('Real', fontsize=12)
            plotter = ax.semilogy if logscale else ax.plot
            plotter(t, dynamic.real)

            plt.tight_layout()
            if filename is not None:
                base, ext = splitext(filename)
                plt.savefig(base + f'_{idx}.pdf')

    def print_summary(self) -> None:
        """
        Print a summary of the DMD model.
        """
        msg = '===== DMD Summary ====='
        header = '='*len(msg)
        print('\n'.join(['', header, msg, header]))
        print(f"{'# of Modes':<20}: {self.n_modes}")
        print(f"{'# of Snapshots':<20}: {self.n_snapshots}")
        print(f"{'Reconstruction Error':<20}: "
              f"{self.reconstruction_error:.3e}")
        print(f"{'Max Snapshot Error':<20}: "
              f"{np.max(self.snapshot_reconstruction_errors):.3e}")
