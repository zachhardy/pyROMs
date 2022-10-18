from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .pod import POD, POD_MCI
    from .dmd import DMD

import os
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm

from typing import Union, Optional
from collections.abc import Iterable


Indices = Components = Union[int, Iterable[int]]
Grid = Union[np.ndarray, Iterable[float]]
Snapshots = Union[np.ndarray, Iterable]

SVDRank = Union[int, float]
Shape = tuple[int, ...]


class ROMBase:
    """
    Base class for reduced-order models.
    """

    def __init__(self) -> None:
        self._svd_rank: SVDRank = None

        self._snapshots: np.ndarray = None
        self._snapshots_shape: Shape = None

        self._rank: int = 0
        self._U: np.ndarray = None
        self._s: np.ndarray = None
        self._Vstar: np.ndarray = None

        self._b: np.ndarray = None

    @property
    def svd_rank(self) -> SVDRank:
        """
        Return the SVD rank.

        Returns
        -------
        int or float
        """
        return self._svd_rank

    @property
    def snapshots(self) -> np.ndarray:
        """
        Return the underlying snapshot data.

        Returns
        -------
        numpy.ndarray (n_features, n_snapshots)
        """
        return self._snapshots

    @property
    def n_snapshots(self) -> int:
        """
        Return the number of snapshots.

        Returns
        -------
        int
        """
        return self._snapshots.shape[1]

    @property
    def n_features(self) -> int:
        """
        Return the number of features per snapshot.

        Returns
        -------
        int
        """
        return self._snapshots.shape[0]

    @property
    def modes(self) -> np.ndarray:
        """
        Return the modes of the ROM.

        Returns
        -------
        numpy.ndarray (n_features, n_modes)
        """
        raise NotImplementedError

    @property
    def n_modes(self) -> int:
        """
        Return the number of modes in the ROM.

        Returns
        -------
        int
        """
        return self.modes.shape[1]

    @property
    def amplitudes(self) -> np.ndarray:
        """
        Return the mode amplitudes.

        Returns
        -------
        numpy.ndarray (n_modes,) or (n_modes, *)
        """
        return self._b

    @property
    def reconstructed_data(self) -> np.ndarray:
        """
        Return the reconstructed training data.

        Returns
        -------
        numpy.ndarray (n_features, n_snapshots)
        """
        raise NotImplementedError

    @property
    def reconstruction_error(self) -> float:
        """
        Return the relative \f$ \ell_2 \f$ reconstruction error.

        Returns
        -------
        float
        """
        X, X_rom = self._snapshots, self.reconstructed_data
        return norm(X - X_rom) / norm(X)

    @property
    def snapshot_errors(self) -> np.ndarray:
        """
        Return the snapshot-wise reconstruction error.

        Returns
        -------
        numpy.ndarray (n_snapshots,)
        """
        X, X_rom = self._snapshots, self.reconstructed_data
        return norm(X - X_rom, axis=0) / norm(X, axis=0)

    @property
    def left_singular_vectors(self) -> np.ndarray:
        """
        Return teh left singular vectors.

        Returns
        -------
        numpy.ndarray (n_features, n_snapshots)
        """
        return self._U

    @property
    def right_singular_vectors(self) -> np.ndarray:
        """
        Return the right singular vectors.

        Returns
        -------
        numpy.ndarray (n_snapshots, n_snapshots)
        """
        return self._Vstar.conj().T

    @property
    def singular_values(self) -> np.ndarray:
        """
        Return the singular values.

        Returns
        -------
        numpy.ndarray (n_snapshots,)
        """
        return self._s

    def _compute_rank(self) -> int:
        """
        Return the POD rank given the singular values and SVD rank.

        Returns
        -------
        int
        """
        # Optimal rank
        if self._svd_rank == 0:
            def omega(x):
                return 0.56 * x ** 3 - 0.95 * x ** 2 + 1.82 * x + 1.43

            beta = np.divide(*sorted(self.snapshots.shape))
            tau = np.median(self._s) * omega(beta)
            return np.sum(self._s > tau)

        # Energy truncation
        elif 0.5 < self._svd_rank < 1:
            s = self._s
            cumulative_energy = np.cumsum(s ** 2 / np.sum(s ** 2))
            return np.searchsorted(cumulative_energy, self._svd_rank) + 1

        elif 0.0 < self._svd_rank <= 0.5:
            s_rel = self._s / max(self._s)
            return len(s_rel[s_rel > self._svd_rank])

        # Fixed rank
        elif self._svd_rank >= 1 and isinstance(self._svd_rank, int):
            return min(self._svd_rank, self._U.shape[1])

        # Full rank
        else:
            return self.snapshots.shape[1]

    @staticmethod
    def _format_2darray(X: Snapshots) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        Private method which formats the training snapshots appropriately
        for an SVD. If the data is already 2D, the original data is returned.
        Otherwise, the data is reshaped into a 2D numpy ndarray with
        column-wise snapshots. When this is done, the reformatted data and
        original snapshot shape is returned.

        Parameters
        ----------
        X : numpy.ndarray or Iterable
            The training data.

        Returns
        -------
        numpy.ndarray (n_features, n_snapshots)
            The formatted 2D training data
        tuple[int, int]
            The original input shape.

        """
        if isinstance(X, np.ndarray) and X.ndim == 2:
            snapshots = X
            snapshots_shape = None
        else:

            input_shapes = [np.asarray(x).shape for x in X]

            if len(set(input_shapes)) != 1:
                raise ValueError("Snapshots have not the same dimension.")

            snapshots_shape = input_shapes[0]
            snapshots = np.transpose([np.asarray(x).flatten() for x in X])
        return snapshots, snapshots_shape

    def plot_singular_values(
            self,
            normalized: bool = True,
            logscale: bool = True,
            show_rank: bool = False,
            filename: Optional[str] = None
    ) -> None:
        """
        Plot the singular values.

        Parameters
        ----------
        normalized : bool, default True
            Flag for normalizing the spectrum to its max value.
        logscale : bool, default True
            Flag for a log scale on the y-axis.
        show_rank : bool, default False
            Flag for showing the truncation location.
        filename : str, default None.
            A location to save the plot to, if specified.
        """

        # Get the singular values
        s = self.singular_values
        s = s / sum(s) if normalized else s

        # Set up the figure
        plt.figure()
        plt.xlabel("n")
        plt.ylabel("Singular Value" if not normalized else
                   "Relative Singular Value")

        # Plot the singular values
        plotter = plt.semilogy if logscale else plt.plot
        plotter(s, '-*b')
        if show_rank:
            s_last = s[len(self.modes[0]) - 1]
            plt.axhline(s_last, color='r',
                        xmin=0, xmax=len(self.snapshots[0]) - 1)
        plt.grid(True)
        plt.tight_layout()

        if filename is not None:
            base, ext = os.path.splitext(filename)
            plt.savefig(f"{base}.pdf")

    def plot_modes_1d(
            self,
            mode_indices: Optional[Indices] = None,
            components: Optional[Indices] = None,
            x: Optional[Grid] = None,
            filename: Optional[str] = None
    ) -> None:
        """
        Plot 1D modes.

        Parameters
        ----------
        mode_indices : int or Iterable[int], default None
            The indices of the modes to plot. The default behavior
            is to plot all modes.
        components : int or Iterable[int], default None
            The components of the modes to plot. The default behavior
            is to plot all components.
        x : numpy.ndarray or Iterable[float], default None
            The grid the modes are defined on. The default behaviors
            is a grid from 0 to n_features - 1.
        filename : str, default None
            A location to save the plot to, if specified.
        """

        ##################################################
        # Check inputs
        ##################################################

        if self.modes is None:
            cls_name = self.__class__.__name__
            msg = f"{cls_name} ROM has not been fit to data."
            raise ValueError(msg)

        if x is None:
            x = np.arange(0, self.snapshots.shape[0], 1)

        n_components = self.snapshots.shape[0] // len(x)
        if not isinstance(n_components, int):
            msg = "The length of the modes must be an integer factor " \
                  "of the length of the grid."
            raise AssertionError(msg)

        if mode_indices is None:
            mode_indices = list(range(len(self.modes[0])))
        elif isinstance(mode_indices, int):
            mode_indices = [mode_indices]
        else:
            for idx in mode_indices:
                if idx < 0 or idx >= len(self.modes[0]):
                    msg = "Invalid mode index encountered."
                    raise ValueError(msg)

        if components is None:
            components = list(range(n_components))
        elif isinstance(components, int):
            components = [components]
        else:
            for c in components:
                if c < 0 or c >= n_components:
                    msg = "Invalid component index encountered."
                    raise ValueError(msg)

        ##################################################
        # Plot the modes
        ##################################################

        for idx in mode_indices:
            mode = self.modes.T[idx].real

            plt.figure()
            plt.title(f"Mode {idx}")
            plt.xlabel("z")
            for c in components:
                start, end = c * len(x), (c + 1) * len(x)
                plt.plot(x, mode[start:end], label=f"Component {c}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if filename is not None:
                base, ext = os.path.splitext(filename)
                plt.savefig(f"{base}_{idx}.pdf")

    def plot_snapshots_1d(
            self,
            snapshot_indices: Optional[Indices] = None,
            components: Optional[Indices] = None,
            x: Optional[Grid] = None,
            filename: Optional[str] = None
    ) -> None:
        """
        Plot 1D snapshots.

        Parameters
        ----------
        snapshot_indices : int or Iterable[int], default None
            The indices of the snapshots to plot. The default behavior
            is to plot all modes.
        components : int or Iterable[int], default None
            The components of the modes to plot. The default behavior
            is to plot all components.
        x : numpy.ndarray or Iterable[float], default None
            The grid the modes are defined on.
        filename : str, default None
            A location to save the plot to, if specified.
        """

        ##################################################
        # Check the inputs
        ##################################################

        if self._snapshots is None:
            cls_name = self.__class__.__name__
            msg = f"No input snapshots attached to {cls_name}."
            raise ValueError(msg)

        if x is None:
            x = np.arange(0, self.n_features, 1)

        n_components = self.n_features // len(x)
        if not isinstance(n_components, int):
            msg = "The length of the snapshots must be an integer factor " \
                  "of the length of the grid."
            raise AssertionError(msg)

        if snapshot_indices is None:
            snapshot_indices = list(range(self.n_snapshots))
        elif isinstance(snapshot_indices, int):
            snapshot_indices = [snapshot_indices]
        else:
            for idx in snapshot_indices:
                if idx < 0 or idx >= self.n_snapshots:
                    msg = "Invalid snapshot index encountered."
                    raise ValueError(msg)

        if components is None:
            components = list(range(n_components))
        elif isinstance(components, int):
            components = [components]
        else:
            for c in components:
                if c < 0 or c >= n_components:
                    msg = "Invalid component index encountered."
                    raise ValueError(msg)

        ##################################################
        # Plot the snapshots
        ##################################################

        for idx in snapshot_indices:
            snapshot = self._snapshots.T[idx].real

            # Plot each snapshot
            plt.figure()
            plt.title(f"Snapshot {idx}")
            plt.xlabel("z")
            for c in components:
                start, end = c * len(x), (c + 1) * len(x)
                plt.plot(x, snapshot[start:end], label=f"Component {c}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if filename is not None:
                base, ext = os.path.splitext(filename)
                plt.savefig(f"{base}_{idx}.pdf")

    def plot_modes_2d(
            self,
            mode_indices: Optional[Indices] = None,
            components: Optional[Indices] = None,
            x: Optional[Grid] = None,
            y: Optional[Grid] = None,
            filename: Optional[str] = None
    ) -> None:
        """
        Plot 2D modes.

        Parameters
        ----------
        mode_indices : int or Iterable[int], default None
            The indices of the modes to plot. The default behavior
            is to plot all modes.
        components : int or Iterable[int], default None
            The components of the modes to plot. The default behavior
            is to plot all components.
        x, y : numpy.ndarray or Iterable[float], default None
            The x,y-nodes the grid is defined on. The default behavior uses
            the stored dimensions in `_snapshots_shape`.
        filename : str, default None
            A location to save the plot to, if specified.
        """

        ##################################################
        # Check inputs
        ##################################################

        if self.modes is None:
            cls_name = self.__class__.__name__
            msg = f"{cls_name} ROM has not been fit to data."
            raise ValueError(msg)

        if x is None and y is None:
            if self._snapshots_shape is None:
                msg = "No information about the snapshot shape."
                raise AssertionError(msg)
            elif len(self._snapshots_shape) != 2:
                msg = "The dimension of the snapshots is not 2D."
                raise AssertionError(msg)
            else:
                x = np.arange(self._snapshots_shape[0])
                y = np.arange(self._snapshots_shape[1])
        X, Y = np.meshgrid(np.unique(x), np.unique(y))
        n_pts = len(x) * len(y)

        n_components = self.n_features // n_pts
        if not isinstance(n_components, int):
            msg = "The length of the modes must be an integer factor " \
                  "of the length of the grid."
            raise AssertionError(msg)

        if mode_indices is None:
            mode_indices = list(range(self.n_modes))
        elif isinstance(mode_indices, int):
            mode_indices = [mode_indices]
        else:
            for idx in mode_indices:
                if idx < 0 or idx >= self.n_modes:
                    msg = "Invalid mode index encountered."
                    raise ValueError(msg)

        if components is None:
            components = list(range(n_components))
        elif isinstance(components, int):
            components = [components]
        else:
            for c in components:
                if c < 0 or c >= n_components:
                    msg = "Invalid component index encountered."
                    raise ValueError(msg)

        ##################################################
        # Plot the modes
        ##################################################

        for idx in mode_indices:
            mode = self.modes.T[idx].real

            # Plot each component separately
            for i, c in enumerate(components):

                plt.figure()
                plt.title(f"Mode {idx}, Component {c}")
                plt.xlabel("X")
                plt.ylabel("Y")

                start, end = c * n_pts, (c + 1) * n_pts
                im = plt.pcolor(X, Y, mode[start:end].reshape(X.shape),
                                cmap='jet', shading='auto',
                                vmin=mode.min(), vmax=mode.max())
                plt.colorbar(im)
                plt.tight_layout()

                if filename is not None:
                    base, ext = os.path.splitext(filename)
                    plt.savefig(f"{base}_{idx}_{c}.pdf")

    def plot_snapshots_2d(
            self,
            snapshot_indices: Optional[Indices] = None,
            components: Optional[Indices] = None,
            x: Optional[Grid] = None,
            y: Optional[Grid] = None,
            filename: str = None
    ) -> None:
        """
        Plot 2D snapshots.

        Parameters
        ----------
        snapshot_indices : list[int], default None
            The indices of the modes to plot. The default behavior
            is to plot all modes.
        components : list[int], default None
            The components of the modes to plot. The default behavior
            is to plot all components.
        x, y : numpy.ndarray or Iterable[float], default None
            The x,y-nodes the grid is defined on. The default behavior uses
            the stored dimensions in `_snapshots_shape`.
        filename : str, default None
            A location to save the plot to, if specified.
        """

        ##################################################
        # Check inputs
        ##################################################

        if self._snapshots is None:
            cls_name = self.__class__.__name__
            msg = f"No input snapshots attached to {cls_name}."
            raise ValueError(msg)

        if x is None and y is None:
            if self._snapshots_shape is None:
                msg = "No information about the snapshot shape."
                raise AssertionError(msg)
            elif len(self._snapshots_shape) != 2:
                msg = "The dimension of the snapshots is not 2D."
                raise AssertionError(msg)
            else:
                x = np.arange(self._snapshots_shape[0])
                y = np.arange(self._snapshots_shape[1])
        X, Y = np.meshgrid(np.unique(x), np.unique(y))

        n_components = self.n_features // (len(x) * len(y))
        if not isinstance(n_components, int):
            msg = "The length of the modes must be an integer factor " \
                  "of the length of the grid."
            raise AssertionError(msg)

        if snapshot_indices is None:
            snapshot_indices = list(range(self.n_snapshots))
        elif isinstance(snapshot_indices, int):
            snapshot_indices = [snapshot_indices]
        else:
            for idx in snapshot_indices:
                if idx < 0 or idx >= self.n_snapshots:
                    msg = "Invalid snapshot index encountered."
                    raise ValueError(msg)

        if components is None:
            components = list(range(n_components))
        elif isinstance(components, int):
            components = [components]
        else:
            for c in components:
                if c < 0 or c >= n_components:
                    msg = "Invalid component index encountered."
                    raise ValueError(msg)

        ##################################################
        # Plot the snapshots
        ##################################################

        for idx in snapshot_indices:
            snapshot = self.snapshots.T[idx].real

            # Plot each component separately
            for i, c in enumerate(components):

                plt.figure()
                plt.title(f"Mode {idx}, Component {c}")
                plt.xlabel("X")
                plt.ylabel("Y")

                start, end = c * n_pts, (c + 1) * n_pts
                im = plt.pcolor(X, Y, snapshot[start:end].reshape(X.shape),
                                cmap='jet', shading='auto',
                                vmin=snapshot.min(), vmax=snapshot.max())
                plt.colorbar(im)
                plt.tight_layout()

                if filename is not None:
                    base, ext = os.path.splitext(filename)
                    plt.savefig(f"{base}_{idx}_{c}.pdf")
