import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Union


class ROMBase:
    """
    Base class for reduced-order models.
    """

    @property
    def snapshots(self) -> np.ndarray:
        """
        Return the underlying training data.

        Returns
        -------
        numpy.ndarray (n_features, n_snapshots)
        """
        cls_name = self.__class__.__name__
        msg = f"Property not implemented in {cls_name}."
        raise NotImplementedError(msg)

    @property
    def singular_values(self) -> np.ndarray:
        """
        Return the singular values.

        Returns
        -------
        numpy.ndarray (n_snapshots,)
        """
        cls_name = self.__class__.__name__
        msg = f"Property not implemented in {cls_name}."
        raise NotImplementedError(msg)

    @property
    def modes(self) -> np.ndarray:
        """
        Return the ROM modes.

        Returns
        -------
        numpy.ndarray (n_features, n_modes)
        """
        cls_name = self.__class__.__name__
        msg = f"Property not implemented in {cls_name}."
        raise NotImplementedError(msg)

    @property
    def fitted(self) -> bool:
        """
        Return whether the ROM has been fit.

        Returns
        -------
        bool
        """
        if (self.snapshots is not None and
                self.modes is not None):
            return True
        return False

    def plot_singular_values(
            self,
            normalized: bool = True,
            logscale: bool = True,
            show_rank: bool = False,
            filename: str = None
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
            base, ext = os.splitext(filename)
            plt.savefig(f"{base}.pdf")

    def plot_modes_1d(
            self,
            mode_indices: Union[int, list[int]] = None,
            components: Union[int, list[int]] = None,
            x: Union[list[float], np.ndarray] = None,
            filename: str = None
    ) -> None:
        """
        Plot 1D modes.

        Parameters
        ----------
        mode_indices : list[int], default None
            The indices of the modes to plot. The default behavior
            is to plot all modes.
        components : list[int], default None
            The components of the modes to plot. The default behavior
            is to plot all components.
        x : numpy.ndarray or list[float], default None
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
                base, ext = os.splitext(filename)
                plt.savefig(f"{base}_{idx}.pdf")

    def plot_snapshots_1d(
            self,
            snapshot_indices: Union[int, list[int]] = None,
            components: Union[int, list[int]] = None,
            x: Union[list[float], np.ndarray] = None,
            filename: str = None
    ) -> None:
        """
        Plot 1D snapshots.

        Parameters
        ----------
        snapshot_indices : list[int], default None
            The indices of the snapshots to plot. The default behavior
            is to plot all modes.
        components : list[int], default None
            The components of the modes to plot. The default behavior
            is to plot all components.
        x : ndarray or list[float], default None
            The grid the modes are defined on.
        filename : str, default None
            A location to save the plot to, if specified.
        """

        ##################################################
        # Check the inputs
        ##################################################

        if self.snapshots is None:
            cls_name = self.__class__.__name__
            msg = f"No input snapshots attached to {cls_name}."
            raise ValueError(msg)

        if x is None:
            x = np.arange(0, self.snapshots.shape[0], 1)

        n_components = self.snapshots.shape[0] // len(x)
        if not isinstance(n_components, int):
            msg = "The length of the snapshots must be an integer factor " \
                  "of the length of the grid."
            raise AssertionError(msg)

        if snapshot_indices is None:
            snapshot_indices = list(range(len(self.snapshots)))
        elif isinstance(snapshot_indices, int):
            snapshot_indices = [snapshot_indices]
        else:
            for idx in snapshot_indices:
                if idx < 0 or idx >= len(self.snapshots):
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
                base, ext = os.splitext(filename)
                plt.savefig(f"{base}_{idx}.pdf")

    def plot_modes_2d(
            self,
            mode_indices: Union[int, list[int]] = None,
            components: Union[int, list[int]] = None,
            x: Union[list[float], np.ndarray] = None,
            y: Union[list[float], np.ndarray] = None,
            filename: str = None
    ) -> None:
        """
        Plot 2D modes.

        Parameters
        ----------
        mode_indices : list[int], default None
            The indices of the modes to plot. The default behavior
            is to plot all modes.
        components : list[int], default None
            The components of the modes to plot. The default behavior
            is to plot all components.
        x, y : ndarray or list[float], default None
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

        n_components = self.snapshots.shape[0] // n_pts
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
                if idx < 0 or idx >= len(self.modes):
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
                    base, ext = os.splitext(filename)
                    plt.savefig(f"{base}_{idx}_{c}.pdf")

    def plot_snapshots_2d(
            self,
            snapshot_indices: Union[int, list[int]] = None,
            components: Union[int, list[int]] = None,
            x: Union[list[float], np.ndarray] = None,
            y: Union[list[float], np.ndarray] = None,
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
        x, y : ndarray or list[float], default None
            The x,y-nodes the grid is defined on. The default behavior uses
            the stored dimensions in `_snapshots_shape`.
        filename : str, default None
            A location to save the plot to, if specified.
        """

        ##################################################
        # Check inputs
        ##################################################

        if self.snapshots is None:
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

        n_components = self.snapshots.shape[0] // (len(x) * len(y))
        if not isinstance(n_components, int):
            msg = "The length of the modes must be an integer factor " \
                  "of the length of the grid."
            raise AssertionError(msg)

        if snapshot_indices is None:
            snapshot_indices = list(range(len(self.snapshots)))
        elif isinstance(snapshot_indices, int):
            snapshot_indices = [snapshot_indices]
        else:
            for idx in snapshot_indices:
                if idx < 0 or idx >= len(self.snapshots):
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
                    base, ext = os.splitext(filename)
                    plt.savefig(f"{base}_{idx}_{c}.pdf")

    @staticmethod
    def _format_2darray(X: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Private method which formats the training snapshots appropriately
        for an SVD. If the data is already 2D, the original data is returned.
        Otherwise, the data is reshaped into a 2D numpy ndarray with
        column-wise snapshots. When this is done, the reformatted data and
        original snapshot shape is returned.

        Parameters
        ----------
        X : numpy.ndarray or list[numpy.ndarray]
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
