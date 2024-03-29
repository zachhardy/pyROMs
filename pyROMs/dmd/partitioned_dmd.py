import itertools
from copy import deepcopy

import numpy as np
from numpy import ndarray
from numpy.linalg import norm

from typing import List, Union, Tuple, Iterable

from .dmd_base import DMDBase
from ..utils import _row_major_2darray


class PartitionedDMD(DMDBase):
    """
    Partitioned Dynamic Mode Decomposition.

    This algorithm partitions the temporal domain and constructs
    DMD models in each to create a piece-wise DMD representation.

    Parameters
    ----------
    dmd : DMDBase
        An instance of a derived class of DMDBase to be used
        in each partition.

    Attributes
    ----------
    options : List[dict]
        Options for individual partitions. This must be the same
        lenght as the number of partitions. Each entry is a
        dictionary with the key-word arguments for the base
        instance. The values in the base instance will be used when
        a key-word is not provided.

    """

    def __init__(self, dmd: DMDBase,
                 partition_points: List[int],
                 options: List[tuple] = None) -> None:
        self._ref_dmd: DMDBase = dmd
        self.dmd_list: List[DMDBase] = []

        self.partition_points: List[int] = partition_points
        self.options: List[dict] = options

        self._original_time: dict = None
        self._dmd_time: dict = None

        # Define partition boundaries
        pb = [0] + partition_points + [-1]
        self._partition_bndrys: List[int] = pb

        # Build the partitions
        self._build_partitions()

    def __iter__(self):
        return self.dmd_list.__iter__()

    def __next__(self) -> 'DMDBase':
        return next(self.dmd_list)

    def __getitem__(self, item: int) -> 'DMDBase':
        return self.dmd_list[item]

    def enumerate(self):
        for i in range(self.n_partitions):
            yield i, self.dmd_list[i]

    @property
    def n_snapshots(self) -> List[int]:
        return [dmd.n_snapshots for dmd in self]

    @property
    def n_modes(self) -> List[int]:
        return [dmd.n_modes for dmd in self]

    @property
    def Atilde(self) -> List[ndarray]:
        return [dmd.Atilde for dmd in self]

    @property
    def eigvals(self) -> List[ndarray]:
        return [dmd.eigvals for dmd in self]

    @property
    def omegas(self) -> List[ndarray]:
        return [dmd.omegas for dmd in self]

    @property
    def frequency(self) -> List[ndarray]:
        return [dmd.frequency for dmd in self]

    @property
    def eigvecs(self) -> List[ndarray]:
        return [dmd.eigvecs for dmd in self]

    @property
    def modes(self) -> List[ndarray]:
        return [dmd.modes for dmd in self]

    @property
    def amplitudes(self) -> List[ndarray]:
        return [dmd.amplitudes for dmd in self]

    @property
    def dynamics(self) -> List[ndarray]:
        return [dmd.dynamics for dmd in self]

    @property
    def n_partitions(self) -> int:
        return len(self.partition_points) + 1

    @property
    def original_time(self) -> dict:
        return self._original_time

    @property
    def dmd_time(self) -> dict:
        return self._dmd_time

    @property
    def svd_rank(self) -> List[Union[int, float]]:
        return [dmd.svd_rank for dmd in self]

    @property
    def exact(self) -> List[bool]:
        return [dmd.exact for dmd in self]

    @property
    def opt(self) -> List[Union[bool, int]]:
        return [dmd.opt for dmd in self]

    @property
    def reconstructed_data(self) -> ndarray:
        """
        Get the reconstructed data.

        Returns
        -------
        ndarray (n_snapshots, n_features)
        """
        for p in range(self.n_partitions):
            Xp = self.dmd_list[p].reconstructed_data
            X = Xp if p == 0 else np.vstack((X, Xp))
        return X

    @property
    def snapshot_errors(self) -> ndarray:
        """
        Get the reconstruction error per snapshot.

        Returns
        -------
        ndarray
        """
        errors = []
        for dmd in self:
            errors.extend(dmd.snapshot_errors)
        return np.array(errors)

    def partial_modes(self, partition: int) -> ndarray:
        """
        Return the modes for the specific `partition`.

        Parameters
        ----------
        partition : int
            The partition index.

        Returns
        -------
        ndarray (n_features, n_modes[partition])
        """
        self._check_partition(partition)
        return self.dmd_list[partition].modes

    def partial_dynamics(self, partition: int) -> ndarray:
        """
        Return the dynamics for the specific `partition`.

        Parameters
        ----------
        partition : int
            The partition index.

        Returns
        -------
        ndarray (n_modes[partition], n_snapshots[partition])
        """
        self._check_partition(partition)
        return self.dmd_list[partition].dynamics

    def partial_eigs(self, partition: int) -> ndarray:
        """
        Return the eigenvalues for the specific `partition`.

        Parameters
        ----------
        partition : int
            The partition index.

        Returns
        -------
        ndarray (n_modes[partition],)
        """
        return self.dmd_list[partition].eigvals

    def partial_reconstructed_data(self, partition: int) -> ndarray:
        """
        Return the reconstructed data over the specific `partition`.

        Parameters
        ----------
         partition : int
            The partition index.

        Returns
        -------
        ndarray (n_features, n_snapshots[partition])
        """
        self._check_partition(partition)
        return self.dmd_list[partition].reconstructed_data

    def partial_reconstruction_error(self, partition: int) -> float:
        """
        Return the reconstruction error over the specified `partition`.

        Parameters
        ----------
        partition : int
            The partition index.

        Returns
        -------
        float
        """
        self._check_partition(partition)
        return self.dmd_list[partition].reconstruction_error

    def partial_time_interval(self, partition: int) -> dict:
        """
        Return a dictionary containing the time information for
        the specified `partition`.

        Parameters
        ----------
        partition : int
            The partition index.

        Returns
        -------
        dict
        """
        self._check_partition(partition)
        if partition == 0:
            return self.dmd_list[partition].original_time
        else:
            t0 = 0
            for p in range(partition):
                dmd = self.dmd_list[p]
                period = dmd.original_time['tend'] - dmd.original_time['t0']
                t0 += period + 1

            dmd = self.dmd_list[partition]
            period = dmd.original_time['tend'] - dmd.original_time['t0']
            return {'t0': t0, 'tend': t0 + period, 'dt': 1}

    def fit(self, X: Union[ndarray, Iterable]) -> 'PartitionedDMD':
        """
        Compute the Partitioned Dynamic Mode Decomposition
        to the input data.

        Parameters
        ----------
        X : ndarray or iterable
            The input snapshots
        """
        # Set the snapshots
        X, Xshape = _row_major_2darray(X)
        self._snapshots = X
        self._snapshots_shape = Xshape

        # Check partitions and options
        if self.partition_points is None:
            raise ValueError('The partition points must be set '
                             'before calling fit.')
        if any([p >= self._snapshots.shape[0] for p in self.partition_points]):
            raise ValueError('Partition indices must be less the number of '
                             'snapshots in the input data.')
        if self.options is not None:
            if len(self.options) != self.n_partitions:
                raise ValueError('If options are provided, they must be '
                                 'provided for each partition.')

        # Loop over each partition and fit the models
        start = 0
        for p in range(self.n_partitions):
            # Next partition point or last snapshot
            if p < self.n_partitions - 1:
                end = self.partition_points[p]
            else:
                end = self._snapshots.shape[0] - 1

            # Define the partitioned dataset
            Xp = self._snapshots[start:end + 1]

            # Fit the submodel
            self.dmd_list[p].fit(Xp)

            # Shift the start point
            start = end + 1

        self._original_time = {"t0": 0,
                               "tend": self._snapshots.shape[0] - 1,
                               "dt": 1}
        self._dmd_time = self.original_time.copy()
        return self

    def find_optimal_parameters(self) -> None:
        """
        Perform a parameter search to find the optimal parameters to
        minimize the error of the DMD model.
        """
        for dmd in self:
            dmd.find_optimal_parameters()

    def print_summary(self, skip_line: bool = False) -> None:
        """
        Print a summary of the DMD model.
        """
        msg = "===== DMD Summary ====="
        header = "="*len(msg)
        if skip_line:
            print()
        print("\n".join([header, msg, header]))
        print(f"{'# of Modes':<20}: {sum(self.n_modes)}")
        print(f"{'# of Snapshots':<20}: {sum(self.n_snapshots)}")
        print(f"{'Reconstruction Error':<20}: "
              f"{self.reconstruction_error:.3e}")
        print(f"{'Mean Snapshot Error':<20}: "
              f"{np.mean(self.snapshot_errors):.3e}")
        print(f"{'Max Snapshot Error':<20}: "
              f"{np.max(self.snapshot_errors):.3e}")

    def print_partition_summaries(self, skip_line: bool = False) -> None:
        labels = ["# of Modes", "# of Snapshots", "Reconstruction Errors",
                  "Mean Snapshot Errors", "Max Snapshot Errors"]

        reconstruction_errors = [dmd.reconstruction_error for dmd in self]
        mean_snapshot_errors = [np.mean(dmd.snapshot_errors) for dmd in self]
        max_snapshot_errors = [np.max(dmd.snapshot_errors) for dmd in self]
        data = [self.n_modes, self.n_snapshots, reconstruction_errors,
                mean_snapshot_errors, max_snapshot_errors]

        msg = f"===== Summary of {self.n_partitions} Partitions ====="
        header = "=" * len(msg)
        if skip_line:
            print()
        print("\n".join([header, msg, header]))
        for i in range(len(labels)):
            msg = f"{labels[i]:<20}: "
            if i < 2:
                msg += f"{data[i]}"
            else:
                msg += "["
                for p in range(self.n_partitions):
                    msg += ", " if p > 0 else ""
                    msg += f"{data[i][p]:.3e}"
                msg += "]"
            print(msg)

    def _build_partitions(self) -> None:
        """
        Construct the submodels on each partition.
        """
        self.dmd_list.clear()
        for p in range(self.n_partitions):
            dmd = deepcopy(self._ref_dmd)
            if self.options is not None:
                for key, val in self.options[p].items():
                    if hasattr(dmd, key):
                        setattr(dmd, key, val)
            self.dmd_list.append(dmd)

    def _check_partition(self, partition: int) -> None:
        """
        Check the partition index.

        Parameters
        ----------
        partition : int
            The partition index.
        """
        if partition >= self.n_partitions:
            raise ValueError(
                f'The partition parameter ({partition}) must be less than '
                f'the total number of partions ({self.n_partitions}).')
