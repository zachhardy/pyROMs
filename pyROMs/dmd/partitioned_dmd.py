import itertools
import numpy as np

from numpy.linalg import norm
from scipy.linalg import block_diag

from copy import deepcopy
from typing import Union, Callable, Optional
from collections.abc import Iterable, Iterator

from .dmd import DMD
from ..utils import format_2darray

SVDRank = Union[int, float]
Shape = tuple[int, ...]

SubModel = Union[DMD, Iterable[DMD], Callable[..., DMD]]
Snapshots = Union[np.ndarray, Iterable]


class PartitionedDMD:
    """
    Implementation of partitioned dynamic mode decomposition.

    This algorithm partitions the temporal domain and constructs
    DMD models in each to create a piece-wise DMD representation.
    """

    def __init__(
            self,
            dmd: SubModel,
            partition_points: Iterable[int],
    ) -> None:
        """
        Parameters
        ----------
        dmd : DMD, Iterable[DMD], or Callable[[int, ...], DMD]
            An instance of a DMD object to be used to initialize
            each partition.
        partition_points : Iterable[int]
            The snapshot indices where the temporal domain is
            partitioned. The specified indices denote the starting
            snapshot for each partition after that starting on the
            first snapshot.
        """

        self._snapshots: np.ndarray = None
        self._snapshots_shape: Shape = None

        self._dmd: SubModel = dmd
        self._dmd_list: list[DMD] = []

        # Define partition boundaries
        pb = partition_points + [-1]
        self._bndrys: list[int] = pb

        # Build the partitions
        self._build_partitions()

    def __iter__(self) -> Iterator[DMD]:
        """
        Return an iterator to a sub-model.

        Returns
        -------
        Iterator[DMD]
        """
        return self._dmd_list.__iter__()

    def __next__(self) -> Iterator[DMD]:
        """
        Return an iterator to the next sub-model.

        Returns
        -------
        Iterator[DMD]
        """
        return next(self._dmd_list)

    def __getitem__(self, index: int) -> DMD:
        """
        Return the sub-model for the specified partition.

        Parameters
        ----------
        index : int
            The partition index.

        Returns
        -------
        DMD
        """
        return self._dmd_list[index]

    @property
    def svd_rank(self) -> list[SVDRank]:
        """
        Return the SVD rank per partition.

        Returns
        -------
        list[int or float]
        """
        return [dmd.svd_rank for dmd in self]

    @property
    def exact(self) -> list[bool]:
        """
        Return the exact modes flag per partition.

        Returns
        -------
        list[bool]
        """
        return [dmd.exact for dmd in self]

    @property
    def opt(self) -> list[bool]:
        """
        Return the optimized amplitudes flag per partition.

        Returns
        -------
        list[bool]
        """
        return [dmd.opt for dmd in self]

    @property
    def sorted_eigs(self) -> list[str]:
        """
        Return the sorting method per partition.

        Returns
        -------
        list[str]
        """
        return [dmd.sorted_eigs for dmd in self]

    @property
    def partition_start_indices(self) -> list[int]:
        """
        Return the indices each partition starts at.

        Returns
        -------
        list[int]
        """
        pb = self._bndrys
        return [pb[i] for i in range(self.n_partitions)]

    @property
    def n_partitions(self) -> int:
        """
        Return the number of partitions.

        Returns
        -------
        int
        """
        return len(self._bndrys)

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
    def n_snapshots(self) -> list[int]:
        """
        Return the number of snapshots per partition.

        Returns
        -------
        list[int]
        """
        return [dmd.n_snapshots for dmd in self]

    @property
    def n_total_snapshots(self) -> int:
        """
        Return the total number of snapshots.

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
    def modes(self) -> list[np.ndarray]:
        """
        Return the DMD modes stored column-wise per partition.

        Returns
        -------
        list[numpy.ndarray]
        """
        return [dmd.modes for dmd in self]

    @property
    def n_modes(self) -> list[int]:
        """
        Return the number of modes per partition.

        Returns
        -------
        list[int]
        """
        return [dmd.n_modes for dmd in self]

    @property
    def n_total_modes(self) -> int:
        """
        Return the total number of modes.

        Returns
        -------
        int
        """
        return sum(self.n_modes)

    @property
    def amplitudes(self) -> list[np.ndarray]:
        """
        Return the amplitudes per partition.

        Returns
        -------
        list[numpy.ndarray]
        """
        return [dmd.amplitudes for dmd in self]

    @property
    def dynamics(self) -> list[np.ndarray]:
        """
        Return the dynamics per partition.

        Returns
        -------
        list[numpy.ndarray]
        """
        return [dmd.dynamics for dmd in self]

    @property
    def reconstructed_data(self) -> np.ndarray:
        """
        Return the reconstructed data.

        To accommodate for non-uniform time steps across partitions,
        each subsequent partition begins where the previous ends.
        Because DMD amplitudes are often computed via a fit to the
        first snapshot, for a snapshot which corresponds to a partition
        point, the representation from the partition in which that
        snapshot is the first snapshot is used and the other discarded.

        Returns
        -------
        numpy.ndarray (n_features, n_total_snapshots)
        """
        X = self[0].reconstructed_data
        for p in range(1, self.n_partitions):
            Xp = self[p].reconstructed_data
            X = np.hstack((X, Xp))
        return X

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
        Return the reconstruction error per snapshot.

        Returns
        -------
        numpy.ndarray (n_total_snapshots,)
        """
        X, X_rom = self._snapshots, self.reconstructed_data
        return norm(X - X_rom, axis=0) / norm(X, axis=0)

    @property
    def Atilde(self) -> list[np.ndarray]:
        """
        Return the low-rank evolution operator per partition.

        Returns
        -------
        list[numpy.ndarray]
        """
        return [dmd.Atilde for dmd in self]

    @property
    def eigvals(self) -> list[np.ndarray]:
        """
        Return the eigenvalues of Atilde per partition.

        Returns
        -------
        list[numpy.ndarray]
        """
        return [dmd.eigvals for dmd in self]

    @property
    def omegas(self) -> list[np.ndarray]:
        """
        Return the continuous eigenvalues of the DMD modes per partition.

        Returns
        -------
        list[numpy.ndarray]
        """
        return [dmd.omegas for dmd in self]

    @property
    def frequency(self) -> list[np.ndarray]:
        """
        Return the frequencies of the DMD mode eigenvalues per partition.

        Returns
        -------
        list[numpy.ndarray]
        """
        return [dmd.frequency for dmd in self]

    @property
    def eigvecs(self) -> list[np.ndarray]:
        """
        Return the eigenvectors of Atilde stored column-wise per partition.

        Returns
        -------
        list[numpy.ndarray]
        """
        return [dmd.eigvecs for dmd in self]

    @property
    def left_singular_vectors(self) -> list[np.ndarray]:
        """
        Return the left singular vectors per partition.

        Returns
        -------
        list[numpy.ndarray]
        """
        return [dmd.left_singular_vectors for dmd in self]

    @property
    def right_singular_vectors(self) -> list[np.ndarray]:
        """
        Return the right singular vectors per partition.

        Returns
        -------
        list[numpy.ndarray]
        """
        return [dmd.right_singular_vectors for dmd in self]

    @property
    def singular_values(self) -> list[np.ndarray]:
        """
        Return the singular values per partition.

        Returns
        -------
        list[numpy.ndarray]
        """
        return [dmd.singular_values for dmd in self]

    def partial_modes(self, index: int) -> np.ndarray:
        """
        Return the modes for the specified partition.

        Parameters
        ----------
        index : int

        Returns
        -------
        numpy.ndarray (n_features, n_modes[index])
        """
        return self[index].modes

    def partial_dynamics(self, index: int) -> np.ndarray:
        """
        Return the dynamics for the specified partition.

        Parameters
        ----------
        index : int

        Returns
        -------
        numpy.ndarray (n_modes[index], *)
        """
        return self[index].dynamics

    def partial_reconstructed_data(self, index: int) -> np.ndarray:
        """
        Return the reconstructed data for the specified partition.

        Parameters
        ----------
        index : int

        Returns
        -------
        numpy.ndarray (n_featrues, n_snapshots[index])
        """
        return self[index].reconstructed_data

    def partial_reconstruction_error(self, index: int) -> float:
        """
        Return the reconstruction error for the specified partition.

        Parameters
        ----------
        index : int

        Returns
        -------
        float
        """
        return self[index].reconstruction_error

    def partial_time_interval(self, index: int) -> dict:
        """
        Return the time dictionary for the specified partition.

        Parameters
        ----------
        index : int

        Returns
        -------
        dict
        """
        return self[index].original_time

    def fit(self, X: Snapshots) -> 'PartitionedDMD':
        """
        Fit the partitioned DMD model to the input data.

        Parameters
        ----------
        X : numpy.ndarray or Iterable
            The input snapshots.

        Returns
        -------
        PartitionedDMD
        """
        X, Xshape = format_2darray(X)

        self._snapshots = X
        self._snapshots_shape = Xshape

        # Check partitions and options
        for i in range(self.n_partitions):
            if self._bndrys[i] == -1:
                self._bndrys[i] = self.n_total_snapshots - 1
            if (self._bndrys[i] < 0 or
                    self._bndrys[i] >= self.n_total_snapshots):
                raise ValueError(f"{pt} is an invalid partition index.")

        # Loop over each partition and fit the sub-models
        start = 0
        for p in range(self.n_partitions):
            end = self._bndrys[p]

            # Fit the sub-models
            Xp = self._snapshots[:, start:end + 1]
            self[p].fit(Xp)
            # self[p].optimize_hyperparameters(True)

            # Shift the start and end points
            start = end + 1

    def _build_partitions(self) -> None:
        """
        Build the sub-models on each partition.
        """
        if isinstance(self._dmd, DMD):
            def builder(*args):
                return deepcopy(self._dmd)

        elif isinstance(self._dmd, list):
            if len(self._dmd) != self.n_partitions:
                msg = "The number of DMD sub-models does not equal " \
                      "the number of partitions."
                raise AssertionError(msg)

            def builder(index, *args):
                return deepcopy(self._dmd[index])

        elif callable(self._dmd):
            builder = self._dmd

        else:
            raise AssertionError("Invalid sub-model input.")

        self._dmd_list.clear()
        for p in range(self.n_partitions):
            self._dmd_list.append(builder(p))

    def print_summary(self, skip_line: bool = False) -> None:
        """
        Print a summary of the DMD model.
        """
        print()
        print("===================================")
        print("===== Partitioned DMD Summary =====")
        print("===================================")
        print(f"{'# of Modes':<20}: {sum(self.n_modes)}")
        print(f"{'# of Snapshots':<20}: {sum(self.n_snapshots)}")
        print(f"{'Reconstruction Error':<20}: "
              f"{self.reconstruction_error:.3e}")
        print(f"{'Mean Snapshot Error':<20}: "
              f"{np.mean(self.snapshot_errors):.3e}")
        print(f"{'Max Snapshot Error':<20}: "
              f"{np.max(self.snapshot_errors):.3e}")
