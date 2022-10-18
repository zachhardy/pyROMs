import itertools
import numpy as np

from numpy.linalg import norm
from copy import deepcopy

from typing import Union, Optional
from collections.abc import Iterable, Iterator

from .dmd import DMD

SVDRank = Union[int, float]
Shape = tuple[int, ...]


class PartitionedDMD:
    """
    Implementation of partitioned dynamic mode decomposition.

    This algorithm partitions the temporal domain and constructs
    DMD models in each to create a piece-wise DMD representation.
    """

    def __init__(
            self,
            dmd: DMD,
            partition_points: Iterable[int],
            options: Optional[Iterable[dict]] = None
    ) -> None:
        """
        Parameters
        ----------
        dmd : DMD
            An instance of a DMD object to be used to initialize
            each partition.
        partition_points : Iterable[int]
            The snapshot indices where the temporal domain is
            partitioned. The specified indices denote the starting
            snapshot for each partition after that starting on the
            first snapshot.
        options : list[dict], default None
            A list of hyper-parameters for each sub-model. If None,
            those from the specified DMD instance are used.
        """

        self._snapshots: np.ndarray = None
        self._snapshots_shape: Shape = None

        self._dmd: DMD = dmd
        self._dmd_list: list[DMD] = []
        self._dmd_options: list[dict] = options

        # Define partition boundaries
        pb = [0] + partition_points + [-1]
        self._partition_bndrys: list[int] = pb

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
        Return the an iterator to the next sub-model.

        Returns
        -------
        Iterator[DMD]
        """
        return next(self._dmd_list)

    def __getitem__(self, partition: int) -> DMD:
        """
        Return the sub-model for the specified partition.

        Parameters
        ----------
        partition : int
            The partition index.

        Returns
        -------
        DMD
        """
        return self._dmd_list[partition]

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
        pb = self._partition_bndrys
        return [pb[i] for i in range(len(pb) - 1)]

    @property
    def n_partitions(self) -> int:
        """
        Return the number of partitions.

        Returns
        -------
        int
        """
        return len(self._dmd_list)

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
        return sum(self.n_snapshots)

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

        To accomodate for non-uniform time steps across partitions,
        each subsequent partition begins where the previous ends.
        Because DMD amplitudes are often computed via a fit to the
        first snapshot, for a snapshot which corresponds to a partition
        point, the representation from the partition in which that
        snapshot is the first snapshot is used and the other discarded.

        Returns
        -------
        numpy.ndarray (n_features, n_total_snapshots)
        """
        X = self[0].reconstructed_data[:, :-1]
        for p in range(1, self.n_partitions):
            Xp = dmd.reconstructed_data
            Xp = Xp if p == self.n_partitions - 1 else Xp[:, :-1]
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
        return [dmd.singular_values for dmd in self]\

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
        return self[self._check_index(index)].modes

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
        return self[self._check_index(index)].dynamics

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
        return self[self._check_index(index)].reconstructed_data

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
        return self[self._check_index(index)].reconstruction_error

    def partial_time_interval(self, index: int) -> dict:
        """
        Return the dictionary containing the time information for
        the specified partition.

        Parameters
        ----------
        index : int

        Returns
        -------
        dict
        """
        self._check_index(index)
        if index == 0:
            return self[index].original_time
        else:
            raise NotImplementedError



    def _check_index(self, index: int) -> int:
        """
        Check that the partition index is valid.

        Parameters
        ----------
        index : int

        Returns
        -------
        int
        """
        if index < 0 or index >= self.n_partitions:
            raise ValueError("Invalid partition index.")
        return index
