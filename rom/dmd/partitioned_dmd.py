from copy import deepcopy

import numpy as np
from numpy import ndarray
from numpy.linalg import norm

from .dmd_base import DMDBase
from typing import List, Union, Tuple, Iterable


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
        self.options: List[tuple] = options

        # Define partition boundaries
        pb = [0] + partition_points + [-1]
        self._partition_bndrys: List[int] = pb

        # Build the partitions
        self._build_partitions()

    def __iter__(self) -> DMDBase:
        return self.dmd_list.__iter__()

    @property
    def n_partitions(self) -> int:
        return len(self.partition_points) + 1

    @property
    def svd_rank(self) -> List[Union[int, float]]:
        return [dmd.svd_rank for dmd in self]

    @property
    def tlsq_rank(self) -> List[int]:
        return [dmd.tlsq_rank for dmd in self]

    @property
    def exact(self) -> List[bool]:
        return [dmd.exact for dmd in self]

    @property
    def opt(self) -> List[Union[bool, int]]:
        return [dmd.opt for dmd in self]

    @property
    def rescale_mode(self) -> List[Union[str, None, ndarray]]:
        return [dmd.rescale_mode for dmd in self]

    @property
    def forward_backward(self) -> List[bool]:
        return [dmd.forward_backward for dmd in self]

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
        return self.dmd_list[partition].eigs

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
        return self.dmd_list[partition].reconstructed_data

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
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

        # Check partitions and options
        if self.partition_points is None:
            raise ValueError('The partition points must be set '
                             'before calling fit.')
        if any([p >= self.n_snapshots for p in self.partition_points]):
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
                end = self.n_snapshots - 1

            # Define the partitioned dataset
            Xp = self._snapshots[:, start:end + 1]

            # Fit the submodel
            self.dmd_list[p].fit(Xp)

            # Shift the start point
            start = end + 1

        return self

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
