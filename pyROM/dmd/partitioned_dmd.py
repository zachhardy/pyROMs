# import itertools
# import numpy as np
#
# from numpy.linalg import norm
# from copy import deepcopy
#
# from .dmd import DMD
# from pydmd.dmdoperator import DMDOperator
#
#
# class PartitionedDMD:
#     """
#     Partitioned Dynamic Mode Decomposition.
#
#     This algorithm partitions the temporal domain and constructs
#     DMD models in each to create a piece-wise DMD representation.
#     """
#
#     def __init__(self, dmd, partition_points, options=None):
#         """
#         Parameters
#         ----------
#         dmd : DMD
#             A reference DMD model with set hyperparameters.
#             This is used to initialize new DMD models.
#         partition_points : list[int]
#             The snapshot indices to partition the snapshots at.
#             Each entry defines the starting snapshot index for
#             a new DMD model.
#         options : list[dict]
#             A list of DMD hyperparameter dictionaries for each
#             DMD model in the partitioning. There must be as many
#             entries as partitions.
#         """
#
#         self._snapshots: np.ndarray = None
#         self._snapshots_shape: np.ndarray = None
#
#         self._ref_dmd: DMD = dmd
#         self._dmd_list: list[DMD] = []
#
#         self._partition_points: list[int] = partition_points
#         self._options: list[dict] = options
#
#         self._original_time: dict = None
#         self._dmd_time: dict = None
#
#         # Define partition boundaries
#         pb = [0] + partition_points + [-1]
#         self._partition_bndrys: list[int] = pb
#
#         # Construct the partitions
#         self._construct_partitions()
#
#     @property
#     def n_partitions(self):
#         """
#         Return the number of partitions.
#
#         Returns
#         -------
#         int
#         """
#         return len(self._partition_points) + 1
#
#     @property
#     def snapshots(self):
#         """
#         Return the training snapshots.
#
#         Returns
#         -------
#         numpy.ndarray (n_features, n_snapshots)
#         """
#         return self._snapshots
#
#     @property
#     def n_total_snapshots(self):
#         """
#         Return the number of training snapshots.
#
#         Returns
#         -------
#         int
#         """
#         return self._snapshots.shape[1]
#
#     @property
#     def n_snapshots(self):
#         """
#         Return the number of training snapshots per partition.
#
#         Returns
#         -------
#         list[numpy.ndarray]
#         """
#         return [dmd.snapshots for dmd in self]
#
#     @property
#     def n_features(self):
#         """
#         Return the number of features per snapshots.
#
#         Returns
#         -------
#         int
#         """
#         return self._snapshots.shape[0]
#
#     @property
#     def singular_values(self):
#         """
#         Return the singular values per partition.
#
#         Returns
#         -------
#         list[numpy.ndarray]
#         """
#         return [dmd.singular_values for dmd in self]
#
#     @property
#     def modes(self):
#         """
#         Return the DMD modes per partition.
#
#         Returns
#         -------
#         list[numpy.ndarray]
#             Each numpy.ndarray is shaped according to the modes
#             of the partition's DMD model.
#         """
#         return [dmd.modes for dmd in self]
#
#     @property
#     def n_total_modes(self):
#         """
#         Return the total number of modes.
#
#         Returns
#         -------
#         int
#         """
#         return sum(self.n_modes_per_partition)
#
#     @property
#     def n_modes(self):
#         """
#         Return the number of modes in per partition.
#
#         Returns
#         -------
#         list[int]
#         """
#         return [dmd.n_modes for dmd in self]
#
#     @property
#     def dynamics(self):
#         """
#         Return the time evolution of each mode per partition.
#
#         Returns
#         -------
#         list[numpy.ndarray]
#         """
#         return [dmd.dynamics for dmd in self]
#
#     @property
#     def Atilde(self):
#         """
#         Return the reduced Koopman operator per partition.
#
#         Returns
#         -------
#         list[numpy.ndarray]
#         """
#         return [dmd.Atilde for dmd in self]
#
#     @property
#     def operator(self):
#         """
#         Return the instance of the DMD operator per partition.
#
#         Returns
#         -------
#         list[DMDOperator]
#         """
#         return [dmd.operator for dmd in self]
#
#     @property
#     def eigs(self):
#         """
#         Return the eigenvalues of the reduced Koopman operator per partition.
#
#         Returns
#         -------
#         list[numpy.ndarray]
#         """
#         return [dmd.eigs for dmd in self]
#
#     @property
#     def omegas(self):
#         """
#         Return the continuous time-eigenvalues per partition.
#
#         Returns
#         -------
#         list[numpy.ndarray]
#         """
#         return [dmd.omegas for dmd in self]
#
#     @property
#     def frequency(self):
#         """
#         Return the frequencies of the eigenvalues per partition.
#
#         Returns
#         -------
#         list[numpy.ndarray]
#         """
#         return [dmd.frequency for dmd in self]
#
#     @property
#     def growth_rate(self):
#         """
#         Return the growth rate of the modes per partition.
#
#         Returns
#         -------
#         list[numpy.ndarray]
#         """
#         return [dmd.growth_rate for dmd in self]
#
#     def __iter__(self):
#         """
#         Iterator over the each partition.
#
#         Returns
#         -------
#         DMD
#         """
#         return self._dmd_list.__iter__()
#
#     def __next__(self):
#         """
#         Return the next partition.
#
#         Returns
#         -------
#         DMD
#         """
#         return next(self._dmd_list)
#
#     def __getitem__(self, index):
#         """
#         Return the partition corresponding to the specified index.
#
#         Parameters
#         ----------
#         index : int
#
#         Returns
#         -------
#         DMD
#         """
#         self._check_partition_index(index)
#         return self._dmd_list[index]
#
#     def enumerate(self):
#         """
#         Alias to an enumeration over the partitions.
#
#         Returns
#         -------
#         int
#             The partition index
#         DMD
#             The partition DMD model.
#         """
#         for i in range(self.n_partitions):
#             yield i, self._dmd_list[i]
#
#     def _construct_partitions(self):
#         """
#         Construct the DMD models on each partition.
#         """
#         self._dmd_list.clear()
#         for p in range(self.n_partitions):
#             dmd = deepcopy(self._ref_dmd)
#             if self._options is not None:
#                 dmd.__init__(**self._options[p])
#             self._dmd_list.append(dmd)
#
#     def _check_partition_index(self, index):
#         """
#         Check the partition index.
#
#         Parameters
#         ----------
#         index : int
#         """
#         if index < 0 or index >= self.n_partitions:
#             msg = f"{index} is not a valid partition index."
#             raise ValueError(msg)
#
