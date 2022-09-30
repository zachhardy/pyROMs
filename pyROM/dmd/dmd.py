# import warnings
# import matplotlib.pyplot as plt
#
#
# from numpy.linalg import norm
# from numpy.linalg import svd
#
# from ..rom_base import ROMBase
# from pydmd.dmd import DMD as PyDMD
# from pydmd.dmdoperator import DMDOperator
#
# warnings.filterwarnings('ignore')
#
#
# class DMD(ROMBase):
#     """
#     Implementation of standard dynamic mode decomposition.
#     """
#
#     plt.rcParams['text.usetex'] = True
#     plt.rcParams['font.size'] = 12
#
#     def __init__(self,
#                  svd_rank=-1,
#                  exact=False,
#                  opt=False,
#                  sorted_eigs=False,
#                  **kwargs):
#         """
#         Parameters
#         ----------
#         svd_rank : int or float, default 0
#             The SVD rank to use for truncation. If a positive integer,
#             the minimum of this number and the maximum possible rank
#             is used. If a float between 0 and 1, the minimum rank to
#             achieve the specified energy content is used. If -1, no
#             truncation is performed.
#         exact : bool, default False
#             A flag for using either exact or projected DMD modes.
#         opt : bool or int, default False
#             If True, amplitudes are computed according to the algorithm
#             for optimized DMD which performs a fit over the whole dataset
#             instead of to a particular snapshot. If False, the standard
#             algorithm is employed and a fit to the initial condition is
#             employed. If an integer, a fit to the snapshot with that index
#             is performed. The user is cautioned when using integer inputs
#             for this argument.
#         sorted_eigs : {'real', 'abs'} or False, default False
#             Sort the eigenvalues (and mode/dynamics) by magnitude if 'abs',
#             by real part (and then imaginary) if 'real'.
#         kwargs : varies
#             Other options within PyDMD.
#         """
#
#         self._dmd = PyDMD(svd_rank=svd_rank,
#                           exact=exact,
#                           opt=opt,
#                           sorted_eigs=sorted_eigs,
#                           **kwargs)
#
#         self._Sigma: np.ndarray = None
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
#         return self._dmd.snapshots
#
#     @property
#     def n_snapshots(self):
#         """
#         Return the number of snapshots.
#
#         Returns
#         -------
#         int
#         """
#         return self.snapshots.shape[0]
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
#         return self.snapshots.shape[1]
#
#     @property
#     def singular_values(self):
#         """
#         Return the singular values.
#
#         Returns
#         -------
#         numpy.ndarray (n_snapshots,)
#         """
#         return self._Sigma
#
#     @property
#     def modes(self):
#         """
#         Return the DMD modes.
#
#         Returns
#         -------
#         numpy.ndarray (n_features, n_modes)
#         """
#         return self._dmd.modes
#
#     @property
#     def n_modes(self):
#         """
#         Return the number of DMD modes.
#
#         Returns
#         -------
#         int
#         """
#         return self._dmd.modes.shape[1]
#
#     @property
#     def dynamics(self):
#         """
#         Return the time evolution of each mode.
#
#         Returns
#         -------
#         numpy.ndarray (n_modes, n_snapshots)
#         """
#         return self._dmd.dynamics
#
#     @property
#     def Atilde(self):
#         """
#         Return the reduced Koopman operator.
#
#         Returns
#         -------
#         numpy.ndarray (n_modes, n_modes)
#         """
#         return self._dmd.atilde
#
#     @property
#     def operator(self):
#         """
#         Return the instance of the DMD operator.
#
#         Returns
#         -------
#         DMDOperator
#         """
#         return self._dmd.operator
#
#     @property
#     def eigs(self):
#         """
#         Return the eigenvalues of the reduced Koopman operator.
#
#         Returns
#         -------
#         numpy.ndarray (n_modes,)
#         """
#         return self._dmd.eigs
#
#     @property
#     def omegas(self):
#         """
#         Return the continuous time-eigenvalues.
#
#         Returns
#         -------
#         numpy.ndarray (n_modes,)
#         """
#         omegas = np.log(self.eigs) / self.original_time["dt"]
#         for i in range(len(omegas)):
#             if omegas[i].imag % np.pi < 1.0e-12:
#                 omegas[i] = omegas[i].real + 0.0j
#         return omegas
#
#     @property
#     def frequency(self):
#         """
#         Return the frequencies of the eigenvalues.
#
#         Returns
#         -------
#         numpy.ndarray (n_modes,)
#         """
#         return self._dmd.frequency
#
#     @property
#     def growth_rate(self):
#         """
#         Return the growth rate of the modes.
#
#         Returns
#         -------
#         numpy.ndarray (n_modes,)
#         """
#         return self._dmd.growth_rate
#
#     @property
#     def reconstructed_data(self):
#         """
#         Return the reconstructed training data.
#
#         Returns
#         -------
#         numpy.ndarray (n_features, n_snapshots)
#         """
#         return self._dmd.reconstructed_data
#
#     @property
#     def reconstruction_error(self):
#         """
#         Return the reconstruction error over all snapshots.
#
#         Returns
#         -------
#         float
#         """
#         X = self.snapshots
#         X_dmd = self.reconstructed_data
#         return norm(X - X_dmd) / norm(X)
#
#     @property
#     def reconstruction_error_per_snapshot(self):
#         """
#         Return the snapshot-wise reconstruction error.
#
#         Returns
#         -------
#         numpy.ndarray (n_snapshots,)
#         """
#         X = self.snapshots
#         X_dmd = self.reconstructed_data
#         return norm(X - X_dmd, axis=0) / norm(X, axis=0)
#
#     @property
#     def svd_rank(self):
#         """
#         Return the singular value decomposition rank used.
#
#         Returns
#         -------
#         float or int
#         """
#         return self._dmd.svd_rank
#
#     @property
#     def opt(self):
#         """
#         Return the optimized DMD flag.
#
#         Returns
#         -------
#         bool or int
#         """
#         return self._dmd.opt
#
#     @property
#     def exact(self):
#         """
#         Return the flag for exact or projected DMD modes.
#
#         Returns
#         -------
#         bool
#         """
#         self._dmd.exact
#
#     @property
#     def original_time(self):
#         """
#         Return the snapshot time dictionary.
#
#         Returns
#         -------
#         dict
#         """
#         return self._dmd.original_time
#
#     @property
#     def original_timesteps(self):
#         """
#         Get the timesteps of the original snapshots.
#
#         Returns
#         -------
#         numpy.ndarray (n_snapshots,)
#         """
#         return self._dmd.original_timesteps
#
#     @property
#     def dmd_time(self):
#         """
#         Return the DMD time dictionary.
#
#         Returns
#         -------
#         dict
#         """
#         return self._dmd.dmd_time
#
#     @dmd_time.setter
#     def dmd_time(self, value):
#         """
#         Set the DMD time dictionary.
#
#         Parameters
#         ----------
#         value : dict
#         """
#         if not isinstance(value, dict):
#             msg = "dmd_time must be set with a dictionary."
#             raise TypeError(msg)
#         if "t0" not in value.keys():
#             msg = "No t0 attribute found in time dictionary."
#             raise KeyError(msg)
#         if "tend" not in value.keys():
#             msg = "No tend attribute found in time dictionary."
#             raise KeyError(msg)
#         if "dt" not in value.keys():
#             msg = "No dt attribute found in time dictionary."
#             raise KeyError(msg)
#         self._dmd.dmd_time = value
#
#     @property
#     def dmd_timesteps(self):
#         """
#         Get the timesteps of the desired DMD reconstructions.
#
#         Returns
#         -------
#         numpy.ndarray (varies,)
#         """
#         return self._dmd.dmd_timesteps
#
#     def fit(self, X):
#         """
#         Fit the standard DMD model to the provided data.
#
#         Parameters
#         ----------
#         X : numpy.ndarray or list[numpy.ndarray]
#
#         Returns
#         -------
#         DMD
#         """
#         X, Xshape = self._dmd._col_major_2darray(X)
#         _, self._Sigma, _ = svd(X[:, :-1], full_matrices=False)
#         self._dmd.fit(X)
#
#         for m in range(len(self._dmd._b)):
#             if self._dmd._b[m].real < 0.0:
#                 self._dmd._b[m] *= -1.0
#                 self._dmd.operator._modes[:, m] *= -1.0
#         return self
#
#     def plot_dynamics(self,
#                       mode_indices=None,
#                       logscale=False,
#                       filenam=None):
#         """
#         Plot the dynamics behaviors of the modes at the DMD time steps.
#
#         Parameters
#         ----------
#         mode_indices : list[int], default None
#             The indices of the modes to plot. The default behavior
#             is to plot all modes.
#         logscale : bool, default False
#             Flag for plotting on a logscale
#         filename : str, default None
#             A location to save the plot to, if specified.
#         """
#         # Check the inputs
#         if self.modes is None:
#             msg = "The DMD model has not been fit."
#             raise AssertionError(msg)
#
#         t = self.dmd_timesteps
#
#         if mode_indices is None:
#             mode_indices = list(range(self.n_modes))
#         elif isinstance(mode_indices, int):
#             mode_indices = [mode_indices]
#         else:
#             for idx in mode_indices:
#                 if idx < 0 or idx >= self.n_modes:
#                     msg = "Invalid mode index encountered."
#                     raise ValueError(msg)
#
#         # Plot each mode dynamic specified
#         for idx in mode_indices:
#             dynamic = self.dynamics[idx] / self._b[idx]
#             omega = np.log(self.eigs[idx]) / self.original_time['dt']
#
#             plt.figure()
#             plt.suptitle(f"DMD Dynamics {idx}\n$\omega$ = "
#                          f"{omega.real:.3g}{omega.imag:+.3g}")
#             plt.xlabel("Time")
#             plt.ylabel("Real")
#
#             plotter = ax.semilogy if logscale else ax.plot
#             plotter(t, dynamic.real)
#             plt.grid(True)
#             plt.tight_layout()
#
#             if filename is not None:
#                 base, ext = splitext(filename)
#                 plt.savefig(f"{base}_{idx}.pdf")
