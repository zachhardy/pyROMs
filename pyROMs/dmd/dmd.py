import itertools

import numpy as np
from numpy.linalg import svd
from numpy import ndarray
from typing import Union, Iterable

from .dmd_base import DMDBase
from ..utils import compute_rank, _row_major_2darray


class DMD(DMDBase):
    """
    Dynamic Mode Decomposition derived from PyDMD

    Parameters
    ----------
    svd_rank : int or float, default 0
        The rank for the truncation. If 0, the method computes the
        optimal rank and uses it for truncation. If positive interger, the
        method uses the argument for the truncation. If float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`. If -1, the method does
        not compute truncation.
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
    sorted_eigs : {'real', 'abs'} or None, default None
        Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`.
    """

    def __init__(self,
                 svd_rank: Union[int, float] = 0,
                 exact: bool = False,
                 opt: Union[bool, int] = False,
                 sorted_eigs: Union[bool, str] = None) -> None:
        super().__init__(svd_rank, exact, opt, sorted_eigs)

    def fit(self, X: Union[ndarray, Iterable],
            svd_rank: Union[int, float] = None) -> 'DMD':
        """
        Fit the DMD model to input snpshots X.

        Parameters
        ----------
        X : ndarray or iterable
            The input snapshots.
        """
        if svd_rank is not None:
            self.svd_rank = svd_rank

        # Format the data
        X, orig_shape = _row_major_2darray(X)
        do_svd = not np.array_equal(X, self._snapshots)
        self._snapshots = X
        self._snapshots_shape = orig_shape

        # Define default time steps
        self.original_time = {'t0': 0, 'tend': self.n_snapshots - 1, 'dt': 1}
        self.dmd_time = self.original_time.copy()

        # Form submatrices
        X = np.transpose(self._snapshots[:-1])
        Y = np.transpose(self._snapshots[1:])

        # Compute SVD
        if do_svd:
            U, s, V = np.linalg.svd(X, full_matrices=False)
            V = np.transpose(np.conj(V))
            self._U, self._s, self._V = U, s, V

        # Compute rank, truncate
        args = (X, self._U, self._s)
        rank = compute_rank(self.svd_rank, *args)
        U = self._U[:, :rank]
        s = self._s[:rank]
        V = self._V[:, :rank]

        # Compute low-dimension structures
        self._Atilde = self._compute_Atilde(U, s, V, Y)
        self._decompose_Atilde()

        # Compute high-dimensional modes
        self._compute_modes(U, s, V, Y)

        # Compute amplitudes
        self._b = self._compute_amplitudes()

        return self

    def find_optimal_parameters(self) -> None:
        """
        Perform a parameter search to find the optimal parameters to
        minimize the error of the DMD model.
        """
        rank = list(range(1, self.n_snapshots))
        exact, opt = [False, True], [False, True]
        cases = list(itertools.product(rank, exact, opt))

        # Loop over each set of parameters
        errors = []
        for rank, exact, opt in cases:
            self.svd_rank = rank
            self.exact = exact
            self.opt = opt
            self.fit(self.snapshots)
            errors.append(self.reconstruction_error)

        argmin = np.nanargmin(errors)
        self.svd_rank = cases[argmin][0]
        self.exact = cases[argmin][1]
        self.opt = cases[argmin][2]
        self.fit(self.snapshots)
