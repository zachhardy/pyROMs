import numpy as np
from numpy import ndarray
from typing import Union

from .dmd_base import DMDBase

SVDRankType = Union[int, float]
OptType = Union[bool, int]
RescaleModeType = Union[str, None, ndarray]
SortedEigsType = Union[str, bool]


class DMD(DMDBase):
    """
    Traditional Dynamic Mode Decomposition

    Parameters
    ----------
    svd_rank : int or float, default -1
        The SVD rank to use for truncation. If a positive integer,
        the minimum of this number and the maximum possible rank
        is used. If a float between 0 and 1, the minimum rank to
        achieve the specified energy content is used. If -1, no
        truncation is performed.
    exact : bool, default False
        A flag for using exact or projected dynamic modes.
    opt : bool, default False
        A flag for using optimal amplitudes or an initial
        condition fit.
    sort_method : str {'eigs', 'amps'} or None
        Mode sorting based on eigenvalues. If 'real', eigenvalues
        are sorted by their real part. If 'abs', eigenvalues are
        sorted by their magnitude. If None, no sorting is performed.
    """

    def __init__(self,
                 svd_rank: SVDRankType = -1,
                 exact: bool = False,
                 opt: bool = False,
                 sort_method: str = None) -> None:
        super().__init__(svd_rank, exact, opt, sort_method)

    def fit(self, X: ndarray, verbose: bool = True) -> 'DMD':
        """
        Fit the traditional DMD model.

        Parameters
        ----------
        X : ndarray (n_features, n_snapshots)
            The snapshot matrix.
        verbose : bool, default True
            Flag for printing DMD summary.

        Returns
        -------
        self
        """
        self._snapshots = np.copy(X)

        # Split the snapshots
        X, Y = self.snapshots[:, :-1], self.snapshots[:, 1:]

        # Compute the SVD of X
        U, s, V = self._compute_svd(X)

        self._compute_operator(Y, U, s, V)
        self._decompose_Atilde()
        self._compute_modes(Y, U, s, V)

        n_snapshots = self.n_snapshots
        self.snapshot_time = {'t0': 0, 'tf': n_snapshots - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tf': n_snapshots - 1, 'dt': 1}

        self._b = self._compute_amplitudes()
        self._sort_modes()

        if verbose:
            msg = '='*10 + ' DMD Summary ' + '='*10
            header = '='*len(msg)
            print('\n'.join(['', header, msg, header]))
            print(f'Number of Modes:\t{self.n_modes}')
            print(f'Reconstruction Error:\t{self.reconstructed_error:.3e}')
            print()
