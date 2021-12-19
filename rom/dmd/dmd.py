from numpy import ndarray
from typing import Union, Iterable

from .dmd_base import DMDBase
from pydmd.dmd import DMD as PyDMD
from pydmd.utils import compute_tlsq


class DMD(DMDBase, PyDMD):
    """
    Dynamic Mode Decomposition derived from pyDMD

    Parameters
    ----------
    svd_rank : int or float, default 0
        The rank for the truncation. If 0, the method computes the
        optimal rank and uses it for truncation. If positive interger, the
        method uses the argument for the truncation. If float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`. If -1, the method does
        not compute truncation.
    tlsq_rank : int, default 0
        Rank truncation computing Total Least Square. Default is 0,
        which means no truncation.
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
    rescale_mode : {'auto'}, None, or ndarray, default None
        Scale Atilde as shown in 10.1016/j.jneumeth.2015.10.010 (section 2.4)
        before computing its eigendecomposition. None means no rescaling,
        'auto' means automatic rescaling using singular values, otherwise the
        scaling factors.
    forward_backward : bool, default False
        If True, the low-rank operator is computed
        like in fbDMD (reference: https://arxiv.org/abs/1507.02264).
    sorted_eigs : {'real', 'abs'} or False, default False
         Sort eigenvalues (and modes/dynamics accordingly) by
        magnitude if `sorted_eigs='abs'`, by real part (and then by imaginary
        part to break ties) if `sorted_eigs='real'`.
    """

    def __init__(self,
                 svd_rank: Union[int, float] = 0,
                 tlsq_rank: int = 0,
                 exact: bool = False,
                 opt: Union[bool, int] = False,
                 rescale_mode: Union[str, None, ndarray] = None,
                 forward_backward: bool = False,
                 sorted_eigs: Union[bool, str] = False) -> None:
        DMDBase.__init__(self, svd_rank, tlsq_rank, exact, opt,
                         rescale_mode, forward_backward, sorted_eigs)

