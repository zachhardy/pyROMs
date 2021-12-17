import numpy as np
from typing import Tuple


def format_subplots(n_plots: int) -> Tuple[int, int]:
    """
    Determine the number of rows and columns for subplots.

    Parameters
    ----------
    n_plots : int

    Returns
    -------
    int, int : n_rows, n_cols
    """
    if n_plots < 4:
        n_rows, n_cols = 1, n_plots
    elif 4 <= n_plots < 9:
        tmp = int(np.ceil(np.sqrt(n_plots)))
        n_rows = n_cols = tmp
        for n in range(1, n_cols + 1):
            if n * n_cols >= n_plots:
                n_rows = n
                break
    else:
        raise AssertionError('Maximum number of subplots is 9. '
                             'Consider modifying the visualization.')
    return n_rows, n_cols
