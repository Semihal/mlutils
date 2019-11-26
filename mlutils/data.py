from typing import Iterable, List

from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np


def chunk_to_batch(x: Iterable, batch_size: int):
    """
    Split list into batches.
    Parameters
    ----------
    x : Iterable
        List to split.
    batch_size : int
        Number of batches.

    Returns
    -------
    List
        batches list of x.

    Examples
    -------
    >>> from mlutils.data import chunk_to_batch

    >>> data = list(range(0, 9))
    >>> batches = chunk_to_batch(data, batch_size=4)
    >>> batches
    [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    """
    return [
        x[i:i+batch_size]
        for i in range(0, len(x), batch_size)
    ]


def sparse_from_series(series: pd.Series) -> csr_matrix:
    """
    Encodes pandas series of lists int sparse format.
    Parameters
    ----------
    series : pd.Series
        Series of int lists.

    Returns
    -------
    scipy.sparse.csr_matrix:
        Sparse format.

    Examples
    -------
    >>> from mlutils.data import sparse_from_series
    >>> import pandas as pd

    >>> transactions = pd.Series([[0, 1, 2], [1, 2, 3], [0, 1]])
    >>> r = sparse_from_series(transactions)
    >>> r.toarray()
    array([[1, 1, 1, 0],
           [0, 1, 1, 1],
           [1, 1, 0, 0]])
    """
    indices = np.hstack(series.values)
    data = [1] * indices.size
    indptr = [0] + np.cumsum(series.apply(len)).tolist()
    return csr_matrix((data, indices, indptr))
