from typing import Iterable, List


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
