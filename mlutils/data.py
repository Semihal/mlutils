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
    """
    return [
        x[i:i+batch_size]
        for i in range(0, len(x), batch_size)
    ]
