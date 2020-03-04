from pathlib import Path
from typing import Union, List


PathType = Union[str, Path]
PathsType = Union[PathType, List[PathType]]


def check_path_type(paths: Union[PathType, PathsType], check_exists=True, as_list=False):
    """
    Convert path(s) to pathlib.Path (if it was str) and check exists.
    Parameters
    ----------
    paths (pathlib.Path or str or List):
        Path to files.
    check_exists (bool):
        Check whether the path exists.
    as_list (bool):
        Return paths as a list type.

    Returns
    -------
        If as_list is True then List[pathlib.Path] else pathlib.Path object.

    Raises
    -------
        ValueError: if path is not exists or if as_list=True but paths contain more than 1 elements.
    """
    new_paths = paths
    if not isinstance(new_paths, List):
        paths = [new_paths]
    elif not as_list and len(paths) > 1:
        raise ValueError(f'Output cannot be not list, because input is list with len={len(paths)}')

    for i, path in enumerate(new_paths):
        if isinstance(new_paths, str):
            path = Path(new_paths)
        if check_exists and not path.exists():
            raise ValueError(f'Path {path} is not exists.')
        new_paths[i] = path

    return paths if as_list else paths[0]
