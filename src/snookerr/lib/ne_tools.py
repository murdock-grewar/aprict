"""Tools related to numexpr usage."""
import pathlib  # noqa: TC003

import numexpr as ne
import numpy as np


def save(path: pathlib.Path, arr: np.ndarray) -> None:
    """Save an array to a path in a faster manner than np.save."""
    mm: np.memmap = np.lib.format.open_memmap(
        path, shape=arr.shape, dtype=arr.dtype, mode='w+')
    ne.evaluate('x', out=mm, local_dict={'x': arr})
