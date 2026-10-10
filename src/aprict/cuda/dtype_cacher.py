# Copyright (C) 2026  Murdock Grewar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tools to generate, cache and load structured dtypes."""
# This is necessary because we have to launch kernels on the device
# to be absolutely sure of the correct offsets/sizes of the various dtypes.
# Well, that's true according to the CuPy documentation at least.
# Launching lots of these little kernels to do these tests is costly,
# and slows down the startup time of the package. So, it's worth caching
# the dtypes.

import os
import pathlib
import shutil
import sys

import numpy as np

from .. import data

# Path to the root directory where the header files will be output.
# The source .cu files will also be copied here.
path_to_dtype_cache = pathlib.Path(str(data.__path__[0])).joinpath("dtypes")

_DISABLE_CACHING: bool = True


def _uid_to_path(uid: str) -> pathlib.Path:
    return path_to_dtype_cache.joinpath(uid + ".txt")


def query(*, uid: str) -> np.dtype | None:
    """Fetch the dtype from the cache by its unique identifier string."""
    if _DISABLE_CACHING:
        return None
    p = _uid_to_path(uid)
    if p.parent.is_dir() and p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            typestr = f.read()
            return np.dtype(eval(typestr))  # nosec
    return None


def save(*, uid: str, dtype: np.dtype) -> None:
    """Save a dtype to the cache."""
    if _DISABLE_CACHING:
        return
    p = _uid_to_path(uid)
    if not p.parent.is_dir():
        os.mkdir(p.parent)
    with open(p, "w+", encoding="utf-8") as f:
        f.write(repr(dtype.descr))


def flush() -> None:
    """Flush the dtype cache."""
    if path_to_dtype_cache.is_dir():
        shutil.rmtree(path_to_dtype_cache)


def main(args: list[str]) -> None:  # ruff: ignore[undocumented-public-function]
    if "flush" in args:
        flush()


if __name__ == "__main__":
    main(sys.argv[1:])
