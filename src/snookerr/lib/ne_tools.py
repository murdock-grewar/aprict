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

"""Tools related to numexpr usage."""

import pathlib  # noqa: TC003

import numexpr as ne
import numpy as np


def save(path: pathlib.Path, arr: np.ndarray) -> None:
    """Save an array to a path in a faster manner than np.save."""
    mm: np.memmap = np.lib.format.open_memmap(
        path, shape=arr.shape, dtype=arr.dtype, mode="w+"
    )
    ne.evaluate("x", out=mm, local_dict={"x": arr})
