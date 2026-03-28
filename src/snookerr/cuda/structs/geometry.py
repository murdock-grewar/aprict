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
"""A struct used to configure the CB projection kernel."""

import numpy as np

from .. import kern_structs
from .base import fvec3, uivec3  # noqa: F401

__all__ = [
    "det_spec",
    "proj_spec",
]

det_spec = kern_structs.KernelStruct(
    struct_name="det_spec",
    names=[
        "pos",  # position of detector (nominally, the middle of it)
        "u",  # vector pointing along the x direction. Also determines pixel width.
        "v",  # vector pointing along the y direction. Also determines pixel height.
        "nu",
        "nv",  # number of columns/rows. i.e. length of x axis; length of y axis.
        "type",  # numeric integer describing the type of detector. Currently, leave it at 0 for a flat panel.
    ],
    dtypes=[fvec3, fvec3, fvec3, np.uint16, np.uint16, np.uint8],
)

proj_spec = kern_structs.KernelStruct(
    struct_name="proj_spec",
    names=["sourcepos", "dspec"],
    dtypes=[fvec3, det_spec],
)
