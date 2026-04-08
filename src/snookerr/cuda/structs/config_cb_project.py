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

__all__ = [
    "config_cb_project",
]


# ruff: disable[D101,N801]
class config_cb_project(
    kern_structs.KernelStruct,
    struct_name="config_CB_project",
    names=[
        "shrink_volume_factor",  # causes the whole volume to be shrunk by this factor, *with respect to the coordinate origin*, after being specified by whatever volume chunk spec object is relevant.
        "multiply_before_adding",  # multiply the projection by this constant before adding it to the data array
    ],
    dtypes=[np.float32, np.float32],
):
    pass  # ty:ignore[invalid-argument-type]


# ruff: enable[D101,N801]
