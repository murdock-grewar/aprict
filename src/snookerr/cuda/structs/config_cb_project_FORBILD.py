# ruff: ignore[invalid-module-name]
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
    "config_cb_project_FORBILD",
]


# ruff: disable[undocumented-public-class,invalid-class-name]
class config_cb_project_FORBILD(
    kern_structs.KernelStruct,
    struct_name="config_CB_project_FORBILD",
    names=[
        "multiply_before_adding",  # multiply the projection by this constant before adding it to the data array
        "overlap_addition",  # if True, then add the values of overlapping shapes. If False, then prioritise the last shape's value
        # Total number of line integral samples is ssu * ssv
        "ssu",  # supersample along the du direction by taking this many samples
        "ssv",  # supersample along the dv direction by taking this many samples
    ],
    dtypes=[np.float32, np.bool, np.uint32, np.uint32],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
):
    pass


# ruff: enable[undocumented-public-class,invalid-class-name]
