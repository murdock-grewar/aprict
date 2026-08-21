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
"""A simple struct used for one or more pytests."""

import numpy as np

from ... import kern_structs

__all__ = [
    "Test_Config_Sum",
]


# ruff: disable[undocumented-public-class,invalid-class-name]
class Test_Config_Sum(
    kern_structs.KernelStruct,
    struct_name="Test_Config_Sum",
    names=["alternate_sign", "multiply_first_term"],
    dtypes=[np.bool, np.int8],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
):
    pass


# ruff: enable[undocumented-public-class,invalid-class-name]
