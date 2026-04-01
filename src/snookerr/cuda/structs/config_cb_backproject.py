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
"""A struct used to configure the CB backprojection kernel."""

import numpy as np

from .. import kern_structs

__all__ = [
    "config_cb_backproject",
]

config_cb_backproject = kern_structs.KernelStruct(
    struct_name="config_CB_backproject",
    names=["exact_line_integ_backproj", "mask_to_cylinder", "mask_to_cyl_rad"],
    dtypes=[np.bool, np.bool, np.float32],
)
