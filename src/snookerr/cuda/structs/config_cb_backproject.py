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


# ruff: disable[D101,N801]
class config_cb_backproject(
    kern_structs.KernelStruct,
    struct_name="config_CB_backproject",
    names=[
        "exact_line_integ_backproj",
        "backproject_all_1s",
        "mask_to_cylinder",
        "mask_to_cyl_rad",
        "apply_summand_multiplier",
        "soften_edge_proportion",
        "Z_on_L_factor_power",
        "Z_on_val_factor_power",
        "Z_on_val_factor_val",
        "approx_mode_direct_raycast",
        "approx_mode_linear_interpolate",
        "backproj_weighting_preset",
    ],
    dtypes=[
        np.bool,
        np.bool,
        np.bool,
        np.float32,
        np.bool,
        np.float32,
        np.int32,
        np.int32,
        np.float32,
        np.bool,
        np.bool,
        np.uint8,
    ],
):
    pass  # ty:ignore[invalid-argument-type]


# ruff: enable[D101,N801]
