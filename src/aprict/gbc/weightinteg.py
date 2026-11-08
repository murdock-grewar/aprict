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

"""A module to compute the expected sum of weights from a perfect cylindrical trajectory. Utilises symmetry and numerical integration."""

import cupy as cp
import numpy as np

from ..cb import flatpanel as cbfp


def backproj_gbc_weight_integ_GPU(  # ruff: ignore[invalid-function-name]
    r_on_R: np.ndarray,  # should be a 1D nonnegative and monotonically increasing array # ruff: ignore[invalid-argument-name]
    config: cbfp.ConfigBackproject,
    num_ang_samps: int,
    num_z_samps: int,
) -> np.ndarray:
    """Integrate the weight*softening over the cylindrical source locus."""
    if r_on_R.ndim != 1:
        msg = "r_on_R should be a 1-dimensional array."
        raise ValueError(msg)
    kern = cbfp._backprojection_module.get_function("get_GBC_weight_integ")  # ruff: ignore[private-member-access]
    input_r_on_R_arr = cp.array(r_on_R.astype(np.float32))  # ruff: ignore[non-lowercase-variable-in-function]
    input_r_on_R_arr_len = np.uint32(len(r_on_R))  # ruff: ignore[non-lowercase-variable-in-function]
    input_output_arr = cp.zeros_like(input_r_on_R_arr)
    input_num_ang_samps = np.uint32(num_ang_samps)
    input_num_z_samps = np.uint32(num_z_samps)
    input_config = config.contents
    args = (
        input_r_on_R_arr,
        input_r_on_R_arr_len,
        input_output_arr,
        input_num_ang_samps,
        input_num_z_samps,
        input_config,
    )

    bX, bY, bZ = (10, 10, 1)  # ruff: ignore[non-lowercase-variable-in-function]
    grid_dims = (
        int((num_z_samps + (bX - 1)) / bX),
        int((num_ang_samps + (bY - 1)) / bY),
        1,
    )
    kern(grid_dims, (bX, bY, bZ), args)
    cp.cuda.get_current_stream().synchronize()  # maybe not necessary?
    return input_output_arr.get()


def backproj_gbc_weight_integ_xygrid_GPU(  # ruff: ignore[invalid-function-name]
    xs: np.ndarray,
    ys: np.ndarray,
    R: float,  # ruff: ignore[invalid-argument-name]
    config: cbfp.ConfigBackproject,
    num_ang_samps: int = 600,
    num_z_samps: int = 600,
) -> np.ndarray:
    """Return a 2D array giving the integrated weights over an xy meshgrid."""
    relRarr = (  # ruff: ignore[non-lowercase-variable-in-function]
        np.sqrt(np.square(xs[:, np.newaxis]) + np.square(ys[np.newaxis, :]))
        / R
    )
    numsamps = 2 * max(len(xs), len(ys))
    r_on_R_input = np.linspace(  # ruff: ignore[non-lowercase-variable-in-function]
        relRarr.min(), relRarr.max(), endpoint=True, num=numsamps
    )

    weight_integ = backproj_gbc_weight_integ_GPU(
        r_on_R_input,
        config,
        num_ang_samps,
        num_z_samps,
    )
    return np.interp(
        relRarr,
        r_on_R_input,
        weight_integ,
    )
