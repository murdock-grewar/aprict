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

"""Tests for the CB projection kernel(s)."""

import cupy as cp
import numpy as np
import pytest

from ... import imview
from ...cuda import header_injection, kern_structs
from ...cuda import structs as kstructs

str_pytest_gen_sources: str = "ks_gen_sources"


@pytest.mark.dependency(name="ks_100", depends=[str_pytest_gen_sources], scope="session")
def test_cb_project() -> None:
    """Test a kernel which is generated dynamically then loaded."""
    ks_config: kern_structs.KernelStruct = (
        kstructs.config_cb_project.config_cb_project
    )
    ks_volchunk_spec: kern_structs.KernelStruct = (
        kstructs.config_cb_project.volume_chunk_spec
    )
    ks_fvec3: kern_structs.KernelStruct = (
        kstructs.config_cb_project.fvec3
    )
    ks_uivec3: kern_structs.KernelStruct = (
        kstructs.config_cb_project.uivec3
    )
    ks_det_spec: kern_structs.KernelStruct = (
        kstructs.config_cb_project.det_spec
    )
    ks_proj_spec: kern_structs.KernelStruct = (
        kstructs.config_cb_project.proj_spec
    )
    injected_module: cp.RawModule = header_injection.load_injected_module(
        "./CB_project.cu",
        name_expressions=[
            "doathing<float,float,4,4,4>",
            "doathing<float,double,4,4,4>",
        ],
    )
    config          = np.array((), dtype=ks_config.dtype)
    vchunk_spec     = np.array((0, 4, 0, 4, 0, 4), dtype=ks_volchunk_spec.dtype)
    sourcepos       = np.array((15, 0, 0), dtype=ks_fvec3.dtype)
    #detpos          = np.array((-3, 0, 0), dtype=ks_fvec3.dtype)
    dspec           = np.array((
        np.array((-3, 0, 0), dtype=ks_fvec3.dtype),
        np.array(( 0, .2, 0), dtype=ks_fvec3.dtype),
        np.array(( 0, 0, .2), dtype=ks_fvec3.dtype),
        np.uint16(100), np.uint16(100),
        np.uint8(0),
        ), dtype=ks_det_spec.dtype)
    projspec = np.array((sourcepos,dspec), dtype=ks_proj_spec.dtype)
    output_projs: cp.ndarray = cp.zeros((1, 100, 100), dtype=cp.float64)
    vol_corner_pos  = np.array((0, 0, 0), dtype=ks_fvec3.dtype)
    vol             = cp.array([
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [0, 1, 1, 0],
            ],
            [
                [0, 1, 1, 0],
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [0, 1, 1, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ]
        ]
        , dtype=cp.float32)
    vol_strides_bytes = np.array(vol.strides, dtype=ks_uivec3.dtype)
    output_projs_strides_bytes = np.array(output_projs.strides, dtype=ks_uivec3.dtype)
    output          = cp.array([0], dtype=cp.float64)
    injected_module.compile()

    fun = injected_module.get_function("doathing<float,double,4,4,4>")
    fun(
        (13, 13,),    # gridDims:  (x,y,z)
        (8, 8),  # blockDims: (x,y,z)
        (  # input args
            config,
            #vchunk_spec,
            projspec,
            #detpos,
            vol_corner_pos,
            vol_strides_bytes,
            vol,
            output_projs_strides_bytes,
            np.uint64(output_projs.data.ptr),  # WARNING: this passes in a np.int32 maybe? Unsure of the int type that is passed. I want to pass whatever is used for a pointer, which I guess may be uint64 for a 64-bit system.
        )
    )
    imview.plot_projections(output_projs)
    expected = cp.array([6.0], dtype=cp.float64)
    cp.testing.assert_array_almost_equal(output_projs, expected)
