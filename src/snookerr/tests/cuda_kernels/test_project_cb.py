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

# from ... import imview
from ...cuda import header_injection
from ...cuda.structs.config_cb_project import config_cb_project as ks_config
from ...cuda.structs.config_cb_project import det_spec as ks_det_spec
from ...cuda.structs.config_cb_project import fvec3 as ks_fvec3
from ...cuda.structs.config_cb_project import proj_spec as ks_proj_spec
from ...cuda.structs.config_cb_project import uivec3 as ks_uivec3
from ...cuda.structs.config_cb_project import vol_spec as ks_vol_spec

str_pytest_gen_sources: str = "ks_gen_sources"


@pytest.mark.dependency(
    name="ks_100", depends=[str_pytest_gen_sources], scope="session"
)
def test_cb_project_single_voxel() -> None:
    """Test a kernel which is generated dynamically then loaded."""
    # Inject and compile module
    injected_module: cp.RawModule = header_injection.load_injected_module(
        "./CB_project.cu",
        name_expressions=[
            "project_CB<float,float,4,4,4>",
            "project_CB<float,double,4,4,4>",
        ],
    )
    injected_module.compile()
    # Config
    config = np.array((), dtype=ks_config.dtype)
    # projspecs
    # The source is here in the center of the volume
    # The detector pixel is one quarter-voxel away from the center
    # Both intersect the volume. This a test of the exactness of the line integrals.
    sourcepos = np.array((0, 0, 0), dtype=ks_fvec3.dtype)
    detwid: int = 501
    dethei: int = 501
    dspec = np.array(
        (
            np.array((-0.25, 0, 0), dtype=ks_fvec3.dtype),
            np.array((0, 0.05, 0), dtype=ks_fvec3.dtype),
            np.array((0, 0, 0.05), dtype=ks_fvec3.dtype),
            np.uint16(detwid),
            np.uint16(dethei),
            np.uint8(0),
        ),
        dtype=ks_det_spec.dtype,
    )
    projspec = np.array((sourcepos, dspec), dtype=ks_proj_spec.dtype)
    projspec_array = cp.array([projspec for _ in range(10)])
    # actual projection data
    output_projs: cp.ndarray = cp.zeros((10, detwid, dethei), dtype=cp.float64)
    output_projs_strides_bytes = np.array(
        output_projs.strides, dtype=ks_uivec3.dtype
    )
    num_projs = np.uint32(output_projs.shape[0])
    # Volume with a single hot voxel
    vol = cp.array(
        [
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ],
        dtype=cp.float32,
    )
    vol_spec = np.array(
        (
            np.uint64(
                vol.data.ptr
            ),  # pointer to the byte beginning the 0,0,0 voxel (can be offset)
            np.array(vol.strides, dtype=ks_uivec3.dtype),  # strides_bytes
            np.array(vol.shape, dtype=ks_uivec3.dtype),  # shape
            np.array(
                tuple(-(np.array(vol.shape, dtype=np.float32) - 1) / 2),
                dtype=ks_fvec3.dtype,
            ),
        ),
        dtype=ks_vol_spec.dtype,
    )

    # Prepare the kernel
    blockx = 8
    blocky = 8
    blocklayer = 10
    fun = injected_module.get_function("project_CB<float,double,4,4,4>")
    fun(
        (
            int(np.ceil(output_projs.shape[1] / blockx)),
            int(np.ceil(output_projs.shape[2] / blocky)),
            int(np.ceil(output_projs.shape[0] / blocklayer)),
        ),  # gridDims:  (x,y,z)
        (blockx, blocky, blocklayer),  # blockDims: (x,y,z)
        (  # input args
            config,
            projspec_array,
            np.uint32(num_projs),
            vol_spec,
            output_projs_strides_bytes,
            np.uint64(output_projs.data.ptr),
        ),
    )
    # imview.plot_projections(output_projs)

    # Tests
    # The center pixel should read exactly 0.25.
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2, dethei // 2], 0.25
    )
    cp.testing.assert_array_almost_equal(cp.amin(output_projs), 0.25)
    # The maximum reading shoud be along the line goin from (0,0,0) to (-0.25, 0.5, 0.5)
    val = np.sqrt(0.25**2 + 0.5**2 + 0.5**2)
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 + 10, dethei // 2 + 10], val
    )
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 - 10, dethei // 2 + 10], val
    )
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 + 10, dethei // 2 - 10], val
    )
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 - 10, dethei // 2 - 10], val
    )
    cp.testing.assert_array_almost_equal(cp.amax(output_projs), val)
