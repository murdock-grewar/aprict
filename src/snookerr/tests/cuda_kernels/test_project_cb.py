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

import itertools

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


@pytest.fixture
def injected_module() -> cp.RawModule:  # noqa: D103
    # Inject and compile module
    module: cp.RawModule = header_injection.load_injected_module(
        "./CB_project.cu",
        name_expressions=[
            "project_CB_sharedmem<float,float,4,4,4>",
            "project_CB_sharedmem<float,double,4,4,4>",
            "project_CB_sharedmem<float,float,6,6,6>",
            "project_CB_sharedmem<float,double,6,6,6>",
            "project_CB_sharedmem<float,float,10,10,10>",
            "project_CB_sharedmem<float,double,10,10,10>",
            "project_CB_sharedmem<float,float,18,18,18>",
            "project_CB_sharedmem<float,double,18,18,18>",
        ],
    )
    module.compile()
    return module


@pytest.mark.dependency(
    name="ks_100", depends=[str_pytest_gen_sources], scope="session"
)
def test_cb_project_single_voxel_offset_detector(
    injected_module: cp.RawModule,
) -> None:
    """Test a kernel which is generated dynamically then loaded."""
    # Config
    config = np.array((), dtype=ks_config.dtype)
    # projspecs
    # The source is here in the center of the volume
    # The detector pixel is one quarter-voxel away from the center
    # Both intersect the volume. This a test of the exactness of the line integrals.
    # Imagine a top-down view with the source on the right-hand side.
    # The x coordinate points to the right.
    # The y coordinate points upwards.
    # The z coordinates points out of the page.
    sourcepos = np.array((0, 0, 0), dtype=ks_fvec3.dtype)
    detwid: int = 501
    dethei: int = 501
    dspec = np.array(
        (
            # Detector is shifted along positive y and z.
            # The output image is viewed from the front of the detector-- the same side as the source.
            # From this perspective, the detector is shifted mostly right and a little up.
            np.array((-0.25, 10, 3), dtype=ks_fvec3.dtype),
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
    fun = injected_module.get_function(
        "project_CB_sharedmem<float,double,4,4,4>"
    )
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
    offy = 200  # 200*0.05 = 10
    offz = 60  # 60*0.05 = 3
    # The center pixel should read exactly 0.25.
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 - offy, dethei // 2 - offz], 0.25
    )
    cp.testing.assert_array_almost_equal(cp.amin(output_projs), 0.25)
    # The maximum reading shoud be along the line goin from (0,0,0) to (-0.25, 0.5, 0.5)
    val = np.sqrt(0.25**2 + 0.5**2 + 0.5**2)
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 + 10 - offy, dethei // 2 + 10 - offz], val
    )
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 - 10 - offy, dethei // 2 + 10 - offz], val
    )
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 + 10 - offy, dethei // 2 - 10 - offz], val
    )
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 - 10 - offy, dethei // 2 - 10 - offz], val
    )
    cp.testing.assert_array_almost_equal(cp.amax(output_projs), val)


# Defunct test. Still useful to have it here to test this method in the future.
@pytest.mark.dependency(depends=["ks_100"], scope="session")
def _test_cb_project_random_noise(injected_module: cp.RawModule) -> None:
    """Test a kernel which is generated dynamically then loaded."""
    # Config
    config = np.array((), dtype=ks_config.dtype)
    # projspecs
    # Imagine a top-down view with the source on the right-hand side.
    # The x coordinate points to the right.
    # The y coordinate points upwards.
    # The z coordinates points out of the page.
    sourcepos = np.array((300, 0, 0), dtype=ks_fvec3.dtype)
    num_projs = np.uint32(10)
    detwid: int = 1001
    dethei: int = 1001
    dspec = np.array(
        (
            # Detector is shifted along positive y and z.
            # The output image is viewed from the front of the detector-- the same side as the source.
            # From this perspective, the detector is shifted mostly right and a little up.
            np.array((-300, 0, 0), dtype=ks_fvec3.dtype),
            np.array((0, 0.3, 0), dtype=ks_fvec3.dtype),
            np.array((0, 0, 0.3), dtype=ks_fvec3.dtype),
            np.uint16(detwid),
            np.uint16(dethei),
            np.uint8(0),
        ),
        dtype=ks_det_spec.dtype,
    )
    projspec = np.array((sourcepos, dspec), dtype=ks_proj_spec.dtype)
    projspec_array = cp.array([projspec for _ in range(num_projs)])
    # actual projection data
    output_projs: cp.ndarray = cp.zeros(
        (num_projs, detwid, dethei), dtype=cp.float32
    )
    output_projs_strides_bytes = np.array(
        output_projs.strides, dtype=ks_uivec3.dtype
    )
    # Volume with a single hot voxel
    vol = cp.abs(cp.random.randn(2501, 2501, 2, dtype=cp.float32))
    cp.mean(vol)
    print("Created the volume. Now launching kernels.", flush=True)

    # Prepare the kernel
    blockx = 10
    blocky = 10
    blocklayer = 10
    # 48 kB shared memory in a block gives a maximum vol block size of 18x18x18
    volblockx = 18
    volblocky = 18
    volblockz = 18
    fun = injected_module.get_function(
        f"project_CB_sharedmem<float,float,{volblockx:d},{volblocky:d},{volblockz:d}>"
    )
    for i, j, k in itertools.product(
        range(int(np.ceil(vol.shape[0] / volblockx))),
        range(int(np.ceil(vol.shape[1] / volblocky))),
        range(int(np.ceil(vol.shape[2] / volblockz))),
    ):
        vol_spec = np.array(
            (
                np.uint64(
                    vol.data.ptr
                    + i * vol.strides[0] * volblockx
                    + j * vol.strides[1] * volblocky
                    + k * vol.strides[2] * volblockz
                ),  # pointer to the byte beginning the 0,0,0 voxel (can be offset)
                np.array(vol.strides, dtype=ks_uivec3.dtype),  # strides_bytes
                np.array(
                    (
                        vol.shape[0] - i * volblockx,
                        vol.shape[1] - j * volblocky,
                        vol.shape[2] - k * volblockz,
                    ),
                    dtype=ks_uivec3.dtype,
                ),  # shape
                np.array(
                    tuple(
                        -(np.array(vol.shape, dtype=np.float32) - 1) / 2
                        + np.array(
                            [i * volblockx, j * volblocky, k * volblockz],
                            dtype=np.float32,
                        )
                    ),
                    dtype=ks_fvec3.dtype,
                ),
            ),
            dtype=ks_vol_spec.dtype,
        )
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
    offy = 200  # 200*0.05 = 10
    offz = 60  # 60*0.05 = 3
    # The center pixel should read exactly 0.25.
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 - offy, dethei // 2 - offz], 0.25
    )
    cp.testing.assert_array_almost_equal(cp.amin(output_projs), 0.25)
    # The maximum reading shoud be along the line goin from (0,0,0) to (-0.25, 0.5, 0.5)
    val = np.sqrt(0.25**2 + 0.5**2 + 0.5**2)
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 + 10 - offy, dethei // 2 + 10 - offz], val
    )
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 - 10 - offy, dethei // 2 + 10 - offz], val
    )
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 + 10 - offy, dethei // 2 - 10 - offz], val
    )
    cp.testing.assert_array_almost_equal(
        output_projs[:, detwid // 2 - 10 - offy, dethei // 2 - 10 - offz], val
    )
    cp.testing.assert_array_almost_equal(cp.amax(output_projs), val)
