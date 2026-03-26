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

"""Tests for the CB backprojection kernel(s)."""

# ruff: disable[D101,D102,N801,ANN201]
import itertools
import time  # noqa: F401

import cupy as cp
import numpy as np
import pytest

from ... import imview  # noqa: F401, RUF100
from ...cuda import header_injection
from ...cuda.structs.config_cb_project import config_cb_project as ks_config
from ...cuda.structs.config_cb_project import det_spec as ks_det_spec
from ...cuda.structs.config_cb_project import fvec3 as ks_fvec3
from ...cuda.structs.config_cb_project import proj_spec as ks_proj_spec
from ...cuda.structs.config_cb_project import uivec3 as ks_uivec3
from ...cuda.structs.config_cb_project import vol_spec as ks_vol_spec

str_pytest_gen_sources: str = "ks_gen_sources"


def create_injected_module_sharedmem_backproject() -> cp.RawModule:  # noqa: D103
    # Inject and compile module
    module: cp.RawModule = header_injection.load_injected_module(
        "./CB_backproject_sharedmem.cu",
        name_expressions=[
            "backproject_CB_sharedmem<float,float,4,4,4>",
            "backproject_CB_sharedmem<float,double,4,4,4>",
            "backproject_CB_sharedmem<float,float,6,6,6>",
            "backproject_CB_sharedmem<float,double,6,6,6>",
            "backproject_CB_sharedmem<float,float,10,10,10>",
            "backproject_CB_sharedmem<float,double,10,10,10>",
            "backproject_CB_sharedmem<float,float,18,18,18>",
            "backproject_CB_sharedmem<float,double,18,18,18>",
        ],
    )
    module.compile()
    return module


@pytest.mark.parametrize("volres", [105, 210])  # , 1250])
class TestCB_BACKPROJ_sharedmem_randomnoise:
    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(self, volres: int):

        # Set up configuration objects etc
        self.config = np.array((), dtype=ks_config.dtype)
        num_projs = np.uint32(10)
        detwid: int = int(4 / 5 * volres * 3)
        dethei: int = int(4 / 5 * volres * 3)
        self.output_projs: cp.ndarray = cp.zeros(
            (num_projs, detwid, dethei), dtype=cp.float32
        )
        self.output_projs_strides_bytes = np.array(
            self.output_projs.strides, dtype=ks_uivec3.dtype
        )
        # Prepare the source/detector positions
        projspec_array = []
        ts = np.linspace(0, 2 * np.pi, endpoint=False, num=num_projs)
        for theta in ts:
            sourcepos = np.array(
                (
                    1750 * volres / 2501 * np.cos(theta),
                    1750 * volres / 2501 * np.sin(theta),
                    0,
                ),
                dtype=ks_fvec3.dtype,
            )
            dspec = np.array(
                (
                    # Detector is shifted along positive y and z.
                    # The output image is viewed from the front of the detector-- the same side as the source.
                    # From this perspective, the detector is shifted mostly right and a little up.
                    np.array(
                        (-volres * np.cos(theta), -volres * np.sin(theta), 0),
                        dtype=ks_fvec3.dtype,
                    ),
                    np.array(
                        (
                            -4.25 * np.sin(theta) / 3,
                            4.25 * np.cos(theta) / 3,
                            0,
                        ),
                        dtype=ks_fvec3.dtype,
                    ),
                    np.array((0, 0, 4.25 / 3), dtype=ks_fvec3.dtype),
                    np.uint16(detwid),
                    np.uint16(dethei),
                    np.uint8(0),
                ),
                dtype=ks_det_spec.dtype,
            )
            projspec = np.array((sourcepos, dspec), dtype=ks_proj_spec.dtype)
            projspec_array.append(projspec)
        self.projspec_array = cp.array(projspec_array)
        self.module = create_injected_module_sharedmem_backproject()

        # actual projection data
        self.input_projs: cp.ndarray = cp.abs(
            cp.random.randn(num_projs, detwid, dethei, dtype=cp.float32)
        )
        self.input_projs_strides_bytes = np.array(
            self.input_projs.strides, dtype=ks_uivec3.dtype
        )
        self.vol: cp.ndarray = cp.zeros(
            (volres, volres, volres), dtype=cp.float32
        )

        # yield to the timed test
        cp.cuda.get_current_stream().synchronize()
        yield

        # imview.plot_volume(self.vol)
        self.vol.fill(0)

    @pytest.mark.parametrize(
        "blockdims",
        [[18, 18, 18]],
        ids=["18x18x18"],
    )
    def test_backproject(self, blockdims: list[int]) -> None:
        vol = self.vol
        config = self.config
        projspec_array = self.projspec_array
        input_projs = self.input_projs
        num_projs = self.input_projs.shape[0]
        input_projs_strides_bytes = self.input_projs_strides_bytes
        # Prepare the kernel
        blockx = 10
        blocky = 10
        blocklayer = 10
        # 48 kB shared memory in a block gives a maximum vol block size of 18x18x18
        volblockx = blockdims[0]
        volblocky = blockdims[1]
        volblockz = blockdims[2]
        fun = self.module.get_function(
            f"backproject_CB_sharedmem<float,float,{volblockx:d},{volblocky:d},{volblockz:d}>"
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
                    np.array(
                        vol.strides, dtype=ks_uivec3.dtype
                    ),  # strides_bytes
                    np.array(
                        (
                            min(volblockx, vol.shape[0] - i * volblockx),
                            min(volblocky, vol.shape[1] - j * volblocky),
                            min(volblockz, vol.shape[2] - k * volblockz),
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
                    int(np.ceil(input_projs.shape[1] / blockx)),
                    int(np.ceil(input_projs.shape[2] / blocky)),
                    int(np.ceil(input_projs.shape[0] / blocklayer)),
                ),  # gridDims:  (x,y,z)
                (blockx, blocky, blocklayer),  # blockDims: (x,y,z)
                (  # input args
                    config,
                    projspec_array,
                    np.uint32(num_projs),
                    vol_spec,
                    input_projs_strides_bytes,
                    np.uint64(input_projs.data.ptr),
                ),
            )
        cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.


# ruff: enable[D101,D102,N801,ANN201]
