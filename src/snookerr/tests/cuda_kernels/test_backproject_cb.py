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
import itertools  # noqa: I001
import time  # noqa: F401

import cupy as cp
import numpy as np
import pytest

from ... import imview  # noqa: F401, RUF100
from ...cuda import header_injection
from ...cuda.structs import (
    config_cb_project as ks_config,
    det_spec as ks_det_spec,
    fvec3 as ks_fvec3,
    uivec3 as ks_uivec3,
    proj_spec as ks_proj_spec,
    vol_spec as ks_vol_spec,
    vol_tex_chunk_spec as ks_vol_tex_chunk_spec,
)
from ...lib import texture_tools

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


def create_injected_module_texture_backproject() -> cp.RawModule:  # noqa: D103
    # Inject and compile module
    module: cp.RawModule = header_injection.load_injected_module(
        "./CB_backproject_texture.cu",
        name_expressions=[
            "backproject_CB_texture<float,float>",
            "backproject_CB_texture<float,double>",
        ],
    )
    module.compile()
    return module


@pytest.fixture()
def kern_BP_sharedmem():  # noqa: D103, N802
    module = create_injected_module_sharedmem_backproject()
    return module.get_function(
        "backproject_CB_sharedmem<float,float,18,18,18>"
    )


@pytest.fixture()
def kern_BP_texture():  # noqa: D103, N802
    module = create_injected_module_texture_backproject()
    return module.get_function("backproject_CB_texture<float,float>")


@pytest.mark.parametrize(
    argnames=["volres", "kernel", "blockdims"],
    argvalues=[
        (105, "kern_BP_sharedmem", [18, 18, 18]),
        (210, "kern_BP_sharedmem", [18, 18, 18]),
        # (105, "kern_BP_texture",   [None, None, None]),
        # (210, "kern_BP_texture",   [None, None, None]),
        # [105, "backproject_CB_texture"],
    ],
    ids=["sharedmem-105", "sharedmem-210"],
)
class TestCB_BACKPROJ_sharedmem_randomnoise:
    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(
        self,
        volres: int,
        kernel,  # noqa: ANN001
        blockdims: list[int | None],
        request,  # noqa: ANN001
    ):
        kernel = request.getfixturevalue(kernel)
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
        self.kern = kernel
        self.blockdims = blockdims

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
        # self.vol.fill(0)

    # @pytest.mark.parametrize(
    #    "blockdims",
    #    [[18, 18, 18]],
    #    ids=["18x18x18"],
    # )
    def test_backproject(self) -> None:
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
        volblockx = (
            vol.shape[0] if self.blockdims[0] is None else self.blockdims[0]
        )
        volblocky = (
            vol.shape[1] if self.blockdims[1] is None else self.blockdims[1]
        )
        volblockz = (
            vol.shape[2] if self.blockdims[2] is None else self.blockdims[2]
        )
        fun = self.kern
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
            args = (  # input args
                config,
                projspec_array,
                np.uint32(num_projs),
                vol_spec,
                input_projs_strides_bytes,
                np.uint64(input_projs.data.ptr),
            )
            fun(
                (
                    int(np.ceil(input_projs.shape[1] / blockx)),
                    int(np.ceil(input_projs.shape[2] / blocky)),
                    int(np.ceil(input_projs.shape[0] / blocklayer)),
                ),  # gridDims:  (x,y,z)
                (blockx, blocky, blocklayer),  # blockDims: (x,y,z)
                args,
            )
        cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.


@pytest.mark.parametrize(
    argnames=["volres", "kernel", "blockdims"],
    argvalues=[
        (105, "kern_BP_texture", [10, 10, 10]),
        (210, "kern_BP_texture", [10, 10, 10]),
        (420, "kern_BP_texture", [10, 10, 10]),
        (840, "kern_BP_texture", [10, 10, 10]),
        # (210, "kern_BP_texture",   [None, None, None]),
        # [105, "backproject_CB_texture"],
    ],
    ids=["texture-105", "texture-210", "texture-420", "texture-840"],
)
class TestCB_BACKPROJ_texture_randomnoise:
    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(
        self,
        volres: int,
        kernel,  # noqa: ANN001
        blockdims: list[int | None],
        request,  # noqa: ANN001
    ):
        kernel = request.getfixturevalue(kernel)
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
        self.kern = kernel
        self.blockdims = blockdims

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
        self.vtex = texture_tools.Texture(
            self.vol,
            address_mode=texture_tools.AddressMode.cudaAddressModeBorder,
            filter_mode=texture_tools.FilterMode.cudaFilterModePoint,
            include_surface=True,
            include_texture=True,
        )

        # yield to the timed test
        cp.cuda.get_current_stream().synchronize()
        yield

        self.vtex.cuda_array.copy_to(self.vol)
        # imview.plot_volume(self.vol)
        # self.vol.fill(0)

    # @pytest.mark.parametrize(
    #    "blockdims",
    #    [[18, 18, 18]],
    #    ids=["18x18x18"],
    # )
    def test_backproject(self) -> None:
        vol = self.vol
        config = self.config
        projspec_array = self.projspec_array
        input_projs = self.input_projs
        num_projs = self.input_projs.shape[0]
        input_projs_strides_bytes = self.input_projs_strides_bytes
        # Prepare the kernel
        proj_chunk_size: int = num_projs  # I haven't yet coded it up to launch separate kernels to get through all the projections
        volblockx = (
            vol.shape[0] if self.blockdims[0] is None else self.blockdims[0]
        )
        volblocky = (
            vol.shape[1] if self.blockdims[1] is None else self.blockdims[1]
        )
        volblockz = (
            vol.shape[2] if self.blockdims[2] is None else self.blockdims[2]
        )
        texchunkspec = np.array(
            (
                np.array(
                    (0, 0, 0),
                    dtype=ks_uivec3.dtype,
                ),  # offsets
                # I don't think I really need the shape after all...
                # the texture fetching will return 0 if out of bounds of the texture.
                np.array(
                    (
                        vol.shape[0],
                        vol.shape[1],
                        vol.shape[2],
                    ),
                    dtype=ks_uivec3.dtype,
                ),  # shape
                np.array(
                    tuple(-np.array(vol.shape, dtype=np.float32) / 2),
                    dtype=ks_fvec3.dtype,
                ),  # texpos
            ),
            dtype=ks_vol_tex_chunk_spec.dtype,
        )
        fun = self.kern
        for i in range(int(np.ceil(num_projs / proj_chunk_size))):
            # Adjust projspec_array here to account for which projections to include...
            args = (  # input args
                config,
                projspec_array,
                np.uint32(num_projs),
                texchunkspec,
                self.vtex.surfobj,
                input_projs_strides_bytes,
                np.uint64(input_projs.data.ptr),
            )
            fun(
                (
                    int(np.ceil(vol.shape[0] / volblockx)),
                    int(np.ceil(vol.shape[1] / volblocky)),
                    int(np.ceil(vol.shape[2] / volblockz)),
                ),  # gridDims:  (x,y,z)
                (volblockx, volblocky, volblockz),  # blockDims: (x,y,z)
                args,
            )
        cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.


# ruff: enable[D101,D102,N801,ANN201]
