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

# ruff: disable[D101,D102,D103,N801,ANN201]
import itertools  # noqa: I001
import time  # noqa: F401

import cupy as cp
import numpy as np
import pytest

from ... import imview  # noqa: F401, RUF100
from ...cuda import header_injection
from ...cuda.structs import (
    config_cb_backproject as ks_config,
    det_spec as ks_det_spec,
    fvec3 as ks_fvec3,
    uivec3 as ks_uivec3,
    proj_spec as ks_proj_spec,
    vol_spec as ks_vol_spec,
    vol_tex_chunk_spec as ks_vol_tex_chunk_spec,
    proj_chunk_spec_maxlen10 as ks_proj_chunk_spec_maxlen10,
)
from ...lib import gpu_tools, texture_tools

str_pytest_gen_sources: str = "ks_gen_sources"

gpu_tools.cupy_cap_default_mem_pool(0)

# import gc
# @pytest.fixture(autouse=True)
# def ensure_gc():
#    gpu_tools.clear_cache()
#    gc.collect()


def create_injected_module_sharedmem_backproject() -> cp.RawModule:
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


def create_injected_module_backproject() -> cp.RawModule:
    # Inject and compile module
    module: cp.RawModule = header_injection.load_injected_module(
        "./CB_backproject.cu",
        name_expressions=[
            "backproject_CB_volmem<float,float>",
            "backproject_CB_volmem<float,half>",
            "backproject_CB_volmem<float,double>",
        ]
        + [
            f"backproject_CB_voltex<{t1:s},{t2:s}>"
            for t1, t2 in itertools.product(
                ["half", "float"], ["half", "float", "double"]
            )
        ],
    )
    module.compile()
    return module


@pytest.fixture()
def kern_BP_sharedmem():  # noqa: N802
    module = create_injected_module_sharedmem_backproject()
    return module.get_function(
        "backproject_CB_sharedmem<float,float,18,18,18>"
    )


@pytest.fixture()
def kern_BP_volmem():  # noqa: N802
    module = create_injected_module_backproject()
    return module.get_function("backproject_CB_volmem<float,float>")


@pytest.fixture()
def kern_BP_voltex():  # noqa: N802
    module = create_injected_module_backproject()
    return module.get_function("backproject_CB_voltex<float,float>")


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
        test_id: str = request.node.callspec.id  # noqa: F841
        # Set up configuration objects etc
        num_projs = np.uint32(10)
        detwid: int = int(4 / 5 * volres * 3)
        dethei: int = int(4 / 5 * volres * 3)
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

        # imview.plot_volume(self.vol, title=test_id)
        # self.vol.fill(0)

    # @pytest.mark.parametrize(
    #    "blockdims",
    #    [[18, 18, 18]],
    #    ids=["18x18x18"],
    # )
    def test_backproject(self) -> None:
        vol = self.vol
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
    argnames=["volres", "numprojs", "use_voltex", "blockdims", "exact"],
    argvalues=[
        (420, 20, False, [10, 10, 10], True),
        (840, 20, False, [10, 10, 10], False),
        (840, 40, False, [10, 10, 10], False),
        (420, 20, True, [10, 10, 10], True),
        (840, 20, True, [10, 10, 10], False),
        (840, 40, True, [10, 10, 10], False),
    ],
    ids=[
        "BP_mem_420-20",
        "BP_mem_840-20-fast",
        "BP_mem_840-40-fast",
        "BP_tex_420-20",
        "BP_tex_840-20-fast",
        "BP_tex_840-40-fast",
    ],
    scope="class",
)
class TestCB_BACKPROJ_randomnoise:
    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(
        self,
        volres: int,
        numprojs: int,
        use_voltex: bool,
        blockdims: list[int | None],
        exact: bool,
        request,  # noqa: ANN001
    ):
        test_id: str = request.node.callspec.id  # noqa: F841
        if use_voltex:
            self.usevoltex: bool = True
            kernel = request.getfixturevalue("kern_BP_voltex")
        else:
            self.usevoltex: bool = False
            kernel = request.getfixturevalue("kern_BP_volmem")
        # Set up configuration objects etc
        self.config = np.array(
            (
                np.bool(exact),
                np.bool(False),
                np.bool(True),
                volres / 2,
                np.bool(False),
                0,
            ),
            dtype=ks_config.dtype,
        )
        num_projs = np.uint32(numprojs)
        detwid: int = int(4 / 5 * volres * 3)
        dethei: int = int(4 / 5 * volres * 3)
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
        self.projspec_array = np.array(projspec_array)
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
        if self.usevoltex:
            self.vtex = texture_tools.Texture(
                self.vol,
                address_mode=texture_tools.AddressMode.cudaAddressModeBorder,
                filter_mode=texture_tools.FilterMode.cudaFilterModePoint,
                include_surface=True,
                include_texture=True,
                layered=False,
            )
            self.dtex = texture_tools.Texture(
                self.input_projs,
                address_mode=texture_tools.AddressMode.cudaAddressModeBorder,
                filter_mode=texture_tools.FilterMode.cudaFilterModePoint,
                include_surface=True,
                include_texture=True,
                layered=True,
            )
        else:
            self.vtex = None
            self.dtex = None

        # yield to the timed test
        cp.cuda.get_current_stream().synchronize()
        yield

        # Copy the data out of the texture
        if self.usevoltex:
            self.vtex.cuda_array.copy_to(self.vol)  # ty:ignore[unresolved-attribute]
        # Plot, if desired
        # imview.plot_volume(self.vol, title=test_id)
        # Clean up the used memory. This indeed seems to be necessary, as pytest doesn't do it automatically for some reason
        del self.input_projs
        del self.vol
        del self.vtex
        del self.dtex
        del self.projspec_array
        # Reset the texture memory to 0s
        # self.vol.fill(0)
        # self.vtex.cuda_array.copy_from(self.vol)
        # gpu_tools.clear_cache()

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
        proj_chunk_size: int = 10
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
        vol_chunk_spec = np.array(
            (
                np.uint64(self.vol.data.ptr),  # ptr_000
                np.array(
                    self.vol.strides, dtype=ks_uivec3.dtype
                ),  # strides_bytes
                np.array(vol.shape, dtype=ks_uivec3.dtype),  # shape
                np.array(
                    tuple((-np.array(vol.shape, dtype=np.float32) + 1) / 2),
                    dtype=ks_fvec3.dtype,
                ),  # pos_000 -- position of the CENTER of the 0,0,0 voxel
            ),
            dtype=ks_vol_spec.dtype,
        )

        input_projs_data_pointer = np.uint64(input_projs.data.ptr)
        grid_dims = (
            int(np.ceil(vol.shape[0] / volblockx)),
            int(np.ceil(vol.shape[1] / volblocky)),
            int(np.ceil(vol.shape[2] / volblockz)),
        )  # gridDims:  (x,y,z)
        block_dims = (volblockx, volblocky, volblockz)
        for i in range(int(np.ceil(num_projs / proj_chunk_size))):
            # Create the projection chunk specification
            chunk_num_projs = min(
                proj_chunk_size, num_projs - i * proj_chunk_size
            )
            projspec_array_chunk = list(
                projspec_array[
                    i * proj_chunk_size : i * proj_chunk_size + chunk_num_projs
                ]
            )
            projspec_array_chunk.extend(
                [projspec_array_chunk[-1]] * (10 - chunk_num_projs)
            )  # repeat the last projspec to fill up a whole chunk of 10.
            input_proj_chunk_spec = np.array(
                (
                    np.array(
                        projspec_array_chunk, dtype=ks_proj_spec
                    ),  # projspecs
                    np.uint8(chunk_num_projs),  # numprojs
                    # np.uint32(i*proj_chunk_size),  # tex start layer
                    np.uint64(
                        input_projs_data_pointer
                        + i * proj_chunk_size * input_projs.strides[0]
                    ),  # projmemaddr
                    input_projs_strides_bytes,  # projmemstrides
                ),
                dtype=ks_proj_chunk_spec_maxlen10,
            )
            if self.usevoltex:
                args = (  # input args
                    config,
                    texchunkspec,
                    self.vtex.surfobj,  # ty:ignore[unresolved-attribute]
                    None,  # the summand_multiplier surface. For these tests, the config is set not to use it.
                    input_proj_chunk_spec,
                )
            else:
                args = (  # input args
                    config,
                    vol_chunk_spec,
                    input_proj_chunk_spec,
                )
            self.kern(grid_dims, block_dims, args)
        cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.


# ruff: enable[D101,D102,D103,N801,ANN201]
