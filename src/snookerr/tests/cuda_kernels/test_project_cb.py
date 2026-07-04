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

# ruff: disable[D101,D102,N801,ANN201]
import itertools  # noqa: I001
import time  # noqa: F401

import cupy as cp
import numexpr as ne
import numpy as np
import pytest

from ... import imview  # noqa: F401, RUF100
from ...cuda import header_injection

from ...cuda.structs import (
    config_cb_project as ks_config,
    det_spec as ks_det_spec,
    fvec3 as ks_fvec3,
    uivec3 as ks_uivec3,
    ulvec3 as ks_ulvec3,
    proj_spec as ks_proj_spec,
    vol_spec as ks_vol_spec,
    vol_tex_chunk_spec as ks_vol_tex_chunk_spec,
)
from ...lib import texture_tools
from ... import phantoms

str_pytest_gen_sources: str = "ks_gen_sources"


def create_injected_module_sharedmem() -> cp.RawModule:  # noqa: D103
    # Inject and compile module
    module: cp.RawModule = header_injection.load_injected_module(
        "./CB_project_sharedmem.cu",
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


@pytest.fixture
def injected_module_sharedmem() -> cp.RawModule:  # noqa: D103
    return create_injected_module_sharedmem()


def injected_module_texture_chunked() -> cp.RawModule:  # noqa: D103
    # Inject and compile module
    module: cp.RawModule = header_injection.load_injected_module(
        "./CB_project_texture_chunked.cu",
        name_expressions=[
            f"project_CB_texture_chunked<{t1:s},{t2:s}>"
            for t1, t2 in itertools.product(
                ["half", "float"], ["float", "double"]
            )
        ],
    )
    module.compile()
    return module


@pytest.mark.dependency(
    name="ks_100", depends=[str_pytest_gen_sources], scope="session"
)
def test_cb_project_single_voxel_offset_detector(
    injected_module_sharedmem: cp.RawModule,
) -> None:
    """Test a kernel which is generated dynamically then loaded."""
    # Config
    config = np.array((1.0, 1.0), dtype=ks_config.dtype)
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
        output_projs.strides, dtype=ks_ulvec3.dtype
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
            np.array(vol.strides, dtype=ks_ulvec3.dtype),  # strides_bytes
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
    fun = injected_module_sharedmem.get_function(
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
    cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.
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


class TestCB_sharedmem_randomnoise:
    def setup_method(self) -> None:
        # Config
        config = np.array((1.0, 1.0), dtype=ks_config.dtype)
        # projspecs
        # Imagine a top-down view with the source on the right-hand side.
        # The x coordinate points to the right.
        # The y coordinate points upwards.
        # The z coordinates points out of the page.
        sourcepos = np.array((500, 0, 0), dtype=ks_fvec3.dtype)
        num_projs = np.uint32(10)
        detwid: int = 1001
        dethei: int = 1001
        dspec = np.array(
            (
                # Detector is shifted along positive y and z.
                # The output image is viewed from the front of the detector-- the same side as the source.
                # From this perspective, the detector is shifted mostly right and a little up.
                np.array((-300, 0, 0), dtype=ks_fvec3.dtype),
                np.array((0, 0.1, 0), dtype=ks_fvec3.dtype),
                np.array((0, 0, 0.1), dtype=ks_fvec3.dtype),
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
            output_projs.strides, dtype=ks_ulvec3.dtype
        )
        # Very short volume.
        vol: cp.ndarray = cp.abs(
            cp.random.randn(501, 500, 20, dtype=cp.float32)
        )
        cp.mean(vol)
        # print("Created the volume. Now launching kernels.", flush=True)
        # Save fields for use in the test method
        self.vol = vol
        self.config = config
        self.projspec_array = projspec_array
        self.module = create_injected_module_sharedmem()
        self.output_projs = output_projs
        self.num_projs = num_projs
        self.output_projs_strides_bytes = output_projs_strides_bytes
        self.dethei = dethei
        self.detwid = detwid

    def teardown_method(self) -> None:
        self.output_projs.fill(0)

    @pytest.mark.parametrize(
        "blockdims",
        [[18, 18, 18]],
        ids=["18x18x18"],
    )
    def test_project(self, blockdims: list[int | None]) -> None:
        vol = self.vol
        config = self.config
        projspec_array = self.projspec_array
        output_projs = self.output_projs
        num_projs = self.num_projs
        output_projs_strides_bytes = self.output_projs_strides_bytes
        dethei = self.dethei
        detwid = self.detwid  # noqa: F841
        # Prepare the kernel
        blockx = 10
        blocky = 10
        blocklayer = 10
        # 48 kB shared memory in a block gives a maximum vol block size of 18x18x18
        volblockx = blockdims[0]
        volblocky = blockdims[1]
        volblockz = blockdims[2]
        fun = self.module.get_function(
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
                    np.array(
                        vol.strides, dtype=ks_ulvec3.dtype
                    ),  # strides_bytes
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
        cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.
        # imview.plot_projections(output_projs)

        # Tests
        # Check that x extremities at z=0 are nonzero
        cp.testing.assert_array_less(
            0, cp.abs(output_projs[:, 0, dethei // 2])
        )
        cp.testing.assert_array_less(
            0, cp.abs(output_projs[:, -1, dethei // 2])
        )
        # Check that z extremities are zero
        cp.testing.assert_array_almost_equal(output_projs[:, :, 0], 0)
        cp.testing.assert_array_almost_equal(output_projs[:, :, -1], 0)


class TestCB_texture_randomnoise:
    def setup_method(self) -> None:
        # Config
        config = np.array((1.0, 1.0), dtype=ks_config.dtype)
        # projspecs
        # Imagine a top-down view with the source on the right-hand side.
        # The x coordinate points to the right.
        # The y coordinate points upwards.
        # The z coordinates points out of the page.
        sourcepos = np.array((500, 0, 0), dtype=ks_fvec3.dtype)
        num_projs = np.uint32(10)
        detwid: int = 1001
        dethei: int = 1001
        dspec = np.array(
            (
                # Detector is shifted along positive y and z.
                # The output image is viewed from the front of the detector-- the same side as the source.
                # From this perspective, the detector is shifted mostly right and a little up.
                np.array((-300, 0, 0), dtype=ks_fvec3.dtype),
                np.array((0, 0.1, 0), dtype=ks_fvec3.dtype),
                np.array((0, 0, 0.1), dtype=ks_fvec3.dtype),
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
            output_projs.strides, dtype=ks_ulvec3.dtype
        )
        # Very short volume.
        vol = np.abs(np.random.randn(501, 500, 20).astype(np.float32)).copy()  # noqa: NPY002
        vtex = texture_tools.Texture(
            vol,
            address_mode=texture_tools.AddressMode.cudaAddressModeBorder,
            filter_mode=texture_tools.FilterMode.cudaFilterModePoint,
        )
        cp.mean(vol)
        # print("Created the volume. Now launching kernels.", flush=True)
        # Save fields for use in the test method
        self.vol = vol
        self.vtex = vtex
        self.config = config
        self.projspec_array = projspec_array
        self.module = injected_module_texture_chunked()
        self.output_projs = output_projs
        self.num_projs = num_projs
        self.output_projs_strides_bytes = output_projs_strides_bytes
        self.dethei = dethei
        self.detwid = detwid

    def teardown_method(self) -> None:
        # imview.plot_projections(self.output_projs)
        self.output_projs.fill(0)

    @pytest.mark.parametrize(
        "blockdims",
        [[18, 18, 18], [None, None, None]],
        ids=["18x18x18", "nochunk"],
    )
    def test_project(self, blockdims: list[int | None]) -> None:
        vol = self.vol
        config = self.config
        vtex = self.vtex
        projspec_array = self.projspec_array
        module = self.module
        output_projs = self.output_projs
        num_projs = self.num_projs
        output_projs_strides_bytes = self.output_projs_strides_bytes
        dethei = self.dethei
        detwid = self.detwid  # noqa: F841
        # Prepare the kernel
        blockx = 10
        blocky = 10
        blocklayer = 10
        # 48 kB shared memory in a block gives a maximum vol block size of 18x18x18
        volchunkx, volchunky, volchunkz = blockdims
        volblockx = vol.shape[0] if volchunkx is None else volchunkx
        volblocky = vol.shape[1] if volchunky is None else volchunky
        volblockz = vol.shape[2] if volchunkz is None else volchunkz
        fun = module.get_function("project_CB_texture_chunked<float,float>")
        for i, j, k in itertools.product(
            range(int(np.ceil(vol.shape[0] / volblockx))),
            range(int(np.ceil(vol.shape[1] / volblocky))),
            range(int(np.ceil(vol.shape[2] / volblockz))),
        ):
            texchunkspec = np.array(
                (
                    np.float32(1),  # voxwid
                    np.array(
                        (
                            i * volblockx,
                            j * volblocky,
                            k * volblockz,
                        ),
                        dtype=ks_uivec3.dtype,
                    ),  # offsets
                    # I don't think I really need the shape after all...
                    # the texture fetching will return 0 if out of bounds of the texture.
                    np.array(
                        (
                            min(volblockx, vol.shape[0] - i * volblockx),
                            min(volblocky, vol.shape[1] - j * volblocky),
                            min(volblockz, vol.shape[2] - k * volblockz),
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
                    texchunkspec,
                    vtex.surfobj,
                    output_projs_strides_bytes,
                    np.uint64(output_projs.data.ptr),
                ),
            )
        cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.
        # imview.plot_projections(output_projs)

        # Tests
        # Check that x extremities at z=0 are nonzero
        cp.testing.assert_array_less(
            0, cp.abs(output_projs[:, 0, dethei // 2])
        )
        cp.testing.assert_array_less(
            0, cp.abs(output_projs[:, -1, dethei // 2])
        )
        # Check that z extremities are zero
        cp.testing.assert_array_almost_equal(output_projs[:, :, 0], 0)
        cp.testing.assert_array_almost_equal(output_projs[:, :, -1], 0)


@pytest.mark.parametrize(
    argnames=["volres", "dtype_vol"],
    argvalues=[
        (156, np.float32),
        (312, np.float32),
        # (625, np.float32),
        (156, np.float16),
        (312, np.float16),
        # (625, np.float16),
    ],  # , 625])  # , 1250])
    ids=[
        "project-FORBILD-156-fp32",
        "project-FORBILD-312-fp32",
        # "project-FORBILD-625-fp32",
        "project-FORBILD-156-fp16",
        "project-FORBILD-312-fp16",
        # "project-FORBILD-625-fp16",
    ],
)
class TestCB_texture_FORBILD:
    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(self, volres: int, dtype_vol: np.dtype, request):  # noqa: ANN001
        test_id: str = request.node.callspec.id  # noqa: F841
        # Setup phase
        # Get the FORBILD phantom.
        v, voxwid_mm = phantoms.FORBILD_head_HU(volres)  # noqa: RUF059
        # Convert from Hounsfield units to attenuation coefficient
        # v = v[:,:,v.shape[2]//4:-(v.shape[2]//4)]
        self.vol = np.empty(v.shape, dtype=dtype_vol)
        ne.evaluate("(v/1000 + 1)*0.02*voxwid_mm", out=self.vol)
        # Create a texture
        self.vtex = texture_tools.Texture(
            self.vol,
            address_mode=texture_tools.AddressMode.cudaAddressModeBorder,
            filter_mode=texture_tools.FilterMode.cudaFilterModePoint,
        )
        # Set up configuration objects etc
        self.config = np.array((1.0, 1.0), dtype=ks_config.dtype)
        num_projs = np.uint32(1000)
        detwid: int = int(4 / 5 * volres)
        dethei: int = int(4 / 5 * volres)
        self.output_projs: cp.ndarray = cp.zeros(
            (num_projs, detwid, dethei), dtype=cp.float32
        )
        self.output_projs_strides_bytes = np.array(
            self.output_projs.strides, dtype=ks_ulvec3.dtype
        )
        # Prepare the source/detector positions
        projspec_array = []
        ts = np.linspace(0, 2 * np.pi, endpoint=False, num=num_projs)
        for theta in ts:
            sourcepos = np.array(
                (
                    1750 * volres / 2501 * np.cos(theta),
                    1750 * volres / 2501 * np.sin(theta),
                    (theta - np.pi) * volres,
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
                        (-4.25 * np.sin(theta), 4.25 * np.cos(theta), 0),
                        dtype=ks_fvec3.dtype,
                    ),
                    np.array((0, 0, 4.25), dtype=ks_fvec3.dtype),
                    np.uint16(detwid),
                    np.uint16(dethei),
                    np.uint8(0),
                ),
                dtype=ks_det_spec.dtype,
            )
            projspec = np.array((sourcepos, dspec), dtype=ks_proj_spec.dtype)
            projspec_array.append(projspec)
        self.projspec_array = cp.array(projspec_array)
        self.module = injected_module_texture_chunked()
        tdict = {np.dtype(np.float32): "float", np.dtype(np.float16): "half"}
        self.kern = self.module.get_function(
            "project_CB_texture_chunked<"
            + tdict[np.dtype(dtype_vol)]
            + ",float>"
        )

        # yield to the timed test
        cp.cuda.get_current_stream().synchronize()
        yield

        # Teardown phase
        # imview.plot_projections(self.output_projs, title=test_id)
        self.output_projs.fill(0)

    def test_project(self, volres: int) -> None:  # noqa: ARG002
        # Prepare the kernel
        blockx = 10
        blocky = 10
        blocklayer = 10
        texchunkspec = np.array(
            (
                np.float32(1),  # voxwid
                np.array((0, 0, 0), dtype=ks_uivec3.dtype),  # offsets
                np.array(self.vol.shape, dtype=ks_uivec3.dtype),  # shape
                np.array(
                    tuple(-np.array(self.vol.shape, dtype=np.float32) / 2),
                    dtype=ks_fvec3.dtype,
                ),  # texpos
            ),
            dtype=ks_vol_tex_chunk_spec.dtype,
        )
        # Launch the kernel
        self.kern(
            (
                int(np.ceil(self.output_projs.shape[1] / blockx)),
                int(np.ceil(self.output_projs.shape[2] / blocky)),
                int(np.ceil(self.output_projs.shape[0] / blocklayer)),
            ),  # gridDims:  (x,y,z)
            (blockx, blocky, blocklayer),  # blockDims: (x,y,z)
            (  # input args
                self.config,
                self.projspec_array,
                np.uint32(self.output_projs.shape[0]),
                texchunkspec,
                self.vtex.surfobj,
                self.output_projs_strides_bytes,
                np.uint64(self.output_projs.data.ptr),
            ),
        )
        cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.


# ruff: enable[D101,D102,N801,ANN201]
