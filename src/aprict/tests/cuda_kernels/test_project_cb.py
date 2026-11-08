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

# ruff: disable[undocumented-public-class,undocumented-public-method,invalid-class-name,missing-return-type-undocumented-public-function]
import itertools  # ruff: ignore[unsorted-imports]
import time  # ruff: ignore[unused-import]

import cupy as cp
import numexpr as ne
import numpy as np
import pytest

from ... import imview  # ruff: ignore[unused-import]
from ...cuda import header_injection

from ...cuda.structs import (
    config_cb_project as ks_config,
    det_spec as ks_det_spec,
    fvec3 as ks_fvec3,
    uivec3 as ks_uivec3,
    ulvec3 as ks_ulvec3,
    proj_spec as ks_proj_spec,
    vol_tex_chunk_spec as ks_vol_tex_chunk_spec,
)
from ...lib import texture_tools, sample_FORBILD
from ... import phantoms

str_pytest_gen_sources: str = "ks_gen_sources"


def injected_module_texture_chunked() -> cp.RawModule:  # ruff: ignore[undocumented-public-function]
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


class TestCB_texture_randomnoise:
    def setup_method(self) -> None:
        # Config
        config = np.array((1.0, 1.0, 1, 1), dtype=ks_config.dtype)
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
                np.uint32(detwid),
                np.uint32(dethei),
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
        vol = np.abs(np.random.randn(501, 500, 20).astype(np.float32)).copy()  # ruff: ignore[numpy-legacy-random]
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
        detwid = self.detwid  # ruff: ignore[unused-variable]
        # Prepare the kernel
        blockx = 8
        blocky = 8
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
    def setup_and_teardown(self, volres: int, dtype_vol: np.dtype, request):  # ruff: ignore[missing-type-function-argument]
        test_id: str = request.node.callspec.id  # ruff: ignore[unused-variable]
        # Setup phase
        # Get the FORBILD phantom.
        fbv = phantoms.FORBILD.HeadPhantom()
        xs = np.arange(volres, dtype=np.float32) - (volres - 1) / 2
        voxwid_mm = 1 / volres * 25
        xs *= voxwid_mm
        ys = xs.copy()
        zs = xs.copy()
        v = sample_FORBILD.sample_FORBILD(
            fbv.shapes, xs=xs, ys=ys, zs=zs
        ).get()

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
        self.config = np.array((1.0, 1.0, 1, 1), dtype=ks_config.dtype)
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
                    np.uint32(detwid),
                    np.uint32(dethei),
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

    def test_project(self, volres: int) -> None:  # ruff: ignore[unused-method-argument]
        # Prepare the kernel
        blockx = 8
        blocky = 8
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


# ruff: enable[undocumented-public-class,undocumented-public-method,invalid-class-name,missing-return-type-undocumented-public-function]
