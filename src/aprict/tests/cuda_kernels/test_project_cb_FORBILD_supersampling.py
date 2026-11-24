# ruff: ignore[invalid-module-name]
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
import numpy as np
import pytest

from ... import imview  # ruff: ignore[unused-import]
from ...cuda import header_injection
from ...lib import gpu_tools
from ... import phantoms

from ...cuda import structs  # ruff: ignore[unused-import]
from ...cuda.structs import (
    config_cb_project_FORBILD as ks_config,
    det_spec as ks_det_spec,
    fvec3 as ks_fvec3,
    ulvec3 as ks_ulvec3,
    proj_spec as ks_proj_spec,
)

str_pytest_gen_sources: str = "ks_gen_sources"


def injected_module_texture_chunked() -> cp.RawModule:  # ruff: ignore[undocumented-public-function]
    # Inject and compile module
    module: cp.RawModule = header_injection.load_injected_module(
        "./CB_project_texture_chunked.cu",
        name_expressions=[
            f"project_CB_FORBILD_shapes<{t1:s},{t2:s}>"
            for t1, t2 in itertools.product(
                ["half", "float", "double"],
                [
                    "FORBILD_" + s
                    for s in [
                        "Box",
                        "Cone",
                        "Cone",
                        "Cone",
                        "Cone",
                        "Cylinder",
                        "Cylinder_x",
                        "Cylinder_y",
                        "Cylinder_z",
                        "Ellipsoid",
                        "Ellipsoid_free",
                        "Ellipt_Cyl",
                        "Shape",
                        "Sphere",
                    ]
                ],
            )
        ],
    )
    module.compile()
    return module


@pytest.mark.parametrize(
    argnames=["numprojs", "ssu", "ssv", "dtype_proj"],
    argvalues=[
        (100, 2, 1, np.float64),
        (100, 1, 2, np.float64),
        (100, 2, 2, np.float64),
        (100, 5, 5, np.float64),
    ],
    ids=[
        "project_FORBILD-head-ssu2-ssv1",
        "project_FORBILD-head-ssu1-ssv2",
        "project_FORBILD-head-ssu2-ssv2",
        "project_FORBILD-head-ssu5-ssv5",
    ],
)
class TestCB_phantom_FORBILD_head:
    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(
        self,
        numprojs: int,
        ssu: int,
        ssv: int,
        dtype_proj: np.dtype,
        request,  # ruff: ignore[missing-type-function-argument]
    ):
        test_id: str = request.node.callspec.id  # ruff: ignore[unused-variable]
        # Setup phase
        self.out_of_mem_count_arr = cp.zeros((1,), dtype=cp.uint32)
        self.shapes_cp_arr = cp.array(phantoms.FORBILD.HeadPhantom().shapes)

        # Set up configuration objects etc
        add_attenuations = False
        self.config_ss1 = np.array(
            (1.0, add_attenuations, 1, 1), dtype=ks_config.dtype
        )
        self.config = np.array(
            (1.0, add_attenuations, ssu, ssv), dtype=ks_config.dtype
        )
        num_projs = np.uint32(numprojs)
        detwid: int = 250
        dethei: int = 250
        self.ssu: int = ssu
        self.ssv: int = ssv
        self.output_projs: cp.ndarray = cp.zeros(
            (num_projs, detwid, dethei), dtype=dtype_proj
        )
        self.output_projs_strides_bytes = np.array(
            self.output_projs.strides, dtype=ks_ulvec3.dtype
        )
        self.output_projs_ss1: cp.ndarray = cp.zeros(
            (num_projs, detwid * ssu, dethei * ssv), dtype=dtype_proj
        )
        self.output_projs_ss1_strides_bytes = np.array(
            self.output_projs_ss1.strides, dtype=ks_ulvec3.dtype
        )
        # Prepare the source/detector positions
        projspec_array = []
        projspec_ss1_array = []
        ts = np.linspace(0, 2 * np.pi, endpoint=False, num=num_projs)
        volres: int = 120  # legacy. Just setting it to a reasonable value.
        for theta in ts:
            sourcepos = np.array(
                (
                    volres * np.cos(theta),
                    volres * np.sin(theta),
                    (theta - np.pi) / np.pi * volres,
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
                        (-0.3 * np.sin(theta), 0.3 * np.cos(theta), 0),
                        dtype=ks_fvec3.dtype,
                    ),
                    np.array((0, 0, 0.3), dtype=ks_fvec3.dtype),
                    np.uint32(detwid),
                    np.uint32(dethei),
                    np.uint8(0),
                ),
                dtype=ks_det_spec.dtype,
            )
            dspec_ss1 = np.array(
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
                            -0.3 * np.sin(theta) / ssu,
                            0.3 * np.cos(theta) / ssu,
                            0 / ssu,
                        ),
                        dtype=ks_fvec3.dtype,
                    ),
                    np.array(
                        (0 / ssv, 0 / ssv, 0.3 / ssv), dtype=ks_fvec3.dtype
                    ),
                    np.uint32(detwid * ssu),
                    np.uint32(dethei * ssv),
                    np.uint8(0),
                ),
                dtype=ks_det_spec.dtype,
            )
            projspec = np.array((sourcepos, dspec), dtype=ks_proj_spec.dtype)
            projspec_array.append(projspec)
            projspec_ss1 = np.array(
                (sourcepos, dspec_ss1), dtype=ks_proj_spec.dtype
            )
            projspec_ss1_array.append(projspec_ss1)
        self.projspec_array = cp.array(projspec_array)
        self.projspec_ss1_array = cp.array(projspec_ss1_array)
        self.module = injected_module_texture_chunked()
        tdict = {
            np.dtype(np.float64): "double",
            np.dtype(np.float32): "float",
            np.dtype(np.float16): "half",
        }
        self.kern = self.module.get_function(
            "project_CB_FORBILD_shapes<"
            + tdict[np.dtype(dtype_proj)]
            + ","
            + "FORBILD_Shape"
            + ">"
        )

        # Set up the non-supersampled reference answer in self.output_projs_ss1
        gpu_tools.clear_cache()
        cp.cuda.get_current_stream().synchronize()
        self.init_ss1_project()

        # yield to the timed test
        gpu_tools.clear_cache()
        cp.cuda.get_current_stream().synchronize()
        yield

        # Teardown phase
        # imview.plot_projections(self.output_projs, title=test_id)
        del self.output_projs_ss1
        del self.output_projs

    def init_ss1_project(self) -> None:
        # Prepare the kernel
        blockx = 5
        blocky = 5
        blocklayer = 5
        # Launch the kernel
        num_shapes: int = self.shapes_cp_arr.size
        # stacksize = cp.cuda.runtime.deviceGetLimit(cp.cuda.runtime.cudaLimitStackSize)
        # print("stack size: ", stacksize)
        # cp.cuda.runtime.deviceSetLimit(cp.cuda.runtime.cudaLimitStackSize, 2048)
        # stacksize = cp.cuda.runtime.deviceGetLimit(cp.cuda.runtime.cudaLimitStackSize)
        # print("stack size: ", stacksize)
        cp.multiply(
            self.out_of_mem_count_arr, 0, out=self.out_of_mem_count_arr
        )
        self.kern(
            (
                int(np.ceil(self.output_projs_ss1.shape[1] / blockx)),
                int(np.ceil(self.output_projs_ss1.shape[2] / blocky)),
                int(np.ceil(self.output_projs_ss1.shape[0] / blocklayer)),
            ),  # gridDims:  (x,y,z)
            (blockx, blocky, blocklayer),  # blockDims: (x,y,z)
            (  # input args
                self.out_of_mem_count_arr,
                self.config_ss1,
                self.projspec_ss1_array,
                np.uint32(self.output_projs_ss1.shape[0]),
                np.uint64(self.shapes_cp_arr.data.ptr),
                np.uint32(num_shapes),
                self.output_projs_ss1_strides_bytes,
                np.uint64(self.output_projs_ss1.data.ptr),
            ),
        )
        cp.cuda.get_current_stream().synchronize()  # Wait for self.output_projs_ss1 to be filled

    def test_project(self) -> None:
        # Prepare the kernel
        blockx = 5
        blocky = 5
        blocklayer = 5
        # Launch the kernel
        num_shapes: int = self.shapes_cp_arr.size
        # stacksize = cp.cuda.runtime.deviceGetLimit(cp.cuda.runtime.cudaLimitStackSize)
        # print("stack size: ", stacksize)
        # cp.cuda.runtime.deviceSetLimit(cp.cuda.runtime.cudaLimitStackSize, 2048)
        # stacksize = cp.cuda.runtime.deviceGetLimit(cp.cuda.runtime.cudaLimitStackSize)
        # print("stack size: ", stacksize)
        cp.multiply(
            self.out_of_mem_count_arr, 0, out=self.out_of_mem_count_arr
        )
        self.kern(
            (
                int(np.ceil(self.output_projs.shape[1] / blockx)),
                int(np.ceil(self.output_projs.shape[2] / blocky)),
                int(np.ceil(self.output_projs.shape[0] / blocklayer)),
            ),  # gridDims:  (x,y,z)
            (blockx, blocky, blocklayer),  # blockDims: (x,y,z)
            (  # input args
                self.out_of_mem_count_arr,
                self.config,
                self.projspec_array,
                np.uint32(self.output_projs.shape[0]),
                np.uint64(self.shapes_cp_arr.data.ptr),
                np.uint32(num_shapes),
                self.output_projs_strides_bytes,
                np.uint64(self.output_projs.data.ptr),
            ),
        )
        cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.

        assert self.out_of_mem_count_arr[0].get() == 0

        # Now check if it matches the downsampled ss1 answer
        downsampled = cp.zeros_like(self.output_projs)
        for i, j in itertools.product(range(self.ssu), range(self.ssv)):
            cp.add(
                downsampled,
                self.output_projs_ss1[:, i :: self.ssu, j :: self.ssv],
                out=downsampled,
            )
        cp.divide(downsampled, self.ssu * self.ssv, out=downsampled)
        norm_ss1 = cp.linalg.norm(downsampled).get()
        assert norm_ss1 > 0  # make sure the test isn't trivial

        # Normalize so that the unit test remains consistent if I change the absolute value of the phantom or integrals later
        divisor = cp.float64(norm_ss1 / np.sqrt(downsampled.size))
        cp.divide(downsampled, divisor, out=downsampled)
        cp.divide(self.output_projs, divisor, out=self.output_projs)

        # imview.plot_projections(downsampled.get())
        # imview.plot_projections(self.output_projs.get())
        # imview.plot_projections((self.output_projs - downsampled).get())
        cp.testing.assert_array_almost_equal(
            self.output_projs, downsampled, decimal=4
        )  # decimal=4 has been tuned so that this test passes when the supersampling was bugfixed, but continues to fail for the non-bugfixed supersampling.


# ruff: enable[undocumented-public-class,undocumented-public-method,invalid-class-name,missing-return-type-undocumented-public-function]
