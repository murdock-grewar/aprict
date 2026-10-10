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
                ["half", "float"],
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
    argnames=["numprojs", "dtype_proj"],
    argvalues=[
        (1000, np.float16),
    ],
    ids=[
        "project_FORBILD-head",
    ],
)
class TestCB_phantom_FORBILD_head:
    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(
        self,
        numprojs: int,
        dtype_proj: np.dtype,
        request,  # ruff: ignore[missing-type-function-argument]
    ):
        test_id: str = request.node.callspec.id  # ruff: ignore[unused-variable]
        # Setup phase
        self.out_of_mem_count_arr = cp.zeros((1,), dtype=cp.uint32)
        self.shapes_cp_arr = cp.array(phantoms.FORBILD.HeadPhantom.shapes)

        # Set up configuration objects etc
        add_attenuations = False
        self.config = np.array(
            (1.0, add_attenuations, 1, 1), dtype=ks_config.dtype
        )
        num_projs = np.uint32(numprojs)
        detwid: int = 800
        dethei: int = 800
        self.output_projs: cp.ndarray = cp.zeros(
            (num_projs, detwid, dethei), dtype=dtype_proj
        )
        self.output_projs_strides_bytes = np.array(
            self.output_projs.strides, dtype=ks_ulvec3.dtype
        )
        # Prepare the source/detector positions
        projspec_array = []
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
                        (-0.1 * np.sin(theta), 0.1 * np.cos(theta), 0),
                        dtype=ks_fvec3.dtype,
                    ),
                    np.array((0, 0, 0.1), dtype=ks_fvec3.dtype),
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
            "project_CB_FORBILD_shapes<"
            + tdict[np.dtype(dtype_proj)]
            + ","
            + "FORBILD_Shape"
            + ">"
        )

        # yield to the timed test
        gpu_tools.clear_cache()
        cp.cuda.get_current_stream().synchronize()
        yield

        # Teardown phase
        # imview.plot_projections(self.output_projs, title=test_id)
        del self.output_projs

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


# ruff: enable[undocumented-public-class,undocumented-public-method,invalid-class-name,missing-return-type-undocumented-public-function]
