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

from ...cuda import structs
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
    argnames=["numspheres", "add_attenuations", "dtype_proj"],
    argvalues=[
        (15, False, np.float16),
        (25, False, np.float32),
        (50, True, np.float16),
        (75, True, np.float32),
        (100, False, np.float16),
        # (1000, False, np.float32),
    ],
    ids=[
        "project-FORBILD-spheres-000015-fp16",
        "project-FORBILD-spheres-000025-fp32",
        "project-FORBILD-spheres-000050-fp16-add",
        "project-FORBILD-spheres-000075-fp32-add",
        "project-FORBILD-spheres-000100-fp16",
        # "project-FORBILD-spheres-001000-fp32",  # breaks due to high recursion
    ],
)
class TestCB_FORBILD_spheres:
    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(
        self,
        numspheres: int,
        add_attenuations: bool,
        dtype_proj: np.dtype,
        request,  # ruff: ignore[missing-type-function-argument]
    ):
        test_id: str = request.node.callspec.id  # ruff: ignore[unused-variable]
        # Setup phase
        # Get the FORBILD boxes
        shapes = []
        spheres = []
        for _ in range(numspheres):
            x, y, z = 10 * np.random.randn(3)  # ruff: ignore[numpy-legacy-random]
            r = np.random.uniform(10, 10)  # ruff: ignore[numpy-legacy-random]
            atten_coeff = np.random.uniform(0, 3)  # ruff: ignore[numpy-legacy-random]
            spheres.append(
                np.array(
                    (atten_coeff, x, y, z, r),
                    dtype=structs.FORBILD_volumes.Sphere,
                )
            )
            s = np.array(0, dtype=structs.FORBILD_volumes.Shape)
            s["type"] = 13  # sphere type is 13
            for key, val in zip(
                ["rho", "x", "y", "z", "r"],
                [atten_coeff, x, y, z, r],
            ):
                s[key] = val
        self.shapes_cp_arr = cp.array(
            np.array(shapes, dtype=structs.FORBILD_volumes.Shape, ndmax=1)
        )
        self.spheres_cp_arr = cp.array(
            np.array(spheres, dtype=structs.FORBILD_volumes.Sphere, ndmax=1)
        )

        # Set up configuration objects etc
        self.config = np.array(
            (1.0, add_attenuations, 1, 1), dtype=ks_config.dtype
        )
        num_projs = np.uint32(1000)
        detwid: int = 1000
        dethei: int = 500
        self.output_projs: cp.ndarray = cp.zeros(
            (num_projs, detwid, dethei), dtype=dtype_proj
        )
        self.output_projs_strides_bytes = np.array(
            self.output_projs.strides, dtype=ks_ulvec3.dtype
        )
        # Prepare the source/detector positions
        projspec_array = []
        ts = np.linspace(0, 2 * np.pi, endpoint=False, num=num_projs)
        volres: int = 100  # legacy. Just setting it to a reasonable value.
        for theta in ts:
            sourcepos = np.array(
                (
                    1750 * volres / 2501 * np.cos(theta),
                    1750 * volres / 2501 * np.sin(theta),
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
            + "FORBILD_Sphere"
            + ">"
        )

        # yield to the timed test
        cp.cuda.get_current_stream().synchronize()
        yield

        # Teardown phase
        # imview.plot_projections(self.output_projs, title=test_id)
        # self.output_projs.fill(0)
        del self.output_projs

    def test_project(self) -> None:
        # Prepare the kernel
        blockx = 8
        blocky = 8
        blocklayer = 10
        # Launch the kernel
        num_shapes: int = self.spheres_cp_arr.size
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
                np.uint64(self.spheres_cp_arr.data.ptr),
                np.uint32(num_shapes),
                self.output_projs_strides_bytes,
                np.uint64(self.output_projs.data.ptr),
            ),
        )
        cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.


@pytest.mark.parametrize(
    argnames=["numboxes", "add_attenuations", "dtype_proj"],
    argvalues=[
        (15, False, np.float16),
        (25, False, np.float32),
        (25, True, np.float32),
        (50, True, np.float16),
        (75, True, np.float32),
        (100, False, np.float16),
        # (1000, False, np.float32),
    ],
    ids=[
        "project-FORBILD-boxes-000015-fp16",
        "project-FORBILD-boxes-000025-fp32",
        "project-FORBILD-boxes-000025-fp32-add",
        "project-FORBILD-boxes-000050-fp16-add",
        "project-FORBILD-boxes-000075-fp32-add",
        "project-FORBILD-boxes-000100-fp16",
        # "project-FORBILD-spheres-001000-fp32",  # breaks due to high recursion
    ],
)
class TestCB_FORBILD_boxes:
    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(
        self,
        numboxes: int,
        add_attenuations: bool,
        dtype_proj: np.dtype,
        request,  # ruff: ignore[missing-type-function-argument]
    ):
        test_id: str = request.node.callspec.id  # ruff: ignore[unused-variable]
        # Setup phase
        # Get the FORBILD spheres
        shapes = []
        boxes = []
        for _ in range(numboxes):
            x, y, z = 10 * np.random.randn(3)  # ruff: ignore[numpy-legacy-random]
            dx = np.random.uniform(1, 10)  # ruff: ignore[numpy-legacy-random]
            dy = np.random.uniform(1, 10)  # ruff: ignore[numpy-legacy-random]
            dz = np.random.uniform(1, 10)  # ruff: ignore[numpy-legacy-random]
            atten_coeff = np.random.uniform(0, 3)  # ruff: ignore[numpy-legacy-random]
            s = np.array(0, dtype=structs.FORBILD_volumes.Shape)
            s["type"] = 13  # sphere type is 13
            boxes.append(
                np.array(
                    (atten_coeff, x, y, z, dx, dy, dz),
                    dtype=structs.FORBILD_volumes.Box,
                )
            )
            for key, val in zip(
                ["rho", "x", "y", "z", "dx", "dy", "dz"],
                [atten_coeff, x, y, z, dx, dy, dz],
            ):
                s[key] = val
            shapes.append(s)
        self.shapes_cp_arr = cp.array(
            np.array(shapes, dtype=structs.FORBILD_volumes.Shape, ndmax=1)
        )
        self.boxes_cp_arr = cp.array(
            np.array(boxes, dtype=structs.FORBILD_volumes.Box, ndmax=1)
        )

        # Set up configuration objects etc
        self.config = np.array(
            (1.0, add_attenuations, 1, 1), dtype=ks_config.dtype
        )
        num_projs = np.uint32(1000)
        detwid: int = 1000
        dethei: int = 500
        self.output_projs: cp.ndarray = cp.zeros(
            (num_projs, detwid, dethei), dtype=dtype_proj
        )
        self.output_projs_strides_bytes = np.array(
            self.output_projs.strides, dtype=ks_ulvec3.dtype
        )
        # Prepare the source/detector positions
        projspec_array = []
        ts = np.linspace(0, 2 * np.pi, endpoint=False, num=num_projs)
        volres: int = 100  # legacy. Just setting it to a reasonable value.
        for theta in ts:
            sourcepos = np.array(
                (
                    1750 * volres / 2501 * np.cos(theta),
                    1750 * volres / 2501 * np.sin(theta),
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
            + "FORBILD_Box"
            + ">"
        )

        # yield to the timed test
        cp.cuda.get_current_stream().synchronize()
        yield

        # Teardown phase
        # imview.plot_projections(self.output_projs, title=test_id)
        # self.output_projs.fill(0)
        del self.output_projs

    def test_project(self) -> None:
        # Prepare the kernel
        blockx = 8
        blocky = 8
        blocklayer = 10
        # Launch the kernel
        num_shapes: int = self.boxes_cp_arr.size
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
                np.uint64(self.boxes_cp_arr.data.ptr),
                np.uint32(num_shapes),
                self.output_projs_strides_bytes,
                np.uint64(self.output_projs.data.ptr),
            ),
        )
        cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.


@pytest.mark.parametrize(
    argnames=["numcyls", "add_attenuations", "dtype_proj"],
    argvalues=[
        (15, False, np.float16),
        (25, False, np.float32),
        (25, True, np.float32),
        (50, True, np.float16),
        (75, True, np.float32),
        (100, False, np.float16),
        # (1000, False, np.float32),
    ],
    ids=[
        "project-FORBILD-cyls-000015-fp16",
        "project-FORBILD-cyls-000025-fp32",
        "project-FORBILD-cyls-000025-fp32-add",
        "project-FORBILD-cyls-000050-fp16-add",
        "project-FORBILD-cyls-000075-fp32-add",
        "project-FORBILD-cyls-000100-fp16",
        # "project-FORBILD-spheres-001000-fp32",  # breaks due to high recursion
    ],
)
class TestCB_FORBILD_cyls:
    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(
        self,
        numcyls: int,
        add_attenuations: bool,
        dtype_proj: np.dtype,
        request,  # ruff: ignore[missing-type-function-argument]
    ):
        test_id: str = request.node.callspec.id  # ruff: ignore[unused-variable]
        # Setup phase
        # Get the FORBILD cylinders
        shapes = []
        cyls = []
        for _ in range(numcyls):
            x, y, z = 6 * np.random.randn(3)  # ruff: ignore[numpy-legacy-random]
            r = np.random.uniform(1, 5)  # ruff: ignore[numpy-legacy-random]
            l = np.random.uniform(1, 10)  # ruff: ignore[numpy-legacy-random]
            ax, ay, az = np.random.randn(3)  # ruff: ignore[numpy-legacy-random]
            atten_coeff = np.random.uniform(0, 3)  # ruff: ignore[numpy-legacy-random]
            s = np.array(0, dtype=structs.FORBILD_volumes.Shape)
            s["type"] = 13  # sphere type is 13
            axis = np.array((ax, ay, az), dtype=structs.fvec3)
            cyls.append(
                np.array(
                    (atten_coeff, x, y, z, r, l, axis),
                    dtype=structs.FORBILD_volumes.Cylinder,
                )
            )
            for key, val in zip(
                ["rho", "x", "y", "z", "r", "l", "axis"],
                [atten_coeff, x, y, z, r, l, axis],
            ):
                s[key] = val
            shapes.append(s)
        self.shapes_cp_arr = cp.array(
            np.array(shapes, dtype=structs.FORBILD_volumes.Shape, ndmax=1)
        )
        self.cyls_cp_arr = cp.array(
            np.array(cyls, dtype=structs.FORBILD_volumes.Cylinder, ndmax=1)
        )

        # Set up configuration objects etc
        self.config = np.array(
            (1.0, add_attenuations, 1, 1), dtype=ks_config.dtype
        )
        num_projs = np.uint32(1000)
        detwid: int = 1000
        dethei: int = 500
        self.output_projs: cp.ndarray = cp.zeros(
            (num_projs, detwid, dethei), dtype=dtype_proj
        )
        self.output_projs_strides_bytes = np.array(
            self.output_projs.strides, dtype=ks_ulvec3.dtype
        )
        # Prepare the source/detector positions
        projspec_array = []
        ts = np.linspace(0, 2 * np.pi, endpoint=False, num=num_projs)
        volres: int = 100  # legacy. Just setting it to a reasonable value.
        for theta in ts:
            sourcepos = np.array(
                (
                    1750 * volres / 2501 * np.cos(theta),
                    1750 * volres / 2501 * np.sin(theta),
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
            + "FORBILD_Cylinder"
            + ">"
        )

        # yield to the timed test
        gpu_tools.clear_cache()
        cp.cuda.get_current_stream().synchronize()
        yield

        # Teardown phase
        # imview.plot_projections(self.output_projs, title=test_id)
        # self.output_projs.fill(0)
        del self.output_projs

    def test_project(self) -> None:
        # Prepare the kernel
        blockx = 8
        blocky = 8
        blocklayer = 10
        # Launch the kernel
        num_shapes: int = self.cyls_cp_arr.size
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
                np.uint64(self.cyls_cp_arr.data.ptr),
                np.uint32(num_shapes),
                self.output_projs_strides_bytes,
                np.uint64(self.output_projs.data.ptr),
            ),
        )
        cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.


@pytest.mark.parametrize(
    argnames=["numellipsoids", "add_attenuations", "dtype_proj"],
    argvalues=[
        (15, False, np.float16),
        (25, False, np.float32),
        (25, True, np.float32),
        (50, True, np.float16),
        # (75, True, np.float32),
        # (100, False, np.float16),
        # (1000, False, np.float32),
    ],
    ids=[
        "project-FORBILD-ellipsoids-000015-fp16",
        "project-FORBILD-ellipsoids-000025-fp32",
        "project-FORBILD-ellipsoids-000025-fp32-add",
        "project-FORBILD-ellipsoids-000050-fp16-add",
        # "project-FORBILD-ellipsoids-000075-fp32-add",
        # "project-FORBILD-ellipsoids-000100-fp16",
        # "project-FORBILD-spheres-001000-fp32",  # breaks due to high recursion
    ],
)
class TestCB_FORBILD_ellipsoids:
    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(
        self,
        numellipsoids: int,
        add_attenuations: bool,
        dtype_proj: np.dtype,
        request,  # ruff: ignore[missing-type-function-argument]
    ):
        test_id: str = request.node.callspec.id  # ruff: ignore[unused-variable]
        # Setup phase
        # Get the FORBILD cylinders
        ellipsoids = []
        for _ in range(numellipsoids):
            x, y, z = 6 * np.random.randn(3)  # ruff: ignore[numpy-legacy-random]
            dx, dy, dz = 3 * np.random.randn(3)  # ruff: ignore[numpy-legacy-random]
            atten_coeff = np.random.uniform(0, 3)  # ruff: ignore[numpy-legacy-random]
            a_x = np.random.randn(3)  # ruff: ignore[numpy-legacy-random]
            a_y = np.random.randn(3)  # ruff: ignore[numpy-legacy-random]
            a_y -= (a_x / np.linalg.norm(a_x)) * np.dot(
                a_x / np.linalg.norm(a_x), a_y
            )
            a_z = np.cross(a_x, a_y)
            a_x = np.array(tuple(a_x), dtype=structs.fvec3)
            a_y = np.array(tuple(a_y), dtype=structs.fvec3)
            a_z = np.array(tuple(a_z), dtype=structs.fvec3)
            ellipsoid = np.array(
                (atten_coeff, x, y, z, dx, dy, dz, a_x, a_y, a_z),
                dtype=structs.FORBILD_volumes.Ellipsoid_free,
            )
            ellipsoids.append(ellipsoid)
        self.ellipsoids_cp_arr = cp.array(
            np.array(
                ellipsoids,
                dtype=structs.FORBILD_volumes.Ellipsoid_free,
                ndmax=1,
            )
        )

        # Set up configuration objects etc
        self.config = np.array(
            (1.0, add_attenuations, 1, 1), dtype=ks_config.dtype
        )
        num_projs = np.uint32(1000)
        detwid: int = 1000
        dethei: int = 500
        self.output_projs: cp.ndarray = cp.zeros(
            (num_projs, detwid, dethei), dtype=dtype_proj
        )
        self.output_projs_strides_bytes = np.array(
            self.output_projs.strides, dtype=ks_ulvec3.dtype
        )
        # Prepare the source/detector positions
        projspec_array = []
        ts = np.linspace(0, 2 * np.pi, endpoint=False, num=num_projs)
        volres: int = 100  # legacy. Just setting it to a reasonable value.
        for theta in ts:
            sourcepos = np.array(
                (
                    1750 * volres / 2501 * np.cos(theta),
                    1750 * volres / 2501 * np.sin(theta),
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
            + "FORBILD_Ellipsoid_free"
            + ">"
        )

        # yield to the timed test
        gpu_tools.clear_cache()
        cp.cuda.get_current_stream().synchronize()
        yield

        # Teardown phase
        # imview.plot_projections(self.output_projs, title=test_id)
        # self.output_projs.fill(0)
        del self.output_projs

    def test_project(self) -> None:
        # Prepare the kernel
        blockx = 8
        blocky = 8
        blocklayer = 10
        # Launch the kernel
        num_shapes: int = self.ellipsoids_cp_arr.size
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
                np.uint64(self.ellipsoids_cp_arr.data.ptr),
                np.uint32(num_shapes),
                self.output_projs_strides_bytes,
                np.uint64(self.output_projs.data.ptr),
            ),
        )
        cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.


@pytest.mark.parametrize(
    argnames=["numcones", "add_attenuations", "dtype_proj"],
    argvalues=[
        # (1, False, np.float16),
        (15, False, np.float16),
        # (25, False, np.float32),
        # (25, True, np.float32),
        (50, True, np.float16),
        # (75, True, np.float32),
        (100, False, np.float16),
        # (1000, False, np.float32),
    ],
    ids=[
        # "project-FORBILD-cones-000001-fp16",
        "project-FORBILD-cones-000015-fp16",
        # "project-FORBILD-cones-000025-fp32",
        # "project-FORBILD-cones-000025-fp32-add",
        "project-FORBILD-cones-000050-fp16-add",
        # "project-FORBILD-cones-000075-fp32-add",
        "project-FORBILD-cones-000100-fp16",
        # "project-FORBILD-spheres-001000-fp32",  # breaks due to high recursion
    ],
)
class TestCB_FORBILD_cones:
    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(
        self,
        numcones: int,
        add_attenuations: bool,
        dtype_proj: np.dtype,
        request,  # ruff: ignore[missing-type-function-argument]
    ):
        test_id: str = request.node.callspec.id  # ruff: ignore[unused-variable]
        # Setup phase
        # Get the FORBILD cylinders
        cones = []
        for _ in range(numcones):
            x, y, z = 6 * np.random.randn(3)  # ruff: ignore[numpy-legacy-random]
            l = np.random.uniform(1, 4)  # ruff: ignore[numpy-legacy-random]
            r1 = np.random.uniform(0, 4)  # ruff: ignore[numpy-legacy-random]
            r2 = np.random.uniform(0, 4)  # ruff: ignore[numpy-legacy-random]
            atten_coeff = np.random.uniform(0, 3)  # ruff: ignore[numpy-legacy-random]
            axis = np.array(tuple(np.random.randn(3)), dtype=structs.fvec3)  # ruff: ignore[numpy-legacy-random]
            cone = np.array(
                (atten_coeff, x, y, z, l, r1, r2, axis),
                dtype=structs.FORBILD_volumes.Cone,
            )
            cones.append(cone)
        self.shapes_cp_arr = cp.array(
            np.array(
                cones,
                dtype=structs.FORBILD_volumes.Cone,
                ndmax=1,
            )
        )

        # Set up configuration objects etc
        self.config = np.array(
            (1.0, add_attenuations, 1, 1), dtype=ks_config.dtype
        )
        num_projs = np.uint32(1000)
        detwid: int = 1000
        dethei: int = 500
        self.output_projs: cp.ndarray = cp.zeros(
            (num_projs, detwid, dethei), dtype=dtype_proj
        )
        self.output_projs_strides_bytes = np.array(
            self.output_projs.strides, dtype=ks_ulvec3.dtype
        )
        # Prepare the source/detector positions
        projspec_array = []
        ts = np.linspace(0, 2 * np.pi, endpoint=False, num=num_projs)
        volres: int = 100  # legacy. Just setting it to a reasonable value.
        for theta in ts:
            sourcepos = np.array(
                (
                    1750 * volres / 2501 * np.cos(theta),
                    1750 * volres / 2501 * np.sin(theta),
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
            + "FORBILD_Cone"
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
        blockx = 8
        blocky = 8
        blocklayer = 10
        # Launch the kernel
        num_shapes: int = self.shapes_cp_arr.size
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
                np.uint64(self.shapes_cp_arr.data.ptr),
                np.uint32(num_shapes),
                self.output_projs_strides_bytes,
                np.uint64(self.output_projs.data.ptr),
            ),
        )
        cp.cuda.get_current_stream().synchronize()  # Needed to ensure that the test doesn't return early before the computation is complete.


# ruff: enable[undocumented-public-class,undocumented-public-method,invalid-class-name,missing-return-type-undocumented-public-function]
