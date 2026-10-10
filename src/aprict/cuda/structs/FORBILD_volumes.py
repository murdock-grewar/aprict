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
"""Structures encoding FORBILD primitives and other related things."""

import pathlib

import numpy as np

from .. import kern_structs
from . import cu
from .base import (  # ruff: ignore[unused-import]
    bbox,
    fvec2,
    fvec3,
    uivec2,
    uivec3,
)

__all__ = [
    "Box",
    "ClipAxisSpec",
    "Cone",
    "Cone_x",
    "Cone_y",
    "Cone_z",
    "Cylinder",
    "Cylinder_x",
    "Cylinder_y",
    "Cylinder_z",
    "Ellipsoid",
    "Ellipsoid_free",
    "Ellipt_Cyl",
    "Ellipt_Cyl_x",
    "Ellipt_Cyl_y",
    "Ellipt_Cyl_z",
    "Shape",
    "Sphere",
]


_cu_src_dir = pathlib.Path(cu.__path__[0])


def _get_extra_code(struct_name: str) -> str:
    with open(
        _cu_src_dir.joinpath(struct_name + ".cu"), mode="r", encoding="utf-8"
    ) as f:
        return f.read()


# ruff: disable[undocumented-public-class,invalid-class-name]
class Sphere(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Sphere",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "r",
    ],
    dtypes=[np.float32, np.float32, np.float32, np.float32, np.float32],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[fvec2, fvec3, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Sphere"),
):
    pass


class Box(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Box",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "dx",  # side length along x axis
        "dy",
        "dz",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[fvec2, fvec3, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Box"),
):
    pass


class Cylinder(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cylinder",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "r",
        "l",
        "axis",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        fvec3,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[fvec2, fvec3, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Cylinder"),
):
    pass


class Cylinder_x(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cylinder_x",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "r",
        "l",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[Cylinder, fvec2, fvec3, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Cylinder_x"),
):
    pass


class Cylinder_y(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cylinder_y",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "r",
        "l",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[Cylinder, fvec2, fvec3, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Cylinder_y"),
):
    pass


class Cylinder_z(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cylinder_z",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "r",
        "l",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[Cylinder, fvec2, fvec3, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Cylinder_z"),
):
    pass


class Ellipsoid(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Ellipsoid",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "dx",  # half-axis length along x axis
        "dy",
        "dz",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[Sphere, fvec2, fvec3, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Ellipsoid"),
):
    pass


class Ellipsoid_free(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Ellipsoid_free",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "dx",
        "dy",
        "dz",
        "a_x",  # these are meant to be three orthogonal vectors. There's no enforcement of it, though.
        "a_y",
        "a_z",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        fvec3,
        fvec3,
        fvec3,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[Sphere, fvec2, fvec3, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Ellipsoid_free"),
):
    pass


class Ellipt_Cyl(  # Cylinder with an elliptical cross-section
    kern_structs.KernelStruct,
    struct_name="FORBILD_Ellipt_Cyl",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",  # length along principal axis
        "dx",  # half-axis length
        "dy",  # half-axis length
        # These vectors should be orthogonal, but there's no enforcement of that:
        "axis",  # direction of the principal direction.
        "a_x",  # direction of an orthogonal direction.
        "a_y",  # direction of an orthogonal direction.
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        fvec3,
        fvec3,
        fvec3,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[Cylinder_z, fvec2, fvec3],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Ellipt_Cyl"),
):
    pass


class Ellipt_Cyl_x(  # Cylinder with an elliptical cross-section
    kern_structs.KernelStruct,
    struct_name="FORBILD_Ellipt_Cyl_x",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",  # length along principal axis
        "dy",  # half-axis length
        "dz",  # half-axis length
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[Ellipt_Cyl, fvec2, fvec3, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Ellipt_Cyl_x"),
):
    pass


class Ellipt_Cyl_y(  # Cylinder with an elliptical cross-section
    kern_structs.KernelStruct,
    struct_name="FORBILD_Ellipt_Cyl_y",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",  # length along principal axis
        "dx",  # half-axis length
        "dz",  # half-axis length
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[Ellipt_Cyl, fvec2, fvec3, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Ellipt_Cyl_y"),
):
    pass


class Ellipt_Cyl_z(  # Cylinder with an elliptical cross-section
    kern_structs.KernelStruct,
    struct_name="FORBILD_Ellipt_Cyl_z",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",  # length along principal axis
        "dx",  # half-axis length
        "dy",  # half-axis length
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[Ellipt_Cyl, fvec2, fvec3, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Ellipt_Cyl_z"),
):
    pass


# Other cone cases may be reduced to this case
class Cone_z(  # Cone with axis parallel to x axis
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cone_z",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",
        "r1",  # radius at z = cone.z - cone.l/2
        "r2",  # radius at z = cone.z + cone.l/2
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[Cylinder_z, fvec2, fvec3, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Cone_z"),
):
    pass


"""Cone
truncated cone, arbitrary orientation in space as given by the axis
axis(expression,expression,expression). Length l, radii r1 and r2 at the two ends of the truncated cone. Moving along the cone axis in the direction given by axis, first the end corresponding to r1, then the end corresponding to r2 is met. x, y and z define the center of the truncated cone axis."""


class Cone(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cone",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",
        "r1",
        "r2",
        "axis",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        fvec3,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[Cone_z, fvec2, fvec3, Cylinder, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Cone"),
):
    pass


class Cone_x(  # Cone with axis parallel to x axis
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cone_x",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",
        "r1",
        "r2",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[Cone_z, fvec2, fvec3, Cylinder_x, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Cone_x"),
):
    pass


class Cone_y(  # Cone with axis parallel to x axis
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cone_y",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",
        "r1",
        "r2",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[Cone_z, fvec2, fvec3, Cylinder_y, bbox],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Cone_y"),
):
    pass


class ClipAxisSpec(  # Cone with axis parallel to x axis
    kern_structs.KernelStruct,
    struct_name="FORBILD_ClipAxisSpec",
    names=[
        "axis",  # note: it should be of unit length. This is NOT enforced!
        "dotprod",
        "mode",  # 0 = don't apply this plane. 1 --> enforce axis.r <= dotprod. 2 --> enforce axis.r > dotprod
    ],
    dtypes=[fvec3, np.float32, np.uint8],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[fvec2, fvec3],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_ClipAxisSpec"),
):
    pass


_dict_shape_name_to_type_num: dict[str, int] = {
    "Box": 1,
    "Cone": 2,
    "Cone_x": 3,
    "Cone_y": 4,
    "Cone_z": 5,
    "Cylinder": 6,
    "Cylinder_x": 7,
    "Cylinder_y": 8,
    "Cylinder_z": 9,
    "Ellipsoid": 10,
    "Ellipsoid_free": 11,
    "Ellipt_Cyl": 12,
    "Sphere": 13,
    "Ellipt_Cyl_x": 14,
    "Ellipt_Cyl_y": 15,
    "Ellipt_Cyl_z": 16,
}


class Shape(  # A generic container for any shape
    kern_structs.KernelStruct,
    struct_name="FORBILD_Shape",
    names=[
        "type",  # specifies which shape this is. 0 is reserved for "nothing", i.e. empty shape
        "rho",
        # up to three clipping planes
        "cas1",
        "cas2",
        "cas3",
        # List all fields which can occur in any shape
        "x",
        "y",
        "z",
        "dx",
        "dy",
        "dz",
        "r",
        "r1",
        "r2",
        "l",
        "a_x",
        "a_y",
        "a_z",
        "axis",
    ],
    dtypes=[
        np.uint8,  # type. value 0 makes this an non-existent shape
        np.float32,  # rho
        ClipAxisSpec,
        ClipAxisSpec,
        ClipAxisSpec,
        np.float32,  # x
        np.float32,  # y
        np.float32,  # z
        np.float32,  # dx
        np.float32,  # dy
        np.float32,  # dz
        np.float32,  # r
        np.float32,  # r1
        np.float32,  # r2
        np.float32,  # l
        fvec3,  # a_x
        fvec3,  # a_y
        fvec3,  # a_z
        fvec3,  # axis
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[
        fvec2,
        fvec3,
        bbox,
        Box,
        Cone,
        Cone_x,
        Cone_y,
        Cone_z,
        Cylinder,
        Cylinder_x,
        Cylinder_y,
        Cylinder_z,
        Ellipsoid,
        Ellipsoid_free,
        Ellipt_Cyl,
        Ellipt_Cyl_x,
        Ellipt_Cyl_y,
        Ellipt_Cyl_z,
        Sphere,
    ],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    extra_code=_get_extra_code("FORBILD_Shape"),
):
    pass


# ruff: enable[undocumented-public-class,invalid-class-name]
