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

# volume.py
"""Class to abstract away the details of a 3D voxelised volume."""
# Developer's note: this class is intended to contain the end-user API
# to the volume data. The user should not have to interact with internal
# classes like texture_tools.Texture, and the like.

import pathlib  # noqa: TC003
import types
import typing

import cupy as cp
import cupyx
import numpy as np

from .cuda.structs import FORBILD_volumes as fv
from .lib import gpu_tools, ne_tools, sample_FORBILD, texture_tools
from .lib.texture_tools import operations as tops

_ShapeType = tuple[int, int, int]
_DType_base = np.float16 | np.float32
_DType = np.dtype[np.float16] | np.dtype[np.float32]


# TODO fix up the typing of the shape and dtype once astral-sh implements the relevant features into ty  # noqa: FIX002


class Volume[
    Shape: _ShapeType,  # Shape will be a subtype, like tuple[Literal[10], Literal[10], Literal[10]]
    DType: _DType,
]:
    """A rectangular prism of voxels. Underpinned by a CUDA surface object."""

    # Used to store the TypeVars which are inferred from the input.
    _shape: _ShapeType
    _dtype: _DType

    # Other fields
    _texture: texture_tools.Texture
    _pos_c: np.ndarray[
        tuple[typing.Literal[3],], np.dtype[np.float32]
    ]  # location of the centre of the volume, in space
    _voxwid: float

    @property
    def voxwid(self) -> float:
        """Return the width of each voxel, in arbitrary units of distance."""
        return self._voxwid

    @property
    def shape(self) -> _ShapeType:
        """Return the shape of the volume array."""
        return self._shape

    @property
    def norm(self) -> float:
        """Return the l2 norm of the volume."""
        return tops.norm(self._texture)

    @property
    def dtype(self) -> _DType:
        """Return the dtype of the volume array."""
        return self._dtype

    @property
    def texture(self) -> texture_tools.Texture:
        """Return the underlying Texture object."""
        return self._texture

    @property
    def pos_c(
        self,
    ) -> np.ndarray[tuple[typing.Literal[3],], np.dtype[np.float32]]:
        """Return the position of the centre of the volume."""
        return self._pos_c.copy()

    def __init__(
        self,
        arr: np.ndarray[
            Shape, DType
        ],  # this typing ensures the correct dimensions (3) and a supported numpy dtype
        *,
        voxwid: float,
        pos_c: np.ndarray[tuple[typing.Literal[3]], np.dtype[np.float32]]
        | tuple[float, float, float] = np.array((0, 0, 0), dtype=np.float32),
    ) -> None:
        # Essentially, we are storing the resolved types of the TypeVars
        self._dtype = arr.dtype
        self._shape = arr.shape
        # Create the texture and load the data
        self._texture = self._new_texture(shape=self._shape, dtype=self._dtype)
        self._voxwid = voxwid
        self._pos_c = np.array(pos_c)
        self.copy_from(arr)

    def __del__(self) -> None:
        """Object cleanup."""
        gpu_tools.clear_cache()  # Ensure that CuPy cleans up the used memory.

    def multiply(self, f: float) -> None:
        """Multiply the volume in-place by a value."""
        tops.multiply(self._texture, f)

    def as_numpy(self, out: np.ndarray | None = None) -> np.ndarray:
        """Copy this volume's data into a numpy array."""
        x = (
            out
            if out is not None
            else cupyx.empty_pinned(self._shape, dtype=self._dtype)
        )
        self._texture.cuda_array.copy_to(
            x
        )  # Warning: this copies asynchronously!
        cp.cuda.get_current_stream().synchronize()  # This SHOULD NOT BE NECESSARY when on the null stream, but it is. CuPy may be bugged, or I am misunderstanding something.
        return x

    def save(self, path: pathlib.Path) -> None:
        """Save this volume into a numpy array on disk."""
        x = self.as_numpy()
        ne_tools.save(path, x)

    def as_cupy(self, out: cp.ndarray | None = None) -> cp.ndarray:
        """Copy this volume's data into a cupy array."""
        x = (
            out
            if out is not None
            else cp.empty(self._shape, dtype=self._dtype)
        )
        self._texture.cuda_array.copy_to(
            x
        )  # Warning: this copies asynchronously!
        cp.cuda.get_current_stream().synchronize()  # This SHOULD NOT BE NECESSARY when on the null stream, but it is. CuPy may be bugged, or I am misunderstanding something.
        return x

    def as_cupy_slice(
        self, axis: int, sl: slice, *, out: cp.ndarray | None = None
    ) -> cp.ndarray:
        """Copy a slice of this volume's data into a cupy array."""
        # Check for supported slice argument
        return tops.slice_nth_axis_to_3d_cupy_array(
            t=self._texture, out=out, axis=axis, sl=sl
        )

    def sum_to_cupy_array(
        self, axis: int, *, out: cp.ndarray | None = None
    ) -> cp.ndarray:
        """Sum the volume along an axis, into a 2D cupy array."""
        return tops.sum_along_nth_axis_to_2d_cupy_array(
            t=self._texture, out=out, axis=axis
        )

    @classmethod
    def _new_texture(cls, shape: Shape, dtype: DType) -> texture_tools.Texture:
        return texture_tools.Texture(
            (dtype, shape),
            address_mode=texture_tools.AddressMode.cudaAddressModeBorder,
            filter_mode=texture_tools.FilterMode.cudaFilterModePoint,
            layered=False,
            normalise_coordinates=False,
            include_texture=True,
            include_surface=True,
        )

    @classmethod
    def zeros(
        cls,
        shape: _ShapeType,
        *,
        voxwid: float,
        dtype: _DType | _DType_base = np.float32,  # ty:ignore[invalid-parameter-default]
        pos_c: np.ndarray[tuple[typing.Literal[3]], np.dtype[np.float32]]
        | tuple[float, float, float] = np.array((0, 0, 0), dtype=np.float32),
    ) -> Volume:
        """Initialise a new volume, filled with zeros."""
        if type(shape) is not tuple or len(shape) != 3:
            msg = "Invalid shape provided. It should be a tuple of three positive integers."
            raise ValueError(msg)
        self: Volume = cls.__new__(cls)
        self._voxwid = voxwid
        self._dtype = np.dtype(dtype)
        self._shape = shape
        self._pos_c = np.array(pos_c)
        self._texture = self._new_texture(shape=self._shape, dtype=self._dtype)
        self.fill(0)
        return self

    def zeroed_copy(self) -> Volume:
        """Return a copy of this volume, but filled with zeros."""
        return Volume.zeros(
            self._shape,
            voxwid=self._voxwid,
            dtype=self._dtype,
            pos_c=self.pos_c,
        )

    def fill(self, value: int | float) -> None:
        """Fill the volume with a constant value."""
        tops.fill(self._texture, value)

    def copy_from(self, arr: np.ndarray[Shape, DType]) -> None:
        """Copy a numpy/cupy array into this Volume object."""
        self._texture.cuda_array.copy_from(
            arr
        )  # Warning: this copies asynchronously!
        cp.cuda.get_current_stream().synchronize()  # This SHOULD NOT BE NECESSARY when on the null stream, but it is. CuPy may be bugged, or I am misunderstanding something.

    def copy_to(self, arr: np.ndarray[Shape, DType]) -> None:
        """Copy this Volume into a numpy/cupy array."""
        self._texture.cuda_array.copy_to(
            arr
        )  # Warning: this copies asynchronously!
        cp.cuda.get_current_stream().synchronize()  # This SHOULD NOT BE NECESSARY when on the null stream, but it is. CuPy may be bugged, or I am misunderstanding something.

    def soften_to_xy_cylinder(
        self,
        relrad_start: float,
    ) -> None:
        """Softmask the volume to a cylinder coaxial with the z axis."""
        tops.softmask_to_cylinder_xy(self._texture, relrad_start)


class FBVolume:
    """A volume specified in FORBILD format."""

    shapes: np.ndarray[np.dtype[fv.Shape]]  # ty:ignore[invalid-type-arguments]
    _kernel_point_sampling: cp.RawKernel

    def __init__(self, shapes: np.ndarray[np.dtype[fv.Shape]]) -> None:  # ty:ignore[invalid-type-arguments]
        self.shapes = shapes

    def to_cupy(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        *,
        out: cp.ndarray | None = None,
        supersamp_cubic_vox_wid: float | None = None,
        supersamp_factor: int | None = None,
        block: bool = True,
    ) -> cp.ndarray[tuple[int, int, int], np.dtype[np.float32]]:
        """Convert to a float32 cupy array by simple point-sampling."""
        return sample_FORBILD.sample_FORBILD(
            self.shapes,
            xs,
            ys,
            zs,
            out=out,
            supersamp_cubic_vox_wid=supersamp_cubic_vox_wid,
            supersamp_factor=supersamp_factor,
            block=block,
        )

    def superimpose_onto(self, vol: FBVolume) -> FBVolume:
        """Superimpose this volume onto another."""
        return FBVolume(
            shapes=np.concatenate((vol.shapes, self.shapes), axis=0)
        )

    def to_numpy(
        self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float32]]:
        """Convert to a float32 numpy array by simple point-sampling."""
        return self.to_cupy(xs, ys, zs).get()

    @classmethod
    def _scale_shape_len(cls, shape, scale: float):  # noqa: ANN001, ANN206
        s = shape.copy()
        for key in ("cas1", "cas2", "cas3"):
            s[key]["dotprod"] *= scale
        for key in (
            "x",
            "y",
            "z",
            "r",
            "l",
            "dx",
            "dy",
            "dz",
            "r1",
            "r2",
        ):
            s[key] *= scale
        return s

    def scale_len(self, factor: float) -> None:
        """Scale up the lengths in the volume by the given factor."""
        self.shapes = np.array(
            [
                FBVolume._scale_shape_len(shape, factor)
                for shape in self.shapes
            ],
            dtype=fv.Shape,
        )

    @classmethod
    def _flip_shape(cls, shape, axis: int | str):  # noqa: ANN001, ANN206
        match axis:
            case 0 | "x":
                axis = "x"
                relevant_number_subkeys = ("x",)
            case 1 | "y":
                axis = "y"
                relevant_number_subkeys = ("y",)
            case 2 | "z":
                axis = "z"
                relevant_number_subkeys = ("z",)
            case _:
                msg = f"Unknown axis: {axis}"
                raise ValueError(msg)
        s = shape.copy()
        for key in ("cas1", "cas2", "cas3"):
            s[key]["axis"][axis] *= -1
        for key in relevant_number_subkeys:
            s[key] *= -1
        for key in (
            "axis",
            "a_x",
            "a_y",
            "a_z",
        ):
            s[key][axis] *= -1
        return s

    def flip(self, axis: int | str) -> None:
        """Flip the object along one of the three axes."""
        self.shapes = [  # ty:ignore[invalid-assignment]
            FBVolume._flip_shape(shape, axis=axis) for shape in self.shapes
        ]

    @classmethod
    def _translate_shape(cls, shape, axis: int | str, amount: float):  # noqa: ANN001, ANN206
        match axis:
            case 0 | "x":
                axis = "x"
                relevant_number_subkeys = ("x",)
            case 1 | "y":
                axis = "y"
                relevant_number_subkeys = ("y",)
            case 2 | "z":
                axis = "z"
                relevant_number_subkeys = ("z",)
            case _:
                msg = f"Unknown axis: {axis}"
                raise ValueError(msg)
        s = shape.copy()
        for key in relevant_number_subkeys:
            s[key] += amount
        return s

    def translate(self, axis: int | str, amount: float) -> None:
        """Translate the shape along the given axis, by the given amount."""
        self.shapes = np.array(
            [
                FBVolume._translate_shape(shape, axis=axis, amount=amount)
                for shape in self.shapes
            ],
            dtype=fv.Shape,
        )

    @classmethod
    def _shape_FORBIRD_str(cls, shape) -> str:  # noqa: ANN001, C901, N802
        type_int = shape["type"]
        id_to_str = {v: k for k, v in fv._dict_shape_name_to_type_num.items()}  # noqa: SLF001
        type_str = id_to_str[int(type_int)]
        ns = types.SimpleNamespace(
            {k: v for k, v in zip(shape.dtype.names, shape)}
        )
        cas1 = types.SimpleNamespace(
            {k: v for k, v in zip(ns.cas1.dtype.names, ns.cas1)}
        )
        cas1_axis = types.SimpleNamespace(
            {k: v for k, v in zip(cas1.axis.dtype.names, cas1.axis)}
        )
        cas2 = types.SimpleNamespace(
            {k: v for k, v in zip(ns.cas2.dtype.names, ns.cas2)}
        )
        cas2_axis = types.SimpleNamespace(
            {k: v for k, v in zip(cas2.axis.dtype.names, cas2.axis)}
        )
        cas3 = types.SimpleNamespace(
            {k: v for k, v in zip(ns.cas3.dtype.names, ns.cas3)}
        )
        cas3_axis = types.SimpleNamespace(
            {k: v for k, v in zip(cas3.axis.dtype.names, cas3.axis)}
        )
        cas_str = (
            (
                ""
                if cas1.mode == 0
                else (
                    " r({x},{y},{z})".format(**cas1_axis.__dict__)
                    + ("<" if cas1.mode == 1 else ">")
                    + "{dotprod}".format(**cas1.__dict__)
                )
            )
            + (
                ""
                if cas2.mode == 0
                else (
                    " r({x},{y},{z})".format(**cas2_axis.__dict__)
                    + ("<" if cas2.mode == 1 else ">")
                    + "{dotprod}".format(**cas2.__dict__)
                )
            )
            + (
                ""
                if cas3.mode == 0
                else (
                    " r({x},{y},{z})".format(**cas3_axis.__dict__)
                    + ("<" if cas3.mode == 1 else ">")
                    + "{dotprod}".format(**cas3.__dict__)
                )
            )
        )
        match type_str:
            case "Box":
                out = (
                    "{ [Box: "
                    + "x={x} y={y} z={z} dx={dx} dy={dy} dz={dz}".format(
                        **ns.__dict__
                    )
                    + cas_str
                    + "] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Cone":
                axis_ns = types.SimpleNamespace(
                    {k: v for k, v in zip(ns.axis.dtype.names, ns.axis)}
                )
                out = (
                    "{ [Cone: "
                    + "x={x} y={y} z={z} l={l} r1={r1} r2={r2}".format(
                        **ns.__dict__
                    )
                    + " axis({x},{y},{z})".format(**axis_ns.__dict__)
                    + cas_str
                    + "] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Cone_x":
                out = (
                    "{ [Cone_x: "
                    + "x={x} y={y} z={z} l={l} r1={r1} r2={r2}".format(
                        **ns.__dict__
                    )
                    + cas_str
                    + " ] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Cone_y":
                out = (
                    "{ [Cone_y: "
                    + "x={x} y={y} z={z} l={l} r1={r1} r2={r2}".format(
                        **ns.__dict__
                    )
                    + cas_str
                    + " ] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Cone_z":
                out = (
                    "{ [Cone_z: "
                    + "x={x} y={y} z={z} l={l} r1={r1} r2={r2}".format(
                        **ns.__dict__
                    )
                    + cas_str
                    + " ] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Cylinder":
                axis_ns = types.SimpleNamespace(
                    {k: v for k, v in zip(ns.axis.dtype.names, ns.axis)}
                )
                out = (
                    "{ [Cylinder: "
                    + "x={x} y={y} z={z} l={l} r={r}".format(**ns.__dict__)
                    + " axis({x},{y},{z})".format(**axis_ns.__dict__)
                    + cas_str
                    + "] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Cylinder_x":
                out = (
                    "{ [Cylinder_x: "
                    + "x={x} y={y} z={z} l={l} r={r}".format(**ns.__dict__)
                    + cas_str
                    + "] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Cylinder_y":
                out = (
                    "{ [Cylinder_y: "
                    + "x={x} y={y} z={z} l={l} r={r}".format(**ns.__dict__)
                    + cas_str
                    + "] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Cylinder_z":
                out = (
                    "{ [Cylinder_z: "
                    + "x={x} y={y} z={z} l={l} r={r}".format(**ns.__dict__)
                    + cas_str
                    + "] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Ellipsoid":
                out = (
                    "{ [Ellipsoid: "
                    + "x={x} y={y} z={z} dx={dx} dy={dy} dz={dz}".format(
                        **ns.__dict__
                    )
                    + cas_str
                    + "] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Ellipsoid_free":
                a_x_ns = types.SimpleNamespace(
                    {k: v for k, v in zip(ns.a_x.dtype.names, ns.a_x)}
                )
                a_y_ns = types.SimpleNamespace(
                    {k: v for k, v in zip(ns.a_y.dtype.names, ns.a_y)}
                )
                a_z_ns = types.SimpleNamespace(
                    {k: v for k, v in zip(ns.a_z.dtype.names, ns.a_z)}
                )
                out = (
                    "{ [Ellipsoid_free: "
                    + "x={x} y={y} z={z} dx={dx} dy={dy} dz={dz}".format(
                        **ns.__dict__
                    )
                    + " a_x({x},{y},{z})".format(**a_x_ns.__dict__)
                    + " a_y({x},{y},{z})".format(**a_y_ns.__dict__)
                    + " a_z({x},{y},{z})".format(**a_z_ns.__dict__)
                    + cas_str
                    + "] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Ellipt_Cyl":
                a_x_ns = types.SimpleNamespace(
                    {k: v for k, v in zip(ns.a_x.dtype.names, ns.a_x)}
                )
                a_y_ns = types.SimpleNamespace(
                    {k: v for k, v in zip(ns.a_y.dtype.names, ns.a_y)}
                )
                axis_ns = types.SimpleNamespace(
                    {k: v for k, v in zip(ns.axis.dtype.names, ns.axis)}
                )
                out = (
                    "{ [Ellipt_Cyl: "
                    + "x={x} y={y} z={z} l={l} dx={dx} dy={dy} dz={dz}".format(
                        **ns.__dict__
                    )
                    + " a_x({x},{y},{z})".format(**a_x_ns.__dict__)
                    + " a_y({x},{y},{z})".format(**a_y_ns.__dict__)
                    + " axis({x},{y},{z})".format(**axis_ns.__dict__)
                    + cas_str
                    + "] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Sphere":
                out = (
                    "{ [Sphere: "
                    + "x={x} y={y} z={z} r={r}".format(**ns.__dict__)
                    + cas_str
                    + " ] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Ellipt_Cyl_x":
                axis_ns = types.SimpleNamespace(
                    {k: v for k, v in zip(ns.axis.dtype.names, ns.axis)}
                )
                out = (
                    "{ [Ellipt_Cyl_x: "
                    + "x={x} y={y} z={z} l={l} dy={dy} dz={dz}".format(
                        **ns.__dict__
                    )
                    + " axis({x},{y},{z})".format(**axis_ns.__dict__)
                    + cas_str
                    + "] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Ellipt_Cyl_y":
                axis_ns = types.SimpleNamespace(
                    {k: v for k, v in zip(ns.axis.dtype.names, ns.axis)}
                )
                out = (
                    "{ [Ellipt_Cyl_y: "
                    + "x={x} y={y} z={z} l={l} dx={dx} dz={dz}".format(
                        **ns.__dict__
                    )
                    + " axis({x},{y},{z})".format(**axis_ns.__dict__)
                    + cas_str
                    + "] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case "Ellipt_Cyl_z":
                axis_ns = types.SimpleNamespace(
                    {k: v for k, v in zip(ns.axis.dtype.names, ns.axis)}
                )
                out = (
                    "{ [Ellipt_Cyl_z: "
                    + "x={x} y={y} z={z} l={l} dx={dx} dy={dy}".format(
                        **ns.__dict__
                    )
                    + " axis({x},{y},{z})".format(**axis_ns.__dict__)
                    + cas_str
                    + "] rho={rho}".format(**ns.__dict__)
                    + " }"
                )
            case _:
                raise NotImplementedError()
        return out

    @property
    def FORBILD_str(self) -> str:  # noqa: N802
        """Return a string representing this volume in FORBILD format."""
        return "\n".join([self._shape_FORBIRD_str(s) for s in self.shapes])
