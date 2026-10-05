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

# flatpanel.py

"""End-user interfaces to the flat-panel cone-beam projection and backprojection CUDA code."""

import dataclasses
import functools
import itertools
import json
import logging  # ruff: ignore[typing-only-standard-library-import]
import math
import pathlib
import typing

import cupy as cp
import numpy as np
import numpy.typing as npt  # ruff: ignore[unused-import]

from .. import logtools
from ..cuda import header_injection, structs
from ..cuda.kern_structs import _np_dtype_to_c_type
from ..volume import FBVolume, Volume

_delay_compile: bool = (
    True  # only compile the CUDA kernels when they are first called.
)


def force_compile() -> None:
    """Force the relevant CUDA kernels to compile."""
    _projection_module.compile()
    _backprojection_module.compile()


def _injected_module_project_texture_chunked() -> cp.RawModule:
    # Inject and compile module
    module: cp.RawModule = header_injection.load_injected_module(
        "./CB_project_texture_chunked.cu",
        name_expressions=[
            f"project_CB_texture_chunked<{t1:s},{t2:s}>"
            for t1, t2 in itertools.product(
                ["half", "float"], ["half", "float", "double"]
            )
        ]
        + [
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
    if not _delay_compile:
        module.compile()
    return module


def _injected_module_backproject() -> cp.RawModule:
    # Inject and compile module
    module: cp.RawModule = header_injection.load_injected_module(
        "./CB_backproject.cu",
        name_expressions=[
            f"backproject_CB_voltex<{t1:s},{t2:s}>"
            for t1, t2 in itertools.product(
                ["half", "float"], ["half", "float", "double"]
            )
        ],
    )
    if not _delay_compile:
        module.compile()
    return module


_projection_module: cp.RawModule = _injected_module_project_texture_chunked()
_backprojection_module: cp.RawModule = _injected_module_backproject()


def _select_projection_kernel(  # ruff: ignore[missing-return-type-private-function]
    config: ConfigProject,  # ruff: ignore[unused-function-argument]
    volume: Volume | FBVolume,
    projs: ProjectionSet,
):
    if projs.data is None:
        msg = "Cannot select the proper kernel when the datatype of the projectionset is as-yet unknown."
        raise RuntimeError(msg)
    if type(volume) is Volume:
        t1 = _np_dtype_to_c_type[volume.dtype]
        t2 = _np_dtype_to_c_type[projs.data.dtype]
        return _projection_module.get_function(
            f"project_CB_texture_chunked<{t1:s},{t2:s}>"
        )
    if type(volume) is FBVolume:
        tp = _np_dtype_to_c_type[projs.data.dtype]
        tv = "FORBILD_Shape"
        return _projection_module.get_function(
            f"project_CB_FORBILD_shapes<{tp:s},{tv:s}>"
        )
    msg = f"Unknown volume type: {type(volume)}"
    raise RuntimeError(msg)


def _select_backprojection_kernel(  # ruff: ignore[missing-return-type-private-function]
    config: ConfigBackproject,  # ruff: ignore[unused-function-argument]
    volume: Volume,
    projs: ProjectionSet,
):
    if projs.data is None:
        msg = "Cannot select the proper kernel when the datatype of the projectionset is as-yet unknown."
        raise RuntimeError(msg)
    t1 = _np_dtype_to_c_type[volume.dtype]
    t2 = _np_dtype_to_c_type[projs.data.dtype]
    return _backprojection_module.get_function(
        f"backproject_CB_voltex<{t1:s},{t2:s}>"
    )


def _assert_flatpanel_geometry_consistency(
    proj_specs: np.ndarray[tuple[int, ...], np.dtype[structs.proj_spec]],  # ty:ignore[invalid-type-arguments]
    projs: cp.ndarray[tuple[int, ...], cp.dtype[cp.float32 | cp.float64]]
    | None,
) -> tuple[int, ...]:
    """Assert consistency/correctness of the projection geometry with the data buffer, and return the expected shape of the output buffer."""
    # Assert correct dtype for the input proj_specs
    if not proj_specs.dtype == structs.proj_spec.dtype:
        msg = "Invalid dtype for the input array. It should be an array of 'proj_spec' objects."
        raise ValueError(msg)
    # Assert that all have the type set to flat panel
    dspecs = np.array(proj_specs["dspec"], dtype=structs.det_spec.dtype)
    if not np.all(0 == dspecs["type"]):
        msg = "Only flat-panel detectors are currently supported. One or more of the provided proj_spec indicates a detector type other than flat panel."
        raise NotImplementedError(msg)
    # Assert that all detector dimensions are the same
    nus = dspecs["nu"].flatten()
    nvs = dspecs["nv"].flatten()
    if not np.all(nus == nus[0]) or not np.all(nvs == nvs[0]):
        msg = "The input detector dimensions are inconsistent between projections. This is not currently implemented."
        raise NotImplementedError(msg)
    if len(proj_specs.shape) < 1:
        msg = "The input proj_specs array is not at least a 1D array. It should be at least a 1D array."
        raise ValueError(msg)
    # Calculate the expected output buffer shape
    det_shape = (nus[0], nvs[0])
    data_shape = proj_specs.shape + det_shape
    # Check consistency with the output array shape
    if projs is not None and projs.shape != data_shape:
        msg = f"Invalid shape for the output_array. The expected shape is {data_shape}, but the provided output_array has shape {projs.shape}."
        raise NotImplementedError(msg)
    return data_shape


@dataclasses.dataclass(
    kw_only=True
)  # making it a dataclass is somewhat useful, despite implementing the __init__ method explicitly
class ProjGeom:  # ruff: ignore[eq-without-hash]
    """A dataclass representing a geometry for a set of cone-beam projections."""

    source_pos: np.ndarray[tuple[int, ...], np.dtype[np.floating]]
    det_pos: np.ndarray[tuple[int, ...], np.dtype[np.floating]]
    det_uvec: np.ndarray[tuple[int, ...], np.dtype[np.floating]]
    det_vvec: np.ndarray[tuple[int, ...], np.dtype[np.floating]]
    nu: np.ndarray[tuple[int, ...], np.dtype[np.integer]]
    nv: np.ndarray[tuple[int, ...], np.dtype[np.integer]]

    def __init__(
        self,
        *,
        source_pos: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        det_pos: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        det_uvec: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        det_vvec: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
        nu: int
        | np.integer
        | np.ndarray[tuple[int, ...], np.dtype[np.integer]],
        nv: int
        | np.integer
        | np.ndarray[tuple[int, ...], np.dtype[np.integer]],
        _skip_regularisation: bool = False,
    ) -> None:
        # Regularise the data structure by encoding then decoding from a serializable format
        self.source_pos = source_pos
        self.det_pos = det_pos
        self.det_uvec = det_uvec
        self.det_vvec = det_vvec
        # Ignore bad typing because in that case, we're about to regularise the object anyway
        self.nu = nu  # ty:ignore[invalid-assignment]
        self.nv = nv  # ty:ignore[invalid-assignment]
        # Regularising the format is important to get the object into a known format.
        # It also makes the fields numpy arrays with expected shapes.
        # This allows to e.g. slice the data by the projection index.
        # We regularise so that the zeroth dimension of each array corresponds to the projection number,
        # except where the zeroth dimension is allowed to have length 1 to indicate broadcasting across
        # all projection indices.
        if not _skip_regularisation:
            pg = ProjGeom._load_from_serialized(self._pgserializable)
            self.source_pos = pg.source_pos
            self.det_pos = pg.det_pos
            self.det_uvec = pg.det_uvec
            self.det_vvec = pg.det_vvec
            self.nu = pg.nu
            self.nv = pg.nv

    def get_bbox(
        self, *, index_tuple: tuple[int, ...] | None = None
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Return a bounding box."""
        # Make a list of points such that the whole projection geometry
        # is contained in the convex hull. Then take the min x,y,z and max x,y,z
        # it = slice(None, None) if index_tuple is None else index_tuple
        it = () if index_tuple is None else index_tuple
        axes = tuple(range(len(self.projs_list_shape) - len(it)))
        min_xyz, max_xyz = None, None
        for points in (
            self.source_pos[it],
            (
                self.det_pos
                + self.det_uvec * self.nu / 2
                + self.det_vvec * self.nv / 2
            )[it],
            (
                self.det_pos
                + self.det_uvec * self.nu / 2
                - self.det_vvec * self.nv / 2
            )[it],
            (
                self.det_pos
                - self.det_uvec * self.nu / 2
                + self.det_vvec * self.nv / 2
            )[it],
            (
                self.det_pos
                - self.det_uvec * self.nu / 2
                - self.det_vvec * self.nv / 2
            )[it],
        ):
            min_ = np.min(points, axis=axes)
            max_ = np.max(points, axis=axes)
            min_xyz = (
                min_
                if min_xyz is None
                else np.min(np.stack((min_xyz, min_), axis=0), axis=0)
            )
            max_xyz = (
                max_
                if max_xyz is None
                else np.max(np.stack((max_xyz, max_), axis=0), axis=0)
            )
        return tuple(min_xyz), tuple(max_xyz)

    @property
    def projs_list_shape(self) -> tuple[int, ...]:
        """The broadcast shape of the projection list. If a flat list was provided, then this will be a 1-tuple like (N,)."""
        return np.broadcast_shapes(
            self.source_pos.shape[:-1],
            self.det_pos.shape[:-1],
            self.det_uvec.shape[:-1],
            self.det_vvec.shape[:-1],
            self.nu.shape,
            self.nv.shape,
        )

    @property
    def num_projs(self) -> int:
        """The number of cone-beam projections."""
        return math.prod(self.projs_list_shape)

    def sliced(
        self, sl: slice[int | None, int | None, None] | list[int]
    ) -> ProjGeom:
        """Return a copy of this geometry, except with the projection index slice applied."""
        if type(sl) not in (slice, list):
            msg = f"Invalid type for slice object {sl}"
            raise ValueError(msg)
        if len(self.projs_list_shape) != 1:
            msg = "Slicing has not been implemented for non-flat projection lists."
            raise NotImplementedError(msg)

        # This assumes that this object has been correctly regularized, so that the zeroth dimension
        # of each array corresponds to the projection index, except where it has length 1, which means
        # it broadcasts across all.
        numprojs: int = self.num_projs

        def copied_slice_maybe_broadcast(
            arr: np.ndarray, sl: slice | list[int]
        ) -> np.ndarray:
            if numprojs == 1:
                # Forget about broadcasting. Treat them like arrays of length 1.
                # Maybe one day this will result in a very specific edge case. Oh well.
                return arr[sl, ...].copy()
            if arr.shape[0] == 1:
                # Do broadcasting-- i.e. return the same array again
                return arr.copy()
            # Do a normal slicing
            return arr[sl, ...].copy()

        f = functools.partial(copied_slice_maybe_broadcast, sl=sl)

        return ProjGeom(
            source_pos=f(self.source_pos),
            det_pos=f(self.det_pos),
            det_uvec=f(self.det_uvec),
            det_vvec=f(self.det_vvec),
            nu=f(self.nu),
            nv=f(self.nv),
        )

    def flipped(self, *, flip_x: bool, flip_y: bool, flip_z: bool) -> ProjGeom:
        """Return a new ProjGeom which is flipped along one or more axes."""
        # Assumes a regularised form of this object, for correct array broadcasting
        fliparr = np.array(
            [-1 if flip_x else 1, -1 if flip_y else 1, -1 if flip_z else 1],
            dtype=np.float32,
        )
        fliparr = np.reshape(fliparr, (1,) * len(self.projs_list_shape) + (3,))
        return ProjGeom(
            source_pos=self.source_pos * fliparr,
            det_pos=self.det_pos * fliparr,
            det_uvec=self.det_uvec * fliparr,
            det_vvec=self.det_vvec * fliparr,
            nu=self.nu,
            nv=self.nv,
        )

    def rotated(self, *, xy_rads: float) -> ProjGeom:
        """Return a new ProjGeom which is rotated about the z axis."""
        rotmat = np.array(
            [
                [np.cos(xy_rads), -np.sin(xy_rads), 0],
                [np.sin(xy_rads), np.cos(xy_rads), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        return ProjGeom(
            source_pos=np.tensordot(self.source_pos, rotmat, axes=(-1, -1)),
            det_pos=np.tensordot(self.det_pos, rotmat, axes=(-1, -1)),
            det_uvec=np.tensordot(self.det_uvec, rotmat, axes=(-1, -1)),
            det_vvec=np.tensordot(self.det_vvec, rotmat, axes=(-1, -1)),
            nu=self.nu,
            nv=self.nv,
        )

    def translated(self, *, x: float, y: float, z: float) -> ProjGeom:
        """Return a new ProjGeom which is translated by the specified amounts."""
        trans = np.array([x, y, z], dtype=np.float32)
        trans = np.reshape(trans, (1,) * len(self.projs_list_shape) + (3,))
        return ProjGeom(
            source_pos=self.source_pos + trans,
            det_pos=self.det_pos + trans,
            det_uvec=self.det_uvec,
            det_vvec=self.det_vvec,
            nu=self.nu,
            nv=self.nv,
        )

    def __eq__(self, other) -> bool:  # ruff: ignore[missing-type-function-argument, complex-structure, undocumented-magic-method]
        if not isinstance(other, ProjGeom):
            return False
        d1 = dataclasses.asdict(self)
        d2 = dataclasses.asdict(other)
        for key in d1.keys():
            if key not in d2.keys():
                return False
        for key in d2.keys():
            if key not in d1.keys():
                return False
        for key in d1.keys():
            v1 = d1[key]
            v2 = d2[key]
            if type(v1) is not type(v2):
                return False
            t = type(v1)
            if t in (int, np.integer, float, np.floating):
                if v1 != v2:
                    return False
            elif t in (np.ndarray,):
                if not np.all(v1 == v2):
                    return False
            else:
                msg = f"Unsupported datatype: {t!s}"
                raise ValueError(msg)
        return True

    def __ne__(self, other) -> bool:  # ruff: ignore[missing-type-function-argument, undocumented-magic-method]
        if not isinstance(other, ProjGeom):
            return True
        return not self.__eq__(other)

    @dataclasses.dataclass(kw_only=True)
    class _ProjGeomSerializable:
        source_pos: list[list | float]
        det_pos: list[list | float]
        det_uvec: list[list | float]
        det_vvec: list[list | float]
        nu: int | list[list | int]
        nv: int | list[list | int]

        def json(self) -> dict:
            return dataclasses.asdict(self)

        def json_str(self) -> str:
            return json.dumps(self.json(), indent=2)

        @staticmethod
        def _load_from_dict(d: dict) -> ProjGeom._ProjGeomSerializable:
            return ProjGeom._ProjGeomSerializable(**d)

        def _save_to_dict(self) -> dict:
            return dataclasses.asdict(self)

    @property
    def _pgserializable(self) -> _ProjGeomSerializable:
        source_pos = self.source_pos
        det_pos = self.det_pos
        det_uvec = self.det_uvec
        det_vvec = self.det_vvec
        nu = np.array(self.nu)
        nv = np.array(self.nv)
        projlistdim: int = max(
            source_pos.ndim - 1,
            det_pos.ndim - 1,
            det_uvec.ndim - 1,
            det_vvec.ndim - 1,
            nu.ndim,
            nv.ndim,
        )
        # Add dimensions of size 1 to the end of the projdims, to enable broadcasting
        shape = source_pos.shape
        shape = (
            shape[:-1] + (1,) * (projlistdim - (source_pos.ndim - 1)) + (3,)
        )
        source_pos = np.reshape(source_pos, shape)
        shape = det_pos.shape
        shape = shape[:-1] + (1,) * (projlistdim - (det_pos.ndim - 1)) + (3,)
        det_pos = np.reshape(det_pos, shape)
        shape = det_uvec.shape
        shape = shape[:-1] + (1,) * (projlistdim - (det_uvec.ndim - 1)) + (3,)
        det_uvec = np.reshape(det_uvec, shape)
        shape = det_vvec.shape
        shape = shape[:-1] + (1,) * (projlistdim - (det_vvec.ndim - 1)) + (3,)
        det_vvec = np.reshape(det_vvec, shape)
        shape = nu.shape
        shape = shape + (1,) * (projlistdim - nu.ndim)
        nu = np.reshape(nu, shape)
        shape = nv.shape
        shape = shape + (1,) * (projlistdim - nv.ndim)
        nv = np.reshape(nv, shape)
        return self._ProjGeomSerializable(
            source_pos=source_pos.tolist(),
            det_pos=det_pos.tolist(),
            det_uvec=det_uvec.tolist(),
            det_vvec=det_vvec.tolist(),
            nu=nu.tolist(),
            nv=nv.tolist(),
        )

    def json(self) -> dict:
        """Get a json dictionary representing this object."""
        # return dataclasses.asdict(self)
        pgs: ProjGeom._ProjGeomSerializable = self._pgserializable
        return pgs._save_to_dict()  # ruff: ignore[private-member-access]

    def json_str(self) -> str:
        """Get a json string representing this object."""
        return json.dumps(self.json(), indent=2)

    @staticmethod
    def _load_from_serialized(pgs: ProjGeom._ProjGeomSerializable) -> ProjGeom:
        return ProjGeom(
            source_pos=np.array(pgs.source_pos, dtype=np.float32),
            det_pos=np.array(pgs.det_pos, dtype=np.float32),
            det_uvec=np.array(pgs.det_uvec, dtype=np.float32),
            det_vvec=np.array(pgs.det_vvec, dtype=np.float32),
            nu=np.array(pgs.nu, dtype=np.uint32),
            nv=np.array(pgs.nv, dtype=np.uint32),
            _skip_regularisation=True,
        )

    @staticmethod
    def load(fpath: pathlib.Path | str) -> ProjGeom:
        """Load a ProjGeom from a file."""
        fpath = pathlib.Path(fpath)
        if not fpath.exists():
            raise FileNotFoundError(fpath)
        with open(fpath, mode="r", encoding="utf-8") as f:
            pgs = ProjGeom._ProjGeomSerializable(**json.load(f))
        return ProjGeom._load_from_serialized(pgs)

    def save(
        self, fpath: pathlib.Path | str, *, allow_overwrite: bool = False
    ) -> None:
        """Save the ProjGeom to a file."""
        fpath = pathlib.Path(fpath)
        if not allow_overwrite and fpath.exists():
            raise FileExistsError(fpath)
        with open(fpath, mode="w+", encoding="utf-8") as f:
            f.write(self.json_str())

    def to_proj_spec_array(
        self,
    ) -> np.ndarray[tuple[int, ...], np.dtype[structs.proj_spec]]:  # ty:ignore[invalid-type-arguments]
        """Create a proj_spec array for use in creating a ProjectionSet object."""
        convert_to_fvec3 = lambda a: np.apply_along_axis(  # ruff: ignore[lambda-assignment]
            lambda arr: np.array(tuple(arr), dtype=structs.fvec3),
            axis=-1,
            arr=a,
        )

        pg = self

        sps = convert_to_fvec3(pg.source_pos)
        dps = convert_to_fvec3(pg.det_pos)
        us = convert_to_fvec3(pg.det_uvec)
        vs = convert_to_fvec3(pg.det_vvec)
        nus = np.array(pg.nu, dtype=np.uint32)
        nvs = np.array(pg.nv, dtype=np.uint32)
        tps = np.array(np.uint(0), dtype=np.uint8)

        def func_det(*args):  # ruff: ignore[missing-type-args, missing-return-type-private-function]
            return np.array(tuple(args), dtype=structs.det_spec)

        func_det_broadcastable = np.frompyfunc(func_det, nin=6, nout=1)

        # func_det_broadcastable = np.vectorize(func_det)
        def func_proj(*args):  # ruff: ignore[missing-type-args, missing-return-type-private-function]
            return np.array(tuple(args), dtype=structs.proj_spec)

        func_proj_broadcastable = np.frompyfunc(func_proj, nin=2, nout=1)
        # func_proj_broadcastable = np.vectorize(func_proj)
        return np.array(
            func_proj_broadcastable(
                sps, func_det_broadcastable(dps, us, vs, nus, nvs, tps)
            ),
            dtype=structs.proj_spec,
        )

    @classmethod
    def new_circular_scan(
        cls,
        *,
        num_source_points: int,
        R: float,  # ruff: ignore[invalid-argument-name]
        L_minus_R: float,  # ruff: ignore[invalid-argument-name]
        pix_widhei_at_isocenter: tuple[float, float],
        # Mutually exclusive:
        det_widhei: tuple[float, float] | None = None,
        num_pix_widhei: tuple[int, int] | None = None,
        # Optionals:
        theta_start: float = 0,
        theta_clockwise: bool = False,
        arc_length_rads: float = 2 * np.pi,
        isocenter: list[float]
        | tuple[float, float, float]
        | np.ndarray[tuple[typing.Literal[3]], np.dtype[np.floating]] = (
            0,
            0,
            0,
        ),
    ) -> ProjGeom:
        """Create a new circular ProjGeom object."""
        # Check validity of mutually exclusive arguments
        if det_widhei is None and num_pix_widhei is None:
            msg = "Exactly one of 'det_widhei' and 'num_pix_widhei' must be specified."
            raise ValueError(msg)
        if det_widhei is not None and num_pix_widhei is not None:
            msg = "Exactly one of 'det_widhei' and 'num_pix_widhei' must be specified."
            raise ValueError(msg)

        # Normalise inputs
        L: float = R + L_minus_R  # ruff: ignore[non-lowercase-variable-in-function]
        if det_widhei is None:
            assert num_pix_widhei is not None
            px, py = pix_widhei_at_isocenter
            nx, ny = num_pix_widhei
            det_widhei = (px * L / R * nx, py * L / R * ny)
        else:
            assert det_widhei is not None
            dx, dy = det_widhei
            px, py = pix_widhei_at_isocenter
            px, py = px * L / R, py * L / R
            num_pix_widhei = (math.ceil(dx / px), math.ceil(dy / py))

        # Unpack tuples
        pix_wid_at_isocenter, pix_hei_at_isocenter = pix_widhei_at_isocenter
        det_wid, det_hei = det_widhei  # ruff: ignore[unused-unpacked-variable]

        pixel_wid = (L / R) * pix_wid_at_isocenter
        pixel_hei = (L / R) * pix_hei_at_isocenter
        det_numpixwid, det_numpixhei = num_pix_widhei

        zs = np.zeros((num_source_points,), dtype=np.float64)
        ts = np.linspace(
            theta_start,
            theta_start + arc_length_rads * (-1 if theta_clockwise else 1),
            endpoint=False,
            num=num_source_points,
            dtype=np.float64,
        )
        source_pos_arr = np.array(
            [R * np.cos(ts), R * np.sin(ts), zs], dtype=np.float64
        ).transpose()
        det_pos_arr = source_pos_arr * np.array(
            [[1 - L / R, 1 - L / R, 1]], dtype=np.float64
        )
        # Translate to isocenter position
        source_pos_arr += np.array(isocenter)[np.newaxis, :]
        det_pos_arr += np.array(isocenter)[np.newaxis, :]
        # Define pixel coordinate vectors
        det_uvec_arr = (
            pixel_wid
            * np.array(
                [-np.sin(ts), np.cos(ts), 0 * ts], dtype=np.float64
            ).transpose()
        )
        det_vvec_arr = pixel_hei * np.array([[0, 0, -1]], dtype=np.float64)
        return cls(
            source_pos=source_pos_arr,
            det_pos=det_pos_arr,
            det_uvec=det_uvec_arr,
            det_vvec=det_vvec_arr,
            nu=det_numpixwid,
            nv=det_numpixhei,
        )


class ProjectionSet:
    """A class to represent a projection geometry, and optionally the associated detector data."""

    _proj_geom: ProjGeom
    _proj_specs: np.ndarray[tuple[int, ...], np.dtype[structs.proj_spec]]  # ty:ignore[invalid-type-arguments]
    _data: (
        cp.ndarray[tuple[int, ...], cp.dtype[cp.float32 | cp.float64]] | None
    )

    @property
    def proj_geom(self) -> ProjGeom:
        """The underlying ProjGeom object."""
        return self._proj_geom

    # @functools.lru_cache(maxsize=100)
    def get_bbox(
        self, *, index_tuple: tuple[int, ...] | None = None
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        """Return a bounding box."""
        return self._proj_geom.get_bbox(index_tuple=index_tuple)

    @property
    def proj_specs(
        self,
    ) -> np.ndarray[tuple[int, ...], np.dtype[structs.proj_spec]]:  # ty:ignore[invalid-type-arguments]
        """The underlying proj_specs object, which is derived from the underlying ProjGeom object."""
        return self._proj_specs

    @property
    def data(
        self,
    ) -> (
        cp.ndarray[
            tuple[int, ...], cp.dtype[cp.float16 | cp.float32 | cp.float64]
        ]
        | None
    ):
        """The underlying cupy array."""
        return self._data  # .copy() -- Too expensive to make a copy.

    def __init__(
        self,
        pg: ProjGeom,
        *,
        data: cp.ndarray[tuple[int, ...], cp.dtype[cp.float32 | cp.float64]]
        | np.ndarray[tuple[int, ...], np.dtype[np.float32 | np.float64]]
        | np.memmap[tuple[int, ...], np.dtype[np.float32 | np.float32]]
        | None,
    ) -> None:
        # Convert numpy input into a CuPy array
        if type(data) in (np.ndarray, np.memmap):
            data = cp.array(data)
        proj_specs = pg.to_proj_spec_array()
        _ = _assert_flatpanel_geometry_consistency(proj_specs, data)
        self._proj_geom = pg
        self._proj_specs = proj_specs
        self._data = data

    def create_buffer(
        self,
        *,
        dtype: type[np.float16 | np.float32 | np.float64],
        existing_ok: bool = False,
    ) -> None:
        """Create an underlying data buffer with the specified dtype."""
        # Handle the case where the data buffer already exists
        if self._data is not None:
            if existing_ok:
                return
            msg = "Data buffer already exists. Pass 'existing_ok=True' to allow this to occur without raising an exception."
            raise RuntimeError(msg)
        # The data buffer does not exist. Create it now.
        shape = _assert_flatpanel_geometry_consistency(
            self._proj_specs, self._data
        )
        self._data = cp.zeros(shape, dtype=dtype)

    def zeroed_copy(
        self, dtype: type[np.float16 | np.float32 | np.float64] | None = None
    ) -> ProjectionSet:
        """Return a copy of this object, except with a buffer filled with zeroes."""
        if self._data is None and dtype is None:
            msg = "Cannot create a zeroed copy without an explicit 'dtype' argument, because the object you are trying to copy does not yet have a determined dtype."
            raise ValueError(msg)
        if self._data is None:
            data_shape = _assert_flatpanel_geometry_consistency(
                self._proj_specs, self._data
            )
            return ProjectionSet(
                self._proj_geom,
                data=cp.zeros(data_shape, dtype=dtype),  # ty:ignore[invalid-argument-type]
            )
        return ProjectionSet(self._proj_geom, data=cp.zeros_like(self._data))


class ConfigProject:
    """Configuration object for projections. Some options will have no effect if a Volume or FBVolume."""

    _contents: np.dtype[structs.config_cb_project]  # ty:ignore[invalid-type-arguments]
    _contents_fb: np.dtype[structs.config_cb_project_FORBILD]  # ty:ignore[invalid-type-arguments]

    @property
    def contents(self) -> np.dtype[structs.config_cb_project]:  # ty:ignore[invalid-type-arguments]
        """The configuration object underlying this object."""
        return self._contents.copy()  # ty:ignore[unresolved-attribute]

    @property
    def contents_fb(self) -> np.dtype[structs.config_cb_project_FORBILD]:  # ty:ignore[invalid-type-arguments]
        """The configuration object underlying this object."""
        return self._contents_fb.copy()  # ty:ignore[unresolved-attribute]

    def __init__(
        self,
        *,
        shrink_volume_factor: int | float | None = None,
        multiply_before_adding: int | float | None = None,
        ssu: int | None = None,
        ssv: int | None = None,
    ) -> None:
        svf = shrink_volume_factor or 1  # coerce 'None' into 1
        mba = multiply_before_adding or 1
        ssu = ssu or 1
        ssv = ssv or 1
        self._contents = np.array(  # ty:ignore[invalid-assignment]
            (np.float32(svf), np.float32(mba), np.uint32(ssu), np.uint32(ssv)),
            dtype=structs.config_cb_project,
        )
        overlap_addition: bool = False  # The default behaviour of FORBILD is to prioritise later volumes, not to add them together. Leave this behaviour as is, by keeping this False.
        self._contents_fb = np.array(  # ty:ignore[invalid-assignment]
            (
                np.float32(mba),
                np.bool(overlap_addition),
                np.uint32(ssu),
                np.uint32(ssv),
            ),
            dtype=structs.config_cb_project_FORBILD,
        )


class ConfigBackproject:
    """Configuration object for backprojections."""

    _contents: np.dtype[structs.config_cb_backproject]  # ty:ignore[invalid-type-arguments]

    @property
    def contents(self) -> np.dtype[structs.config_cb_backproject]:  # ty:ignore[invalid-type-arguments]
        """The configuration object underlying this object."""
        return self._contents.copy()  # ty:ignore[unresolved-attribute]

    def __init__(
        self,
        *,
        fast_approx: bool = False,
        cyl_mask_rad: float | int | None = None,
        backproject_1s: bool = False,
        apply_summand_multiplier: bool = False,
        soften_det_edges_proportion: float | int | None = None,
        Z_on_L_factor_power: int = 0,  # ruff: ignore[invalid-argument-name]
        Z_on_val_factor_power: int = 0,  # ruff: ignore[invalid-argument-name]
        Z_on_val_factor_val: float = 1,  # ruff: ignore[invalid-argument-name]
        approx_mode_direct_raycast: bool = False,
        approx_mode_bilinear_interpolate: bool = False,
        backproj_weighting_preset: int = 0,
    ) -> None:
        r = cyl_mask_rad or 0  # coerce 'None' to 0
        sep = soften_det_edges_proportion or 0  # coerce 'None' to 0
        self._contents = np.array(  # ty:ignore[invalid-assignment]
            (
                np.bool(
                    not fast_approx
                ),  # whether to compute the backprojection exactly
                np.bool(backproject_1s),
                np.bool(
                    r
                ),  # whether to apply a circular mask to the backprojection
                np.float32(r),  # the radius of the circular mask, in voxels
                np.bool(
                    apply_summand_multiplier
                ),  # whether the kernel should use the summand_multiplier argument to multiply the backprojection before adding it to the volume
                sep,
                np.int32(Z_on_L_factor_power),
                np.int32(Z_on_val_factor_power),
                np.float32(Z_on_val_factor_val),
                np.bool(approx_mode_direct_raycast),
                np.bool(approx_mode_bilinear_interpolate),
                np.uint8(backproj_weighting_preset),
            ),
            dtype=structs.config_cb_backproject,
        )


def project(
    config: ConfigProject,
    volume: Volume | FBVolume,
    projs: ProjectionSet,
    block: bool = False,
) -> ProjectionSet:
    """Add the projection of the input volume to the output array."""
    projs.create_buffer(dtype=np.float32, existing_ok=True)
    assert projs.data is not None
    is_fb: bool = type(volume) is FBVolume
    # Prepare the kernel and the static kernel args
    if is_fb:
        blockx = 32  # needs to be at least 4
        blocky = 20
        blocklayer = 1
    else:
        blockx = min(10, projs.data.shape[-2])
        blocky = min(10, projs.data.shape[-1])
        blocklayer = min(10, projs.data.shape[-3])
    kern = _select_projection_kernel(config, volume, projs)
    block_dims = (blockx, blocky, blocklayer)
    grid_dims = (
        int(np.ceil(projs.data.shape[-2] / blockx)),
        int(np.ceil(projs.data.shape[-1] / blocky)),
        int(np.ceil(projs.data.shape[-3] / blocklayer)),
    )  # gridDims:  (x,y,z)
    if not is_fb:
        assert type(volume) is Volume
        texchunkspec = np.array(
            (
                np.float32(volume.voxwid),  # voxwid
                np.array((0, 0, 0), dtype=structs.uivec3.dtype),  # offsets
                np.array(volume.shape, dtype=structs.uivec3.dtype),  # shape
                np.array(
                    tuple(
                        volume.pos_c
                        - volume.voxwid
                        * np.array(volume.shape, dtype=np.float32)
                        / 2
                    ),
                    dtype=structs.fvec3.dtype,
                ),  # texpos
            ),
            dtype=structs.vol_tex_chunk_spec.dtype,
        )
    if is_fb:
        assert type(volume) is FBVolume
        # Prune the volume
        volume = volume.prune_to_bbox(*projs.get_bbox())
    input_proj_array_len = np.uint32(projs.data.shape[-3])
    input_proj_array_strides = np.array(
        projs.data.strides[-3:], dtype=structs.ulvec3.dtype
    )
    # Compute the projection. Loop through higher dimensions of the passed array, if applicable.
    for index_tuple in itertools.product(
        *map(range, projs.proj_specs.shape[:-1])
        # The below line should be equivalent and more idiomatic, but may be slightly more flaky
        # *map(range, projs.proj_geom.projs_list_shape[:-1])
    ):
        # If a flat proj_specs was passed, then 'index_tuple' will only take on the value of an  empty tuple.
        input_proj_specs_slice = cp.array(projs.proj_specs[index_tuple])
        input_proj_array_ptr = np.uint64(
            projs.data.data.ptr
            + sum(
                map(
                    lambda index, stride: index * stride,
                    index_tuple,
                    projs.data.strides[:-3],
                ),
                0,
            )
        )
        # Launch the kernel
        if is_fb:
            assert type(volume) is FBVolume
            # Opportunity for significant optimisation:
            # Prune the shapes list for each thread block.
            # This can be accomplished by computing the bounding box
            # for each individual shape, and their projections onto the detector.
            # If the silhouette of a shape intersects the thread block, then keep it.
            # Otherwise, discard it from the list and launch the kernel with
            # the thinned-down list of shapes.
            #
            # We'll launch the kernel separately for each block.
            # Each block will be given only the relevant shapes to deal with
            # Never mind: launching the kernel separately for each block
            # incurs massive overhead. 307.8ms --> 22837.4ms

            # Prune the volume to the bounding box containing all the source points
            # and corners of the detectors.
            min_xyz, max_xyz = projs.get_bbox(index_tuple=index_tuple)
            v_pruned: FBVolume = volume.prune_to_bbox(
                tuple(min_xyz), tuple(max_xyz)
            )
            input_shapes_pruned = cp.array(
                v_pruned.shapes, dtype=structs.FORBILD_volumes.Shape
            )
            input_num_shapes_pruned = np.uint32(input_shapes_pruned.size)

            args = (  # input args
                config.contents_fb,
                input_proj_specs_slice,
                input_proj_array_len,
                input_shapes_pruned,  # input_shapes,
                input_num_shapes_pruned,  # input_num_shapes,
                input_proj_array_strides,  # strides in bytes
                input_proj_array_ptr,  # pointer to the beginning of the det data
            )
            kern(grid_dims, block_dims, args)
        else:
            assert type(volume) is Volume
            args = (  # input args
                config.contents,
                input_proj_specs_slice,
                input_proj_array_len,
                texchunkspec,
                volume._texture.surfobj,  # ruff: ignore[private-member-access]
                input_proj_array_strides,  # strides in bytes
                input_proj_array_ptr,  # pointer to the beginning of the det data
            )
            kern(grid_dims, block_dims, args)
    if block:
        cp.cuda.get_current_stream().synchronize()
    return projs


def backproject(
    config: ConfigBackproject,
    volume: Volume,
    projs: ProjectionSet,
    block: bool = False,
    summand_multiplier: Volume | None = None,
    logger: logging.Logger | None = None,
    enable_pbar: bool = False,
) -> Volume:
    """Add the backprojection of the projection array to the input volume."""
    # Assert that the projection data is supplied
    if projs.data is None:
        msg = "You must provide projection data to this method."
        raise ValueError(msg)
    # Check that the multiplier mask has a matching shape and dtype.
    # Note that aspects of the multiplier may be ignored. It will just
    # multiply element-wise.
    if summand_multiplier is not None:
        if summand_multiplier.shape != volume.shape:
            msg = f"Shape mismatch between volume and multiplier. Volume shape: {volume.shape}. Multiplier shape: {summand_multiplier.shape}"
            raise ValueError(msg)
        if summand_multiplier.dtype != volume.dtype:
            msg = f"dtype mismatch between summand multiplier and the volume. Volume dtype: {volume.dtype}. Multiplier dtype: {summand_multiplier.dtype}. Technically, this could be supported, but it is not currently implemented."
            raise NotImplementedError(msg)
    # Check for consistency between the config and the presence of a summand_multiplier
    if config.contents["apply_summand_multiplier"] != (
        summand_multiplier is not None
    ):
        msg = "To prevent unintended behaviour, the config 'apply_summand_multiplier' must be true if and only if a summand_multiplier is included."
        raise ValueError(msg)
    # Prepare the kernel and the static kernel args
    blockx = min(8, volume.shape[0])
    blocky = min(8, volume.shape[1])
    blockz = min(8, volume.shape[2])
    kern = _select_backprojection_kernel(config, volume, projs)
    block_dims = (blockx, blocky, blockz)
    grid_dims = (
        int(np.ceil(volume.shape[0] / blockx)),
        int(np.ceil(volume.shape[1] / blocky)),
        int(np.ceil(volume.shape[2] / blockz)),
    )  # gridDims:  (x,y,z)
    texchunkspec = np.array(
        (
            np.float32(volume.voxwid),  # voxwid
            np.array((0, 0, 0), dtype=structs.uivec3.dtype),  # offsets
            np.array(volume.shape, dtype=structs.uivec3.dtype),  # shape
            np.array(
                tuple(
                    volume.pos_c
                    - volume.voxwid
                    * np.array(volume.shape, dtype=np.float32)
                    / 2
                ),
                dtype=structs.fvec3.dtype,
            ),  # texpos
        ),
        dtype=structs.vol_tex_chunk_spec.dtype,
    )

    projs_3d_strides_bytes = np.array(
        projs.data.strides[-3:], dtype=structs.uivec3
    )
    # Compute the projection. Loop through higher dimensions of the passed array, if applicable.
    index_tuple_list = list(
        itertools.product(*map(range, projs.proj_specs.shape[:-1]))
    )
    istop: int = int(np.ceil(projs.proj_specs.shape[-1] / 10))
    with logtools.ProgressBar(
        desc="backprojection:",
        maxval=len(index_tuple_list) * projs.proj_specs.shape[-1],
        valunit="proj",
        logger=logger,
        enable=enable_pbar,
    ) as pbar:
        for index_tuple_i, index_tuple in enumerate(index_tuple_list):
            # If a flat proj_specs was passed, then 'index_tuple' will only take on the value of an  empty tuple.
            for i in range(istop):
                # Prepare a chunk of 10 projections
                proj_spec_sl = slice(
                    i * 10, min((i + 1) * 10, projs.proj_specs.shape[-1])
                )
                proj_specs_10chunk = list(
                    projs.proj_specs[index_tuple][
                        proj_spec_sl
                    ]  # [i * 10 : (i + 1) * 10]
                )
                num_projs_in_chunk = len(proj_specs_10chunk)
                proj_specs_10chunk.extend(
                    [proj_specs_10chunk[-1]] * (10 - num_projs_in_chunk)
                )  # repeat the last projspec to fill up a whole chunk of 10.
                #
                input_proj_chunk_spec = np.array(
                    (
                        np.array(
                            proj_specs_10chunk, dtype=structs.proj_spec
                        ),  # projspecs
                        np.uint8(num_projs_in_chunk),  # numprojs
                        np.uint64(
                            projs.data.data.ptr
                            + sum(
                                map(
                                    lambda index, stride: index * stride,
                                    index_tuple,
                                    projs.data.strides[:-3],
                                ),
                                i * 10 * projs.data.strides[-3],
                            )
                        ),  # projmemaddr
                        projs_3d_strides_bytes,  # projmemstrides
                    ),
                    dtype=structs.proj_chunk_spec_maxlen10,
                )
                # Launch the kernel
                args = (  # input args
                    config.contents,
                    texchunkspec,
                    volume._texture.surfobj,  # ruff: ignore[private-member-access]
                    None
                    if summand_multiplier is None
                    else summand_multiplier._texture.surfobj,  # ruff: ignore[private-member-access]
                    input_proj_chunk_spec,
                )
                kern(grid_dims, block_dims, args)
                pbar.update_to(
                    index_tuple_i * projs.proj_specs.shape[-1]
                    + proj_spec_sl.stop
                )
        if block:
            cp.cuda.get_current_stream().synchronize()
    return volume
