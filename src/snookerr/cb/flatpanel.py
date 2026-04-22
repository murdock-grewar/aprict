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

import itertools

import cupy as cp
import numpy as np
import numpy.typing as npt  # noqa: F401

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


def _select_projection_kernel(  # noqa: ANN202
    config: ConfigProject,  # noqa: ARG001
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


def _select_backprojection_kernel(  # noqa: ANN202
    config: ConfigBackproject,  # noqa: ARG001
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


def assert_flatpanel_geometry_consistency(
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


def create_proj_spec_array(
    source_pos: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    det_pos: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    det_uvec: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    det_vvec: np.ndarray[tuple[int, ...], np.dtype[np.floating]],
    nu: int | np.integer | np.ndarray[tuple[int, ...], np.dtype[np.integer]],
    nv: int | np.integer | np.ndarray[tuple[int, ...], np.dtype[np.integer]],
) -> np.ndarray[tuple[int, ...], np.dtype[structs.proj_spec]]:  # ty:ignore[invalid-type-arguments]
    """Create a proj_spec array for use in creating a ProjectionSet object."""
    convert_to_fvec3 = lambda a: np.apply_along_axis(  # noqa: E731
        lambda arr: np.array(tuple(arr), dtype=structs.fvec3),
        axis=-1,
        arr=a,
    )

    sps = convert_to_fvec3(source_pos)
    dps = convert_to_fvec3(det_pos)
    us = convert_to_fvec3(det_uvec)
    vs = convert_to_fvec3(det_vvec)
    nus = np.array(nu, dtype=np.uint16)
    nvs = np.array(nv, dtype=np.uint16)
    tps = np.array(np.uint(0), dtype=np.uint8)

    def func_det(*args):  # noqa: ANN002, ANN202
        return np.array(tuple(args), dtype=structs.det_spec)

    func_det_broadcastable = np.frompyfunc(func_det, nin=6, nout=1)

    # func_det_broadcastable = np.vectorize(func_det)
    def func_proj(*args):  # noqa: ANN002, ANN202
        return np.array(tuple(args), dtype=structs.proj_spec)

    func_proj_broadcastable = np.frompyfunc(func_proj, nin=2, nout=1)
    # func_proj_broadcastable = np.vectorize(func_proj)
    return np.array(
        func_proj_broadcastable(
            sps, func_det_broadcastable(dps, us, vs, nus, nvs, tps)
        ),
        dtype=structs.proj_spec,
    )


class ProjectionSet:
    """A class to represent a projection geometry, and optionally the associated detector data."""

    _proj_specs: np.ndarray[tuple[int, ...], np.dtype[structs.proj_spec]]  # ty:ignore[invalid-type-arguments]
    _data: (
        cp.ndarray[tuple[int, ...], cp.dtype[cp.float32 | cp.float64]] | None
    )

    @property
    def proj_specs(
        self,
    ) -> np.ndarray[tuple[int, ...], np.dtype[structs.proj_spec]]:  # ty:ignore[invalid-type-arguments]
        """Return the underlying proj_specs object."""
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
        """Return the underlying cupy array."""
        return self._data  # .copy() -- Too expensive to make a copy.

    def __init__(
        self,
        proj_specs: np.ndarray[tuple[int, ...], np.dtype[structs.proj_spec]],  # ty:ignore[invalid-type-arguments]
        *,
        data: cp.ndarray[tuple[int, ...], cp.dtype[cp.float32 | cp.float64]]
        | None,
    ) -> None:
        _ = assert_flatpanel_geometry_consistency(proj_specs, data)
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
        shape = assert_flatpanel_geometry_consistency(
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
            data_shape = assert_flatpanel_geometry_consistency(
                self._proj_specs, self._data
            )
            return ProjectionSet(
                self._proj_specs,
                data=cp.zeros(data_shape, dtype=dtype),  # ty:ignore[invalid-argument-type]
            )
        return ProjectionSet(self._proj_specs, data=cp.zeros_like(self._data))


class ConfigProject:
    """Configuration object for projections. Some options will have no effect if a Volume or FBVolume."""

    _contents: np.dtype[structs.config_cb_project]  # ty:ignore[invalid-type-arguments]
    _contents_fb: np.dtype[structs.config_cb_project_FORBILD]  # ty:ignore[invalid-type-arguments]

    @property
    def contents(self) -> np.dtype[structs.config_cb_project]:  # ty:ignore[invalid-type-arguments]
        """Get the configuration object underlying this object."""
        return self._contents.copy()  # ty:ignore[unresolved-attribute]

    @property
    def contents_fb(self) -> np.dtype[structs.config_cb_project_FORBILD]:  # ty:ignore[invalid-type-arguments]
        """Get the configuration object underlying this object."""
        return self._contents_fb.copy()  # ty:ignore[unresolved-attribute]

    def __init__(
        self,
        *,
        shrink_volume_factor: int | float | None = None,
        multiply_before_adding: int | float | None = None,
    ) -> None:
        svf = shrink_volume_factor or 1  # coerce 'None' into 1
        mba = multiply_before_adding or 1
        self._contents = np.array(  # ty:ignore[invalid-assignment]
            (np.float32(svf), np.float32(mba)), dtype=structs.config_cb_project
        )
        overlap_addition: bool = False  # The default behaviour of FORBILD is to prioritise later volumes, not to add them together. Leave this behaviour as is, by keeping this False.
        self._contents_fb = np.array(  # ty:ignore[invalid-assignment]
            (np.float32(mba), np.bool(overlap_addition)),
            dtype=structs.config_cb_project_FORBILD,
        )


class ConfigBackproject:
    """Configuration object for backprojections."""

    _contents: np.dtype[structs.config_cb_backproject]  # ty:ignore[invalid-type-arguments]

    @property
    def contents(self) -> np.dtype[structs.config_cb_backproject]:  # ty:ignore[invalid-type-arguments]
        """Get the configuration object underlying this object."""
        return self._contents.copy()  # ty:ignore[unresolved-attribute]

    def __init__(
        self,
        *,
        fast_approx: bool = False,
        cyl_mask_rad_vox: float | int | None = None,
        backproject_1s: bool = False,
        apply_summand_multiplier: bool = False,
    ) -> None:
        r = cyl_mask_rad_vox or 0  # coerce 'None' to 0
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
    blockx = min(5 if is_fb else 10, projs.data.shape[-2])
    blocky = min(5 if is_fb else 10, projs.data.shape[-1])
    blocklayer = min(5 if is_fb else 10, projs.data.shape[-3])
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
                np.array((0, 0, 0), dtype=structs.uivec3.dtype),  # offsets
                np.array(volume.shape, dtype=structs.uivec3.dtype),  # shape
                np.array(
                    tuple(
                        volume.pos_c
                        - np.array(volume.shape, dtype=np.float32) / 2
                    ),
                    dtype=structs.fvec3.dtype,
                ),  # texpos
            ),
            dtype=structs.vol_tex_chunk_spec.dtype,
        )
    if is_fb:
        assert type(volume) is FBVolume
        input_shapes = cp.array(
            volume.shapes, dtype=structs.FORBILD_volumes.Shape
        )
        input_num_shapes = np.uint32(input_shapes.size)
    input_proj_array_len = np.uint32(projs.data.shape[-3])
    input_proj_array_strides = np.array(
        projs.data.strides[-3:], dtype=structs.ulvec3.dtype
    )
    # Compute the projection. Loop through higher dimensions of the passed array, if applicable.
    for index_tuple in itertools.product(
        *map(range, projs.proj_specs.shape[:-1])
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
            args = (  # input args
                config.contents_fb,
                input_proj_specs_slice,
                input_proj_array_len,
                input_shapes,
                input_num_shapes,
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
                volume._texture.surfobj,  # noqa: SLF001
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
    blockx = min(10, volume.shape[0])
    blocky = min(10, volume.shape[1])
    blockz = min(10, volume.shape[2])
    kern = _select_backprojection_kernel(config, volume, projs)
    block_dims = (blockx, blocky, blockz)
    grid_dims = (
        int(np.ceil(volume.shape[0] / blockx)),
        int(np.ceil(volume.shape[1] / blocky)),
        int(np.ceil(volume.shape[2] / blockz)),
    )  # gridDims:  (x,y,z)
    texchunkspec = np.array(
        (
            np.array((0, 0, 0), dtype=structs.uivec3.dtype),  # offsets
            np.array(volume.shape, dtype=structs.uivec3.dtype),  # shape
            np.array(
                tuple(
                    volume.pos_c - np.array(volume.shape, dtype=np.float32) / 2
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
    for index_tuple in itertools.product(
        *map(range, projs.proj_specs.shape[:-1])
    ):
        # If a flat proj_specs was passed, then 'index_tuple' will only take on the value of an  empty tuple.
        for i in range(int(np.ceil(projs.proj_specs.shape[-1] / 10))):
            # Prepare a chunk of 10 projections
            proj_specs_10chunk = list(
                projs.proj_specs[index_tuple][i * 10 : (i + 1) * 10]
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
                volume._texture.surfobj,  # noqa: SLF001
                None
                if summand_multiplier is None
                else summand_multiplier._texture.surfobj,  # noqa: SLF001
                input_proj_chunk_spec,
            )
            kern(grid_dims, block_dims, args)
    if block:
        cp.cuda.get_current_stream().synchronize()
    return volume
