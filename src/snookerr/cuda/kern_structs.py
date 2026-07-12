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

"""A helper module to produce custom np.dtypes compatible with cupy."""

from __future__ import annotations

import abc
import inspect
import pathlib
import re
import typing

import cupy as cp
import numpy as np
import numpy.typing as npt
from cuda.core import (  # noqa: F401
    Device,
    LaunchConfig,
    ObjectCode,
    Program,
    ProgramOptions,
    launch,
)

from . import dtype_cacher, ptx_cacher

_valid_C_identifier_regex = re.compile(r"[_a-zA-Z]+\w*")  # noqa: N816

# fmt: off
_np_dtype_to_c_type = {
    np.dtype(np.bool):          "bool",
    np.dtype(np.int8):          "char",
    np.dtype(np.uint8):         "unsigned char",
    np.dtype(np.int16):         "short",
    np.dtype(np.uint16):        "unsigned short",
    np.dtype(np.int32):         "int",
    np.dtype(np.uint32):        "unsigned int",
    np.dtype(np.int64):         "long",
    np.dtype(np.uint64):        "unsigned long",
    np.dtype(np.float16):       "half",
    np.dtype(np.float32):       "float",
    np.dtype(np.float64):       "double",
    np.dtype(np.complex64):     "complex<float>",
    np.dtype(np.complex128):    "complex<double>",
}
# fmt: on

_dev = Device()
_dev.set_current()


class KernelStruct[T: np.dtype[typing.Any]](
    abc.ABC
):  # , np.generic):  # adding 'np.generic' here helps with dtyping with ty, but it breaks the creation of the data structures, for some reason.
    """A custom numpy dtype with a corresponding C structure."""

    # class fields
    struct_name: str
    dtype: type[
        T
    ]  # this might be incorrect... ty does not know how to check these kinds of things at the moment
    struct_definition_c: str
    dependent_structs: list[KernelStruct]

    # instance fields
    ndarray: np.ndarray[tuple[()], T]

    # Note this useful way of creating arrays of these objects:
    # np.array([[(1,2,3), (7,8,9)]], ndmax=2, dtype=fvec3)
    # That will preserve the tuples in the last dimension and create a (1,2)-shaped array of fvec3 objects.

    # COMMENTED THIS OUT. MAY BE USEFUL LATER. I WILL TRY TO AVOID IT CURRENTLY.
    # # This will be inherited by the subclasses. It's a shortcut, enabling us to call
    # # fv3 = fvec3((1.0, 2, 3))
    # # to create an object with a backing ndarray like
    # # fv3.ndarray == np.array((1.0, 2, 3), dtype=fvec3.dtype)
    # def __init__(self, input: tuple | T) -> None:
    #     # Try to create the array directly from the input
    #     a = np.array(input, dtype=self.dtype)
    #     # Check that the created object is valid.
    #     if not a.shape == ():
    #         msg = "Invalid object passed as input parameter."
    #         raise ValueError(msg)
    #     if not a.dtype == self.dtype:
    #         msg = "Invalid object passed as input parameter."
    #         raise ValueError(msg)
    #     # Assign the validated object to the underlying data array
    #     self.ndarray = a  # ty:ignore[invalid-assignment, unused-ignore-comment]

    @classmethod
    def _struct_definition(
        cls,
        dtype_to_cname_dict: dict[np.dtype | KernelStruct, str],
        struct_name: str,
        insert_into_struct: str | None,
        *tuples_type_name: typing.Iterable[
            tuple[np.dtype | KernelStruct | KernelArray, str]
        ],
    ) -> str:
        """Produce C++ code for a custom datatype."""

        def cname(dtype: np.dtype | KernelStruct | KernelArray) -> str:
            if type(dtype) is KernelArray:
                actual_dtype = dtype.member_dtype
            else:
                actual_dtype = dtype
            if issubclass(actual_dtype, KernelStruct):  # ty:ignore[invalid-argument-type]
                return dtype_to_cname_dict[actual_dtype]  # ty:ignore[invalid-argument-type]
            return dtype_to_cname_dict[np.dtype(actual_dtype)]

        def get_fieldname(
            dtype: np.dtype | KernelStruct | KernelArray, name: str
        ) -> str:
            s = name
            if type(dtype) is KernelArray:
                for i in dtype.shape:
                    s += f"[{i:d}]"
            return s

        if not cls._validate_struct_name(struct_name):
            msg = f"Invalid structure name: '{struct_name:s}'. Use something which is a valid type name in C++."
            raise ValueError(msg)
        try:
            return (
                f"struct {struct_name:s} "
                + "{\n"
                + "".join(
                    [
                        f"    {cname(dtype)} {get_fieldname(dtype, name)};\n"  # ty:ignore[invalid-argument-type]
                        for dtype, name in tuples_type_name
                    ]
                )
                + ("" if insert_into_struct is None else insert_into_struct)
                + "};"
            )
        except KeyError as e:
            msg = f"Invalid choice of np.dtype. The implemented choices are: {[str(k) for k in dtype_to_cname_dict.keys()]}. The provided types were was {[dtype for dtype, name in tuples_type_name]}."
            raise ValueError(msg) from e

    @classmethod
    def _validate_struct_name(cls, struct_name: str) -> bool:
        """Return True if the proposed struct_name is OK to use as a class name in C++."""
        return re.fullmatch(_valid_C_identifier_regex, struct_name) is not None

    @classmethod
    def _issubclass(cls, obj: object) -> bool:
        return inspect.isclass(obj) and issubclass(obj, KernelStruct)

    @classmethod
    def _kernel_get_struct_layout(
        cls,
        dtype_to_cname_dict: dict[np.dtype | KernelStruct, str],
        struct_name: str,
        force_dependencies: list[KernelStruct] | None,
        insert_into_struct: str | None,
        *tuples_type_name: typing.Iterable[
            tuple[np.dtype | KernelStruct | KernelArray, str]
        ],
    ) -> cp.RawKernel:
        tuples: list = [*tuples_type_name]
        struct_definition: str = cls._struct_definition(
            dtype_to_cname_dict, struct_name, insert_into_struct, *tuples
        )
        names = [n for t, n in tuples]
        struct_layout_code = ""
        # Create an ordered list of dependencies. First are the base dependencies.
        dtypes = [tup[0] for tup in tuples]
        dependent_ks = [
            k if KernelStruct._issubclass(k) else k.member_dtype
            for k in dtypes
            if (
                KernelStruct._issubclass(k)
                or (
                    type(k) is KernelArray
                    and KernelStruct._issubclass(k.member_dtype)
                )
            )
        ]
        if force_dependencies is not None:
            dependent_ks.extend(list(force_dependencies))
        for i in list(reversed(range(len(dependent_ks)))):
            ks: KernelStruct = dependent_ks[i]
            dependent_ks = (
                dependent_ks[:i] + ks.dependent_structs + dependent_ks[i:]
            )
        # Remove duplicates. Keep first appearance of each symbol.
        seen = set()
        dependent_ks = [
            ks for ks in dependent_ks if not (ks in seen or seen.add(ks))
        ]
        # Add the structure layout code for each dependency, in the proper order
        for ks in dependent_ks:
            struct_layout_code += ks.struct_definition_c
        struct_layout_code += (
            """
{struct_definition}

extern "C" __global__ void get_struct_layout_{struct_name}(
                                unsigned long long *itemsize,
                                unsigned long long *sizes,
                                unsigned long long *offsets) {{
    const {struct_name}* ptr = nullptr;

    itemsize[0] = sizeof({struct_name});
""".format(struct_definition=struct_definition, struct_name=struct_name)
            + "".join(
                [
                    f"    sizes[{i}] = sizeof(ptr->{fname});\n"
                    for i, fname in enumerate(names)
                ]
            )
            + "\n"
            + "".join(
                [
                    f"    offsets[{i}] = (unsigned long long)&ptr->{fname};\n"
                    for i, fname in enumerate(names)
                ]
            )
            + "}"
        )

        module_unique_name = f"module_get_struct_layout_{struct_name}"
        module_ptx = ptx_cacher.compile_module(
            code_cpp=struct_layout_code,
            uid=module_unique_name,
            name_expressions=(f"get_struct_layout_{struct_name}",),
        )
        return module_ptx.get_kernel(f"get_struct_layout_{struct_name}")

        # return cp.RawKernel(
        #     struct_layout_code, f"get_struct_layout_{struct_name}"
        # )

    @classmethod
    def save_c_struct_to_file(
        cls, path: pathlib.Path | str, overwrite: bool = False
    ) -> None:
        """Save the C structure definition to a file."""
        p: pathlib.Path = pathlib.Path(path)
        if not overwrite and p.is_file():
            msg = f"The file '{p.resolve()!s}' already exists. You may use the overwrite=True option to overwrite it."
            raise FileExistsError(msg)
        with open(path, "w+", encoding="utf-8") as f:
            f.write(cls.struct_definition_c)

    @classmethod
    def append_c_struct_to_file(
        cls, path: pathlib.Path | str, create_ok: bool = True, end: str = ""
    ) -> None:
        """Append the C structure definition to a file."""
        p: pathlib.Path = pathlib.Path(path)
        mode: str = "a+"
        if not p.is_file():
            if create_ok:
                mode = "w+"
            else:
                msg = f"The file '{p.resolve()!s}' does not exist. You may use the option create_ok=True to create it."
                raise FileNotFoundError(msg)
        with open(path, mode=mode, encoding="utf-8") as f:
            f.write(cls.struct_definition_c)
            f.write(end)

    def __init_subclass__(  # noqa: D105
        cls,
        struct_name: str,
        *,
        names: tuple[str] | list[str],
        dtypes: tuple[npt.DTypeLike | KernelStruct | KernelArray]
        | list[
            npt.DTypeLike | KernelStruct | KernelArray
        ],  # input a list/tuple of np.dtypes and/or KernelStructs.
        force_dependencies: list[KernelStruct] | None = None,
        insert_into_struct: str
        | None = None,  # extra code to put INSIDE the struct. Useful for defining implicit constuctors and the like
        extra_code: str
        | None = None,  # extra code to put after the structure definiton. Useful for e.g. defining operators.
    ) -> None:

        # Validate inputs #
        if len(names) != len(dtypes):
            msg = f"Mismatch in length of `names` and `dtypes`. Names: {names}. dtypes: {dtypes}"
            raise ValueError(msg)
        # Test that all 'dtypes' can actually be interpreted by numpy as dtypes
        for dtype in dtypes:
            np.dtype(dtype)  # May raise a TypeError. This is intentional.
        ###################

        tuples_type_name = list(zip(dtypes, names))

        dtype_to_cname_dict: dict[np.dtype | KernelStruct, str] = dict(
            _np_dtype_to_c_type.items()
        )  # Create copy

        cls.dependent_structs = [  # ty:ignore[invalid-assignment]
            k if KernelStruct._issubclass(k) else k.member_dtype  # ty:ignore[unresolved-attribute] # ty is wrong here
            for k in dtypes
            if (
                KernelStruct._issubclass(k)
                or (
                    type(k) is KernelArray
                    and KernelStruct._issubclass(k.member_dtype)
                )
            )
        ]
        if force_dependencies is not None:
            cls.dependent_structs.extend([t for t in force_dependencies])
        # Add the dependents of dependents. Make sure it's ordered, with base dependencies on the left.
        for i in list(reversed(range(len(cls.dependent_structs)))):
            ks: KernelStruct = cls.dependent_structs[i]
            cls.dependent_structs = (
                cls.dependent_structs[:i]
                + ks.dependent_structs
                + cls.dependent_structs[i:]
            )
        # Now remove duplicates, keeping the first-encountered copy of each
        seen = set()
        cls.dependent_structs = [
            ks
            for ks in cls.dependent_structs
            if not (ks in seen or seen.add(ks))
        ]

        dtype_to_cname_dict.update(
            {ks: ks.struct_name for ks in cls.dependent_structs}
        )

        cls.struct_name = struct_name
        cls.struct_definition_c = cls._struct_definition(
            dtype_to_cname_dict,
            struct_name,
            insert_into_struct,
            *tuples_type_name,
        ) + ("" if extra_code is None else extra_code)

        # If the dtype isn't in the cache, then generate it now.
        uid: str = cls.struct_name
        dtype = dtype_cacher.query(uid=uid)
        if dtype is not None:
            cls.dtype = dtype  # ty:ignore[invalid-assignment]
        else:
            # Run a CUDA kernel with the appropriate set of arguments, and find out
            # how CUDA arranges the offsets in memory of the various members. We'll
            # need this to define the custom np.dtype.
            #  itemsize -> overall struct size
            #  sizes    -> individual struct member sizes, determined with sizeof
            #  offsets  -> individual struct member offsets, determined with offsetof
            num_members = len(tuples_type_name)
            # fmt: off
            itemsize = cp.ndarray(shape=(1,),           dtype=np.uint64)
            sizes    = cp.ndarray(shape=(num_members,), dtype=np.uint64)
            offsets  = cp.ndarray(shape=(num_members,), dtype=np.uint64)
            # fmt: on
            kernel = cls._kernel_get_struct_layout(
                dtype_to_cname_dict,
                struct_name,
                force_dependencies,
                insert_into_struct,
                *tuples_type_name,
            )
            # kernel((1,), (1,), (itemsize, sizes, offsets))

            kconfig = LaunchConfig(grid=(1,), block=(1,))
            stream = _dev.create_stream()
            launch(
                stream,
                kconfig,
                kernel,
                itemsize.data.ptr,
                sizes.data.ptr,
                offsets.data.ptr,
            )
            stream.sync()

            (itemsize, sizes, offsets) = map(
                cp.asnumpy, (itemsize, sizes, offsets)
            )
            itemsize = itemsize.item()
            names = [name for t, name in tuples_type_name]
            formats = [t for t, name in tuples_type_name]

            cls.dtype = np.dtype(  # ty:ignore[invalid-assignment]
                {
                    "names": names,
                    "formats": formats,
                    "offsets": offsets,
                    "itemsize": itemsize,
                }
            )
            dtype_cacher.save(uid=uid, dtype=cls.dtype)  # ty:ignore[invalid-argument-type]


class KernelArray:
    """A type to represent an array of a fixed numpy dtype and length."""

    dtype: np.dtype
    member_dtype: type | KernelStruct | KernelArray
    shape: tuple[int, ...]

    def __init__(
        self,
        *,
        dtype: type | KernelStruct | KernelArray,
        shape: tuple[int, ...],
    ) -> None:
        self.member_dtype = dtype
        self.dtype = np.dtype((dtype, shape))
        self.shape = shape

        # Validate inputs #
        # Test that the 'dtype' can actually be interpreted by numpy as a dtype.
        np.dtype(self.dtype)  # May raise a TypeError. This is intentional.
        ###################
