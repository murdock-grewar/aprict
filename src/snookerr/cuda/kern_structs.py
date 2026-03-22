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

import re
import typing

import cupy as cp
import numpy as np

_valid_C_identifier_regex = re.compile(r"[_a-zA-Z]+\w*")  # noqa: N816

# fmt: off
_np_dtype_to_c_type = {
    np.dtype(np.bool):          "bool",
    np.dtype(np.int8):          "char",
    np.dtype(np.int16):         "short",
    np.dtype(np.int32):         "int",
    np.dtype(np.int64):         "long",
    np.dtype(np.float16):       "half",
    np.dtype(np.float32):       "float",
    np.dtype(np.float64):       "double",
    np.dtype(np.complex64):     "complex<float>",
    np.dtype(np.complex128):    "complex<double>",
}
# fmt: on


class KernelStruct:
    """A custom numpy dtype with a corresponding C structure."""

    dtype: np.dtype
    struct_definition_c: str

    def __init__(
        self,
        struct_name: str,
        *,
        names: tuple[str] | list[str],
        dtypes: tuple[typing.Any] | list[typing.Any],
        # *tuples_type_name: typing.Iterable[tuple[np.dtype, str]],
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

        self.struct_definition_c = self._struct_definition(
            struct_name, *tuples_type_name
        )

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
        kernel = self._kernel_get_struct_layout(struct_name, *tuples_type_name)
        kernel((1,), (1,), (itemsize, sizes, offsets))
        (itemsize, sizes, offsets) = map(
            cp.asnumpy, (itemsize, sizes, offsets)
        )
        itemsize = itemsize.item()
        names = [name for t, name in tuples_type_name]
        formats = [t for t, name in tuples_type_name]

        self.dtype = np.dtype(
            {
                "names": names,
                "formats": formats,
                "offsets": offsets,
                "itemsize": itemsize,
            }
        )

    @classmethod
    def _struct_definition(
        cls,
        struct_name: str,
        *tuples_type_name: typing.Iterable[tuple[np.dtype, str]],
    ) -> str:
        """Produce C++ code for a custom datatype."""
        if not cls._validate_struct_name(struct_name):
            msg = f"Invalid structure name: '{struct_name:s}'. Use something which is a valid type name in C++."
            raise ValueError(msg)
        try:
            return (
                f"struct {struct_name:s} "
                + "{\n"
                + "".join(
                    [
                        f"    {_np_dtype_to_c_type[np.dtype(dtype)]} {fieldname:s};\n"
                        for dtype, fieldname in tuples_type_name
                    ]
                )
                + "};"
            )
        except KeyError as e:
            msg = f"Invalid choice of np.dtype. The implemented choices are: {list(_np_dtype_to_c_type.keys())}."
            raise ValueError(msg) from e

    @classmethod
    def _validate_struct_name(cls, struct_name: str) -> bool:
        """Return True if the proposed struct_name is OK to use as a class name in C++."""
        return re.fullmatch(_valid_C_identifier_regex, struct_name) is not None

    @classmethod
    def _kernel_get_struct_layout(
        cls,
        struct_name: str,
        *tuples_type_name: typing.Iterable[tuple[np.dtype, str]],
    ) -> cp.RawKernel:
        tuples = [*tuples_type_name]
        struct_definition: str = cls._struct_definition(struct_name, *tuples)
        names = [n for t, n in tuples]
        struct_layout_code = (
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
        return cp.RawKernel(
            struct_layout_code, f"get_struct_layout_{struct_name}"
        )
