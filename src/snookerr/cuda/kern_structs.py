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

import pathlib
import re
import typing

import cupy as cp
import numpy as np

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


class KernelStruct:
    """A custom numpy dtype with a corresponding C structure."""

    struct_name: str
    dtype: np.dtype
    struct_definition_c: str
    dependent_structs: list[KernelStruct]

    def __init__(
        self,
        struct_name: str,
        *,
        names: tuple[str] | list[str],
        dtypes: tuple[type | KernelStruct] | list[type | KernelStruct],  # input a list/tuple of np.dtypes and/or KernelStructs.
        extra_code: str | None = None,  # extra code to put after the structure definiton. Useful for e.g. defining operators.
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

        dtype_to_cname_dict: dict[np.dtype | KernelStruct, str] = dict(_np_dtype_to_c_type.items())  # Create copy
        self.dependent_structs = [t for t in dtypes if type(t) is KernelStruct]
        dtype_to_cname_dict.update({ks: ks.struct_name for ks in self.dependent_structs})

        self.struct_name = struct_name
        self.struct_definition_c = self._struct_definition(
            dtype_to_cname_dict, struct_name, *tuples_type_name
        ) + ("" if extra_code is None else extra_code)

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
        kernel = self._kernel_get_struct_layout(dtype_to_cname_dict, struct_name, *tuples_type_name)
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
        dtype_to_cname_dict: dict[np.dtype | KernelStruct, str],
        struct_name: str,
        *tuples_type_name: typing.Iterable[tuple[np.dtype | KernelStruct, str]],
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
                        f"    {dtype_to_cname_dict[dtype if type(dtype) is KernelStruct else np.dtype(dtype)]} {fieldname:s};\n"
                        for dtype, fieldname in tuples_type_name
                    ]
                )
                + "};"
            )
        except KeyError as e:
            msg = f"Invalid choice of np.dtype. The implemented choices are: {[str(k) for k in dtype_to_cname_dict.keys()]}."
            raise ValueError(msg) from e

    @classmethod
    def _validate_struct_name(cls, struct_name: str) -> bool:
        """Return True if the proposed struct_name is OK to use as a class name in C++."""
        return re.fullmatch(_valid_C_identifier_regex, struct_name) is not None

    @classmethod
    def _kernel_get_struct_layout(
        cls,
        dtype_to_cname_dict: dict[np.dtype | KernelStruct, str],
        struct_name: str,
        *tuples_type_name: typing.Iterable[tuple[np.dtype, str]],
    ) -> cp.RawKernel:
        tuples: list = [*tuples_type_name]
        struct_definition: str = cls._struct_definition(dtype_to_cname_dict, struct_name, *tuples)
        names = [n for t, n in tuples]
        struct_layout_code = ""
        # Create an ordered list of dependencies. First are the base dependencies.
        dependent_ks = [tup[0] for tup in tuples if type(tup[0]) is KernelStruct]
        for i in list(reversed(range(len(dependent_ks)))):
            ks: KernelStruct = dependent_ks[i]
            dependent_ks = dependent_ks[:i] + ks.dependent_structs + dependent_ks[i:]
        # Remove duplicates. Keep first appearance of each symbol.
        seen = set()
        dependent_ks = [ks for ks in dependent_ks if not (ks in seen or seen.add(ks))]
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
        return cp.RawKernel(
            struct_layout_code, f"get_struct_layout_{struct_name}"
        )

    def save_c_struct_to_file(
        self, path: pathlib.Path | str, overwrite: bool = False
    ) -> None:
        """Save the C structure definition to a file."""
        p: pathlib.Path = pathlib.Path(path)
        if not overwrite and p.is_file():
            msg = f"The file '{p.resolve()!s}' already exists. You may use the overwrite=True option to overwrite it."
            raise FileExistsError(msg)
        with open(path, "w+", encoding="utf-8") as f:
            f.write(self.struct_definition_c)

    def append_c_struct_to_file(
        self, path: pathlib.Path | str, create_ok: bool = True, end: str = ""
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
            f.write(self.struct_definition_c)
            f.write(end)

    def __str__(self) -> str:  # noqa: D105
        return f"KernelStruct({self.struct_name})"
