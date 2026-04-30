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

"""Tools to generate, cache and load CUDA PTX code."""

import functools
import os
import pathlib
import shutil
import sys

import cuda.core

from .. import data

# Path to the root directory where the header files will be output.
# The source .cu files will also be copied here.
path_to_generated_cuptx = pathlib.Path(str(data.__path__[0])).joinpath("ptx")

_DISABLE_CACHING: bool = False


@functools.cache
def _get_default_device() -> cuda.core.Device:
    dev = cuda.core.Device()
    dev.set_current()
    return dev


def _path_to_module(uid: str) -> pathlib.Path:
    """Recipe for constructing the path to a PTX file given its unqiue identifier string."""
    return path_to_generated_cuptx.joinpath(uid + ".ptx")


def flush_cache() -> None:
    """Flush the PTX cache."""
    if path_to_generated_cuptx.is_dir():
        shutil.rmtree(path_to_generated_cuptx)


def compile_module(
    code_cpp: str,
    *,
    uid: str,
    name_expressions: list[str] | tuple[str] | None = None,
    dev: cuda.core.Device | None = None,
) -> cuda.core.ObjectCode:
    """Compile a module, or fetch it from the cache if available."""
    if not _DISABLE_CACHING:
        p = _path_to_module(uid)
        # Make ptx cache directory if it doesn't exist
        if not p.parent.is_dir():
            os.mkdir(p.parent)
        # Return cached ptx if available
        if p.is_file():
            # print("ptx_cacher: returning cached PTX")
            return cuda.core.ObjectCode.from_ptx(
                module=str(p.resolve()), symbol_mapping=None
            )
    # Compile the ptx, save it in the cache, and return it
    program_options = cuda.core.ProgramOptions(
        std="c++17",
        arch=f"sm_{(dev if dev is not None else _get_default_device()).arch}",
        device_as_default_execution_space=True,
    )
    prog_ptx = cuda.core.Program(
        code_cpp, code_type="c++", options=program_options
    )
    module_ptx = prog_ptx.compile("ptx", name_expressions=name_expressions)
    if not _DISABLE_CACHING:
        with open(p, "wb") as f:
            f.write(module_ptx.code)
            # print("ptx_cacher: generated new PTX at", p)
            # TODO: add a log line here indicating that a new PTX has been generated  # noqa: FIX002
    return module_ptx
    # module_ptx.symbol_mapping  # doesn't seem to be necessary to get this mapping or do anything with it.


def main(args: list[str]) -> None:  # noqa: D103
    if "flush" in args:
        flush_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
