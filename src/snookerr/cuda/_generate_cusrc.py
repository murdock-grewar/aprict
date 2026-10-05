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

"""Tools to generate CUDA source code."""

import importlib.util
import inspect
import pathlib
import pkgutil
import shutil
import sys

from .. import data
from ..cuda.kern_structs import KernelStruct
from . import cusrc_templates, structs

# Path to the root directory where the header files will be output.
# The source .cu files will also be copied here.
path_to_generated_cusrc = pathlib.Path(str(data.__path__[0])).joinpath("cusrc")
# Path to the original .cu source file directory which will be copied
path_to_template_cusrc = pathlib.Path(str(cusrc_templates.__path__[0]))


def import_module_rel_name(module_rel_name: str, anchor, is_pkg: bool = True):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
    module_name = anchor.__name__ + module_rel_name
    anchor_path = pathlib.Path(anchor.__path__[0])
    if is_pkg:
        file_path = anchor_path.joinpath(*module_rel_name.split(".")).joinpath(
            "__init__.py"
        )
    else:
        path_pieces = module_rel_name.split(".")
        path_pieces[-1] += ".py"
        file_path = anchor_path.joinpath(*path_pieces)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)  # ty:ignore[invalid-argument-type]
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)  # ty:ignore[unresolved-attribute]
    except ModuleNotFoundError as e:
        msg = f"There was an error loading the file at '{file_path}'."
        raise RuntimeError(msg) from e
    return module


def import_submodules(package, anchor, recursive=True):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        rel_name = full_name[len(anchor.__name__) :]
        subpackage = import_module_rel_name(rel_name, anchor, is_pkg=is_pkg)
        results[full_name] = subpackage
        if recursive and is_pkg:
            results.update(
                import_submodules(subpackage, anchor, recursive=True)
            )
    return results


def generate_headers(
    output_dir: pathlib.Path, include_dependencies: bool = False
) -> None:
    prefix: str = structs.__name__
    for submodname, submod in import_submodules(
        package=structs, anchor=structs, recursive=True
    ).items():
        # Skip __init__.py / modules that are just directories
        if submod.__file__[-11:] == "__init__.py":
            continue
        # Get the KernelStruct objects from the file
        kernstructs: list[tuple[str, KernelStruct]] = list(
            inspect.getmembers(submod, KernelStruct._issubclass)  # ruff: ignore[private-member-access]
        )
        if not include_dependencies:
            # Filter out the structs imported into the file.
            # Retain only those explicitly exported with __all__
            kernstructs = list(
                filter(lambda nameks: nameks[0] in submod.__all__, kernstructs)
            )
        # Construct the path for the output header file
        output_path = output_dir.joinpath(
            submodname[len(prefix) + 1 :].replace(".", "/") + ".h"
        )
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Make directory if it doesn't exist

        # Don't get dependencies recursively. This results in duplicate definitions.
        # The generated header files will not import their dependencies, however, so is not
        # really valid C++.
        if include_dependencies:
            # Now order the structs so that the dependencies are first
            kslist = [ks for _, ks in kernstructs]
            for i in list(reversed(range(len(kslist)))):
                ks: KernelStruct = kslist[i]
                kslist = kslist[:i] + ks.dependent_structs + kslist[i:]
            # Now remove duplicates, keeping the first-encountered copy of each
            seen = set()
            kslist = [ks for ks in kslist if not (ks in seen or seen.add(ks))]  # ty: ignore[redundant-condition]
        else:
            # Order the structs within the one file so that they have correct dependencies
            kslist = [ks for _, ks in kernstructs]
            ksset = set(kslist)
            for i in list(reversed(range(len(kslist)))):
                ks: KernelStruct = kslist[i]
                kslist = kslist[:i] + ks.dependent_structs + kslist[i:]
            # Now remove duplicates, keeping the first-encountered copy of each
            seen = set()
            kslist = [
                ks
                for ks in kslist
                if (ks in ksset) and not (ks in seen or seen.add(ks))  # ty: ignore[redundant-condition]
            ]
        # Add the structure code, in order of dependency
        for ks in kslist:
            ks.append_c_struct_to_file(
                path=output_path, create_ok=True, end="\n\n"
            )


def delete_generated_sources() -> None:
    # Delete the path if it exists.
    if path_to_generated_cusrc.is_dir():
        shutil.rmtree(path_to_generated_cusrc)


def main() -> None:
    delete_generated_sources()
    # Copy cusrc to the generated source directory
    shutil.copytree(
        src=path_to_template_cusrc,
        dst=path_to_generated_cusrc,
        dirs_exist_ok=False,
    )
    # Generate header files into the generated source directory
    generate_headers(output_dir=path_to_generated_cusrc)


if __name__ == "__main__":
    main()
