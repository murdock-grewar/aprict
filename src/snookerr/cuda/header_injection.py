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

"""Module to inject local header files into .cu files."""

import pathlib
import re

import cupy as cp

from ..data import cusrc

cusrc_path = pathlib.Path(cusrc.__path__[0])

re_include_pattern = re.compile(r'^[\w]*#include "([^"]*)"(.*)$')


def inject_headers(path_to_c_file: pathlib.Path | str) -> str:
    """Inject local #includes and return the new file string."""
    s = ""
    p = pathlib.Path(path_to_c_file)
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if m := re_include_pattern.match(line):
                fn, eol = m.groups()
                pdep = p.parent.joinpath(fn)
                s += eol + "\n"
                with open(pdep, "r", encoding="utf-8") as fdep:
                    [(s := s + li) for li in fdep]  # ty: ignore[unresolved-reference] # ty is bugged. See issues #3077 and #162 on the ty GitHub page.
                s += "\n"
            else:
                s += line
    return s


def load_injected_module(
    rel_path_to_uninjected_cu: pathlib.Path | str,
    *,
    # Add -lineinfo to make the cached kernel readable by nsight-compute
    options: tuple[str, ...] = (
        "-std=c++17",
        "-use_fast_math",
    ),  # "-lineinfo"),  # "-maxrregcount", "128"),
    extraoptions: tuple[str, ...] | None = None,
    name_expressions: list[str] | None = None,
) -> cp.RawModule:
    """Create a cupy.RawModule object from a .cu source file.

    Local header files will be injected into the file before compilation.
    """  # noqa: DOC201
    if extraoptions is not None:
        options: tuple[str, ...] = options + extraoptions
    return cp.RawModule(
        code=inject_headers(cusrc_path.joinpath(rel_path_to_uninjected_cu)),
        options=options,
        name_expressions=name_expressions,
    )
