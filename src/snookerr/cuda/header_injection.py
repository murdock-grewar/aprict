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
