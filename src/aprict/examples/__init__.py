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

"""
A submodule containing example scripts using the Python API.

Browse the `aprict` source code and navigate to this directory to see the examples.

Examples can be run with e.g. `python -m aprict.examples.EXAMPLE_NAME`.

"""

from . import (
    composite_human_sirt,
    fan_beam_sirt,
    helical_sirt,
    lungs_sirt,
    trajtest_A_sirt,
    triangle_exp_fb,
)

__all__ = [
    "composite_human_sirt",
    "fan_beam_sirt",
    "helical_sirt",
    "lungs_sirt",
    "trajtest_A_sirt",
    "triangle_exp_fb",
]
