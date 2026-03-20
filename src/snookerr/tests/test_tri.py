#!/usr/bin/env python
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

"""Run all available tests with pytest."""

import pytest

from .. import tritest


def test_triangle_numbers() -> None:
    """Test that triangle numbers are computed correctly."""
    inputs = [0, 1, 2, 3, 4, 5]
    outputs = [0, 1, 3, 6, 10, 15]
    for i, o in zip(inputs, outputs):
        assert o == tritest.triangle_number(i)
    with pytest.raises(ValueError):
        tritest.triangle_number(-1)


if __name__ == "__main__":
    pytest.main()
