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

"""Tests for the Volume class."""

import numpy as np
import pytest

from .. import Volume  # import directoly from root snookerr namespace

rng = np.random.default_rng()


@pytest.mark.parametrize(
    argnames=["dtype"],
    argvalues=[(np.float16,), (np.float32,)],
    ids=["Volume-fp16", "Volume-fp32"],
)
def test_volume(dtype: np.dtype) -> None:
    """Test the creation of the Volume class, and filling it with data."""
    a1 = rng.standard_normal((100, 121, 139)).astype(dtype)
    # Create the volume directly from a1
    v = Volume(a1)
    # Check shape and type
    assert v.dtype == np.dtype(dtype)
    assert v.shape == (100, 121, 139)
    # Check that the data was set properly
    a2 = np.empty_like(a1)
    v.copy_to(a2)
    np.testing.assert_array_equal(a1, a2)
    # Creating the volume by copying in data
    del v
    v = Volume.zeros(a1.shape, dtype)
    a1 = rng.standard_normal(v.shape).astype(v.dtype)
    v.copy_from(a1)
    v.copy_to(a2)
    np.testing.assert_array_equal(a1, a2)
    # Filling the array with data
    v.fill(2.0)
    a1.fill(2)
    v.copy_to(a2)
    np.testing.assert_array_equal(a1, a2)
