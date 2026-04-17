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

"""Simulate a projection of a collection of spheres."""

import cupy as cp
import numpy as np

import snookerr
from snookerr.cb import flatpanel as cbfp

cbfp.force_compile()  # force the relevant CUDA kernels to compile first, so that the compilation time isn't included in the timings

# List of shape objects
shapes = []

# Make sphere shapes
numspheres: int = 3
for _ in range(numspheres):
    x, y, z = 10 * np.random.randn(3)  # noqa: NPY002
    r = np.random.uniform(0, 3)  # noqa: NPY002
    atten_coeff = np.random.uniform(0, 3)  # noqa: NPY002
    s = np.array(0, dtype=snookerr.dtypes.FORBILD_volumes.Shape)
    s["type"] = 13  # sphere type is 13
    s["rho"] = atten_coeff
    s["Sphere"] = np.array(
        (x, y, z, r), dtype=snookerr.dtypes.FORBILD_volumes.Sphere
    )
    shapes.append(s)

arr = cp.array(
    np.array(shapes, dtype=snookerr.dtypes.FORBILD_volumes.Shape, ndmax=1)
)
print(arr)
