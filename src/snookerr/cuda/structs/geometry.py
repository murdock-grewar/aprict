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
"""A struct used to configure the CB projection kernel."""

import numpy as np

from .. import kern_structs
from .base import fvec2, fvec3, uivec2, uivec3  # noqa: F401

__all__ = [
    "det_spec",
    "proj_spec",
]

det_spec = kern_structs.KernelStruct(
    struct_name="det_spec",
    names=[
        "pos",  # position of detector (nominally, the middle of it)
        "u",  # vector pointing along the x direction. Also determines pixel width.
        "v",  # vector pointing along the y direction. Also determines pixel height.
        "nu",
        "nv",  # number of columns/rows. i.e. length of x axis; length of y axis.
        "type",  # numeric integer describing the type of detector. Currently, leave it at 0 for a flat panel.
    ],
    dtypes=[fvec3, fvec3, fvec3, np.uint16, np.uint16, np.uint8],
)

proj_spec = kern_structs.KernelStruct(
    struct_name="proj_spec",
    names=["sourcepos", "dspec"],
    dtypes=[fvec3, det_spec],
    force_dependencies=[
        fvec2,
    ],
    extra_code="""
/*
 * In order to mantain O(N^3) scaling in the backprojection, we need a finite, closed-form equation
 * which maps a voxel to its projection onto the detector, in terms of the detector indices.
 *
 * (The alterantive is to run a thread for each detector index, but the lack of atomic addition
 * for surfaces/textures leads to the need to use a lot of global memory atomicAdds for large
 * volumes, which is unacceptably slow. Maybe it's possible to make the memory accesses more
 * efficient for that case; I'm not sure.)
 */

__device__ fvec2 get_projected_detector_coords(
        const fvec3       point,
        const proj_spec   projspec
)
{
    if (projspec.dspec.type == 0) {
        /* Solve equation for p, lambda:
         * p . n = detector . n
         * p = sourcepos + lambda(point - sourcepos)
         * ==> detector . n = sourcepos . n + lambda (point - sourcepos) . n
         * ==> (detector - sourcepos) . n = lambda (point - sourcepos) . n
         * ==> lambda = [(detector - sourcepos) . n] / [(point - sourcepos) . n]
         * ==> p = sourcepos + lambda(point - sourcepos) with lambda as above
         */
        // Compute actual point of intersection, in 3D space
        const fvec3 n = cross_prod(projspec.dspec.u, projspec.dspec.v);
        fvec3 p = projspec.sourcepos + (point - projspec.sourcepos) * (
            dot_prod(projspec.dspec.pos - projspec.sourcepos, n) /
            dot_prod(point              - projspec.sourcepos, n)
        );
        // Compute the point's detector coordinates, from the center.
        p = p - projspec.dspec.pos;
        const float unorm = norm_fvec3(projspec.dspec.u);
        const float vnorm = norm_fvec3(projspec.dspec.v);
        return fvec2(
            dot_prod(projspec.dspec.u, p)/unorm/unorm,
            dot_prod(projspec.dspec.v, p)/vnorm/vnorm
        );
    } else {
        // UNIMPLEMENTED!!
        return fvec2(0.f, 0.f);
    }
}""",
)
