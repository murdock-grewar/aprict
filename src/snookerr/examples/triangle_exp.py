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

"""Simulate an example experiment with a triangular scanning apparatus."""

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import snookerr

### Configurable parameters ###
voxwid_mm: float = 0.25  # the voxel width of the head phantom
D_mm: float = 600  # shortest distance from center to the edge of the equilateral triangle
flattened_corner_width_mm: float = 430  # also the detector width and height
num_sources_per_side: int = 15
pixwid_mm: float = 0.5
###############################

# Equilateral triangle
# - 'radius' r is the distance from the center to a corner of the triangle
# - side length of equilateral triangle is r * sqrt(3)
# - D = Distance from center to closest point on edge is r / sqrt(2)

# Derived quantities
D = D_mm / voxwid_mm
flattened_corner_width = flattened_corner_width_mm / voxwid_mm
pixwid = pixwid_mm / voxwid_mm
fcw: float = flattened_corner_width
r: float = D * np.sqrt(2)
l: float = r * np.sqrt(3)

# Define the corners of the triangle
v1, v2, v3 = r * np.array(
    [
        [np.cos(ang), np.sin(ang), 0]
        for ang in [np.deg2rad(angdeg) for angdeg in (90, 210, 330)]
    ],
    dtype=np.float32,
)
# The corners are 'blunted' and flattened to make way for the detectors.
# The three original edges of the triangle are thereby shrunk.
# Detectors are dotted along these shortened edges.
# The first edge is from point e11 to e12. Likewise for (e21, e22) and (e31, e32).
e11 = v1 + fcw * np.array(
    (np.sin(np.deg2rad(210)), np.cos(np.deg2rad(210)), 0), dtype=np.float32
)
e12 = v2 - fcw * np.array(
    (np.sin(np.deg2rad(210)), np.cos(np.deg2rad(210)), 0), dtype=np.float32
)
e21 = v2 + fcw * np.array((1, 0, 0), dtype=np.float32)
e22 = v3 - fcw * np.array((1, 0, 0), dtype=np.float32)
e31 = v3 + fcw * np.array(
    (np.cos(np.deg2rad(120)), np.sin(np.deg2rad(120)), 0), dtype=np.float32
)
e32 = v1 - fcw * np.array(
    (np.cos(np.deg2rad(120)), np.sin(np.deg2rad(120)), 0), dtype=np.float32
)
# Source and detector parameters
rdet: float = float(np.linalg.norm((e12 + e21) / 2))
detwid: float = flattened_corner_width
dethei: float = detwid
source_z_spread: float = detwid * 2 / 3

# Print some information about the geometry
print("Detector width (mm):     ", detwid * voxwid_mm)
print(
    "Source strip width (mm): ", voxwid_mm * float(np.linalg.norm(e12 - e11))
)

# Generate source positions
# These are scattered along the three edges, and staggered at high/low positions.
sps = np.array(
    [
        [
            tuple(
                v + (1 - 2 * (i % 2)) * np.array([0, 0, source_z_spread / 2])
            )
            for i, v in enumerate(
                np.linspace(
                    endpoint1,
                    endpoint2,
                    endpoint=True,
                    num=(num_sources_per_side + 2),
                )[1:-1]
            )
        ]
        for endpoint1, endpoint2 in zip((e11, e21, e31), (e12, e22, e32))
    ],
    ndmax=2,
    dtype=snookerr.dtypes.fvec3,
)
assert sps.dtype == snookerr.dtypes.fvec3


# Define center coordinates of the three flat-panel detectors
p1, p2, p3 = np.array(
    [
        (rdet * np.cos(ang), rdet * np.sin(ang), 0)
        for ang in [np.deg2rad(angdeg) for angdeg in (330, 90, 210)]
    ],
    ndmax=1,
    dtype=snookerr.dtypes.fvec3,
)
# Define the horizontal vector pointing from one pixel to the next along the same row
u1, u2, u3 = np.array(
    [
        (pixwid * np.cos(ang - np.pi / 2), pixwid * np.sin(ang - np.pi / 2), 0)
        for ang in [np.deg2rad(angdeg) for angdeg in (330, 90, 210)]
    ],
    ndmax=1,
    dtype=snookerr.dtypes.fvec3,
)
# Define the vertical vector pointing from one pixel to the next along the same column
v1, v2, v3 = np.array(
    [(0, 0, pixwid) for _ in range(3)],
    ndmax=1,
    dtype=snookerr.dtypes.fvec3,
)
nu = np.uint16(int(detwid / pixwid))  # pixel width  of detectors
nv = np.uint16(int(detwid / pixwid))  # pixel height of detectors
print("Detector resolution:     ", (int(nu), int(nv)))

# Construct the 'detector specification' objects.
# One is required for every cone-beam projection. So there's one for each source position,
# and detector combination. Here, I've chosen to arrange them in an array as follows:
# [
#   [det_1_source_00, det_1_source_01, ...],
#   [det_2_source_10, det_1_source_11, ...],
#   [det_3_source_20, det_1_source_21, ...],
# ]
# Each detector is exposed to different source points, but each is exposed to exactly 15 sources,
# so this type of arrangement is possible. They could all be arranged in a flattened array if desired.
# The exact layout doesn't matter, as long as it's an array with dtype 'det_spec'.
dss = np.array(
    [
        [
            (
                p,
                u,
                v,
                nu,
                nv,
                np.uint8(
                    0
                ),  # a developer-reserved number. Leave at 0 for a flat-panel detector.
            )
            for _ in range(num_sources_per_side)
        ]
        for p, u, v in zip(
            (p1, p2, p3),
            (u1, u2, u3),
            (v1, v2, v3),
        )
    ],
    ndmax=2,
    dtype=snookerr.dtypes.det_spec,
)
assert dss.dtype == snookerr.dtypes.det_spec.dtype
print("Number of cone-beam projections: ", dss.size)


# Construct an array of 'projection specification' objects.
# These are pairings of `detector specification' objects with corresponding 'source_pos' objects.
def convert_to_projspec(sourcepos, det_spec):  # noqa: ANN001, ANN201, D103
    return np.array(
        (sourcepos, det_spec), dtype=snookerr.dtypes.proj_spec.dtype
    )


pss = np.empty(sps.shape, dtype=snookerr.dtypes.proj_spec.dtype)
for index_tuple in itertools.product(*map(range, sps.shape)):
    pss[index_tuple] = convert_to_projspec(sps[index_tuple], dss[index_tuple])
assert pss.dtype == snookerr.dtypes.proj_spec.dtype
# Check that all have the type set to flat panel
dspecs = np.array(pss["dspec"], dtype=snookerr.dtypes.det_spec.dtype)
assert np.all(0 == dspecs["type"])
# Check that all detector dimensions are the same
nus = dspecs["nu"]
nvs = dspecs["nv"]
assert np.all(nus == nus[0])
assert np.all(nvs == nvs[0])


# Create the volume to be projected
volwid: int = int(np.ceil(250 / voxwid_mm))
print("Volume   resolution:     ", (int(volwid), int(volwid), int(volwid)))
print("Generating FORBILD head phantom... ", end="", flush=True)
vnp, voxwid_mm = snookerr.lib.phantoms.FORBILD_head_HU(
    volwid
)  # intentionally overwriting voxwid_mm
print("done.")
volwid_mm: float = volwid * voxwid_mm
print("Voxel width (mm):        ", voxwid_mm)
# Remove negative offset from the HU
vnp += 1000
# Convert to (very approximate) attenuation coefficient of 0.0184 per mm
vnp *= 0.0184 * voxwid_mm / 1000
# Convert volume to 16-bit to save on VRAM...
vnp = vnp.astype(np.float16)
# The volume should now be encoded so that the value in a voxel is the attenuation through one voxel-width.

### Do plotting of the imaging geometry
# Prepare the 3D plot
ax = plt.figure().add_subplot(projection="3d")
ax.set_xlim(-D_mm, D_mm)
ax.set_ylim(-D_mm, D_mm)
ax.set_zlim(-D_mm, D_mm)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.view_init(elev=20.0, azim=-35, roll=0)
ax.set_aspect("equal")
# Plot X-ray sources
ax.scatter(
    voxwid_mm * sps["x"],
    voxwid_mm * sps["y"],
    zs=voxwid_mm * sps["z"],
    marker="x",
    label="X-ray source",
)
# Plot detectors
# Convert the data back to plain np.float32 format so that we can do plotting with matplotlib
p1, p2, p3, u1, u2, u3, v1, v2, v3 = [
    np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
    for v in [p1, p2, p3, u1, u2, u3, v1, v2, v3]
]
verts1_mm = voxwid_mm * np.array(
    [
        p1 + nu * u1 / 2 + nv * v1 / 2,
        p1 - nu * u1 / 2 + nv * v1 / 2,
        p1 - nu * u1 / 2 - nv * v1 / 2,
        p1 + nu * u1 / 2 - nv * v1 / 2,
        p1 + nu * u1 / 2 + nv * v1 / 2,
    ]
)
verts2_mm = voxwid_mm * np.array(
    [
        p2 + nu * u2 / 2 + nv * v2 / 2,
        p2 - nu * u2 / 2 + nv * v2 / 2,
        p2 - nu * u2 / 2 - nv * v2 / 2,
        p2 + nu * u2 / 2 - nv * v2 / 2,
        p2 + nu * u2 / 2 + nv * v2 / 2,
    ]
)
verts3_mm = voxwid_mm * np.array(
    [
        p3 + nu * u3 / 2 + nv * v3 / 2,
        p3 - nu * u3 / 2 + nv * v3 / 2,
        p3 - nu * u3 / 2 - nv * v3 / 2,
        p3 + nu * u3 / 2 - nv * v3 / 2,
        p3 + nu * u3 / 2 + nv * v3 / 2,
    ]
)
ax.add_collection(
    Poly3DCollection([verts1_mm, verts2_mm, verts3_mm], alpha=0.7)
)
# Plot the sphere representing the object
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = volwid_mm / 2 * np.outer(np.cos(u), np.sin(v))
y = volwid_mm / 2 * np.outer(np.sin(u), np.sin(v))
z = volwid_mm / 2 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color="black", alpha=0.3, label="object")
ax.legend()
plt.show()


## Call the projection code on the prepared geometry ##
config = np.array((), dtype=snookerr.dtypes.config_cb_project)
vol = snookerr.Volume(vnp)
output_array = None
t1 = time.time()
out = snookerr.cb.project_flatpanel(config, vol, pss, output_array, block=True)  # ty:ignore[invalid-argument-type]
t2 = time.time()
elapsed_milliseconds: float = (t2 - t1) * 1000
print(f"Projection time (ms):     {elapsed_milliseconds:.1f}")
snookerr.imview.plot_projections(out.reshape(-1, *out.shape[-2:]))
