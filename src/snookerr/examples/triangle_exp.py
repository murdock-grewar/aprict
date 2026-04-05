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

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import snookerr
from snookerr.cb import flatpanel as cbfp

cbfp.force_compile()  # force the relevant CUDA kernels to compile first, so that the compilation time isn't included in the timings

### Configurable parameters ###
voxwid_mm: float = 2  # the voxel width of the head phantom
scale_phantom_size: float = (
    2  # make the phantom bigger, without reducing its attenuation coefficient
)
D_mm: float = 652.5  # shortest distance from center to the edge of the equilateral triangle
flattened_corner_width_mm: float = 430  # also the detector width and height
num_sources_per_side: int = 17
pixwid_mm: float = 0.5
curved_source_edges: bool = True
superscale_volume_factor: int = 2  # factor by which to upscale the volume resolution, for the purpose of generating data.
translate_phantom_mm = (
    0  # scale_phantom_size*np.array([0, 0, 125], dtype=np.float32)
)
total_clearfield_photons: float = 1e13  # total number of clearfield photons in the experiment, to be distributed evenly among every detector pixel of every projection.
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
source_z_spread: float = detwid * 0

# Print some information about the geometry
print("Distance from center to detector (mm): ", rdet * voxwid_mm)
print("Detector width (mm):     ", detwid * voxwid_mm)
print(
    "Source strip width (mm): ", voxwid_mm * float(np.linalg.norm(e12 - e11))
)

# Generate source positions
# These are scattered along the three edges, and staggered at high/low positions.
if curved_source_edges:
    c2 = rdet * np.array(
        [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90)), 0], dtype=np.float32
    )
    ang_start2 = np.arctan2((e21 - c2)[1], (e21 - c2)[0])
    ang_end2 = np.arctan2((e22 - c2)[1], (e22 - c2)[0])
    c3 = rdet * np.array(
        [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210)), 0], dtype=np.float32
    )
    ang_start3 = np.arctan2((e31 - c3)[1], (e31 - c3)[0])
    ang_end3 = np.arctan2((e32 - c3)[1], (e32 - c3)[0])
    c1 = rdet * np.array(
        [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330)), 0], dtype=np.float32
    )
    ang_start1 = np.arctan2((e11 - c1)[1], (e11 - c1)[0])
    ang_end1 = np.arctan2((e12 - c1)[1], (e12 - c1)[0])
    sps = np.array(
        [
            [
                tuple(
                    (
                        c
                        + 2
                        * rdet
                        * np.array(
                            [np.cos(ang), np.sin(ang), 0], dtype=np.float32
                        )
                    )
                    + (1 - 2 * (i % 2)) * np.array([0, 0, source_z_spread / 2])
                )
                for i, ang in enumerate(
                    np.linspace(
                        ang_start,
                        ang_end,
                        endpoint=True,
                        num=(num_sources_per_side + 2),
                    )[1:-1]
                )
            ]
            for c, ang_start, ang_end in zip(
                (c1, c2, c3),
                (ang_start1, ang_start2, ang_start3),
                (ang_end1, ang_end2, ang_end3),
            )
        ],
        ndmax=2,
        dtype=snookerr.dtypes.fvec3,
    )
else:
    sps = np.array(
        [
            [
                tuple(
                    v
                    + (1 - 2 * (i % 2)) * np.array([0, 0, source_z_spread / 2])
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

photons_per_pixel: float = (
    total_clearfield_photons / nu / nv / num_sources_per_side / 3
)
print("Photons (clearfield) per pixel: ", photons_per_pixel)

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
volwid: int = int(np.ceil(250 / voxwid_mm * scale_phantom_size))
print("Volume   resolution:     ", (int(volwid), int(volwid), int(volwid)))
print(
    "Volume   resolution (superscaled):     ",
    (
        superscale_volume_factor * int(volwid),
        superscale_volume_factor * int(volwid),
        superscale_volume_factor * int(volwid),
    ),
)
print()
print("Generating FORBILD head phantom... ", end="", flush=True)
vnp, voxwid_mm = snookerr.lib.phantoms.FORBILD_head_HU(
    volwid * superscale_volume_factor
)  # intentionally overwriting voxwid_mm
print("done.")
print()
voxwid_mm *= scale_phantom_size
volwid_mm: float = superscale_volume_factor * volwid * voxwid_mm
print("Voxel width (mm):        ", voxwid_mm)
# Remove negative offset from the HU
vnp += 1000
# Convert to (very approximate) attenuation coefficient of 0.0184 per mm
vnp *= 0.0184 * voxwid_mm / 1000
# Convert volume to 16-bit to save on VRAM...
vnp = vnp.astype(np.float16)
# The volume should now be encoded so that the value in a voxel is the attenuation through one voxel-width.
# adjust the voxwid_mm to represent the voxel width of the non-supersampled volume
voxwid_mm_supersamp = voxwid_mm
voxwid_mm *= superscale_volume_factor

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
config_project = cbfp.ConfigProject(
    shrink_volume_factor=superscale_volume_factor
)
projs = cbfp.ProjectionSet(pss, data=None)
vol = snookerr.Volume(
    vnp,
    pos_c=np.array(
        translate_phantom_mm / voxwid_mm_supersamp, dtype=np.float32
    ),
)
print("Generating the linearised projection data... ", end="", flush=True)
t1 = time.time()
out = cbfp.project(
    config_project,
    vol,
    projs,
    block=True,
)
assert projs.data is not None  # helpful assertion for type-checking
t2 = time.time()
elapsed_milliseconds: float = (t2 - t1) * 1000
print(f"done. Projection time (ms): {elapsed_milliseconds:.1f}", flush=True)
print("Applying Poisson noise... ", end="", flush=True)
out_ = projs.data.copy()
cp.exp(-out_, out=out_)
cp.multiply(out_, photons_per_pixel, out=out_)
out_ = cp.random.poisson(lam=out_).astype(
    cp.float32
)  # no option to run this in-place with cupy...
cp.true_divide(out_, cp.float32(photons_per_pixel), out=out_)
out_noised_transmission = out_.copy()
cp.log(out_, out=out_)
cp.multiply(out_, -1, out=out_)
print("done.", flush=True)
print()
cap_max_atten = cp.log(
    photons_per_pixel
)  # make the minimum photon count 1 here
out_noised_attenuation = out_
out_noised_attenuation_capped = cp.nan_to_num(
    out_noised_attenuation, posinf=cap_max_atten
)
# snookerr.imview.plot_projections(out_noised_transmission.reshape(-1, *out_noised_transmission.shape[-2:]))
snookerr.imview.plot_projections(
    out_noised_attenuation_capped.reshape(
        -1, *out_noised_attenuation_capped.shape[-2:]
    ),
    title="max attenuation capped for plotting purposes",
)


# Set up the working volume
working_vol = snookerr.Volume.zeros(
    (
        vol.shape[0] // superscale_volume_factor,
        vol.shape[1] // superscale_volume_factor,
        vol.shape[2] // superscale_volume_factor,
    ),
    dtype=np.float16,  # ty:ignore[invalid-argument-type]
)


# Backproject into the working volume
config_backproject = cbfp.ConfigBackproject(
    fast_approx=True, cyl_mask_rad_vox=working_vol.shape[0] / 2
)
projs = cbfp.ProjectionSet(pss, data=out_noised_attenuation_capped)
t1 = time.time()
working_vol = cbfp.backproject(
    config_backproject,
    working_vol,
    projs,
    block=True,
)
t2 = time.time()
elapsed_milliseconds: float = (t2 - t1) * 1000
print(f"Backrojection time (ms): {elapsed_milliseconds:.1f}", flush=True)
snookerr.imview.plot_volume(working_vol.as_numpy())
