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

import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import snookerr
from snookerr import liveplot
from snookerr.cb import flatpanel as cbfp


def main() -> None:  # ruff: ignore[undocumented-public-function]
    cbfp.force_compile()  # force the relevant CUDA kernels to compile first, so that the compilation time isn't included in the timings

    ### Configurable parameters ###
    voxwid_mm: float = 1.0  # the voxel width of the phantom
    scale_phantom_size: float = (
        # 2*4/5  # make the phantom bigger, without reducing its attenuation coefficient
        1
    )
    fast_backproj: bool = (
        True  # whether to use the fast approximate backrpojection
    )
    numits: int = 100  # Number of SIRT iterations to apply
    apply_noise: bool = True
    D_mm: float = 658.353  # shortest distance from center to the edge of the equilateral triangle  # ruff: ignore[non-lowercase-variable-in-function]
    flattened_corner_width_mm: float = 440  # also the detector width
    det_aspect_ratio: float = 1  # how many times taller than wide
    num_sources_per_side: int = 17 * 1
    pixwid_mm: float = 0.3
    curved_source_edges: bool = True
    total_clearfield_photons: float = 5e12  # total number of clearfield photons in the experiment, to be distributed evenly among every detector pixel of every projection.
    ###############################

    # Equilateral triangle
    # - 'radius' r is the distance from the center to a corner of the triangle
    # - side length of equilateral triangle is r * sqrt(3)
    # - D = Distance from center to closest point on edge is r / sqrt(2)

    # Derived quantities
    D = D_mm / voxwid_mm  # ruff: ignore[non-lowercase-variable-in-function]
    flattened_corner_width = flattened_corner_width_mm / voxwid_mm
    pixwid = pixwid_mm / voxwid_mm
    fcw: float = flattened_corner_width
    r: float = D * np.sqrt(2)
    # l: float = r * np.sqrt(3)

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
    dethei: float = detwid * det_aspect_ratio
    source_z_spread: float = dethei * 0.7

    # Print some information about the geometry
    print("Distance from center to detector (mm): ", rdet * voxwid_mm)
    print("Detector width (mm):     ", detwid * voxwid_mm)
    print(
        "Source strip width (mm): ",
        voxwid_mm * float(np.linalg.norm(e12 - e11)),
    )

    # Generate source positions
    # These are scattered along the three edges.
    if curved_source_edges:
        c2 = rdet * np.array(
            [np.cos(np.deg2rad(90)), np.sin(np.deg2rad(90)), 0],
            dtype=np.float32,
        )
        ang_start2 = np.arctan2((e21 - c2)[1], (e21 - c2)[0])
        ang_end2 = np.arctan2((e22 - c2)[1], (e22 - c2)[0])
        rad2 = np.linalg.norm(e22 - c2)
        c3 = rdet * np.array(
            [np.cos(np.deg2rad(210)), np.sin(np.deg2rad(210)), 0],
            dtype=np.float32,
        )
        ang_start3 = np.arctan2((e31 - c3)[1], (e31 - c3)[0])
        ang_end3 = np.arctan2((e32 - c3)[1], (e32 - c3)[0])
        rad3 = np.linalg.norm(e32 - c3)
        c1 = rdet * np.array(
            [np.cos(np.deg2rad(330)), np.sin(np.deg2rad(330)), 0],
            dtype=np.float32,
        )
        ang_start1 = np.arctan2((e11 - c1)[1], (e11 - c1)[0])
        ang_end1 = np.arctan2((e12 - c1)[1], (e12 - c1)[0])
        rad1 = np.linalg.norm(e12 - c1)
        sps = np.array(
            [
                [
                    (
                        (
                            c
                            + 1
                            * rad
                            * np.array(
                                [np.cos(ang), np.sin(ang), 0], dtype=np.float32
                            )
                        )
                        + (1 - 2 * (i % 2))
                        * np.array([0, 0, source_z_spread / 2])
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
                for c, ang_start, ang_end, rad in zip(
                    (c1, c2, c3),
                    (ang_start1, ang_start2, ang_start3),
                    (ang_end1, ang_end2, ang_end3),
                    (rad1, rad2, rad3),
                )
            ],
            dtype=np.float32,
        )
    else:
        sps = np.array(
            [
                [
                    (
                        v
                        + (1 - 2 * (i % 2))
                        * np.array([0, 0, source_z_spread / 2])
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
                for endpoint1, endpoint2 in zip(
                    (e11, e21, e31), (e12, e22, e32)
                )
            ],
            dtype=np.float32,
        )

    # Define center coordinates of the three flat-panel detectors
    p1, p2, p3 = np.array(
        [
            [rdet * np.cos(ang), rdet * np.sin(ang), 0]
            for ang in [np.deg2rad(angdeg) for angdeg in (330, 90, 210)]
        ],
        dtype=np.float32,
    )
    # Define the horizontal vector pointing from one pixel to the next along the same row
    u1, u2, u3 = np.array(
        [
            [
                pixwid * np.cos(ang - np.pi / 2),
                pixwid * np.sin(ang - np.pi / 2),
                0,
            ]
            for ang in [np.deg2rad(angdeg) for angdeg in (330, 90, 210)]
        ],
        dtype=np.float32,
    )
    # Define the vertical vector pointing from one pixel to the next along the same column
    v1, v2, v3 = np.array(
        [[0, 0, pixwid] for _ in range(3)],
        dtype=np.float32,
    )
    nu = np.uint16(int(detwid / pixwid))  # pixel width  of detectors
    nv = np.uint16(int(dethei / pixwid))  # pixel height of detectors
    print("Detector resolution:     ", (int(nu), int(nv)))

    photons_per_pixel: float = (
        total_clearfield_photons / nu / nv / num_sources_per_side / 3
    )
    print("Photons (clearfield) per pixel: ", photons_per_pixel)

    # Construct the 'projection specification' array.
    pg: cbfp.ProjGeom = cbfp.ProjGeom(
        source_pos=sps,  # shape (3, N, 3) -- last axis is x,y,z
        det_pos=np.array(
            [[p1], [p2], [p3]]
        ),  # shape (3, 1, 3). Last axis is x,y,z. axis 1 will be broadcast
        det_uvec=np.array(
            [[u1], [u2], [u3]]
        ),  # shape (3, 1, 3). Last axis is x,y,z. axis 1 will be broadcast
        det_vvec=np.array(
            [[v1], [v2], [v3]]
        ),  # shape (3, 1, 3). Last axis is x,y,z. axis 1 will be broadcast
        nu=nu,  # integer. Will be broadcast.
        nv=nv,  # integer. Will be broadcast.
    )
    print("Number of cone-beam projections: ", pg.num_projs)

    # Create the volume to be projected
    print("Voxel width (mm):        ", voxwid_mm)
    # vol: snookerr.FBVolume = snookerr.phantoms.FORBILD.HeadPhantom
    vol: snookerr.FBVolume = snookerr.phantoms.FORBILD.CustomComposite
    vol.scale_len(
        1 / voxwid_mm
    )  # factor to change from units of mm to units of voxel-lengths
    volwid_mm = 300  # in the future, fetch from the phantom
    # Scale up the volume
    vol.scale_len(scale_phantom_size)
    volwid_mm *= scale_phantom_size

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
        voxwid_mm * sps[..., 0],
        voxwid_mm * sps[..., 1],
        zs=voxwid_mm * sps[..., 2],
        marker="x",
        label="X-ray source",
    )
    # Plot detectors
    # Convert the data back to plain np.float32 format so that we can do plotting with matplotlib
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
        # convert from attenuation as a multiple of water's attenuation to attenuation per voxel-width
        multiply_before_adding=voxwid_mm
        * 0.0184,  # attenuation of water ~ 0.0184 per mm
    )
    projs = cbfp.ProjectionSet(pg, data=None)
    print("Generating the linearised projection data... ", end="", flush=True)
    t1 = time.time()
    _ = cbfp.project(
        config_project,
        vol,
        projs,
        block=True,
    )
    assert projs.data is not None  # helpful assertion for type-checking
    t2 = time.time()
    elapsed_milliseconds: float = (t2 - t1) * 1000
    print(
        f"done. Projection time (ms): {elapsed_milliseconds:.1f}", flush=True
    )
    if apply_noise:
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
        cap_max_atten = cp.log(
            photons_per_pixel
        )  # make the minimum photon count 1 here
        out_noised_attenuation = out_
        out_noised_attenuation_capped = cp.nan_to_num(
            out_noised_attenuation, posinf=cap_max_atten
        )
        print("done.", flush=True)
        print()
        proj_array = out_noised_attenuation_capped
    else:
        proj_array = projs.data
    # snookerr.imview.plot_projections(out_noised_transmission.reshape(-1, *out_noised_transmission.shape[-2:]))
    snookerr.imview.plot_projections(
        proj_array.reshape(-1, *proj_array.shape[-2:]),
        title="max attenuation capped for plotting purposes",
    )

    # Free up memory by deleting various unneeded variables
    del vol, projs
    if apply_noise:
        out_noised_attenuation, out_noised_transmission, out_

    # Set up the working volume
    vol_shape: tuple[int, int, int] = tuple(
        int(volwid_mm // voxwid_mm) for _ in range(3)
    )  # ty:ignore[invalid-assignment]  # ty is bugged here
    working_vol = snookerr.Volume.zeros(vol_shape, voxwid=1, dtype=np.float32)  # ty:ignore[invalid-argument-type]  # ty is bugged

    # Backproject into the working volume
    config_backproject = cbfp.ConfigBackproject(
        fast_approx=fast_backproj,
        cyl_mask_rad=working_vol.shape[0] / 2 * working_vol.voxwid,
    )
    projs = cbfp.ProjectionSet(pg, data=proj_array)
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

    # Set up the liveplot object
    _lp_process, lp_dispatcher = liveplot.start_live_plot_2d(
        title="Live plot -- reconstructed volume",
        histogram_label="Line Attenuation",
        plotmin=0,
        plotmax=1.8 * 0.0184 * voxwid_mm,
        # plotmin=0.0184 * voxwid_mm * (1 + default_plot_range_line_attenuationdefault_plot_range_HU[0] / 1000),
        # plotmax=0.0184 * voxwid_mm * (1 + default_plot_range_HU[1] / 1000),
        show=True,
    )

    # Make a SIRT object
    working_vol.fill(0)
    recon = snookerr.iterators.SIRT_flatpanel(
        config=snookerr.iterators.SIRT_flatpanel.Config(
            fast_backproject=fast_backproj,
            soft_from_cylinder_relrad=0.95,
            bp_soft_from_detedge_relrad=0.95,
        ),
        volume=working_vol,
        projs=projs,
    )
    for i in range(numits + 1):
        s = f"Iteration {i:04d}/{numits:04d}"
        print(s + "   -- UNIX time: ", time.time(), flush=True)
        recon.iterate()
        lp_dispatcher.update_image(
            cp.flip(
                recon.volume.as_cupy().transpose((2, 0, 1)), axis=(1, 2)
            ).get()
        )
    v = recon.volume.as_numpy()
    snookerr.imview.plot_projections(
        np.sum(v, axis=1)[np.newaxis, :, :],
        title=s + " -- projection through y axis",
    )
    snookerr.imview.plot_volume(v, title=s + " -- top-down")
    snookerr.imview.plot_volume(
        v.transpose((0, 2, 1)), title=s + " -- through y axis"
    )
    snookerr.imview.plot_volume(
        v.transpose((2, 1, 0)), title=s + " -- through x axis"
    )


if __name__ == "__main__":
    main()
