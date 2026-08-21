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

"""Simulate an example experiment with a helical scan."""

import itertools
import math
import pathlib
import sys
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

import snookerr
from snookerr import liveplot
from snookerr.cb import flatpanel as cbfp


def main() -> None:  # ruff: ignore[complex-structure, undocumented-public-function]
    cbfp.force_compile()  # force the relevant CUDA kernels to compile first, so that the compilation time isn't included in the timings

    # Set the region of interest, in millimeters
    # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    # roi = [[-150, 0], [-75, 75], [-25, 25]]
    roi = [[-150, 0], [-75, 75], [-100, 100]]

    ### Configurable parameters ###
    output_projs_path: pathlib.Path | None = (
        None if len(sys.argv) < 4 else pathlib.Path(sys.argv[3])
    )
    output_points_zt_path: pathlib.Path | None = (
        None if len(sys.argv) < 5 else pathlib.Path(sys.argv[4])
    )

    reconstruct: bool = False
    live_plot: bool = True
    # default_plot_range_HU: list[int | float] = [0, 1000]
    default_plot_range_line_attenuation: list[int | float] = [0.0, 0.4]
    voxwid_mm: float = (
        0.75  # the voxel width of the head phantom--- for reconstruction only
    )
    scale_phantom_size: float = (
        # 2*4/5  # make the phantom bigger, without reducing its attenuation coefficient
        1
    )
    plot_volume: bool = False
    apply_noise: bool = False
    # R_mm: float = 500
    R_mm: float = 500  # ruff: ignore[non-lowercase-variable-in-function]
    L_mm: float = R_mm + 500  # ruff: ignore[non-lowercase-variable-in-function]
    num_sources: int = 30
    total_clearfield_photons: float = 1e14  # total number of clearfield photons in the experiment, to be distributed evenly among every detector pixel of every projection.
    detwid_mm: float = 1200
    dethei_mm: float = 600
    pixel_wid_mm: float = 1
    fast_approx_backprojection: bool = True
    numits: int = 100 if len(sys.argv) < 2 else int(sys.argv[1])
    vol_dtype = np.float32  # either np.float16 or np.float32
    proj_dtype = np.float32  # either np.float16 or np.float32 or np.float64. np.float16 seems to not have enough precision
    save_volume_path: pathlib.Path | None = (
        None if len(sys.argv) < 3 else pathlib.Path(sys.argv[2])
    )
    show_plots_before_recon: bool = True
    ###############################

    if save_volume_path is not None:
        if save_volume_path.exists():
            msg = (
                f"File at output path '{save_volume_path!s:s}' already exists."
            )
            raise FileExistsError(msg)

    # Derived quantities
    det_numpixwid: int = int(detwid_mm / pixel_wid_mm)
    det_numpixhei: int = int(dethei_mm / pixel_wid_mm)
    photons_per_pixel: float = (
        total_clearfield_photons / det_numpixwid / det_numpixhei / num_sources
    )

    # Print some information about the geometry
    print("Distance from source to axis     (mm): ", R_mm)
    print("Distance from source to detector (mm): ", L_mm)
    print("Detector width  (mm):       ", detwid_mm)
    print("Detector height (mm):       ", dethei_mm)
    print("Detector width  (px):       ", det_numpixwid)
    print("Detector height (px):       ", det_numpixhei)
    print("Detector pixel width  (mm): ", pixel_wid_mm)
    print("Detector pixel height (mm): ", pixel_wid_mm)  # same as pixel width

    # Generate projection specifications
    # cbfp.create_proj_spec_array(
    #   source_pos_arr,
    #   det_pos_arr,
    #   det_uvec_arr,
    #   det_vvec_arr,
    #   detpixwid,
    #   detpixhei,
    # )
    roi_maxrad = np.sqrt(
        np.square(max(abs(roi[0][0]), abs(roi[0][1])))
        + np.square(max(abs(roi[1][0]), abs(roi[1][1])))
    )
    # h = height of source cylinder
    zeta = np.arctan2(
        # Vertical distance to corner of detector
        dethei_mm / 2,
        # Lateral distance to corner of detector
        np.sqrt(np.square(L_mm) + np.square(detwid_mm / 2)),
    )
    print(
        f"Vertical cone angle (after Colsher window): {np.rad2deg(2 * zeta):.1f} degs"
    )
    h = (
        # z height of ROI
        (roi[2][1] - roi[2][0])
        # Lateral distance from source point to furthest point in ROI
        + 2 * np.tan(zeta) * (roi_maxrad + R_mm)
    )
    zmid_mm = (roi[2][0] + roi[2][1]) / 2
    zmin_mm = zmid_mm - h / 2
    zmax_mm = zmid_mm + h / 2

    zs, ts = np.array(
        list(
            itertools.islice(
                snookerr.lib.lds.lds_cyl(zmin=zmin_mm, zmax=zmax_mm, rad=R_mm),
                num_sources,
            )
        ),
        dtype=np.float32,
    ).transpose()

    if output_points_zt_path is not None:
        np.save(output_points_zt_path, np.array([zs, ts]).transpose())

    source_pos_arr = np.array(
        [R_mm * np.cos(ts), R_mm * np.sin(ts), zs], dtype=np.float32
    ).transpose()
    det_pos_arr = source_pos_arr * np.array(
        [[1 - L_mm / R_mm, 1 - L_mm / R_mm, 1]], dtype=np.float32
    )
    det_uvec_arr = (
        pixel_wid_mm
        * np.array(
            [-np.sin(ts), np.cos(ts), 0 * ts], dtype=np.float32
        ).transpose()
    )
    det_vvec_arr = pixel_wid_mm * np.array([[0, 0, 1]], dtype=np.float32)
    pg: cbfp.ProjGeom = cbfp.ProjGeom(
        source_pos=source_pos_arr / voxwid_mm,
        det_pos=det_pos_arr / voxwid_mm,
        det_uvec=det_uvec_arr / voxwid_mm,
        det_vvec=det_vvec_arr / voxwid_mm,
        nu=det_numpixwid,
        nv=det_numpixhei,
    )
    print("Number of cone-beam projections: ", pg.num_projs)

    # Create the volume to be projected
    print("Voxel width (mm):        ", voxwid_mm)
    print()
    print("Creating volume... ", end="", flush=True)
    vol: snookerr.FBVolume = snookerr.phantoms.FORBILD.CustomComposite
    if plot_volume:
        print("plotting volume... ", end="", flush=True)
        xs = np.linspace(-300, 300, num=601, endpoint=True)
        ys = np.linspace(-200, 200, num=401, endpoint=True)
        zs = np.linspace(-750, 500, num=1251, endpoint=True)
        xvol = vol.to_numpy(xs, ys, zs)
        snookerr.imview.plot_volume(
            xvol.transpose((2, 1, 0)), title="Side profile of whole volume"
        )
        # Plot region of interest (ROI)
        xs = np.linspace(
            roi[0][0],
            roi[0][1],
            num=1 + math.ceil(4 * (roi[0][1] - roi[0][0])),
            endpoint=True,
        )
        ys = np.linspace(
            roi[1][0],
            roi[1][1],
            num=1 + math.ceil(4 * (roi[1][1] - roi[1][0])),
            endpoint=True,
        )
        zs = np.linspace(
            roi[2][0],
            roi[2][1],
            num=1 + math.ceil(4 * (roi[2][1] - roi[2][0])),
            endpoint=True,
        )
        xvol = vol.to_numpy(xs, ys, zs)
        snookerr.imview.plot_volume(
            xvol, title="Region of interest (axial slices)"
        )
        del xvol
    vol.scale_len(1 / voxwid_mm)
    print("done.")
    if False:
        vol_numpy = vol.to_numpy(
            xs=np.linspace(
                -400, 400, num=1001, endpoint=True, dtype=np.float32
            ),
            ys=np.linspace(
                -400, 400, num=1001, endpoint=True, dtype=np.float32
            ),
            zs=np.linspace(
                2 * math.floor(zmin_mm),
                2 * math.floor(zmax_mm),
                endpoint=True,
                num=1 + math.floor(zmax_mm) - math.floor(zmin_mm),
                dtype=np.float32,
            ),
        )
        snookerr.imview.plot_volume(vol_numpy)
        exit(0)
        del vol_numpy
    print("Number of shapes in volume: ", len(vol.shapes))
    volwid_mm = 500  # in the future, fetch from the phantom
    # Scale up the volume
    vol.scale_len(scale_phantom_size)
    volwid_mm *= scale_phantom_size

    ### Do plotting of the imaging geometry
    if show_plots_before_recon:
        # Prepare the 3D plot
        ax = plt.figure().add_subplot(projection="3d")
        ax.set_xlim(-R_mm, R_mm)
        ax.set_ylim(-R_mm, R_mm)
        ax.set_zlim(zs.min(), zs.max())
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.view_init(elev=20.0, azim=-35, roll=0)
        ax.set_aspect("equal")
        # Plot X-ray sources
        ax.scatter(
            source_pos_arr[..., 0],
            source_pos_arr[..., 1],
            source_pos_arr[..., 2],
            marker="x",
            label="X-ray source",
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
    projs.create_buffer(dtype=proj_dtype)
    print("Generating the linearised projection data... ", end="", flush=True)
    t1 = time.time()
    out = cbfp.project(  # ruff: ignore[unused-variable]
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
        proj_array = out_noised_attenuation_capped.astype(proj_dtype)
    else:
        proj_array = projs.data
    if show_plots_before_recon:
        snookerr.imview.plot_projections(
            proj_array.reshape(-1, *proj_array.shape[-2:]),
            title="max attenuation capped for plotting purposes",
        )

    # Write the projections to disk
    if output_projs_path is not None:
        np.save(
            output_projs_path, proj_array.reshape(-1, *proj_array.shape[-2:])
        )

    if numits == 0:
        return

    # Free up memory by deleting various unneeded variables
    del vol, projs
    if apply_noise:
        del out_noised_attenuation, out_noised_transmission, out_

    if reconstruct:
        # Set up the working volume
        vol_shape: tuple[int, int, int] = (
            int(volwid_mm // voxwid_mm),
            int(volwid_mm // voxwid_mm),
            int((volwid_mm // voxwid_mm) * 2 / 4),
        )
        print("Working volume shape: ", vol_shape)
        working_vol = snookerr.Volume.zeros(
            vol_shape, voxwid=1, dtype=vol_dtype
        )

        # Backproject into the working volume
        config_backproject = cbfp.ConfigBackproject(
            fast_approx=fast_approx_backprojection,
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
        print(
            f"Backrojection time (ms): {elapsed_milliseconds:.1f}", flush=True
        )
        # snookerr.imview.plot_volume(working_vol.as_numpy())

        # Make a SIRT object
        working_vol = snookerr.Volume.zeros(
            vol_shape, voxwid=1, dtype=vol_dtype
        )
        working_vol.fill(0)
        recon = snookerr.iterators.SIRT_flatpanel(
            config=snookerr.iterators.SIRT_flatpanel.Config(
                fast_backproject=fast_approx_backprojection,
                soft_from_cylinder_relrad=0.95,
            ),
            volume=working_vol,
            projs=projs,
        )

        # Set up live plot
        if live_plot:
            _lp_process, lp_dispatcher = liveplot.start_live_plot_2d(
                title="Live plot -- projection along Y axis",
                histogram_label="Line Attenuation",
                plotmin=default_plot_range_line_attenuation[0],
                plotmax=default_plot_range_line_attenuation[1],
                # plotmin=0.0184 * voxwid_mm * (1 + default_plot_range_line_attenuationdefault_plot_range_HU[0] / 1000),
                # plotmax=0.0184 * voxwid_mm * (1 + default_plot_range_HU[1] / 1000),
                show=False,
            )

        # Run the recon
        liveplot_arr = cp.empty(
            (working_vol.shape[0], working_vol.shape[2]), dtype=vol_dtype
        )
        for i in range(numits + 1):
            s = f"Iteration {i:04d}/{numits:04d}"
            print(s + "   -- UNIX time: ", time.time(), flush=True)
            recon.iterate()
            # Live plot update
            if live_plot:
                recon.volume.sum_to_cupy_array(axis=1, out=liveplot_arr)
                a = liveplot_arr.get()
                lp_dispatcher.update_image(a[:, :, np.newaxis])
                if i == 0:
                    lp_dispatcher.set_visible(True)
        v = recon.volume.as_numpy()
        if save_volume_path is not None:
            np.save(save_volume_path, v)
        if False:
            snookerr.imview.plot_volume(
                v.transpose(0, 2, 1),
                title=s + " -- view of y slices",
            )
            snookerr.imview.plot_volume(
                v,
                title=s + " -- view of z slices",
            )


if __name__ == "__main__":
    main()
