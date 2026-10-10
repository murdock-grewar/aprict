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

import sys
import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

import aprict
from aprict import liveplot
from aprict.cb import flatpanel as cbfp


def main() -> None:  # ruff: ignore[undocumented-public-function]
    cbfp.force_compile()  # force the relevant CUDA kernels to compile first, so that the compilation time isn't included in the timings

    ### Configurable parameters ###
    live_plot: bool = True
    default_plot_range_HU: list[int | float] = [0, 100]  # ruff: ignore[non-lowercase-variable-in-function]
    voxwid_mm: float = 0.5  # the voxel width of the head phantom
    scale_phantom_size: float = (
        # 2*4/5  # make the phantom bigger, without reducing its attenuation coefficient
        1
    )
    apply_noise: bool = False
    R_mm: float = 1000  # ruff: ignore[non-lowercase-variable-in-function]
    L_mm: float = R_mm + 200  # ruff: ignore[non-lowercase-variable-in-function]
    num_sources: int = 1000
    total_clearfield_photons: float = 1e14  # total number of clearfield photons in the experiment, to be distributed evenly among every detector pixel of every projection.
    detwid_mm: float = 350  # The reconstruction is not good when the reconstruction support is smaller than the masked volume region... there's a dark ring that appears when setting detwid_mm = 500, and the exterior of the reconstruction is then poor. Looks much better with detwid_mm = 700. I need to look into this. It might be improved by softening the edges of the detector when backprojecting.
    dethei_mm: float = 350
    pixel_wid_mm: float = 0.25
    fast_approx_backprojection: bool = True
    numits: int = 100 if len(sys.argv) < 2 else int(sys.argv[1])
    vol_dtype = np.float32  # either np.float16 or np.float32
    proj_dtype = np.float32  # either np.float16 or np.float32 or np.float64. np.float16 seems to not have enough precision
    ###############################

    # Derived quantities
    det_numpixwid: int = int(detwid_mm / pixel_wid_mm)
    det_numpixhei: int = int(dethei_mm / pixel_wid_mm)
    photons_per_pixel: float = (
        total_clearfield_photons / det_numpixwid / det_numpixhei / num_sources
    )

    # Print some information about the geometry
    print("Distance from center to detector (mm): ", L_mm)
    print("Detector width  (mm):     ", detwid_mm)
    print("Detector height (mm):     ", dethei_mm)
    print("Detector width  (px):     ", det_numpixwid)
    print("Detector height (px):     ", det_numpixhei)

    # Generate projection specifications
    # cbfp.create_proj_spec_array(
    #   source_pos_arr,
    #   det_pos_arr,
    #   det_uvec_arr,
    #   det_vvec_arr,
    #   detpixwid,
    #   detpixhei,
    # )
    # zs = np.linspace(-300, 300, endpoint=True, num=num_sources, dtype=np.float32)
    zs = np.linspace(
        -250, 250, endpoint=True, num=num_sources, dtype=np.float32
    )
    ts = 0.618 * 2 * np.pi * np.arange(num_sources, dtype=np.float32)
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
    pg = cbfp.ProjGeom(
        source_pos=source_pos_arr / voxwid_mm,
        det_pos=det_pos_arr / voxwid_mm,
        det_uvec=det_uvec_arr / voxwid_mm,
        det_vvec=det_vvec_arr / voxwid_mm,
        nu=det_numpixwid,
        nv=det_numpixhei,
    )
    print("Number of fan-beam projections: ", pg.num_projs)

    # Create the volume to be projected
    print("Voxel width (mm):        ", voxwid_mm)
    vol: aprict.FBVolume = aprict.phantoms.FORBILD.HeadPhantom
    vol.scale_len(1 / voxwid_mm)
    volwid_mm = 275  # in the future, fetch from the phantom
    # Scale up the volume
    vol.scale_len(scale_phantom_size)
    volwid_mm *= scale_phantom_size

    ### Do plotting of the imaging geometry
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
        proj_array = out_noised_attenuation_capped
    else:
        proj_array = projs.data
    # aprict.imview.plot_projections(
    #     proj_array.reshape(-1, *proj_array.shape[-2:]),
    #     title="max attenuation capped for plotting purposes",
    # )

    # Free up memory by deleting various unneeded variables
    del vol, projs
    if apply_noise:
        out_noised_attenuation, out_noised_transmission, out_

    # Set up the working volume
    vol_shape: tuple[int, int, int] = tuple(
        int(volwid_mm // voxwid_mm) for _ in range(3)
    )  # ty:ignore[invalid-assignment]  # ty is bugged here
    print("Working volume shape: ", vol_shape)
    working_vol = aprict.Volume.zeros(vol_shape, voxwid=1, dtype=vol_dtype)  # ty:ignore[invalid-argument-type]  # ty is bugged

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
    print(f"Backrojection time (ms): {elapsed_milliseconds:.1f}", flush=True)
    # aprict.imview.plot_volume(working_vol.as_numpy())

    # Make a SIRT object
    working_vol = aprict.Volume.zeros(vol_shape, voxwid=1, dtype=vol_dtype)  # ty:ignore[invalid-argument-type]
    working_vol.fill(0)
    recon = aprict.iterators.SIRT_flatpanel(
        config=aprict.iterators.SIRT_flatpanel.Config(
            fast_backproject=fast_approx_backprojection,
            soft_from_cylinder_relrad=0.95,
        ),
        volume=working_vol,
        projs=projs,
    )

    # Set up live plot
    if live_plot:
        _lp_process, lp_dispatcher = liveplot.start_live_plot_2d(
            title="Live plot",
            histogram_label="Attenuation coefficient",
            plotmin=0.0184 * voxwid_mm * (1 + default_plot_range_HU[0] / 1000),
            plotmax=0.0184 * voxwid_mm * (1 + default_plot_range_HU[1] / 1000),
            show=False,
        )

    # Run the recon
    liveplot_sl = slice(
        working_vol.shape[2] // 2, working_vol.shape[2] // 2 + 1
    )
    liveplot_arr = cp.empty(
        (working_vol.shape[0], working_vol.shape[1], 1), dtype=vol_dtype
    )
    for i in range(numits + 1):
        s = f"Iteration {i:04d}/{numits:04d}"
        print(s + "   -- UNIX time: ", time.time(), flush=True)
        recon.iterate()
        # Live plot update
        if live_plot:
            recon.volume.as_cupy_slice(
                axis=2, sl=liveplot_sl, out=liveplot_arr
            )
            lp_dispatcher.update_image(liveplot_arr.get())
            if i == 0:
                lp_dispatcher.set_visible(True)
    aprict.imview.plot_volume(recon.volume.as_numpy(), title=s)


if __name__ == "__main__":
    main()
