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

"""An example script to generate flat-panel fan-beam transmission data from a 2D phantom."""

import argparse
import pathlib
import sys
import time

import numexpr as ne
import numpy as np

import snookerr
from snookerr import imview, lib
from snookerr.cb import flatpanel as cbfp
from snookerr.logtools import ProgressBar


def parse_args() -> argparse.Namespace:
    """Parse commandline args using argparse."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("num_sources", type=int)
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("./projection_artefacts"),
        help="The output path in which to write the artefact directory.",
    )
    parser.add_argument(
        "--R",
        type=float,
        default=20,
        help="Distance from source to isocenter.",
    )
    parser.add_argument(
        "--L-minus-R",
        type=float,
        default=20,
        help="Distance from isocenter to detector.",
    )
    parser.add_argument(
        "--pixwid-fovc",
        type=float,
        default=0.03,
        help="The pixel width at isocenter. The physical detector width will be this times L/R.",
    )
    parser.add_argument(
        "--detwid", type=float, default=35, help="Width of the detector."
    )
    parser.add_argument(
        "--isocenter",
        type=lib.argparse_tools.FloatTupleArgTypeFactory(numfloats=3),
        default=(0.0, 0.0, 0.0),
        help="Position of isocenter.",
    )
    parser.add_argument(
        "--ss",
        type=int,
        default=1,
        help="The number of sub-line-integrals to sample for each pixel. Increases projection accuracy; this becomes relevant when low-resolution detector data is used to reconstruct features on the order of the pixel-at-isocenter size, or smaller.",
    )
    parser.add_argument(
        "--theta-start",
        type=lib.argparse_tools.MaybeType(float),
        default=None,
        help="Starting angle of the source, in radians.",
    )
    parser.add_argument(
        "--arc-length",
        type=lib.argparse_tools.MaybeType(float),
        default=2 * np.pi,
        help="Length of circular arc, in radians. Defaults to a full circle.",
    )
    parser.add_argument(
        "--noise-cfc",
        type=lib.argparse_tools.MaybeType(float),
        help="If provided, then the generated data is subject to Poisson noise. The clearfield photon count is equal to the provided value, and is uniform across all pixels.",
    )
    parser.add_argument(
        "--phantom",
        type=lib.argparse_tools.MaybeType(pathlib.Path),
        default=None,
        metavar="./path/to/phantom.pha",
        help="Path to FORBILD file (*.pha) specifying the phantom. If unspecified, then it will default to using the FORBILD head phantom, with distances measured in centimeters.",
    )
    parser.add_argument(
        "--plot",
        type=lib.argparse_tools.BoolArgType,
        default=False,
        help="Supply 'True' to plot the projection data after it's generated.",
    )
    parser.add_argument(
        "--print-info",
        type=lib.argparse_tools.BoolArgType,
        default=False,
        help="Supply 'True' to additionally print some more information on stderr.",
    )
    parser.add_argument(
        "--multiply-attenuation",
        type=float,
        default=1,
        help="Multiply the input phantom's attenuation by this amount. The modified phantom will be saved to the artefact directory.",
    )

    return parser.parse_args()


def make_proj_geom(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, cbfp.ProjGeom, cbfp.ProjGeom]:
    """Take in commandline arguments and output the (supersampled, non-supersampled) projection geometry as a 2-tuple of projection geometries."""
    # Input/Output paths
    path_pg = args.output.joinpath("./projgeom.dict")
    path_pg_ss = args.output.joinpath("./projgeom_ss.dict")

    # Generate projections
    detwid_mm: float = args.detwid
    pixel_wid_mm_ss: float = (
        args.pixwid_fovc / (args.R / (args.L_minus_R + args.R)) / args.ss
    )

    ss: int = args.ss

    # Derived quantities
    det_numpixwid_ss: int = int(np.ceil(detwid_mm / pixel_wid_mm_ss))
    det_numpixwid_ss = ss * int(np.ceil(det_numpixwid_ss / ss))
    det_numpixhei: int = 1

    theta_start = (
        args.theta_start
        if args.theta_start is not None
        else np.random.rand() * 2 * np.pi  # ruff: ignore[numpy-legacy-random]
    )

    pg_ss, pg = [
        cbfp.ProjGeom.new_circular_scan(
            num_source_points=args.num_sources,
            R=args.R,
            L_minus_R=args.L_minus_R,
            pix_widhei_at_isocenter=(
                pixel_wid_mm,
                pixel_wid_mm,
            ),  # the actual pixel height isn't used in generating the projections in this script, due to the implementation details. The supersampling only happens laterally, because it's meant to simulate a true fan-beam geometry.
            num_pix_widhei=(det_numpixwid, det_numpixhei),
            theta_start=theta_start,
            arc_length_rads=args.arc_length,
            isocenter=args.isocenter,
        )
        for (det_numpixwid, pixel_wid_mm) in zip(
            (det_numpixwid_ss, det_numpixwid_ss // ss),
            (pixel_wid_mm_ss, pixel_wid_mm_ss * ss),
        )
    ]

    pg_ss.save(path_pg_ss)
    pg.save(path_pg)

    return args, pg_ss, pg


def make_projection_data(
    args: argparse.Namespace,
    pg_ss: cbfp.ProjGeom,
    pg: cbfp.ProjGeom,
    supply_output_in_ram: bool,
) -> tuple[argparse.Namespace, cbfp.ProjGeom, np.ndarray]:
    """Generate the projection data."""
    # Define paths
    path_fbv_pha = args.output.joinpath("phantom.pha")
    path_fbv_pruned_pha = args.output.joinpath("phantom_pruned.pha")
    path_projs = args.output.joinpath("projs.npy")

    # Load and save phantom
    # v = snookerr.phantoms.FORBILD.RockCanisterAl
    if args.phantom is None:
        v = snookerr.phantoms.FORBILD.HeadPhantom
        # convert distances from millimeters to centimeters
        v.scale_len(1 / 10)
        # convert attenuation coefficient from multiples of water to multiples of 0.184 per cm
        v.multiply_attenuation(0.184)
    else:
        v = snookerr.FBVolume.load(args.phantom)
    v.multiply_attenuation(args.multiply_attenuation)
    with open(path_fbv_pha, mode="w+", encoding="utf-8") as f:
        f.write(v.FORBILD_str)
    v = v.prune_to_bbox(*pg.get_bbox())  # Prune unused shapes in phantom
    with open(path_fbv_pruned_pha, mode="w+", encoding="utf-8") as f:
        f.write(v.FORBILD_str)

    # path_projs_ss = pathlib.Path('./tmp_projs_ss.npy')
    # path_projs_unnoised = pathlib.Path('./tmp_projs_unnoised.npy')
    # Generate projections
    proj_dtype = np.float64  # either np.float16 or np.float32 or np.float64. np.float16 seems to not have enough precision

    num_sources = int(pg_ss.num_projs)
    det_numpixwid = int(pg_ss.nu.flatten()[0])
    det_numpixhei = int(pg_ss.nv.flatten()[0])

    ss: int = int(pg_ss.nu.flatten()[0]) // int(pg.nu.flatten()[0])

    ## Call the projection code on the prepared geometry ##
    chunksize: int = min(10, num_sources)
    config_project = cbfp.ConfigProject()
    output_projs_ss = np.zeros(
        (chunksize, det_numpixwid, det_numpixhei), dtype=proj_dtype
    )
    output_projs = np.lib.format.open_memmap(
        path_projs,
        "w+",
        dtype=proj_dtype,
        shape=(num_sources, det_numpixwid // ss, det_numpixhei),
    )
    output_projs[...] = 0
    mincount: float | None = None
    t1 = time.time()
    with ProgressBar(
        maxval=num_sources, valunit="proj", desc="Projection", persist=True
    ) as pbar:
        for i in range((num_sources + chunksize - 1) // chunksize):
            sl = slice(i * chunksize, min(num_sources, (i + 1) * chunksize))
            pg_sl = pg_ss.sliced(sl)
            projs = cbfp.ProjectionSet(pg_sl, data=None)
            projs.create_buffer(dtype=proj_dtype)
            _ = cbfp.project(
                config_project,
                v,
                projs,
                block=True,
            )
            vo = output_projs_ss[0 : (sl.stop - sl.start)]
            assert (
                projs.data is not None
            )  # helpful assertion for type-checking
            vi = projs.data.get()
            ne.evaluate("vi", out=vo, local_dict={"vi": vi})

            # Apply noise, if applicable
            if args.noise_cfc is None:
                vo = output_projs[sl]
                for offidx in range(ss):
                    vi = output_projs_ss[0 : (sl.stop - sl.start), offidx::ss]
                    ne.evaluate(
                        "vi + vo", out=vo, local_dict={"vi": vi, "vo": vo}
                    )
                ne.evaluate("vo/ss", out=vo, local_dict={"vo": vo, "ss": ss})
            else:
                p = output_projs_ss[0 : (sl.stop - sl.start)]
                # print("Maximum attenuation encountered: ", p.max())
                # vo = np.lib.format.open_memmap(path_projs, 'w+', dtype=proj_dtype, shape=(num_sources, det_numpixwid//ss, det_numpixhei))
                counts_ss = np.random.poisson(args.noise_cfc / ss * np.exp(-p))  # ruff: ignore[numpy-legacy-random]

                # Downbin
                counts = np.zeros(
                    (sl.stop - sl.start, det_numpixwid // ss, det_numpixhei),
                    dtype=counts_ss.dtype,
                )
                for offidx in range(ss):
                    vi = counts_ss[:, offidx::ss]
                    ne.evaluate(
                        "vi + counts",
                        out=counts,
                        local_dict={"vi": vi, "counts": counts},
                    )

                mincount = (
                    counts.min()
                    if mincount is None
                    else min(mincount, counts.min())
                )
                vo = output_projs[sl]
                ne.evaluate(
                    "log(cfc/counts)",
                    out=vo,
                    local_dict={"cfc": args.noise_cfc, "counts": counts},
                )

            pbar.update_to(sl.stop)
    t2 = time.time()
    elapsed_milliseconds: float = (t2 - t1) * 1000  # ruff: ignore[unused-variable]
    # print(
    #     f"Done. Projection time (sec): {elapsed_milliseconds/1000:.1f}", flush=True, file=sys.stderr
    # )
    if args.noise_cfc is not None:
        print("Minimum photon count encountered: ", mincount, file=sys.stderr)

    if supply_output_in_ram:
        output = np.zeros_like(output_projs)
        ne.evaluate(
            "output_projs",
            out=output,
            local_dict={"output_projs": output_projs},
        )
    else:
        output = output_projs
    return args, pg, output


def main() -> None:
    """Process commandline input and generate the projection data."""
    args = parse_args()
    # Check if output directory already exists
    if args.output.is_dir():
        msg = f"Output directory '{args.output}' already exists."
        raise FileExistsError(msg)
    args.output.mkdir()

    # print("Cupy version: ", cp.__version__, file=sys.stderr, flush=True)
    t1 = time.time()
    print(
        "Compiling CUDA sources (or fetching cubin from cache)... ",
        end="",
        flush=True,
        file=sys.stderr,
    )
    cbfp.force_compile()  # Compile it now so the compilation time isn't counted in other timings
    t2 = time.time()
    print(f"done. Took {t2 - t1:.3f} seconds.", flush=True, file=sys.stderr)
    _, pg_ss, pg = make_proj_geom(args)
    _, _, projs = make_projection_data(
        args, pg_ss, pg, supply_output_in_ram=args.plot
    )

    if args.print_info:
        print(
            f"Output written to directory '{args.output.resolve()}'.",
            flush=True,
            file=sys.stderr,
        )
        print(file=sys.stderr)
        print("Miscellaneous information:", file=sys.stderr)
        print(f"Number of projections: {projs.shape[0]:d}", file=sys.stderr)
        print(
            f"Pixels in each projection: {projs.shape[1]:d}", file=sys.stderr
        )

    # Plot, if chosen
    if args.plot:
        assert projs is not None
        imview.plot_projections(projs.transpose((2, 1, 0)))


if __name__ == "__main__":
    main()
