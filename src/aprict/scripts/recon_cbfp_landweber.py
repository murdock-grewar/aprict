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

"""Script to perform Landweber reconstruction from flat-panel cone-beam transmission data."""

import argparse
import logging
import os
import pathlib

import cupy as cp
import cupyx
import numexpr as ne
import numpy as np

# Try to load some modules
try:
    from .. import image_view as imview

    mod_avail_imview: bool = True
except ModuleNotFoundError:
    mod_avail_imview: bool = False
try:
    from .. import liveplot  # ruff: ignore[unused-import]

    mod_avail_liveplot: bool = True
except ModuleNotFoundError:
    mod_avail_liveplot: bool = False

from .. import Volume, lib, logtools
from ..cb import flatpanel as cbfp
from ..iterators import Landweber_flatpanel

_progdesc = "Perform an iterative Landweber reconstruction from flat-panel cone-beam transmission data."


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_progdesc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "projgeom",
        metavar="proj_geom.json",
        type=pathlib.Path,
        help="Path to saved ProjGeom object.",
    )
    parser.add_argument(
        "projdata",
        metavar="proj_data.npy",
        type=pathlib.Path,
        help="Path to the numpy array containing the transmission data corresponding to the ProjGeom object.",
    )

    group_mandatory_options = parser.add_argument_group(
        title="required options"
    )
    group_mandatory_options.add_argument(
        "--voxwid",
        metavar="1.0",
        type=float,
        required=True,
        help="The cubic voxel width, in the same distance units as the projection data is specified in proj_geom.json",
        default=argparse.SUPPRESS,
    )
    group_mandatory_options.add_argument(
        "--volshape",
        metavar="'(nX,nY,nZ)'",
        type=lib.argparse_tools.IntTupleArgTypeFactory(numints=3),
        help="Dimensions of the voxel grid on which to reconstruct",
        required=True,
        default=argparse.SUPPRESS,
    )

    group_geometry_options = parser.add_argument_group(
        title="geometry options"
    )
    group_geometry_options.add_argument(
        "--volpos",
        metavar="'(0.,0.,0.)'",
        type=lib.argparse_tools.FloatTupleArgTypeFactory(numfloats=3),
        help="Position of the middle of the volume.",
        required=False,
        default=(0.0, 0.0, 0.0),
    )

    group_data_preprocessing = parser.add_argument_group(
        title="data preprocessing"
    )
    group_data_preprocessing.add_argument(
        "--autorange-z",
        metavar="False",
        type=lib.argparse_tools.BoolArgType,
        default=False,
        help="Autorange the z coordinates of the trajectory, so that the highest source point and the lowest source point are +-z0. This autoranging takes place before slicing of projection data.",
    )
    group_data_preprocessing.add_argument(
        "--slice-projs",
        metavar="'(start, stop)'",
        type=lib.argparse_tools.ContiguousSliceArgType,
        default=slice(None, None),
        help="Slice the projections to include those of index start to (stop-1), inclusive.",
    )
    # group_data_preprocessing.add_argument('--rot-xy', metavar="NUMRADIANS", type=lib.argparse_tools.MaybeType(float), default=None, help="Rotate the projection geometry about the z axis by this many radians.")
    group_data_preprocessing.add_argument(
        "--flip-x",
        metavar="False",
        type=lib.argparse_tools.BoolArgType,
        default=False,
        help="Flip the projection geometry about the xz plane.",
    )
    group_data_preprocessing.add_argument(
        "--flip-y",
        metavar="False",
        type=lib.argparse_tools.BoolArgType,
        default=False,
        help="Flip the projection geometry about the yz plane.",
    )
    group_data_preprocessing.add_argument(
        "--flip-z",
        metavar="False",
        type=lib.argparse_tools.BoolArgType,
        default=False,
        help="Flip the projection geometry about the xy plane.",
    )

    group_landweber_options = parser.add_argument_group(
        title="landweber options"
    )
    group_landweber_options.add_argument(
        "--numits",
        metavar="10",
        type=lib.argparse_tools.MaybeType(int),
        default=None,
        required=False,
        help="The number of iterations to use. None = iterate indefinitely, until a SIGINT is sent.",
    )
    group_landweber_options.add_argument(
        "--relaxation-factor",
        "--rf",
        metavar="1.0",
        type=float,
        required=False,
        default=1.0,
        help="The relaxation factor to add to the correction terms. Should be in the open interval (0,2)",
    )
    group_landweber_options.add_argument(
        "--numits-for-rayleigh-quotient",
        "--rqits",
        metavar="10",
        type=int,
        required=False,
        default=10,
        help="Number of iterations to spend on computing the Rayleigh quotient",
    )
    group_landweber_options.add_argument(
        "--os-size",
        metavar="42",
        type=lib.argparse_tools.MaybeType(int),
        required=False,
        default=None,
        help="Split the projection dataset into ordered subsets of this many cone-beam projections each. (Be careful: if there are leftover projections at the end, these will be lumped into an ordered subset of smaller size than specified.)",
    )
    group_landweber_options.add_argument(
        "--exact-adjoint",
        metavar="True",
        type=lib.argparse_tools.BoolArgType,
        default=True,
        help="Whether to use the exact adjoint. (If False, then a much faster approximate adjoint is used.)",
    )

    group_cyl_options = parser.add_argument_group(
        title="cylindrical trajectory options",
        description="These options are specific to the cylindrical trajectory. They will not make sense as options for arbitrary geometries.",
    )
    group_cyl_options.add_argument(
        "--cyl-confine-proj-lines-z",
        metavar="False",
        type=lib.argparse_tools.BoolArgType,
        default=False,
        help="A special option to mask out projection lines whose intersection with the solid cylinder is not completely contained in the reconstruction volume. This option assumes that the reconstruction volume has x,y = 0 at its centre.",
        required=False,
    )

    group_io_options = parser.add_argument_group(title="I/O options")
    group_io_options.add_argument(
        "--output",
        metavar="path/to/new/dir/",
        type=lib.argparse_tools.MaybeType(pathlib.Path),
        default=None,
        help="Path to a new directory to create and place output artefacts and log files.",
    )
    group_io_options.add_argument(
        "--verbosity",
        metavar="INFO",
        type=logtools.LogLevel,
        action=lib.argparse_tools.EnumAction,
        default=logtools.LogLevel.INFO,
        help="The verbosity of output logging to stderr. Log files will not be affected by this setting.",
    )
    group_io_options.add_argument(
        "--output-intermediates",
        metavar="False",
        type=lib.argparse_tools.BoolArgType,
        default=False,
        help="Whether to output intermediate reconstructions from each iteration.",
        required=False,
    )
    group_io_options.add_argument(
        "--plotmin",
        "--pmin",
        metavar="0.0",
        type=lib.argparse_tools.MaybeType(float),
        default=None,
        required=False,
        help="The minimum for the plotting range. Set to None to select minimum value in each plot.",
    )
    group_io_options.add_argument(
        "--plotmax",
        "--pmax",
        metavar="1.0",
        type=lib.argparse_tools.MaybeType(float),
        default=None,
        required=False,
        help="The maximum for the plotting range. Set to None to select maximum value in each plot.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate a set of args."""
    # Ensure that the output directory doesn't already exist
    if args.output is not None and args.output.exists():
        msg = f"Output path '{args.output!s}' already exists."
        raise ValueError(msg)

    # If outputting intermediates, then the user must also specify the --output directory
    if args.output_intermediates and not args.output:
        msg = "Cannot output intermediate tomograms when no output directory was specified. Specify an output directory with --output path/to/new/dir"
        raise ValueError(msg)


def get_ordered_subsets_and_masks(
    projdata_path: pathlib.Path,
    pg: cbfp.ProjGeom,
    *,
    slice_projs: slice[int | None, int | None, None] | None,
    autorange_z: bool,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
    os_size: int | None,
    cyl_confine_proj_lines_z_zminmax: tuple[float, float] | None,
) -> tuple[list[cbfp.ProjectionSet], list[cp.ndarray] | None]:
    """Process input data and output something which can be given to a Landweber recon object."""
    if autorange_z:
        pg = pg.translated(
            x=0,
            y=0,
            z=-(pg.source_pos[:, 2].max() + pg.source_pos[:, 2].min()) / 2,
        )
    if slice_projs is not None:
        pg = pg.sliced(slice_projs)
    if flip_x or flip_y or flip_z:
        pg = pg.flipped(flip_x=flip_x, flip_y=flip_y, flip_z=flip_z)
    data_mm: np.memmap = np.lib.format.open_memmap(
        projdata_path, mode="r"
    )[
        slice_projs
    ]  # Note: a contiguous slice is still a memmap object  # ty:ignore[invalid-assignment]
    ps_lst: list[cbfp.ProjectionSet] = []
    os_size = (
        os_size or pg.num_projs
    )  # Null-coalesce 'None' to be pg.num_projs
    sls: list[slice] = []
    sl_start: int = 0
    while sl_start < pg.num_projs:
        sl = slice(sl_start, sl_start + os_size)
        sls.append(sl)
        ps_lst.append(cbfp.ProjectionSet(pg.sliced(sl), data=data_mm[sl]))
        sl_start += os_size

    # Make masks to zero out projection lines which do not intersect the cylinder twice
    masks = None
    if cyl_confine_proj_lines_z_zminmax is not None:
        masks = []
        # Step 1: compute the lateral chord length based on lateral position of detector pixel
        # Step 2: compute the dz from source to intersection with cylinder, as a 2D array
        # Step 3: compute mask as min_z <= (source_z + dz_2D) <= max_z
        unit = np.float64(1)
        pixel_wid = np.mean(
            np.sqrt(
                ne.evaluate(
                    "sum(unit*vec*vec, 1)",
                    local_dict={"vec": pg.det_uvec, "unit": unit},
                )
            )
        )
        pixel_hei = np.mean(
            np.sqrt(
                ne.evaluate(
                    "sum(unit*vec*vec, 1)",
                    local_dict={"vec": pg.det_vvec, "unit": unit},
                )
            )
        )
        det_numpixwid = pg.nu[0]
        det_numpixhei = pg.nv[0]
        R = np.mean(  # ruff: ignore[non-lowercase-variable-in-function]
            np.sqrt(
                ne.evaluate(
                    "sum(unit*vec*vec, 1)",
                    local_dict={"vec": pg.source_pos, "unit": unit},
                )
            )
        )
        L = np.mean(  # ruff: ignore[non-lowercase-variable-in-function]
            np.sqrt(
                ne.evaluate(
                    "sum(unit*vec*vec, 1)",
                    local_dict={
                        "vec": pg.source_pos - pg.det_pos,
                        "unit": unit,
                    },
                )
            )
        )
        det_y = pixel_wid * (
            np.arange(det_numpixwid, dtype=np.float32)
            - ((det_numpixwid - 1) / 2)
        )
        det_z = pixel_hei * (
            np.arange(det_numpixhei, dtype=np.float32)
            - ((det_numpixhei - 1) / 2)
        )
        cl = 2 * R / np.sqrt(1 + np.square(det_y / L))
        lateral_length_source_to_detector = np.sqrt(
            np.square(L) + np.square(det_y)
        )
        dz = ne.evaluate(
            "a*b",
            local_dict={
                "a": (cl / lateral_length_source_to_detector)[:, np.newaxis],
                "b": det_z[np.newaxis, :],
            },
        )
        zs = pg.source_pos[..., 2]
        zs_mask_min, zs_mask_max = cyl_confine_proj_lines_z_zminmax
        for sl in sls:
            z_ = zs[sl][:, np.newaxis, np.newaxis]
            z_of_cyl_intersection = ne.evaluate(
                "a+b",
                local_dict={
                    "a": z_,
                    "b": dz[np.newaxis, :, :],
                },
            )
            mask = cupyx.empty_pinned(
                z_of_cyl_intersection.shape, dtype=np.bool
            )
            ne.evaluate(
                "where(zmin <= zi, True, False) & where(zi <= zmax, True, False) & where(zmin <= z_, True, False) & where(z_ <= zmax, True, False)",
                out=mask,
                local_dict={
                    "zi": z_of_cyl_intersection,
                    "z_": z_,
                    "zmin": zs_mask_min,
                    "zmax": zs_mask_max,
                },
            )
            mask_cp = cp.array(mask)
            masks.append(mask_cp)
    return ps_lst, masks


def main() -> None:
    """Run the script."""
    # Parse the input arguments
    parser = _make_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    ##########################################################
    ################ Create the output path ##################
    ##########################################################
    artefact_path: pathlib.Path | None = args.output
    artefact_path and artefact_path.mkdir()
    path_output: pathlib.Path | None = (
        artefact_path and artefact_path.joinpath("output")
    )
    path_output and path_output.mkdir()
    path_intms: pathlib.Path | None = (
        artefact_path
        and args.output_intermediates
        and artefact_path.joinpath("intermediates")
    )
    path_intms and path_intms.mkdir()
    path_intms_tomos: pathlib.Path | None = path_intms and path_intms.joinpath(
        "tomograms"
    )
    path_intms_tomos and path_intms_tomos.mkdir()
    path_intms_slx: pathlib.Path | None = path_intms and path_intms.joinpath(
        "slices_x"
    )
    path_intms_slx and path_intms_slx.mkdir()
    path_intms_sly: pathlib.Path | None = path_intms and path_intms.joinpath(
        "slices_y"
    )
    path_intms_sly and path_intms_sly.mkdir()
    path_intms_slz: pathlib.Path | None = path_intms and path_intms.joinpath(
        "slices_z"
    )
    path_intms_slz and path_intms_slz.mkdir()

    ##########################################################
    ##################### Logging setup ######################
    ##########################################################
    logging.root.setLevel(
        logging.NOTSET
    )  # Prevent root logger from squashing child loggers from outputting
    logger = logtools.make_logger(
        path=artefact_path, stderr_loglevel=args.verbosity
    )

    ##########################################################
    ##### Set numexpr to use as many threads as possible #####
    ##########################################################
    # Set numexpr to have the appropriate number of threads
    # This should have already worked at import time, but it doesn't for some reason
    preferred_thread_count: int = int(2 * os.cpu_count())  # ty:ignore[unsupported-operator]
    nthreads: int = int(
        os.environ.get("NUMEXPR_MAX_THREADS", default=None)
        or preferred_thread_count
    )
    logger.debug(f"Setting numexpr num_threads to {nthreads:d}")
    ne.set_num_threads(nthreads)

    ##########################################################
    ##### Prepare the input data for the Landweber recon #####
    ##########################################################
    # Create the ProjectionSet objects corresponding to each ordered subset
    # pg: cbfp.ProjGeom = cbfp.ProjGeom.load(args.projgeom)
    ps_lst, masks = get_ordered_subsets_and_masks(
        projdata_path=args.projdata,
        pg=cbfp.ProjGeom.load(args.projgeom),
        slice_projs=args.slice_projs,
        autorange_z=args.autorange_z,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        flip_z=args.flip_z,
        os_size=args.os_size,
        cyl_confine_proj_lines_z_zminmax=(
            None
            if args.cyl_confine_proj_lines_z is None
            else (
                -args.volshape[2] / 2 * args.voxwid + args.volpos[2],
                args.volshape[2] / 2 * args.voxwid + args.volpos[2],
            )
        ),
    )

    # Create the working volume
    vol = Volume.zeros(
        args.volshape,
        voxwid=args.voxwid,
        dtype=np.float32,  # ty:ignore[invalid-argument-type]
        pos_c=args.volpos,
    )

    ##########################################################
    ######### Create the Landweber iteration object ##########
    ##########################################################
    recon_config = Landweber_flatpanel.Config(
        rayleigh_quotient_num_iterations=args.numits_for_rayleigh_quotient,
        acceleration_factor=args.relaxation_factor,
        fast_backproject=not args.exact_adjoint,
        soft_from_cylinder_relrad=1,  # Don't soften
        bp_soft_from_detedge_relrad=1,  # Don't soften
    )
    logger.info("Initializing Landweber iterator...")
    recon = Landweber_flatpanel(
        config=recon_config,
        logger=logger,
        volume=vol,
        projs=ps_lst,
        proj_zero_masks=masks,
    )

    ##########################################################
    ################# Define helper methods ##################
    ##########################################################
    def save_current_volume(id_str: str, is_intermediate: bool) -> None:
        # Do nothing if the output path is None.
        if path_output is None:
            return

        # ty is not smart enough to understand that path_intms can be None only when path_output is None
        assert path_intms is not None

        # Get the current volume
        x = recon.volume.as_numpy()

        # Get the 'root' directory of where this volume will be saved
        save_path = path_intms if is_intermediate else path_output

        # Save the volume as a .npy file
        tomo_path = (
            save_path.joinpath("tomograms") if is_intermediate else save_path
        )
        tomo_path.mkdir(exist_ok=True)
        lib.ne_tools.save(tomo_path.joinpath(id_str + ".npy"), x, flush=True)

        # Save midslices from volume as .png files
        # The method used here can be a bit slow, because I get a full numpy array and process that.
        # I've decided that I prefer the simplicity and robustness of this approach, instead of trying
        # to squeeze optimal performance by directly extracting slices from cupy arrays.
        for axis, axis_str in enumerate("xyz"):
            if is_intermediate:
                save_path_sl = save_path.joinpath("slices_" + axis_str)
                save_path_sl.mkdir(exist_ok=True)
            else:
                save_path_sl = save_path
            imview.save_volslice(
                save_path_sl.joinpath(
                    "midslice_" + axis_str + "_" + id_str + ".png"
                ),
                x,
                axis=axis,
                index_of_slice=x.shape[axis] // 2,
                pmin=args.plotmin,
                pmax=args.plotmax,
            )

    ##########################################################
    ################# Begin reconstruction ###################
    ##########################################################
    numits: int = args.numits or 1
    logger.info("Beginning iterations...")
    with logtools.ProgressBar(
        desc="Landweber iteration:",
        maxval=numits,
        valunit="iteration",
        logger=logger,
    ) as pbar:
        for itnum in range(1, 1 + numits):
            # Perform a Landweber iteration
            recon.iterate()

            # Save intermediate results
            if path_intms and itnum < numits:
                save_current_volume(
                    id_str=f"itnum_{itnum:05d}", is_intermediate=True
                )

            # Perform logging updates
            pbar.update_to(itnum)
            logger.info(f"Completed iteration {itnum:04d}/{numits:04d}")

    # Save final output artefacts
    if path_output:
        save_current_volume(id_str="recon", is_intermediate=False)

    # Perform logging updates
    logger.info("Done.")
