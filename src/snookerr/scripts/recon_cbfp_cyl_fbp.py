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

"""Script to perform FBP reconstruction from cylindrical flat-panel cone-beam transmission data."""

import argparse
import logging
import pathlib
import typing

import cupy as cp
import numpy as np

# Try to load some modules
try:
    from .. import image_view as imview

    _mod_avail_imview = True
except ModuleNotFoundError:
    _mod_avail_imview = False
try:
    from .. import liveplot  # noqa: F401

    _mod_avail_liveplot = True
except ModuleNotFoundError:
    _mod_avail_liveplot = False

from .. import Volume, cb, lib, logtools
from ..cb import flatpanel as cbfp

_progdesc = "Perform an FBP reconstruction from cylindrical flat-panel cone-beam transmission data."


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

    group_fbp_options = parser.add_argument_group(title="FBP options")
    # group_fbp_options.add_argument(
    #     "--numits",
    #     metavar="10",
    #     type=lib.argparse_tools.MaybeType(int),
    #     default=None,
    #     required=False,
    #     help="The number of iterations to use. None = iterate indefinitely, until a SIGINT is sent.",
    # )
    group_fbp_options.add_argument(
        "--fan-beam",
        metavar="False",
        type=lib.argparse_tools.BoolArgType,
        default=False,
        required=False,
        help="Set to True to activate fan-beam reconstruction mode. This may avoid some other settings to make it work on 2D circular fan-beam data.",
    )
    group_fbp_options.add_argument(
        "--two-sided",
        metavar="True",
        type=lib.argparse_tools.BoolArgType,
        default=True,
        required=False,
        help="When True, this uses a combination of the ramp filter and Hilbert transform to correct for angular undersampling.",
    )
    group_fbp_options.add_argument(
        "--limit-zeta",
        metavar="1.23",
        type=lib.argparse_tools.MaybeType(float),
        default=None,
        required=False,
        help="Set this to impose a maximum value for zeta. This will limit the data used by this method.",
    )
    group_fbp_options.add_argument(
        "--num-rot-angs",
        metavar="None",
        type=lib.argparse_tools.MaybeType(int),
        default=None,
        required=False,
        help="How many filtration directions to sample. Set to None to use a random single direction for each CB projection.",
    )
    group_fbp_options.add_argument(
        "--interpolate-measurements",
        metavar="True",
        type=lib.argparse_tools.BoolArgType,
        default=True,
        required=False,
        help="Whether to apply bilinear interpolation between measurement pixels, when backprojecting.",
    )
    group_fbp_options.add_argument(
        "--superres",
        metavar="1",
        type=int,
        default=1,
        required=False,
        help="Reconstruct with a voxel size this many times smaller, then downsample.",
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

    group_resource_options = parser.add_argument_group(
        title="resource options"
    )
    group_resource_options.add_argument(
        "--ne-threads",
        type=lib.ne_tools.NETHREADS,
        action=lib.argparse_tools.EnumAction,
        default=lib.ne_tools.NETHREADS.ALL_LOGICAL_CORES,
        required=False,
        help="Set the number of threads to be used by numexpr. This will attempt to override the env NUMEXPR_MAX_THREADS as necessary.",
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


def get_filtered_projectionset_generator(  # noqa: C901
    projdata_path: pathlib.Path,
    pg: cbfp.ProjGeom,
    num_rot_angs: int
    | None,  # set to None to use a single random angle per CB projection
    *,
    fan_beam_mode: bool,
    limit_zeta: float | None,
    slice_projs: slice[int | None, int | None, None] | None,
    autorange_z: bool,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
    num_projs_per_chunk: int = 9,
    filt_chunk_size: int = 3,
    logger: logging.Logger,
    two_sided: bool,
) -> tuple[
    cbfp.ProjGeom,
    typing.Iterable[
        tuple[slice[int, int, None], tuple[cbfp.ProjectionSet, ...]]
    ],
    float,
    float,
    float,
]:
    """Process input data and output something which can be given to an FBP recon object."""
    if fan_beam_mode:
        num_rot_angs = 1
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

    # Extract the needed information from the projection geometry
    if len(pg.projs_list_shape) != 1:
        msg = f"Input projection geometry has a non-flat shape: {pg.projs_list_shape}. This isn't supported by this script."
        raise NotImplementedError(msg)
    if not np.all(pg.nu == pg.nu[0]):
        msg = "Inconsistent horizontal resolution of detector."
        raise NotImplementedError(msg)
    nu = pg.nu[0]
    if not np.all(pg.nv == pg.nv[0]):
        msg = "Inconsistent vertical resolution of detector."
        raise NotImplementedError(msg)
    nv = pg.nv[0]
    pixwids = np.linalg.norm(pg.det_uvec, axis=1)
    if not np.all(np.isclose(pixwids, pixwids[0])):
        msg = "Inconsistent pixel widths of detector."
        raise NotImplementedError(msg)
    pixwid = pixwids[0]
    pixheis = np.linalg.norm(pg.det_vvec, axis=1)
    if not np.all(np.isclose(pixheis, pixheis[0])):
        msg = "Inconsistent pixel heights of detector."
        raise NotImplementedError(msg)
    pixhei = pixheis[0]
    if not np.isclose(pixwid, pixhei):
        msg = "Non-square pixels of detector."
        raise NotImplementedError(msg)
    Rs = np.sqrt(  # noqa: N806
        np.square(pg.source_pos[:, 0]) + np.square(pg.source_pos[:, 1])
    )
    if not np.all(np.isclose(Rs, Rs[0])):
        msg = "Inconsistent source radius."
        raise NotImplementedError(msg)
    R = Rs[0]  # noqa: N806
    Ls = Rs + np.sqrt(  # noqa: N806
        np.square(pg.det_pos[:, 0]) + np.square(pg.det_pos[:, 1])
    )
    if not np.all(np.isclose(Ls, Ls[0])):
        msg = "Inconsistent source-to-detector length."
        raise NotImplementedError(msg)
    L = Ls[0]  # noqa: N806
    logger.info(f"R: {R!s:s}")
    logger.info(f"L: {L!s:s}")

    # Set up the filtering object
    filt = cb.FBPFilterer(
        proj_2d_shape=(nu, nv),
        projs_per_batch=filt_chunk_size,
        L=L,
        physpixwid=pixwid,
        limit_zeta=limit_zeta,
        include_hilbert_transform_in_output=two_sided,
        _hacky_fan_beam_mode=fan_beam_mode,
    )

    zs_max = pg.source_pos[:, 2].max()
    zs_min = pg.source_pos[:, 2].min()
    if fan_beam_mode:
        final_multiplier = (
            # 1
            # / (8 * np.square(np.pi) * R)
            # * (R * 2 * np.pi / pg.num_projs)
            1 / (4 * np.pi * R * pg.num_projs)
        )
    else:
        final_multiplier = (
            # R  # cancel the factor of R in the physpixwid of the ramp kernel
            # / (32 * np.float_power(np.pi, 3) * np.float_power(R, 3))
            # * (R * 2 * np.pi * (zs_max - zs_min) / pg.num_projs)
            (zs_max - zs_min) / (16 * np.square(np.pi) * R**2 * pg.num_projs)
        )
    logger.debug(f"Height of source cylinder: {zs_max - zs_min!s:s}")

    def _gen() -> typing.Generator[  # noqa: C901
        tuple[slice[int, int, None], tuple[cbfp.ProjectionSet, ...]]
    ]:
        if num_rot_angs is not None:  # None means a single random angle.
            if num_rot_angs == 1:
                angs = np.array([0.0], dtype=np.float32)
            else:
                angs = np.linspace(
                    -filt.zeta + (filt.zeta / num_rot_angs / 2),
                    filt.zeta - (filt.zeta / num_rot_angs / 2),
                    endpoint=True,
                    num=num_rot_angs,
                )
        sl_start: int = 0
        while sl_start < pg.num_projs:
            sl = slice(
                sl_start, min(pg.num_projs, sl_start + num_projs_per_chunk)
            )
            # Create ProjectionSet with filtered data in buffer
            if num_rot_angs is None:
                with cp.cuda.Stream(non_blocking=False) as stream:
                    filt_arrs, event_completed = filt.filter_bulk_axis1_concat(
                        cp.array(data_mm[sl]),
                        force_rotation_angle=None,
                        block=True,
                    )
                    stream.wait_event(event_completed)
                    assert not fan_beam_mode
                    mult = (
                        2
                        * filt.zeta
                        / np.float_power(np.sin(filt.zeta / 2), 2)
                    )
                    for index in range(len(filt_arrs)):
                        cp.multiply(
                            filt_arrs[index], mult, out=filt_arrs[index]
                        )
                pss = [
                    cbfp.ProjectionSet(pg.sliced(sl), data=filt_arr)
                    for filt_arr in filt_arrs
                ]
            else:
                with cp.cuda.Stream(non_blocking=False) as stream:
                    pss = [
                        cbfp.ProjectionSet(pg.sliced(sl), data=None)
                        for _ in range(2 if two_sided else 1)
                    ]
                    for ps in pss:
                        ps.create_buffer(dtype=cp.float32)
                        assert ps.data is not None  # useful for type-checking
                        ps.data.fill(0)
                    for ang in angs:
                        filt_arrs, event_completed = (
                            filt.filter_bulk_axis1_concat(
                                cp.array(data_mm[sl]),
                                force_rotation_angle=ang,
                                block=True,
                            )
                        )
                        stream.wait_event(event_completed)
                        for index in range(len(filt_arrs)):
                            cp.add(
                                filt_arrs[index],
                                pss[index].data,
                                out=pss[index].data,
                            )
                    if not fan_beam_mode:
                        mult = (
                            2
                            * filt.zeta
                            / np.float_power(np.sin(filt.zeta / 2), 2)
                            / len(angs)
                        )
                        for index in range(len(filt_arrs)):
                            cp.multiply(
                                pss[index].data, mult, out=pss[index].data
                            )
            # Yield
            yield sl, tuple(pss)
            # Increment the slice
            sl_start += num_projs_per_chunk

    return pg, _gen(), R, L, final_multiplier


def main() -> None:  # noqa: C901
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
    # Prevent root logger from squashing child loggers from outputting
    logging.root.setLevel(logging.NOTSET)
    # Create custom logger object
    logger = logtools.make_logger(
        path=artefact_path, stderr_loglevel=args.verbosity
    )

    ##########################################################
    ##### Set numexpr to use as many threads as possible #####
    ##########################################################
    # Set numexpr to have the appropriate number of threads
    lib.ne_tools.try_set_nethreads(args.ne_threads, logger=logger)

    ##########################################################
    ######## Prepare the input data for the FBP recon ########
    ##########################################################
    # Create the ProjectionSet objects corresponding to each ordered subset
    pg, sl_filtps_gen, R, L, final_multiplier = (  # noqa: N806, RUF059
        get_filtered_projectionset_generator(
            projdata_path=args.projdata,
            pg=cbfp.ProjGeom.load(args.projgeom),
            fan_beam_mode=args.fan_beam,
            num_rot_angs=args.num_rot_angs,
            limit_zeta=args.limit_zeta,
            slice_projs=args.slice_projs,
            autorange_z=args.autorange_z,
            flip_x=args.flip_x,
            flip_y=args.flip_y,
            flip_z=args.flip_z,
            logger=logger,
            two_sided=args.two_sided,
        )
    )

    # Create the working volume
    vol = Volume.zeros(
        tuple(n * args.superres for n in args.volshape),
        voxwid=args.voxwid / args.superres,
        dtype=np.float32,  # ty:ignore[invalid-argument-type]
        pos_c=args.volpos,
    )

    # Initialise the configuration used for backprojection
    config_bp_ramp_onesided = cbfp.ConfigBackproject(
        fast_approx=True,
        cyl_mask_rad=vol.shape[0] / 2 * vol.voxwid,
        Z_on_val_factor_power=(-2 if args.fan_beam else -3),
        Z_on_val_factor_val=R,
        backproj_weighting_preset=0,
        approx_mode_direct_raycast=True,
        approx_mode_bilinear_interpolate=args.interpolate_measurements,
    )
    # Configurations to backproject the ramp- and Hilbert-filtered data
    # This two-sided formula is based on a formula by F. Noo.
    config_bp_ramp_twosided = cbfp.ConfigBackproject(
        fast_approx=True,
        cyl_mask_rad=vol.shape[0] / 2 * vol.voxwid,
        Z_on_val_factor_power=(0 if args.fan_beam else -1),
        Z_on_val_factor_val=R,
        backproj_weighting_preset=1,  # 1,#3,
        approx_mode_direct_raycast=True,
        approx_mode_bilinear_interpolate=args.interpolate_measurements,
    )
    config_bp_hilb_twosided = cbfp.ConfigBackproject(
        fast_approx=True,
        cyl_mask_rad=vol.shape[0] / 2 * vol.voxwid,
        Z_on_val_factor_power=(0 if args.fan_beam else -1),
        Z_on_val_factor_val=R,
        backproj_weighting_preset=2,  # 2,#4,
        approx_mode_direct_raycast=True,
        approx_mode_bilinear_interpolate=args.interpolate_measurements,
    )

    ##########################################################
    ########## Initialisation of FBP reconstruction ##########
    ##########################################################
    logger.info(f"superres voxel width: {vol.voxwid!s:s}")
    logger.info(f"superres volume shape: {vol.shape!s:s}")
    logger.info(f"superres factor: {args.superres!s:s}")
    logger.debug(
        f"normalization multiplier on volume after filtered backprojection: {final_multiplier!s:s}"
    )
    logger.info("Initializing FBP reconstruction...")

    ##########################################################
    ################# Define helper methods ##################
    ##########################################################
    def save_current_volume(id_str: str, is_intermediate: bool) -> None:
        # Do nothing if the output path is None.
        if path_output is None:
            return

        # Get the current volume
        x = vol.as_numpy()
        x = lib.ne_tools.downbin(x, n=args.superres, preserve_sum=False)

        # Get the 'root' directory of where this volume will be saved
        save_path = path_intms if is_intermediate else path_output

        # Save the volume as a .npy file
        tomo_path = (
            save_path.joinpath("tomograms") if is_intermediate else save_path
        )
        tomo_path.mkdir(exist_ok=True)
        lib.ne_tools.save(tomo_path.joinpath(id_str + ".npy"), x)

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
            imview.save_volslice(
                save_path_sl.joinpath(
                    "midslice_" + axis_str + "_fullrange_" + id_str + ".png"
                ),
                x,
                axis=axis,
                index_of_slice=x.shape[axis] // 2,
                pmin=x.min(),
                pmax=x.max(),
            )

    ##########################################################
    ################# Begin reconstruction ###################
    ##########################################################
    logger.info("Beginning reconstruction...")
    with logtools.ProgressBar(
        desc="Filtered Backproj.:",
        maxval=pg.num_projs,
        valunit="proj",
        logger=logger,
        persist=False,
    ) as pbar:
        if args.fan_beam:
            sum_of_c = 0  # Will be used to do a quick calculation of the approximate attenuation in the middle voxel
        for sl, pss in sl_filtps_gen:
            if args.two_sided:
                ps_ramp, ps_hilb = pss
                cbfp.backproject(
                    config_bp_ramp_twosided, vol, ps_ramp, block=True
                )
                cbfp.backproject(
                    config_bp_hilb_twosided, vol, ps_hilb, block=True
                )
                assert (
                    ps_ramp.data is not None
                )  # useful assertion for type-checking
                assert (
                    ps_hilb.data is not None
                )  # useful assertion for type-checking
                if args.fan_beam:
                    sum_of_c += float(
                        cp.sum(ps_ramp.data[:, ps_ramp.data.shape[1] // 2, :])
                    )
                    sum_of_c += float(
                        cp.sum(ps_hilb.data[:, ps_ramp.data.shape[1] // 2, :])
                    )
            else:
                (ps_ramp,) = pss
                cbfp.backproject(
                    config_bp_ramp_onesided, vol, ps_ramp, block=True
                )
                assert (
                    ps_ramp.data is not None
                )  # useful assertion for type-checking
                if args.fan_beam:
                    sum_of_c += float(
                        cp.sum(ps_ramp.data[:, ps_ramp.data.shape[1] // 2, :])
                    )
            # Perform logging updates
            pbar.update_to(sl.stop)

    if args.fan_beam:
        logger.debug(
            f"Approximate attenuation coefficient of middle voxel: {sum_of_c * final_multiplier:.5f}"
        )

    # Scale the answer correctly
    vol.multiply(final_multiplier)

    # Save final output artefacts
    if path_output:
        save_current_volume(id_str="recon", is_intermediate=False)

    # Perform logging updates
    logger.info("Done.")
