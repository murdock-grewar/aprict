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

"""Script to perform GBC reconstruction from cylindrical flat-panel cone-beam transmission data."""

import enum
import logging
import math
import pathlib
import types
import typing

import click
import click_extra as ce
import cupy as cp
import numpy as np
import rich.console

# Try to load some modules
try:
    from .. import liveplot  # ruff: ignore[unused-import]

    _mod_avail_liveplot = True
except ModuleNotFoundError:
    _mod_avail_liveplot = False

try:
    from ..lib import ng_tools

    _mod_avail_neuroglancer = True
except ModuleNotFoundError:
    _mod_avail_neuroglancer = False

from .. import Volume, gbc, imview, lib, logtools
from ..cb import flatpanel as cbfp

_mod_avail_imview = imview._pyqtgraph_installed  # ruff: ignore[private-member-access]


class ALG(enum.Enum):  # ruff: ignore[undocumented-public-class]
    GREWAR = 1


class PLOTMODE(enum.Enum):  # ruff: ignore[undocumented-public-class]
    NONE = 1
    PYQTGRAPH = 2
    NEUROGLANCER = 3


_progdesc = "Perform an FBP reconstruction from cylindrical flat-panel cone-beam transmission data."


@ce.command(
    params=None,
    help=_progdesc,
    context_settings={
        "show_default": True,
        "show_choices": True,
        "terminal_width": None,
        "max_content_width": 300,
    },
)
@ce.option_group(
    "required options",
    ce.option(
        "--geom",
        required=True,
        metavar="projgeom.json",
        type=click.Path(path_type=pathlib.Path),
        help="Path to saved ProjGeom object.",
    ),
    ce.option(
        "--projs",
        required=True,
        metavar="projdata.npy",
        type=click.Path(path_type=pathlib.Path),
        help="Path to the numpy array containing the transmission data.",
    ),
    ce.option(
        "--voxwid",
        metavar="1.0",
        type=float,
        help="The cubic voxel width, in the same distance units as the projection data is specified in projgeom.json.",
        required=True,
    ),
    ce.option(
        "--volshape",
        metavar="'(nX,nY,nZ)'",
        type=lib.click_tools.IntTupleArgTypeFactory(numints=3),
        help="Dimensions of the voxel grid on which to reconstruct.",
        required=True,
    ),
)
@ce.option_group(
    "geometry options",
    ce.option(
        "--volpos",
        metavar="'(0.,0.,0.)'",
        type=lib.click_tools.FloatTupleArgTypeFactory(numfloats=3),
        help="Position of the middle of the volume.",
        default=(0.0, 0.0, 0.0),
    ),
    ce.option(
        "--cyl-mask-rad",
        metavar="None",
        type=lib.click_tools.MaybeType(click.FLOAT),
        default="None",
        help="Mask out the reconstruction outside the cylinder of this radius. Choose 'None' to have it match the radius of the reconstruction support.",
    ),
)
@ce.option_group(
    "data preprocessing",
    ce.option(
        "--autorange-z",
        type=bool,
        help="Autorange the z coordinates of the trajectory, so that the highest source point and the lowest source point are +-z0. This autoranging takes place before slicing of projection data.",
    ),
    ce.option(
        "--slice-projs",
        metavar="'(start, stop)'",
        type=lib.click_tools.ContiguousSliceArgType(),
        default=slice(None, None),
        help="Slice the projections to include those of index start to (stop-1), inclusive.",
    ),
    ce.option(
        "--flip-x",
        type=bool,
        default=False,
        help="Flip the projection geometry about the yz plane.",
    ),
    ce.option(
        "--flip-y",
        type=bool,
        default=False,
        help="Flip the projection geometry about the xz plane.",
    ),
    ce.option(
        "--flip-z",
        type=bool,
        default=False,
        help="Flip the projection geometry about the xy plane.",
    ),
    ce.option(
        "--rotate-xy",
        metavar="0.0",
        type=float,
        default=0.0,
        help="Rotate the projection geometry by this many radians, around the z axis.",
    ),
)
@ce.option_group(
    "GBC options",
    ce.option(
        "--alg",
        type=click.Choice(ALG),
        default=ALG.GREWAR,
        help="The FBP algorithm to use.",
    ),
    ce.option(
        "--f1",
        type=click.FLOAT,
        default=1.2,
        required=True,
        help="Padding factor for full-scale backprojection.",
    ),
    ce.option(
        "--f2",
        type=click.FLOAT,
        default=6,
        required=True,
        help="Padding factor for full-scale backprojection.",
    ),
    ce.option(
        "--n",
        type=click.INT,
        metavar="ODD_INTEGER",
        default=9,
        required=True,
        help="Downscale factor for the mini recon used to correct low-frequency errors.",
    ),
    ce.option(
        "--eps",
        type=click.FLOAT,
        default=1e-6,
        required=True,
        help="Small positive value to use for softening division by zero in the weight normalisation.",
    ),
    ce.option(
        "--softspan-v",
        type=click.FLOAT,
        default=0.05,
        required=True,
        help="Vertical softspan, in radians.",
    ),
    ce.option(
        "--softspan-h",
        type=click.FLOAT,
        default=0.05,
        required=True,
        help="Horizontal softspan, in radians.",
    ),
    ce.option(
        "--interpolate-measurements",
        type=bool,
        default=True,
        help="Whether to apply bilinear interpolation between measurement pixels, when backprojecting.",
    ),
    ce.option(
        "--ssx",
        type=int,
        default=1,
        help="Sample this many x positions within a voxel. (Multplies to a grid with --ssy, --ssz.)",
    ),
    ce.option(
        "--ssy",
        type=int,
        default=1,
        help="Sample this many x positions within a voxel. (Multplies to a grid with --ssx, --ssz.)",
    ),
    ce.option(
        "--ssz",
        type=int,
        default=1,
        help="Sample this many x positions within a voxel. (Multplies to a grid with --ssx, --ssy.)",
    ),
)
@ce.option_group(
    "I/O options",
    ce.option(
        "--output",
        # Note that setting file_okay=False and dir_okay=False causes it to only accept uncreated paths
        type=lib.click_tools.MaybeType(
            click.Path(
                path_type=pathlib.Path,
                writable=True,
                file_okay=False,
                dir_okay=False,
            )
        ),
        default=None,
        help="Path to a new directory to create and place output artefacts and log files.",
    ),
    ce.option(
        "--verbosity",
        type=click.Choice(logtools.LogLevel),
        default=logtools.LogLevel.WARN,
        help="The verbosity of output logging to stderr. Log files will not be affected by this setting.",
    ),
    ce.option(
        "--output-intermediates",
        type=bool,
        default=False,
        help="Whether to output intermediate reconstructions from each iteration.",
    ),
    ce.option(
        "--plotmin",
        "--pmin",
        metavar="0.0",
        type=lib.click_tools.MaybeType(click.FLOAT),
        default="None",
        help="The minimum for the plotting range. Set to None to select minimum value in each plot.",
    ),
    ce.option(
        "--plotmax",
        "--pmax",
        metavar="1.0",
        type=lib.click_tools.MaybeType(click.FLOAT),
        default="None",
        help="The maximum for the plotting range. Set to None to select maximum value in each plot.",
    ),
    ce.option(
        "--plot-mode",
        type=click.Choice(PLOTMODE),
        default=PLOTMODE.NONE,
        help="How to plot.",
    ),
    ce.option(
        "--web-port",
        type=click.INT,
        default=8080,
        help="Port to use for serving things via a web interface, if applicable.",
    ),
)
@ce.option_group(
    "resource options",
    ce.option(
        "--ne-threads",
        type=click.Choice(lib.ne_tools.NETHREADS),
        default=lib.ne_tools.NETHREADS.ALL_LOGICAL_CORES,
        help="Set the number of threads to be used by numexpr. This will attempt to override the env NUMEXPR_MAX_THREADS as necessary.",
    ),
)
def main(**kwargs) -> None:  # ruff: ignore[missing-type-kwargs]
    """Parse input arguments using click."""
    args = types.SimpleNamespace(**kwargs)
    main_from_args(args)


def validate_args(args: types.SimpleNamespace) -> None:
    """Validate a set of args."""
    # Ensure that the output directory doesn't already exist
    if args.output is not None and args.output.exists():
        msg = f"Output path '{args.output!s}' already exists."
        raise ValueError(msg)

    # If outputting intermediates, then the user must also specify the --output directory
    if args.output_intermediates and not args.output:
        msg = "Cannot output intermediate tomograms when no output directory was specified. Specify an output directory with --output path/to/new/dir"
        raise ValueError(msg)

    if not args.output and args.plot_mode is PLOTMODE.NONE:
        msg = "No output directory specified and no plotting option chosen. There is no point running the recon."
        raise ValueError(msg)

    if args.ssx <= 0 or args.ssy <= 0 or args.ssz <= 0:
        msg = "--ssx, --ssy, --ssz must all be positive integers."
        raise ValueError(msg)

    if args.n < 1 or args.n % 2 == 0:
        msg = "--n must be a positive odd integer."
        raise ValueError(msg)

    if args.plot_mode is PLOTMODE.PYQTGRAPH and not _mod_avail_imview:
        msg = "Cannot use --plot-mode PYQTGRAPH because the relevant modules are not installed."
        raise ValueError(msg)

    if args.plot_mode is PLOTMODE.NEUROGLANCER and not _mod_avail_neuroglancer:
        msg = "Cannot use --plot-mode NEUROGLANCER because the relevant modules are not installed."
        raise ValueError(msg)


def get_projectionset_generator(  # ruff: ignore[complex-structure]
    projdata_path: pathlib.Path,
    pg: cbfp.ProjGeom,
    *,
    slice_projs: slice[int | None, int | None, None] | None,
    autorange_z: bool,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
    rotate_xy: float,
    num_projs_per_chunk: int = 9,
    logger: logging.Logger,
    vol_minxyz_maxxyz: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ],  # used to quickly skip projections out of range for this algorithm
) -> tuple[
    cbfp.ProjGeom,
    typing.Iterable[
        tuple[slice[int, int, None], tuple[cbfp.ProjectionSet, ...]]
    ],
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    """Process input data and output something which can be given to an FBP recon object."""
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
    if rotate_xy != 0:
        pg = pg.rotated(xy_rads=rotate_xy)
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
    nu = int(pg.nu[0])
    if not np.all(pg.nv == pg.nv[0]):
        msg = "Inconsistent vertical resolution of detector."
        raise NotImplementedError(msg)
    nv = int(pg.nv[0])
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
    Rs = np.sqrt(  # ruff: ignore[non-lowercase-variable-in-function]
        np.square(pg.source_pos[:, 0]) + np.square(pg.source_pos[:, 1])
    )
    if not np.all(np.isclose(Rs, Rs[0])):
        msg = "Inconsistent source radius."
        raise NotImplementedError(msg)
    R = Rs[0]  # ruff: ignore[non-lowercase-variable-in-function]
    Ls = Rs + np.sqrt(  # ruff: ignore[non-lowercase-variable-in-function]
        np.square(pg.det_pos[:, 0]) + np.square(pg.det_pos[:, 1])
    )
    if not np.all(np.isclose(Ls, Ls[0])):
        msg = "Inconsistent source-to-detector length."
        raise NotImplementedError(msg)
    L = Ls[0]  # ruff: ignore[non-lowercase-variable-in-function]

    # Compute vartheta_v and friends
    _dethei = nv * pixhei  # ruff: ignore[used-dummy-variable]
    _detwid = nu * pixwid  # ruff: ignore[used-dummy-variable]
    vartheta_v = np.arctan2(
        _dethei / 2 - pixhei / 2,
        np.sqrt(np.square(L) + np.square(_detwid / 2 - pixwid / 2)),
    )
    vartheta_h = np.arctan2(
        _detwid / 2 - pixwid / 2,
        L,
    )

    zs_max = pg.source_pos[:, 2].max()
    zs_min = pg.source_pos[:, 2].min()
    recon_support_radius: float = R * np.sin(np.arctan(pixwid * nu / 2 / L))
    recon_support_height: float = max(
        0,
        (zs_max - zs_min)
        - 2 * (R + recon_support_radius) * np.tan(vartheta_v),
    )
    max_r_of_any_volpoint: float = np.sqrt(
        np.square(
            max(abs(vol_minxyz_maxxyz[0][0]), abs(vol_minxyz_maxxyz[1][0]))
        )
        + np.square(
            max(abs(vol_minxyz_maxxyz[0][1]), abs(vol_minxyz_maxxyz[1][1]))
        )
    )
    # _hdiff will be used to quickly prune projections that don't intersect the volume
    hdiff_: float = np.tan(vartheta_v) * (R + max_r_of_any_volpoint)

    logger.info(f"R: {R!s:s}")
    logger.info(f"L: {L!s:s}")
    logger.info(f"recon_support_radius: {recon_support_radius:.5g}")
    logger.info(f"recon_support_height: {recon_support_height:.5g}")

    # Do not apply this here!
    # The soft-zeroing of the weights needs to apply to the weights with
    # this factor included. So, just set this to 1 here and do that soft-zeroing
    # elsewhere in the script.
    # final_multiplier = (
    #     # 1/(mu R^2) = ((zs_max - zs_min)*(2*pi*R)/num_projs) / R^2
    #     (zs_max - zs_min) * 2*np.square(np.pi) / (R * pg.num_projs)
    # )
    mu = pg.num_projs / (2 * np.pi * R * (zs_max - zs_min))
    logger.debug(f"Height of source cylinder: {zs_max - zs_min!s:s}")

    def _gen() -> typing.Generator[
        tuple[slice[int, int, None], tuple[cbfp.ProjectionSet, ...]]
    ]:
        sl_start: int = 0
        while sl_start < pg.num_projs:
            sl = slice(
                sl_start, min(pg.num_projs, sl_start + num_projs_per_chunk)
            )

            # If projections are totally out of range of the volume, then skip
            pg_sl = pg.sliced(sl)
            sl_zs_max = pg_sl.source_pos[:, 2].max()
            sl_zs_min = pg.source_pos[:, 2].min()
            if (vol_minxyz_maxxyz[1][2] < sl_zs_min - hdiff_) or (
                vol_minxyz_maxxyz[0][2] > sl_zs_max + hdiff_
            ):
                # Projections are totally out of range of the volume. Skip loading these projections.
                # Increment the slice
                sl_start += num_projs_per_chunk
                continue

            # Create ProjectionSet with filtered data in buffer
            filt_arrs = [cp.array(data_mm[sl])]
            pss = [
                cbfp.ProjectionSet(pg.sliced(sl), data=filt_arr)
                for filt_arr in filt_arrs
            ]
            # Yield
            yield sl, tuple(pss)
            # Increment the slice
            sl_start += num_projs_per_chunk

    return (
        pg,
        _gen(),
        R,
        L,
        recon_support_radius,
        recon_support_height,
        mu,
        vartheta_v,
        vartheta_h,
    )


def main_from_args(args: types.SimpleNamespace) -> None:
    """Run the script."""
    # Validate the input arguments
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
    logger.info(f"Currently executing file: '{__file__}'")
    logger.info("Complete list of arguments passed to this method:")
    keys = sorted(args.__dict__.keys())
    for k in keys:
        v = args.__dict__[k]
        logger.info(f"    '{k:s}' value: '{v!s:s}'")

    # Set up the rich console object
    # Needed for showing spinners , etc.
    rich_console = rich.console.Console(stderr=True)

    ##########################################################
    ##### Set numexpr to use as many threads as possible #####
    ##########################################################
    # Set numexpr to have the appropriate number of threads
    lib.ne_tools.try_set_nethreads(args.ne_threads, logger=logger)

    ##########################################################
    ######## Prepare the input data for the FBP recon ########
    ##########################################################

    # Create the working volumes
    # We need to pad the volume to a multiple of args.n in every axis, so that we can apply the low-pad correction. We will then crop the output. However, to crop the output, we need to be able to crop symmetrically, i.e. remove an even number of voxels from each axis. Therefore, we need the padded shape to be odd if and only if the input shape is odd.
    # This is why we require the input arg --n to be odd-valued, so that we can ensure these conditions can always be met.
    padded_volshape = tuple(
        args.n * (4 + math.ceil(n * args.f1 / args.n)) for n in args.volshape
    )  # adding 4*args.n because we'll have to move in by 2*args.n from the edges to estimate the DC component. We need to add extra padding to ensure that we can, in fact, move in by 2*args.n from all edges.
    # Now, check each axis. If one of them has a different parity to the original spec, then add args.n to the length of that axis. Since we've ensured that --n is odd, this will change the parity.
    padded_volshape = tuple(
        (n if (n - m) % 2 == 0 else n + args.n)
        for m, n in zip(args.volshape, padded_volshape)
    )
    padx, pady, padz = tuple(
        (n - m) // 2 for m, n in zip(args.volshape, padded_volshape)
    )
    padded_volshape_mini = tuple(
        math.ceil(n * args.f2 / args.n) for n in args.volshape
    )
    # Ensure that the mini volshape is at least big enough to support the padded_volshape.
    padded_volshape_mini = tuple(
        max(n, m // args.n)  # note: m here is an integer multiple of args.n
        for m, n in zip(padded_volshape, padded_volshape_mini)
    )
    # For similar reasons as above, make sure that the padded volshape_mini is an even number of voxels bigger than padded_volshape*args.n, on each axis. It's OK if it's the same size.
    padded_volshape_mini = tuple(
        (n if (n - m // args.n) % 2 == 0 else n + 1)
        for m, n in zip(padded_volshape, padded_volshape_mini)
    )
    mvc_padx, mvc_pady, mvc_padz = tuple(
        (n - m // args.n) // 2
        for m, n in zip(padded_volshape, padded_volshape_mini)
    )
    logger.debug(f"padded_volshape:      {padded_volshape}")
    logger.debug(f"padded_volshape_mini: {padded_volshape_mini}")
    logger.debug(f"minivol padx, pady, padz: {(mvc_padx, mvc_pady, mvc_padz)}")
    vol = Volume.zeros(
        padded_volshape,
        voxwid=args.voxwid,
        dtype=np.float32,  # ty:ignore[invalid-argument-type]
        pos_c=args.volpos,
    )
    minivol = Volume.zeros(
        padded_volshape_mini,
        voxwid=args.voxwid * args.n,
        dtype=np.float32,  # ty: ignore[invalid-argument-type]
        pos_c=args.volpos,
    )
    vol_weightsum = Volume.zeros(
        padded_volshape,
        voxwid=args.voxwid,
        dtype=np.float32,  # ty:ignore[invalid-argument-type]
        pos_c=args.volpos,
    )
    minivol_weightsum = Volume.zeros(
        padded_volshape_mini,
        voxwid=args.voxwid * args.n,
        dtype=np.float32,  # ty: ignore[invalid-argument-type]
        pos_c=args.volpos,
    )

    # Create the ProjectionSet objects corresponding to each ordered subset
    (
        pg,
        sl_filtps_gen,
        R,  # ruff: ignore[non-lowercase-variable-in-function]
        _L,  # ruff: ignore[non-lowercase-variable-in-function]
        _recon_support_radius,
        _recon_support_height,
        mu,
        vartheta_v,
        vartheta_h,
    ) = get_projectionset_generator(
        projdata_path=args.projs,
        pg=cbfp.ProjGeom.load(args.geom),
        slice_projs=args.slice_projs,
        autorange_z=args.autorange_z,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        flip_z=args.flip_z,
        rotate_xy=args.rotate_xy,
        logger=logger,
        # These two are supplied to prune irrelevant projections
        vol_minxyz_maxxyz=minivol.minxyz_maxxyz,  # this should be the largest volume
    )

    # Initialise the configuration used for backprojection
    config_bp = cbfp.ConfigBackproject(
        fast_approx=True,
        cyl_mask_rad=None,
        Z_on_val_factor_power=0,
        Z_on_val_factor_val=R,  # This must be equal to R for the kernel to operate correctly.
        backproj_weighting_preset=7,
        approx_mode_direct_raycast=True,
        approx_mode_bilinear_interpolate=args.interpolate_measurements,
        ssx=args.ssx,
        ssy=args.ssy,
        ssz=args.ssz,
        soft_vartheta_v=vartheta_v,
        soft_vartheta_v_softspan=args.softspan_v,
        soft_vartheta_h=vartheta_h,
        soft_vartheta_h_softspan=args.softspan_h,
    )

    ##########################################################
    ########## Initialisation of FBP reconstruction ##########
    ##########################################################
    logger.info("Initializing FBP reconstruction...")

    ##########################################################
    ################# Define helper methods ##################
    ##########################################################
    pass  # ruff: ignore[unnecessary-placeholder]

    ##########################################################
    ################# Begin reconstruction ###################
    ##########################################################
    logger.info("Beginning reconstruction...")
    with logtools.ProgressBar(
        desc="Backproj.:",
        maxval=pg.num_projs,
        valunit="proj",
        logger=logger,
        persist=True,
    ) as pbar:
        for sl, pss in sl_filtps_gen:
            (ps_ramp,) = pss
            cbfp.backproject(
                config_bp,
                vol,
                ps_ramp,
                secondary_volume=vol_weightsum,
                block=True,
            )
            cbfp.backproject(
                config_bp,
                minivol,
                ps_ramp,
                secondary_volume=minivol_weightsum,
                block=True,
            )
            assert (
                ps_ramp.data is not None
            )  # useful assertion for type-checking
            # Perform logging updates
            pbar.update_to(sl.stop)

    # Normalise to the correct weighting.
    # This will multiply both a numerator and a divisor by the same amount.
    # Howevever, it will affect how the denominator is soft-zeroed.
    msg = "Applying constant factor to volume and weights..."
    logger.info(msg)
    with rich_console.status(msg, spinner="dots"):
        factor = 1 / (mu * R**2)
        vol.multiply(factor)
        minivol.multiply(factor)
        vol_weightsum.multiply(factor)
        minivol_weightsum.multiply(factor)

    # Normalize weights by the expected weightsums
    msg = "Normalizing the weightsums by the expected sum of backprojection weights..."
    logger.info(msg)
    with rich_console.status(msg, spinner="dots"):
        # Full-scale volume
        xs, ys, _ = vol_weightsum.xs_ys_zs
        xygrid = np.power(
            gbc.weightinteg.backproj_gbc_weight_integ_xygrid(
                xs,
                ys,
                R,
                soft_vartheta_v=vartheta_v,
                soft_vartheta_v_softspan=args.softspan_v,
                soft_vartheta_h=vartheta_h,
                soft_vartheta_h_softspan=args.softspan_h,
            ),
            -1,
        ).astype(np.float32)
        vol_weightsum.multiply_by_xygrid(xygrid)
        # Mini volume
        xs, ys, _ = minivol_weightsum.xs_ys_zs
        xygrid = np.power(
            gbc.weightinteg.backproj_gbc_weight_integ_xygrid(
                xs,
                ys,
                R,
                soft_vartheta_v=vartheta_v,
                soft_vartheta_v_softspan=args.softspan_v,
                soft_vartheta_h=vartheta_h,
                soft_vartheta_h_softspan=args.softspan_h,
            ),
            -1,
        ).astype(np.float32)
        minivol_weightsum.multiply_by_xygrid(xygrid)

    # Divide by the weightsums
    msg = "Normalizing volumes by the sum of backprojection weights..."
    logger.info(msg)
    with rich_console.status(msg, spinner="dots"):
        vol.divide_softzero(vol_weightsum, eps=args.eps)
        minivol.divide_softzero(minivol_weightsum, eps=args.eps)
        del vol_weightsum
        del minivol_weightsum

    msg = "Filtering volumes..."
    logger.info(msg)
    with rich_console.status(msg, spinner="dots"):
        # Filter the mini recon first.
        minivol_cp = minivol.as_cupy()
        minivol_voxwid = minivol.voxwid
        ndarray_container = lib.np_tools.NDArrayContainer(array=minivol_cp)
        del minivol  # allows memory to be freed, because the gbc.filters.ramp_filter_conebeam may be free to overwrite the ndarray_container.array
        minivol_padded_filtered_cp = gbc.filters.ramp_filter_conebeam(
            ndarray_container,
            minivol_voxwid,
            2 * vartheta_v,
            theta_soft=args.softspan_v,
            auto_dc_adjust=False,
            allow_overwrite_arr=False,  # we're going to reuse the minivol_cp in the next filtration
            logger=logger,
            copy_to_array=None,
        )
        ndarray_container = lib.np_tools.NDArrayContainer(
            array=minivol_cp[
                mvc_padx : (-mvc_padx) or None,
                mvc_pady : (-mvc_pady) or None,
                mvc_padz : (-mvc_padz) or None,
            ]
        )
        del minivol_cp  # not going to reuse this, so delete this reference so the filtration can free some memory.
        minivol_unpadded_filtered_cp = gbc.filters.ramp_filter_conebeam(
            ndarray_container,
            minivol_voxwid,
            2 * vartheta_v,
            theta_soft=args.softspan_v,
            auto_dc_adjust=False,
            allow_overwrite_arr=True,  # we're not going to reuse the 'minivol_cp' object after this.
            logger=logger,
            copy_to_array=None,
        )
        minivol_correction_cp = (
            minivol_padded_filtered_cp[
                mvc_padx : (-mvc_padx) or None,
                mvc_pady : (-mvc_pady) or None,
                mvc_padz : (-mvc_padz) or None,
            ]
            - minivol_unpadded_filtered_cp
        )
        # Delete some references to free up memory, if possible
        del (
            minivol_padded_filtered_cp,
            minivol_unpadded_filtered_cp,
            ndarray_container,
        )
        # Filter the full recon now
        ndarray_container = lib.np_tools.NDArrayContainer(array=vol.as_cupy())
        vol_voxwid = vol.voxwid
        del vol  # allows memory to be freed, because the gbc.filters.ramp_filter_conebeam is free to overwrite the ndarray_container.array
        vol_filtered_cp = gbc.filters.ramp_filter_conebeam(
            ndarray_container,
            vol_voxwid,
            2 * vartheta_v,
            theta_soft=args.softspan_v,
            auto_dc_adjust=False,
            allow_overwrite_arr=True,
            logger=logger,
            copy_to_array=None,  # Should be a contiguous array. E.g., not an arbitrary view.
            lowres_correction_array=minivol_correction_cp,
        )
        del minivol_correction_cp, ndarray_container

        # Now estimate the DC component by looking at the voxels on the boundary.
        # Due to the lowres correction interpolating on the edges, we want to move in
        # by 2*args.n (empirically determined) before we get an accurate result.
        cropped_view_cp = vol_filtered_cp[
            2 * args.n : -2 * args.n,
            2 * args.n : -2 * args.n,
            2 * args.n : -2 * args.n,
        ]
        # Average the four rectangular boundaries
        dc_component = (
            1
            / 4
            * (
                cp.mean(cropped_view_cp[:, 0, :])
                + cp.mean(cropped_view_cp[:, -1, :])
                + cp.mean(cropped_view_cp[0, :, :])
                + cp.mean(cropped_view_cp[-1, :, :])
            )
        )
        logger.debug(
            f"Estimated DC component to be subtracted: {dc_component:.5g}"
        )
        # Correct the DC component
        cp.subtract(vol_filtered_cp, dc_component, out=vol_filtered_cp)

        # Now crop back to the original shape
        vol_filtered_cp = vol_filtered_cp[
            padx : (-padx) or None,
            pady : (-pady) or None,
            padz : (-padz) or None,
        ]
        logger.debug(f"Output volume shape: {vol_filtered_cp.shape}")

        # Fetch it now as a numpy array
        vol_filtered = vol_filtered_cp.get()  # ty: ignore[unresolved-attribute]

        # Clean up
        del vol_filtered_cp
        lib.gpu_tools.clear_cache()

    if path_output is not None:
        msg = "Saving volume to disk..."
        logger.info(msg)
        with rich_console.status(msg, spinner="dots"):
            x = vol_filtered
            # Save the volume as a .npy file
            tomo_path = path_output.joinpath("recon.npy")
            lib.ne_tools.save(tomo_path, x)
            # Save slices
            for axis, axis_str in enumerate("xyz"):
                save_path_sl = path_output
                imview.save_volslice(
                    save_path_sl.joinpath(
                        "midslice_" + axis_str + "_" + "recon" + ".png"
                    ),
                    x,
                    axis=axis,
                    index_of_slice=x.shape[axis] // 2,
                    pmin=args.plotmin,
                    pmax=args.plotmax,
                )
                imview.save_volslice(
                    save_path_sl.joinpath(
                        "midslice_"
                        + axis_str
                        + "_fullrange_"
                        + "recon"
                        + ".png"
                    ),
                    x,
                    axis=axis,
                    index_of_slice=x.shape[axis] // 2,
                    pmin=x.min(),
                    pmax=x.max(),
                )

    # Perform logging updates
    logger.info("Done.")

    # Plot, if applicable
    match args.plot_mode:
        case PLOTMODE.NONE:
            pass
        case PLOTMODE.PYQTGRAPH:
            msg = "Displaying plot..."
            logger.info(msg)
            with rich_console.status(msg, spinner="dots10"):
                imview.plot_volume(vol_filtered, title="Filtered volume")
                imview.plot_volume(
                    vol_filtered.transpose((2, 1, 0)), title="Filtered volume"
                )
        case PLOTMODE.NEUROGLANCER:
            msg = "Displaying plot..."
            logger.info(msg)
            with ng_tools.ArrayViewer3D(
                vol_filtered,
                plotmin=args.plotmin,
                plotmax=args.plotmax,
                port=args.web_port,
                voxwid_mm=vol_voxwid,
                wait_for_input_before_exit_prompt=None,
            ) as av:
                msg = f"Displaying plot at URL: {av.url} ..."
                logger.info(msg)
                with rich_console.status(
                    msg + " press Enter to quit.", spinner="dots10"
                ):
                    input()
