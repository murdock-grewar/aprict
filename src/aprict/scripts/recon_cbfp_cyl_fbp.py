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

import enum
import logging
import pathlib
import time
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


from .. import Volume, cb, imview, lib, logtools
from ..cb import flatpanel as cbfp

_mod_avail_imview = imview._pyqtgraph_installed  # ruff: ignore[private-member-access]


class ALG(enum.Enum):  # ruff: ignore[undocumented-public-class]
    STANDARD = 1
    NOOGREWAR = 2


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
    "FBP options",
    ce.option(
        "--alg",
        type=click.Choice(ALG),
        default=ALG.STANDARD,
        help="The FBP algorithm to use.",
    ),
    ce.option(
        "--fan-beam",
        type=bool,
        default=False,
        help="Set to True to activate fan-beam reconstruction mode. This may avoid some other settings to make it work on 2D circular fan-beam data.",
    ),
    ce.option(
        "--limit-zeta",
        metavar="1.23",
        type=lib.click_tools.MaybeType(click.FLOAT),
        default="None",
        help="Set this to impose a maximum value for zeta. This will limit the data used by this method.",
    ),
    ce.option(
        "--num-rot-angs",
        metavar="None",
        type=lib.click_tools.MaybeType(click.INT),
        default="None",
        help="How many filtration directions to sample. Set to None to use a random single direction for each CB projection.",
    ),
    ce.option(
        "--seed-proj-rots",
        default=None,
        type=lib.click_tools.MaybeType(click.INT),
        help="Seed to use for generating the volume preset, if applicable.",
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
        type=lib.click_tools.BoolArgType(),
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
    ce.option(
        "--memmap-vols",
        type=lib.click_tools.BoolArgType(),
        default=True,
        help="Memory-map volumes, rather than keeping them in RAM.",
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
        raise ce.UsageError(msg)

    if args.output is None and args.memmap_vols:
        msg = "Cannot use --memmap-vols True while --output is not specified."
        raise ce.UsageError(msg)

    # If outputting intermediates, then the user must also specify the --output directory
    if args.output_intermediates and not args.output:
        msg = "Cannot output intermediate tomograms when no output directory was specified. Specify an output directory with --output path/to/new/dir"
        raise ce.UsageError(msg)

    if not args.output and args.plot_mode is PLOTMODE.NONE:
        msg = "No output directory specified and no plotting option chosen. There is no point running the recon."
        raise ce.UsageError(msg)

    if args.fan_beam and args.ssz != 1:
        msg = "--ssz must be set to 1 for fan-beam mode, as otherwise the backprojection would sample things toward the top/bottom of a voxel, instead of just the center."
        raise ce.UsageError(msg)

    if args.ssx <= 0 or args.ssy <= 0 or args.ssz <= 0:
        msg = "--ssx, --ssy, --ssz must all be positive integers."
        raise ce.UsageError(msg)

    if args.plot_mode is PLOTMODE.PYQTGRAPH and not _mod_avail_imview:
        msg = "Cannot use --plot-mode PYQTGRAPH because the relevant modules are not installed."
        raise ce.UsageError(msg)

    if args.plot_mode is PLOTMODE.NEUROGLANCER and not _mod_avail_neuroglancer:
        msg = "Cannot use --plot-mode NEUROGLANCER because the relevant modules are not installed."
        raise ce.UsageError(msg)


def get_filtered_projectionset_generator(  # ruff: ignore[complex-structure]
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
    rotate_xy: float,
    num_projs_per_chunk: int = 9,
    filt_chunk_size: int = 3,
    logger: logging.Logger,  # ruff: ignore[unused-function-argument]
    two_sided: bool,
    vol_minxyz_maxxyz: tuple[
        tuple[float, float, float], tuple[float, float, float]
    ],  # used to quickly skip projections out of range for this algorithm
    cyl_mask_rad: float
    | None,  # set to None to use the reconstruction support radius
    seed_proj_rots: int,
) -> tuple[
    cbfp.ProjGeom,
    typing.Iterable[
        tuple[slice[int, int, None], tuple[cbfp.ProjectionSet, ...] | None]
    ],
    float,
    float,
    float,
    float,
    float,
    float,
    int,
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
    vartheta_h = np.arctan2(  # ruff: ignore[unused-variable]
        _detwid / 2 - pixwid / 2,
        L,
    )

    zs_max = pg.source_pos[:, 2].max()
    zs_min = pg.source_pos[:, 2].min()
    recon_support_radius: float = R * np.sin(np.arctan(pixwid * nu / 2 / L))
    cyl_mask_rad = cyl_mask_rad or recon_support_radius
    max_r_of_any_volpoint: float = min(
        cyl_mask_rad,
        np.sqrt(
            np.square(
                max(abs(vol_minxyz_maxxyz[0][0]), abs(vol_minxyz_maxxyz[1][0]))
            )
            + np.square(
                max(abs(vol_minxyz_maxxyz[0][1]), abs(vol_minxyz_maxxyz[1][1]))
            )
        ),
    )
    vartheta_v_effective: float = (
        min(limit_zeta, vartheta_v) if limit_zeta is not None else vartheta_v
    )
    recon_support_height: float = max(
        0,
        (zs_max - zs_min)
        - 2 * (R + recon_support_radius) * np.tan(vartheta_v_effective),
    )
    # _hdiff will be used to quickly prune projections that don't intersect the volume
    hdiff_: float = np.tan(vartheta_v_effective) * (R + max_r_of_any_volpoint)

    # Set up the filtering object
    rng_proj_rots: np.random.Generator = np.random.default_rng(
        seed=seed_proj_rots
    )
    filt = cb.FBPFilterer(
        proj_2d_shape=(nu, nv),
        projs_per_batch=filt_chunk_size,
        L=L,
        physpixwid=pixwid,
        limit_zeta=limit_zeta,
        include_hilbert_transform_in_output=two_sided,
        _hacky_fan_beam_mode=fan_beam_mode,
        rng=rng_proj_rots,
    )

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

    def is_inrange(pg_sliced: cbfp.ProjGeom) -> bool:
        pg_sl = pg_sliced
        sl_zs_max = pg_sl.source_pos[:, 2].max()
        sl_zs_min = pg_sl.source_pos[:, 2].min()
        return not (
            (vol_minxyz_maxxyz[1][2] < sl_zs_min - hdiff_)
            or (vol_minxyz_maxxyz[0][2] > sl_zs_max + hdiff_)
        )

    # Figure out ahead of time how many projections need to be considered.
    # This is helpful progress bar / time estimates.
    tot_num_nonskipped_projs: int = 0
    sl_start: int = 0
    while sl_start < pg.num_projs:
        sl = slice(sl_start, min(pg.num_projs, sl_start + num_projs_per_chunk))
        pg_sl = pg.sliced(sl)
        if is_inrange(pg_sl):
            tot_num_nonskipped_projs += sl.stop - sl.start
        sl_start += num_projs_per_chunk

    def _gen() -> typing.Generator[  # ruff: ignore[complex-structure]
        tuple[slice[int, int, None], tuple[cbfp.ProjectionSet, ...] | None]
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

            # If projections are totally out of range of the volume, then skip
            pg_sl = pg.sliced(sl)
            if not is_inrange(pg_sl):
                # Projections are totally out of range of the volume. Skip loading these projections.
                # Increment the slice
                sl_start += num_projs_per_chunk
                filt.advance_rng(amount=sl.stop - sl.start)
                continue

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

    return (
        pg,
        _gen(),
        R,
        L,
        recon_support_radius,
        recon_support_height,
        cyl_mask_rad,
        final_multiplier,
        tot_num_nonskipped_projs,
        zs_min,
        zs_max,
    )


def main_from_args(args: types.SimpleNamespace) -> None:  # ruff: ignore[complex-structure]
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
    vol_dtype = np.float32
    pg_raw: cbfp.ProjGeom = cbfp.ProjGeom.load(args.geom)

    # Resolve the seed. We need to do this because a seed value of 'None' will cause
    # a different seed to be used each time we generate projections.
    # We want to fix a definite seed value right now.
    BIG_INT: int = int(1e10)  # ruff: ignore[non-lowercase-variable-in-function]
    seed_proj_rots: int = int(
        BIG_INT * np.random.default_rng(seed=args.seed_proj_rots).random()
    )

    # Create a working volume -- this will only be used to compute some geometry information.
    # Then it will be discarded. This is not very efficient.
    volshape: tuple[int, int, int] = (
        (*tuple(n for n in args.volshape[0:2]), 1)
        if args.fan_beam
        else tuple(n for n in args.volshape)
    )

    two_sided = args.alg is ALG.NOOGREWAR

    # Do some computations to figure out geometry attributes, and log them
    dummy_vol: Volume = Volume.zeros(
        volshape,
        voxwid=args.voxwid,
        dtype=vol_dtype,  # ty:ignore[invalid-argument-type]
        pos_c=args.volpos,
        initialize_texture=False,  # just a dummy volume. Don't try to allocate room on the GPU for it.
    )
    (
        _pg,
        _sl_filtps_gen,
        R,  # ruff: ignore[non-lowercase-variable-in-function]
        L,  # ruff: ignore[non-lowercase-variable-in-function]
        recon_support_radius,
        recon_support_height,
        _cyl_mask_rad,
        _final_multiplier,
        _tot_num_nonskipped_projs,
        zs_min,
        zs_max,
    ) = get_filtered_projectionset_generator(
        projdata_path=args.projs,
        pg=pg_raw,
        fan_beam_mode=args.fan_beam,
        num_rot_angs=args.num_rot_angs,
        limit_zeta=args.limit_zeta,
        slice_projs=args.slice_projs,
        autorange_z=args.autorange_z,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        flip_z=args.flip_z,
        rotate_xy=args.rotate_xy,
        logger=logger,
        two_sided=two_sided,
        # These two are supplied to prune irrelevant projections
        vol_minxyz_maxxyz=dummy_vol.minxyz_maxxyz,
        cyl_mask_rad=args.cyl_mask_rad,
        seed_proj_rots=seed_proj_rots,
    )

    ##########################################################
    ########## Initialisation of FBP reconstruction ##########
    ##########################################################
    # logger.info(f"cyl mask radius: {cyl_mask_rad:.5g}")
    # logger.debug(
    #     f"normalization multiplier on volume after filtered backprojection: {final_multiplier!s:s}"
    # )
    logger.info("Initializing FBP reconstruction...")

    ##########################################################
    ################# Define helper methods ##################
    ##########################################################

    def bytes_at_index(index: int) -> int:  # ruff: ignore[unused-function-argument]
        return volshape[0] * volshape[1] * np.dtype(vol_dtype).itemsize

    # List the number of bytes corresponding to each z index of the minivol, from bottom to top.
    bytes_by_index: list[int] = [
        bytes_at_index(zind) for zind in range(volshape[2])
    ]
    # Cumulative sum of the above
    bytes_cumsum = np.cumsum(bytes_by_index)
    _, gpu_VRAM_capacity_MiB = lib.gpu_tools.get_gpu_mb_used_and_total()  # ruff: ignore[non-lowercase-variable-in-function]
    GiB_per_chunk: float = (2 / 4) * gpu_VRAM_capacity_MiB / 1024  # ruff: ignore[non-lowercase-variable-in-function]

    def next_slice_zindex(last_slice: slice | None) -> slice | None:
        last_slice = slice(0, 0) if last_slice is None else last_slice
        if last_slice.stop >= volshape[2]:
            return None
        idxs = np.argwhere(
            (
                (
                    bytes_cumsum.astype(np.float32)
                    - bytes_cumsum[max(0, last_slice.stop)].astype(np.float32)
                )
                / (1024**3 * GiB_per_chunk)
            )
            < 1
        )
        if idxs.size == 0:
            sl = slice(last_slice.stop, last_slice.stop + 1)
        else:
            sl = slice(last_slice.stop, int(idxs[-1][0]) + 1)
        sl = slice(
            last_slice.stop,
            min(volshape[2], max(last_slice.stop + 1, sl.stop)),
        )
        if sl.stop == sl.start:
            return None
        return sl

    def make_vols(
        slz_: slice,
    ) -> tuple[slice, Volume | None]:  # volume and corresponding slice
        """Produce volume corresponding to a volume chunk."""
        # The 'zindex' gives the index beginning from the bottom (0) of the volume.

        zindex: int = slz_.start
        znumlayers: int = slz_.stop - slz_.start

        slz = slice(
            min(volshape[2], zindex),
            min(volshape[2], zindex + znumlayers),
        )

        volpos = (
            args.volpos[0],
            args.volpos[1],
            args.volpos[2]
            - args.voxwid
            * ((volshape[2] - (slz.stop - slz.start)) / 2 - slz.start),
        )
        vol = (
            None
            if ((slz.stop - slz.start) <= 0)
            else Volume.zeros(
                (*volshape[:2], slz.stop - slz.start),
                voxwid=args.voxwid,
                dtype=vol_dtype,  # ty:ignore[invalid-argument-type]
                pos_c=volpos,
            )
        )
        return (slz, vol)

    def save_volume(
        vol: np.ndarray,
        id_str: str,
        is_intermediate: bool,
        skip_if_exists: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        # Do nothing if the output path is None.
        if path_output is None:
            return

        # Get the current volume
        x = vol

        # ty is not smart enough to understand that path_intms can be None only when path_output is None
        assert path_intms is not None

        # Get the 'root' directory of where this volume will be saved
        save_path = path_intms if is_intermediate else path_output

        # Save the volume as a .npy file
        tomo_path = (
            save_path.joinpath("tomograms") if is_intermediate else save_path
        )
        tomo_path.mkdir(exist_ok=True)
        savepath = tomo_path.joinpath(id_str + ".npy")
        if not savepath.is_file() or not skip_if_exists:
            if logger is not None:
                msg = f"Saving full volume to '{savepath}'..."
                logger.debug(msg)
            lib.ne_tools.save(savepath, x)
            if logger is not None:
                msg = "Full volume now saved to disk."
                logger.debug(msg)
        else:
            if logger is not None:
                msg = f"Volume already present at '{savepath}'. Skipping save."
                logger.debug(msg)

        # Save midslices from volume as .png files
        # The method used here can be a bit slow, because I get a full numpy array and process that.
        # I've decided that I prefer the simplicity and robustness of this approach, instead of trying
        # to squeeze optimal performance by directly extracting slices from cupy arrays.
        for axis, axis_str in enumerate("xyz"):
            if logger is not None:
                msg = f"Saving '{axis_str}' slice..."
                logger.debug(msg)
            if is_intermediate:
                save_path_sl = save_path.joinpath("slices_" + axis_str)
                save_path_sl.mkdir(exist_ok=True)
            else:
                save_path_sl = save_path
            arr_slice = imview.save_volslice(
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
                arr_slice,
                axis=axis,
                index_of_slice=0,
                pmin=arr_slice.min(),
                pmax=arr_slice.max(),
            )

    ##########################################################
    ################# Begin reconstruction ###################
    ##########################################################
    logger.info(f"R: {R!s:s}")
    logger.info(f"L: {L!s:s}")
    logger.info(f"recon_support_radius: {recon_support_radius:.5g}")
    logger.info(f"recon_support_height: {recon_support_height:.5g}")
    logger.info(f"Height of source cylinder: {zs_max - zs_min!s:s}")

    msg = "Allocating and zeroing the backprojection arrays..."
    logger.info(msg)
    # Put the z axis first, so that we can slice it in Z and have a C-contiguous array. This is necessary to copy directly from a cupy array to a numpy array.
    if args.memmap_vols:
        assert path_output is not None  # ensured by validate_args(...)
        savepath = path_output.joinpath("recon" + ".npy")
        vol_np = np.lib.format.open_memmap(
            savepath, shape=volshape, dtype=vol_dtype, mode="w+"
        )
    else:
        vol_np = np.empty(volshape, dtype=vol_dtype)
    lib.ne_tools.set_array_to_constant_pbar(
        vol_np,
        const=0,
        pbar_desc="Allocating room for big backprojection...",
        pbar_valunit=None,
        pbar_logger=logger,
        pbar_persist=False,
    )

    logger.info("Beginning reconstruction...")
    curr_slice: slice | None = None
    with logtools.ProgressBar(
        desc="Filtered Backproj.:",
        maxval=int(bytes_cumsum[-1]),
        valunit="proj",
        logger=logger,
        persist=True,
        unit_is_bytes=True,
    ) as pbar:
        # I don't want the progress bar to increment until a piece of work is actually finished by the GPU,
        # not merely queued on the GPU. For this reason, I will add Events to the CUDA stream and this function
        # will check which events have completed, and how many bytes to add to the progress bar.
        prog_evs_bytes: list[tuple[cp.cuda.Event, float]] = []

        def update_pbar(pbar: logtools.ProgressBar) -> None:
            # Loop backwards through list so we can safely delete things as we go
            for i in reversed(range(len(prog_evs_bytes))):
                ev, advance = prog_evs_bytes[i]
                if ev.done:
                    pbar.advance(advance)
                    del prog_evs_bytes[i]

        while (curr_slice := next_slice_zindex(curr_slice)) is not None:
            # Log information about which volume chunk is being worked on
            msg = f"Computing chunk with slice: {curr_slice}; maximum index {volshape[2]:d}"
            logger.debug(msg)

            # Loop through each volume chunk.
            sl_vol, vol = make_vols(curr_slice)
            if vol is None:
                msg = "Something has gone wrong. The script has a bug."
                logger.error(msg)
                return

            # Create the ProjectionSet objects
            (
                _pg,
                sl_filtps_gen,
                R,  # ruff: ignore[non-lowercase-variable-in-function]
                L,  # ruff: ignore[non-lowercase-variable-in-function]
                recon_support_radius,
                recon_support_height,
                cyl_mask_rad,
                final_multiplier,
                tot_num_nonskipped_projs,
                zs_min,
                zs_max,
            ) = get_filtered_projectionset_generator(
                projdata_path=args.projs,
                pg=pg_raw,
                fan_beam_mode=args.fan_beam,
                num_rot_angs=args.num_rot_angs,
                limit_zeta=args.limit_zeta,
                slice_projs=args.slice_projs,
                autorange_z=args.autorange_z,
                flip_x=args.flip_x,
                flip_y=args.flip_y,
                flip_z=args.flip_z,
                rotate_xy=args.rotate_xy,
                logger=logger,
                two_sided=two_sided,
                # These two are supplied to prune irrelevant projections
                vol_minxyz_maxxyz=vol.minxyz_maxxyz,
                cyl_mask_rad=args.cyl_mask_rad,
                seed_proj_rots=seed_proj_rots,
            )

            # Initialise the configuration used for backprojection
            config_bp_ramp_onesided = cbfp.ConfigBackproject(
                fast_approx=True,
                cyl_mask_rad=cyl_mask_rad,
                Z_on_val_factor_power=(-2 if args.fan_beam else -3),
                Z_on_val_factor_val=R,
                backproj_weighting_preset=0,
                approx_mode_direct_raycast=True,
                approx_mode_bilinear_interpolate=args.interpolate_measurements,
                ssx=args.ssx,
                ssy=args.ssy,
                ssz=args.ssz,
            )
            # Configurations to backproject the ramp- and Hilbert-filtered data
            # Combine one of these preset pairs:
            #  - (1, 2) for a formula derived from Kudo's result
            #  - (3, 4) for a formula derived by F. Noo.
            #  - (1, 6) for a formula derived from a result by F. Noo.
            config_bp_ramp_twosided = cbfp.ConfigBackproject(
                fast_approx=True,
                cyl_mask_rad=cyl_mask_rad,
                Z_on_val_factor_power=(0 if args.fan_beam else -1),
                Z_on_val_factor_val=R,
                backproj_weighting_preset=1,  # 1,#3,
                approx_mode_direct_raycast=True,
                approx_mode_bilinear_interpolate=args.interpolate_measurements,
                ssx=args.ssx,
                ssy=args.ssy,
                ssz=args.ssz,
            )
            config_bp_hilb_twosided = cbfp.ConfigBackproject(
                fast_approx=True,
                cyl_mask_rad=cyl_mask_rad,
                Z_on_val_factor_power=(0 if args.fan_beam else -1),
                Z_on_val_factor_val=R,
                backproj_weighting_preset=6,  # 2,#4,
                approx_mode_direct_raycast=True,
                approx_mode_bilinear_interpolate=args.interpolate_measurements,
                # Note: it probably isn't necessary to supersample the Hilbert transform part, since it's low frequency.
                # I'll do it anyway, for simplicity in describing this method.
                ssx=args.ssx,
                ssy=args.ssy,
                ssz=args.ssz,
            )

            with cp.cuda.Stream(non_blocking=True) as stream:
                for sl, pss in sl_filtps_gen:
                    # pss is None if there was nothing to do for this slice
                    if pss is not None:
                        if two_sided:
                            ps_ramp, ps_hilb = pss
                            cbfp.backproject(
                                config_bp_ramp_twosided,
                                vol,
                                ps_ramp,
                                block=False,
                                logger=logger,
                            )
                            cbfp.backproject(
                                config_bp_hilb_twosided,
                                vol,
                                ps_hilb,
                                block=False,
                                logger=logger,
                            )
                            assert (
                                ps_ramp.data is not None
                            )  # useful assertion for type-checking
                            assert (
                                ps_hilb.data is not None
                            )  # useful assertion for type-checking
                        else:
                            (ps_ramp,) = pss
                            cbfp.backproject(
                                config_bp_ramp_onesided,
                                vol,
                                ps_ramp,
                                block=False,
                            )
                            assert (
                                ps_ramp.data is not None
                            )  # useful assertion for type-checking

                        # ProgressBar stuff
                        ev = cp.cuda.Event()
                        ev.record(stream=stream)
                        prog_evs_bytes.append(
                            (
                                ev,
                                (
                                    (
                                        (sl.stop - sl.start)
                                        / tot_num_nonskipped_projs
                                    )
                                    * (
                                        int(bytes_cumsum[curr_slice.stop - 1])
                                        - int(
                                            ([0, *bytes_cumsum])[
                                                curr_slice.start
                                            ]
                                        )
                                    )
                                ),
                            )
                        )
                    update_pbar(
                        pbar
                    )  # update from this and previous queued work

                # Don't synchronize the ordinary way, because then we get no progress updates
                # stream.synchronize()
                while not stream.done:
                    update_pbar(
                        pbar
                    )  # update from this and previous queued work
                    time.sleep(0.05)
                update_pbar(pbar)

            if tot_num_nonskipped_projs > 0:
                # Scale the answer correctly
                vol.multiply(final_multiplier)

                # Save the backprojections into the arrays
                msg = "Transferring backprojected volume chunk from GPU..."
                logger.debug(msg)
                if vol is not None:
                    # Not efficient. Can't copy from CUDAArray directly to the chunk because it's not C-contiguous.
                    vol.as_numpy(out=vol_np[..., sl_vol], allow_indirect_copy=True)
                    del vol
                    lib.gpu_tools.clear_cache()
            else:
                msg = "No need to transfer volume chunk from GPU. No work performed."
                logger.debug(msg)
                del vol

            # Free some memory
            lib.gpu_tools.clear_cache()
            lib.gpu_tools.clear_pinned_cache()

        pbar.update_to(int(bytes_cumsum[-1]))

    # Save final output artefacts
    if path_output:
        msg = "Saving volume to disk..."
        logger.info(msg)
        save_volume(
            vol=vol_np,
            id_str="recon",
            is_intermediate=False,
            skip_if_exists=True,
            logger=logger,
        )  # the skip_if_exists is there in case memory-mapping was enabled, in which case the volume will already be there on disk.

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
                x = vol_np
                imview.plot_volume(
                    x, plotmin=args.plotmin, plotmax=args.plotmax
                )
                if not args.fan_beam:
                    imview.plot_volume(
                        x.transpose((2, 1, 0)),
                        title="Filtered volume",
                        plotmin=args.plotmin,
                        plotmax=args.plotmax,
                    )
        case PLOTMODE.NEUROGLANCER:
            msg = "Displaying plot..."
            logger.info(msg)
            with ng_tools.ArrayViewer3D(
                vol_np,
                plotmin=args.plotmin,
                plotmax=args.plotmax,
                port=args.web_port,
                voxwid_mm=args.voxwid,
                wait_for_input_before_exit_prompt=None,
            ) as av:
                msg = f"Displaying plot at URL: {av.url} ..."
                logger.info(msg)
                with rich_console.status(
                    msg + " press Enter to quit.", spinner="dots10"
                ):
                    input()
