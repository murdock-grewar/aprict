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

"""Produce projection data on-disk, from a geometry description and a phantom description."""

import enum
import logging
import math
import pathlib
import types

import click
import click_extra as ce
import cloup
import cupy as cp
import numexpr as ne
import numpy as np

from snookerr.lib import click_tools

from .. import FBVolume, imview, logtools, phantoms
from ..cb import flatpanel as cbfp

_progdesc = "Compute flat-panel projection data."


class PHANTOM_PRESET(enum.Enum):  # ruff: ignore[invalid-class-name, undocumented-public-class]
    HEAD = 1
    ROCK_CAN = 2
    UNIT_Z_CYL = 3


class OUTPUT_DTYPE(enum.Enum):  # ruff: ignore[invalid-class-name, undocumented-public-class]
    FP16 = 1
    FP32 = 2
    FP64 = 3


def _ph_preset_to_phantom(preset: PHANTOM_PRESET) -> FBVolume:
    match preset:
        case PHANTOM_PRESET.HEAD:
            v = phantoms.FORBILD.HeadPhantom
            v.scale_len(1 / 10)  # convert from cm to mm
            v.multiply_attenuation(
                0.0184
            )  # convert from multiples of water to a reasonable attenuation, corresponding to an X-ray energy of ~80 keV.
            return v
        case PHANTOM_PRESET.ROCK_CAN:
            return phantoms.FORBILD.RockCanisterAlWithHoles
        case PHANTOM_PRESET.UNIT_Z_CYL:
            return phantoms.FORBILD.UnitCyl_z
        case _:
            raise ValueError()


def _outdtype_to_dtype(
    outdt: OUTPUT_DTYPE,
) -> type[np.float16 | np.float32 | np.float64]:
    match outdt:
        case OUTPUT_DTYPE.FP16:
            return np.float16
        case OUTPUT_DTYPE.FP32:
            return np.float32
        case OUTPUT_DTYPE.FP64:
            return np.float64
        case _:
            raise ValueError()


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
    "mandatory options",
    ce.option(
        "--geom",
        required=True,
        metavar="projgeom.json",
        type=click.Path(path_type=pathlib.Path),
        help="Path to saved ProjGeom object.",
    ),
    ce.option(
        "--output",
        "-o",
        required=True,
        # Note that setting file_okay=False and dir_okay=False causes it to only accept uncreated paths
        type=click.Path(
            path_type=pathlib.Path,
            writable=True,
            file_okay=False,
            dir_okay=False,
        ),
        help="Path at which to create the new artefact directory.",
    ),
)
@ce.option_group(
    "phantom specification",
    ce.option(
        "--phantom",
        metavar="vol.pha",
        type=click.Path(path_type=pathlib.Path),
        help="Path to the FORBILD phantom.",
    ),
    ce.option(
        "--phantom-preset",
        type=click.Choice(PHANTOM_PRESET),
        help="Choose from among a set of preset phantoms.",
    ),
    constraint=cloup.constraints.RequireExactly(1),
)
@ce.option_group(
    "accuracy options",
    ce.option(
        "--ssu",
        type=click.INT,
        default=1,
        help="Supersample along du direction of detector. Compute this many line integrals per pixel. Total number of samples multiplies with --ssv.",
    ),
    ce.option(
        "--ssv",
        type=click.INT,
        default=1,
        help="Supersample along dv direction of detector. Compute this many line integrals per pixel. Total number of samples multiplies with --ssu.",
    ),
)
@ce.option_group(
    "phantom preprocessing",
    ce.option(
        "--multiply-attenuation",
        default=1.0,
        type=click.FLOAT,
        help="Multiplies the phantom's attenuation by this amount.",
    ),
    ce.option(
        "--scale-vol-len",
        type=click.FLOAT,
        default=1.0,
        help="Multiply the length scale used in the volume by this amount. Takes effect BEFORE translations with --translate-vol. E.g. if the volume is recorded with the units of distance as cm, then using --scale-vol-len 10 will change it to mm. The attenuation coefficient will also change, so to preserve the same coefficient, one would also have to supply --multiply-attenuation 0.1.",
    ),
    ce.option(
        "--translate-vol",
        type=click_tools.FloatTupleArgTypeFactory(numfloats=3),
        default=(
            0.0,
            0.0,
            0.0,
        ),
        help="Translate the volume by (x,y,z).",
    ),
    help="Options to preprocess the input phantom. The preprocessed phantom will be saved to the output directory.",
)
@ce.option_group(
    "random seeds",
    ce.option(
        "--seed-noise",
        type=click_tools.MaybeType(click.INT),
        help="Seed to use for generating Poisson noise.",
    ),
    help="Seeds to use for random number generation. Leave an option as None to use a pseudorandom seed.",
)
@ce.option_group(
    "noise",
    ce.option(
        "--cfc-pixel",
        type=click_tools.MaybeType(click.FLOAT),
        default=None,
        help="Apply noise, with this many clearfield photons per pixel.",
    ),
    constraint=cloup.constraints.AcceptAtMost(1),
)
@ce.option_group(
    "I/O",
    ce.option(
        "--verbosity",
        type=click.Choice(logtools.LogLevel),
        default=logtools.LogLevel.INFO,
        help="The verbosity of output logging to stderr. Log files will not be affected by this setting.",
    ),
    ce.option(
        "--out-dtype",
        type=click.Choice(OUTPUT_DTYPE),
        default=OUTPUT_DTYPE.FP64,
        help="The dtype of the output.",
    ),
    ce.option(
        "--plot-afterwards",
        type=click_tools.BoolArgType(),
        default=False,
        help="Attempt to display a plot of the data after it has been generated.",
    ),
)
@ce.option_group(
    "strategy",
    ce.option(
        "--chunk-size",
        default=10,
        type=click.INT,
        help="How many projections to compute in each step. This number is only a suggestion; it will be rounded up to the nearest usable value.",
    ),
    help="Options affecting the implementation, but not the output. E.g. memory management options.",
)
def main(**kwargs) -> None:  # ruff: ignore[missing-type-kwargs]
    """Parse input arguments using click."""
    args = types.SimpleNamespace(**kwargs)
    main_from_args(args)


def main_from_args(args: types.SimpleNamespace) -> None:  # ruff: ignore[complex-structure, undocumented-public-function]

    artefact_path = args.output
    artefact_path.mkdir(exist_ok=False)

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

    ##########################################################
    ###################### Preparation #######################
    ##########################################################

    # Get the output dtype
    proj_dtype = _outdtype_to_dtype(args.out_dtype)

    # Load the geometry
    logger.debug("Loading projection geometry...")
    pg: cbfp.ProjGeom = cbfp.ProjGeom.load(args.geom)

    # Load the phantom
    logger.debug("Loading the phantom...")
    if args.phantom_preset is not None:
        v = _ph_preset_to_phantom(args.phantom_preset)
    else:
        v = FBVolume.load(args.phantom)
    logger.debug("Preprocessing the phantom...")
    v_org = v.copy()
    v = _preprocess_phantom(v, args)

    # Validate args
    # validate_args(args)
    if args.ssu < 1 or args.ssv < 1:
        msg = "--ssu and --ssv must be positive integers."
        raise ValueError(msg)

    ##########################################################
    ############### Preparation for projection ###############
    ##########################################################
    logger.debug(f"ProjGeom projs_list_shape: {pg.projs_list_shape}")
    num_projs: int = math.prod(pg.projs_list_shape)
    logger.info(f"Total number of projections: {num_projs:d}")
    ## Call the projection code on the geometry
    projs_per_unit: int = math.prod(
        pg.projs_list_shape[:-1]
    )  # gives 1 for a flat list
    num_units: int = pg.projs_list_shape[-1]
    units_per_chunk: int = min(
        num_units,
        max(1, (args.chunk_size + (projs_per_unit - 1)) // projs_per_unit),
    )
    config_project = cbfp.ConfigProject(ssu=args.ssu, ssv=args.ssv)

    rng: cp.random.Generator = cp.random.default_rng(args.seed_noise)

    def apply_noise_in_place(atten_arr: cp.ndarray) -> None:
        # Make sure to preserve the dtype of arr
        if args.cfc_pixel is not None:
            cp.multiply(atten_arr, -1, out=atten_arr)  # -mu
            cp.exp(atten_arr, out=atten_arr)  # exp(-mu)
            cp.multiply(
                atten_arr, args.cfc_pixel, out=atten_arr
            )  # cfc_pixel * exp(-mu)
            measured_photons = rng.poisson(atten_arr)  # measured_photons
            cp.true_divide(
                measured_photons, args.cfc_pixel, out=atten_arr
            )  # Transmittance = exp(-mu_measured)
            cp.log(atten_arr, out=atten_arr)  # -mu_measured
            cp.multiply(atten_arr, -1, out=atten_arr)  # mu_measured
        else:  # check for other cfc args, if implemented...
            pass

    # At present, only a uniform choice for nu and nv is supported by this script. They should each be integers, the same across all projections.
    nu = pg.nu.flatten()[0]
    nv = pg.nv.flatten()[0]
    if not np.all(pg.nu == nu) or not np.all(pg.nv == nv):
        msg = "The input projection geometry has a complex structure for nu or nv. This is currently not supported by this script, because it is not possible to write the result to a single array, since the last and/or second-to-last axis will not have a uniform length across the array. In the future, this may be resolved by allowing the script to write multiple arrays."
        raise NotImplementedError(msg)
    if len(pg.projs_list_shape) != 1:
        msg = "The projection list is non-flat. At present, slicing of projection geometries for non-flat lists is not implemented. This script relies on slicing the geometry, so the required functionality is not yet implemented. Only a flat-list geometry is supported."
        raise NotImplementedError(msg)
    logger.info(f"Detector shape: ({nu:d},{nv:d})")

    ##########################################################
    ################### Saving input data ####################
    ##########################################################
    for v_, tag in zip([v_org, v], ["original", "preprocessed"]):
        s: str = "" if tag == "original" else f"_{tag:s}"
        vol_path_npy: pathlib.Path = artefact_path.joinpath(
            "vol" + s + "_fbv_shapes.npy"
        )
        np.save(vol_path_npy, v_.shapes)
        logger.info(
            f"Saved {tag:s} volume shapes array at: '{vol_path_npy.name}'"
        )
        vol_path_pha: pathlib.Path = artefact_path.joinpath("vol" + s + ".pha")
        v_.save(vol_path_pha)
        logger.info(f"Saved {tag:s} volume .pha at: '{vol_path_pha.name}'")
        logger.info(
            "(In the event of a conflict, the *.npy file is authoritative; the *.pha is meant to be an exact copy, but there is always a possibility for a software bug.)"
        )

    geom_path: pathlib.Path = artefact_path.joinpath("projgeom.json")
    pg.save(geom_path)
    logger.info(f"Saved projection geometry at: '{geom_path.name}'")

    ##########################################################
    #################### Begin projection ####################
    ##########################################################
    logger.info("")
    logger.info("Allocating the projection array and filling with zeros...")
    output_projs = np.lib.format.open_memmap(
        args.output.joinpath("projs.npy"),
        "w+",
        dtype=proj_dtype,
        shape=tuple(map(int, (*pg.projs_list_shape, nu, nv))),
    )
    output_projs[...] = 0
    logger.info("Done with allocating projection array.")
    logger.info("Beginning projection...")
    with logtools.ProgressBar(
        desc="Projection:",
        maxval=num_projs,
        valunit="proj",
        logger=logger,
        persist=False,
    ) as pbar:
        for chunk_index in range(
            (num_units + units_per_chunk - 1) // units_per_chunk
        ):
            # We will compute output_projs[sls], with sls as this:
            sls = (slice(None, None),) * len(pg.projs_list_shape[:-1]) + (
                slice(
                    chunk_index * units_per_chunk,
                    min(num_units, (chunk_index + 1) * units_per_chunk),
                ),
            )
            # ARBITRARY SHAPES NOT IMPLEMENTED!
            # pg_sliced = pg.sliced(sls)
            # The only implemented case is where 'sls' above is a 1-tuple containing the necessary slice.
            pg_sl: cbfp.ProjGeom = pg.sliced(sls[-1])
            # NOTE: not optimized!
            # - the below is likely to be unoptimized, because we are creating a new buffer every time.
            # - we are using block=True
            projs = cbfp.ProjectionSet(pg_sl, data=None)
            projs.create_buffer(dtype=proj_dtype)
            _ = cbfp.project(
                config_project,
                v,  # the preprocessed volume, which is not the same as v_org which is the original volume before transformations!
                projs,
                block=True,
            )
            assert (
                projs.data is not None
            )  # helpful assertion for type-checking
            pout = output_projs[*sls]
            apply_noise_in_place(projs.data)
            pin = projs.data.get()
            ne.evaluate("pin", out=pout, local_dict={"pin": pin})

            # Perform logging updates
            pbar.update_to(sls[-1].stop * projs_per_unit)
    logger.info("Done.")
    if args.plot_afterwards:
        logger.info("Displaying plot...")
        # Be clever: if the data has the shape of fan-beam data, then plot the sinogram instead.
        if len(output_projs.shape) == 3 and output_projs.shape[2] == 1:
            imview.plot_projections(
                output_projs.copy()[np.newaxis, :, :, 0].transpose((0, 2, 1)),
                title="Sinogram",
            )
        else:
            imview.plot_projections(output_projs.copy())


def _preprocess_phantom(v: FBVolume, args: types.SimpleNamespace) -> FBVolume:
    v.scale_len(args.scale_vol_len)
    v.multiply_attenuation(args.multiply_attenuation)
    v.translate(axis="x", amount=args.translate_vol[0])
    v.translate(axis="y", amount=args.translate_vol[1])
    v.translate(axis="z", amount=args.translate_vol[2])
    return v
