
import argparse

# Import numexpr with the appropriate number of threads
import os
import pathlib

import cupy as cp
import numexpr as ne
import numpy as np

# Try to load some modules
try:
    from .. import image_view as imview
    mod_avail_imview: bool = True
except ModuleNotFoundError:
    mod_avail_imview: bool = False
try:
    from .. import liveplot
    mod_avail_liveplot: bool = True
except ModuleNotFoundError:
    mod_avail_liveplot: bool = False

from .. import Volume, lib
from ..cb import flatpanel as cbfp
from ..iterators import Landweber_flatpanel

_progdesc = "Perform an iterative Landweber reconstruction from flat-panel cone-beam transmission data."


def _make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_progdesc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('projgeom', metavar="proj_geom.json", type=pathlib.Path, help="Path to saved ProjGeom object.")
    parser.add_argument('projdata', metavar="proj_data.npy", type=pathlib.Path, help="Path to the numpy array containing the transmission data corresponding to the ProjGeom object.")

    group_mandatory_options = parser.add_argument_group(title="required options")
    group_mandatory_options.add_argument('--voxwid', metavar="1.0", type=float, required=True, help="The cubic voxel width, in the same distance units as the projection data is specified in proj_geom.json", default=argparse.SUPPRESS)
    group_mandatory_options.add_argument('--volshape', metavar="'(nX,nY,nZ)'", type=lib.argparse_tools.IntTupleArgTypeFactory(numints=3), help="Dimensions of the voxel grid on which to reconstruct", required=True, default=argparse.SUPPRESS)

    group_geometry_options = parser.add_argument_group(title="geometry options")
    group_geometry_options.add_argument('--volpos', metavar="'(0.,0.,0.)'", type=lib.argparse_tools.FloatTupleArgTypeFactory(numfloats=3), help="Position of the middle of the volume.", required=False, default=(0., 0., 0.))

    group_data_preprocessing = parser.add_argument_group(title="data preprocessing")
    group_data_preprocessing.add_argument('--autorange-z', metavar="False", type=lib.argparse_tools.BoolArgType, default=False, help="Autorange the z coordinates of the trajectory, so that the highest source point and the lowest source point are +-z0. This autoranging takes place before slicing of projection data.")
    group_data_preprocessing.add_argument('--slice-projs', metavar="'(start, stop)'", type=lib.argparse_tools.ContiguousSliceArgType, default=slice(None, None), help="Slice the projections to include those of index start to (stop-1), inclusive.")
    # group_data_preprocessing.add_argument('--rot-xy', metavar="NUMRADIANS", type=lib.argparse_tools.MaybeType(float), default=None, help="Rotate the projection geometry about the z axis by this many radians.")
    group_data_preprocessing.add_argument('--flip-x', metavar="False", type=lib.argparse_tools.BoolArgType, default=False, help="Flip the projection geometry about the xz plane.")
    group_data_preprocessing.add_argument('--flip-y', metavar="False", type=lib.argparse_tools.BoolArgType, default=False, help="Flip the projection geometry about the yz plane.")
    group_data_preprocessing.add_argument('--flip-z', metavar="False", type=lib.argparse_tools.BoolArgType, default=False, help="Flip the projection geometry about the xy plane.")

    group_landweber_options = parser.add_argument_group(title="landweber options")
    group_landweber_options.add_argument('--numits', metavar="10", type=lib.argparse_tools.MaybeType(int), default=None, required=False, help="The number of iterations to use. None = iterate indefinitely, until a SIGINT is sent.")
    group_landweber_options.add_argument('--relaxation-factor', '--rf', metavar="1.0", type=float, required=False, default=1.0, help="The relaxation factor to add to the correction terms. Should be in the open interval (0,2)")
    group_landweber_options.add_argument('--numits-for-rayleigh-quotient', '--rqits', metavar="10", type=int, required=False, default=10, help="Number of iterations to spend on computing the Rayleigh quotient")
    group_landweber_options.add_argument('--os-size', metavar="42", type=lib.argparse_tools.MaybeType(int), required=False, default=None, help="Split the projection dataset into ordered subsets of this many cone-beam projections each. (Be careful: if there are leftover projections at the end, these will be lumped into an ordered subset of smaller size than specified.)")
    group_landweber_options.add_argument('--exact-adjoint', metavar="True", type=lib.argparse_tools.BoolArgType, default=True, help="Whether to use the exact adjoint. (If False, then a much faster approximate adjoint is used.)")

    group_cyl_options = parser.add_argument_group(title="cylindrical trajectory options", description="These options are specific to the cylindrical trajectory. They will not make sense as options for arbitrary geometries.")
    group_cyl_options.add_argument('--cyl-confine-proj-lines-z', metavar="False", type=lib.argparse_tools.BoolArgType, default=False, help="A special option to mask out projection lines whose intersection with the solid cylinder is not completely contained in the reconstruction volume. This option assumes that the reconstruction volume has x,y = 0 at its centre.", required=False)

    return parser


def main() -> None:
    """Run the script."""
    # Set numexpr to have the appropriate number of threads
    # This should have already worked at import time, but it doesn't for some reason
    if 'NUMEXPR_MAX_THREADS' in os.environ:
        ne.set_num_threads(min(2*os.cpu_count(), int(os.environ['NUMEXPR_MAX_THREADS'])))  # ty:ignore[unsupported-operator]
    else:
        ne.set_num_threads(2*os.cpu_count())  # ty:ignore[unsupported-operator]

    # Parse the input arguments
    parser = _make_arg_parser()
    args = parser.parse_args()

    # Create the ProjectionSet objects corresponding to each ordered subset
    pg: cbfp.ProjGeom = cbfp.ProjGeom.load(args.projgeom)
    if args.autorange_z:
        pg = pg.translated(x=0, y=0, z=-(pg.source_pos[:, 2].max() + pg.source_pos[:, 2].min())/2)
    pg = pg.sliced(args.slice_projs)
    # if args.rot_xy is not None:
    #     pg = pg.rotated(xy_rads=args.rot_xy)
    if args.flip_x or args.flip_y or args.flip_z:
        pg = pg.flipped(flip_x=args.flip_x, flip_y=args.flip_y, flip_z=args.flip_z)
    data_mm = np.lib.format.open_memmap(args.projdata, mode='r')[args.slice_projs]  # Note: a contiguous slice is still a memmap object
    ps_lst: list[cbfp.ProjectionSet] = []
    os_size = args.os_size or pg.num_projs  # Null-coalesce 'None' to be pg.num_projs
    sls: list[slice] = []
    sl_start: int = 0
    while sl_start < pg.num_projs:
        sl = slice(sl_start, sl_start + os_size)
        sls.append(sl)
        ps_lst.append(cbfp.ProjectionSet(pg.sliced(sl), data=data_mm[sl]))
        sl_start += os_size

    # Make masks to zero out projection lines which do not intersect the cylinder twice
    masks = None
    if args.cyl_confine_proj_lines_z:
        masks = []
        # Step 1: compute the lateral chord length based on lateral position of detector pixel
        # Step 2: compute the dz from source to intersection with cylinder, as a 2D array
        # Step 3: compute mask as min_z <= (source_z + dz_2D) <= max_z
        pixel_wid = np.mean(np.linalg.norm(pg.det_uvec, axis=-1))
        pixel_hei = np.mean(np.linalg.norm(pg.det_vvec, axis=-1))
        det_numpixwid = pg.nu[0]
        det_numpixhei = pg.nv[0]
        R = np.mean(np.linalg.norm(pg.source_pos, axis=-1))  # noqa: N806
        L = np.mean(np.linalg.norm(pg.source_pos - pg.det_pos, axis=-1))  # noqa: N806
        det_y = pixel_wid * (np.arange(det_numpixwid, dtype=np.float32) - ((det_numpixwid-1)/2))
        det_z = pixel_hei * (np.arange(det_numpixhei, dtype=np.float32) - ((det_numpixhei-1)/2))
        cl = 2*R / np.sqrt(1 + np.square(det_y / L))
        lateral_length_source_to_detector = np.sqrt(np.square(L) + np.square(det_y))
        dz = (
            (cl/lateral_length_source_to_detector)[:, np.newaxis]
            * det_z[np.newaxis, :]
        )
        zs = pg.source_pos[..., 2]
        zs_mask_min = -args.volshape[2]/2*args.voxwid + args.volpos[2]
        zs_mask_max =  args.volshape[2]/2*args.voxwid + args.volpos[2]
        for sl in sls:
            z_of_cyl_intersection = zs[sl][:, np.newaxis, np.newaxis] + dz[np.newaxis, :, :]
            z_ = zs[sl][:, np.newaxis, np.newaxis]
            mask = np.logical_and(zs_mask_min <= z_of_cyl_intersection, z_of_cyl_intersection <= zs_mask_max)
            mask2 = np.logical_and(zs_mask_min <= z_, z_ <= zs_mask_max)
            mask = np.logical_and(mask, mask2)
            mask_cp = cp.array(mask)
            masks.append(mask_cp)

    # Create the working volume
    vol = Volume.zeros(args.volshape, voxwid=args.voxwid, dtype=np.float32, pos_c=args.volpos)  # ty:ignore[invalid-argument-type]

    # Create the Landweber iteration object
    recon_config = Landweber_flatpanel.Config(
        rayleigh_quotient_num_iterations=args.numits_for_rayleigh_quotient,
        acceleration_factor=args.relaxation_factor,
        fast_backproject=not args.exact_adjoint,
        soft_from_cylinder_relrad=1,  # Don't soften
        bp_soft_from_detedge_relrad=1,  # Don't soften
    )
    print("Initializing Landweber iterator...")
    recon = Landweber_flatpanel(
        config=recon_config,
        volume=vol,
        projs=ps_lst,
        proj_zero_masks=masks,
    )

    numits: int = args.numits or 1
    print("Beginning iterations...")
    for itnum in range(1, 1+numits):
        recon.iterate()
        print(f"Completed iteration {itnum:04d}/{numits:04d}")
    print("Done.")
    x = recon.volume.as_numpy()
    if mod_avail_imview:
        imview.plot_volume(x[x.shape[0]//2-1: x.shape[0]//2+2, ...].transpose((1, 2, 0)).copy())
        imview.plot_volume(x[:, :, x.shape[2]//2-1: x.shape[2]//2+2, ...].copy())
