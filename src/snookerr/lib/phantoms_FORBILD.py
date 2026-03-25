# FORBILD_head.py
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

# This file includes extracts from the LEAP library made available at https://github.com/llnl/LEAP, released under the MIT license, copyright 2022-2023

# The only intended public function of this file is 'FORBILD_HU'.

import dataclasses
import enum
import typing

import numexpr as ne
import numpy as np
from scipy.spatial.transform import Rotation as R


class GeomObjectType(enum.Enum):
    ELLIPSOID     = 0
    PARALLELPIPED = 1
    CYLINDER_X    = 2
    CYLINDER_Y    = 3
    CYLINDER_Z    = 4
    CONE_X        = 5
    CONE_Y        = 6
    CONE_Z        = 7

@dataclasses.dataclass(kw_only=True)
class GeomObject:
    t:  GeomObjectType
    c:  np.ndarray # centre of object, (3,) array
    r:  np.ndarray # x,y,z radii of object, (3,) array
    val: float     # attenuation coefficient of this object
    A:  typing.Optional[np.ndarray] # inverse of rotation matrix to rotate the object about its centre, (3,3) array
    clip: typing.Optional[np.ndarray] # specifies the 'clipping planes', if any, (3,) array
    def sample(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, out_mask: typing.Optional[np.ndarray] = None) -> typing.Tuple[float, np.ndarray, typing.Tuple[int,int,int]]:
        """Output: value, out_mask, indices_offset"""

        # Initiate as self.val, then set to nan outside the object
        out_shape = (len(xs), len(ys), len(zs))
        if out_mask is None:
            out_mask = np.ones(out_shape, dtype=np.bool_)
        else:
            if out_mask.shape != out_shape:
                raise ValueError("Wrong shape for out_mask.")
            ne.evaluate('out_mask | True', out=out_mask)

        c1, c2, c3 = self.c[0], self.c[1], self.c[2]

        # Start by making a Cartesian pruning box, so we don't have to process the entire volume
        obj_rad = np.amax(self.r)
        crop_index_offset = [0,0,0]
        # Crop x
        w = np.argwhere(np.logical_and(xs >= c1 - obj_rad, xs <= c1 + obj_rad))
        if w.size == 0:
            return np.zeros((0,0,0), dtype=np.float32), np.zeros((0,0,0), dtype=np.bool_), (0,0,0)  # ty:ignore[invalid-return-type]
        xs_cropped = xs[w[0,0]:1+w[-1,0]]
        crop_index_offset[0] = w[0,0]
        # Crop y
        w = np.argwhere(np.logical_and(ys >= c2 - obj_rad, ys <= c2 + obj_rad))
        if w.size == 0:
            return np.zeros((0,0,0), dtype=np.float32), np.zeros((0,0,0), dtype=np.bool_), (0,0,0)  # ty:ignore[invalid-return-type]
        ys_cropped = ys[w[0,0]:1+w[-1,0]]
        crop_index_offset[1] = w[0,0]
        # Crop z
        w = np.argwhere(np.logical_and(zs >= c3 - obj_rad, zs <= c3 + obj_rad))
        if w.size == 0:
            return np.zeros((0,0,0), dtype=np.float32), np.zeros((0,0,0), dtype=np.bool_), (0,0,0)  # ty:ignore[invalid-return-type]
        zs_cropped = zs[w[0,0]:1+w[-1,0]]
        crop_index_offset[2] = w[0,0]
        # Record the offsets
        crop_index_offset = tuple(crop_index_offset)
        crop_shape = (len(xs_cropped), len(ys_cropped), len(zs_cropped))

        # Take views of 'out' and 'out_mask'
        out_mask = out_mask[
            crop_index_offset[0]:crop_index_offset[0]+crop_shape[0],
            crop_index_offset[1]:crop_index_offset[1]+crop_shape[1],
            crop_index_offset[2]:crop_index_offset[2]+crop_shape[2]
        ]

        # Make broadcastable coordinate arrays
        X = xs_cropped[:,np.newaxis,np.newaxis].copy()
        Y = ys_cropped[np.newaxis,:,np.newaxis].copy()
        Z = zs_cropped[np.newaxis,np.newaxis,:].copy()

        # Set to zero if there are clipping planes
        if self.clip is not None:
            if c1 != 0:
                ne.evaluate('out_mask & (abs(X) <= abs(c1))', out=out_mask)
            if c2 != 0:
                ne.evaluate('out_mask & (abs(Y) <= abs(c2))', out=out_mask)
            if c3 != 0:
                ne.evaluate('out_mask & (abs(Z) <= abs(c3))', out=out_mask)

        # Transform vector to coordinates of the object
        ne.evaluate('X - c1', out=X)
        ne.evaluate('Y - c2', out=Y)
        ne.evaluate('Z - c3', out=Z)

        # Set coefficients of rotation matrix
        if self.A is None:
            A11, A12, A13 = 1, 0, 0
            A21, A22, A23 = 0, 1, 0
            A31, A32, A33 = 0, 0, 1
        else:
            A11, A12, A13 = self.A[0,:]
            A21, A22, A23 = self.A[1,:]
            A31, A32, A33 = self.A[2,:]

        # Check if point is within the object
        match self.t:
            case GeomObjectType.ELLIPSOID:
                rx, ry, rz = self.r[0], self.r[1], self.r[2]
                ne.evaluate("""out_mask & ((
                    ((A11*X + A12*Y + A13*Z)/rx)**2
                    + ((A21*X + A22*Y + A23*Z)/ry)**2
                    + ((A31*X + A32*Y + A33*Z)/rz)**2
                    ) < 1)""", out=out_mask)
            case GeomObjectType.PARALLELPIPED:
                raise NotImplementedError()
            case GeomObjectType.CYLINDER_X:
                rx, ry, rz = self.r[0], self.r[1], self.r[2]
                ne.evaluate("""out_mask & ((
                    ((A21*X + A22*Y + A23*Z)/ry)**2
                    + ((A31*X + A32*Y + A33*Z)/rz)**2
                    ) < 1)""", out=out_mask)
                ne.evaluate("""out_mask & (abs(A11*X + A12*Y + A13*Z) < rx)""", out=out_mask)
            case GeomObjectType.CYLINDER_Y:
                rx, ry, rz = self.r[0], self.r[1], self.r[2]
                ne.evaluate("""out_mask & ((
                    ((A11*X + A12*Y + A13*Z)/rx)**2
                    + ((A31*X + A32*Y + A33*Z)/rz)**2
                    ) < 1)""", out=out_mask)
                ne.evaluate("""out_mask & (abs(A21*X + A22*Y + A23*Z) < ry)""", out=out_mask)
            case GeomObjectType.CYLINDER_Z:
                rx, ry, rz = self.r[0], self.r[1], self.r[2]  # noqa: F841
                ne.evaluate("""out_mask & ((
                    ((A11*X + A12*Y + A13*Z)/rx)**2
                    + ((A21*X + A22*Y + A23*Z)/ry)**2
                    ) < 1)""", out=out_mask)
                ne.evaluate("""out_mask & (abs(A31*X + A32*Y + A33*Z) < rz)""", out=out_mask)
            case GeomObjectType.CONE_X:
                r1, l, r2 = self.r[0], self.r[1], self.r[2]  # noqa: E741
                ne.evaluate("""out_mask & (sqrt(
                    ((A21*X + A22*Y + A23*Z))**2
                    + ((A31*X + A32*Y + A33*Z))**2
                    ) < (
                        (r1+r2)/2 - (A11*X + A12*Y + A13*Z)/l*(r1-r2)/2
                    ))""", out=out_mask)
                ne.evaluate("""out_mask & (abs(A11*X + A12*Y + A13*Z) < l)""", out=out_mask)
            # NOTE ON the cone_Y:
            # (Similar for cone_X and cone_Z)
            # rx, ry, rz actually corresponds to [r1, l, r2], where "...truncated cone of length l, radius r1 at the end with smaller y, radius r2 at the end with larger y, axis parallel to the y axis. x, y and z define the center of the truncated cone axis."
            case GeomObjectType.CONE_Y:
                r1, l, r2 = self.r[0], self.r[1], self.r[2]  # noqa: E741
                ne.evaluate("""out_mask & (sqrt(
                    ((A11*X + A12*Y + A13*Z))**2
                    + ((A31*X + A32*Y + A33*Z))**2
                    ) < (
                        (r1 + r2)/2 - (A21*X + A22*Y + A23*Z)/l*(r1-r2)/2
                    ))""", out=out_mask)
                ne.evaluate("""out_mask & (abs(A21*X + A22*Y + A23*Z) < l)""", out=out_mask)
            case GeomObjectType.CONE_Z:
                r1, l, r2 = self.r[0], self.r[1], self.r[2]  # noqa: E741, F841
                ne.evaluate("""out_mask & (sqrt(
                    ((A11*X + A12*Y + A13*Z))**2
                    + ((A21*X + A22*Y + A23*Z))**2
                    ) < (
                        (r1+r2)/2 - (A31*X + A32*Y + A33*Z)/l*(r1-r2)/2
                    ))""", out=out_mask)
                ne.evaluate("""out_mask & (abs(A31*X + A32*Y + A33*Z) < l)""", out=out_mask)
            case _:
                raise NotImplementedError()
        val = np.float32(self.val)
        return val, out_mask, crop_index_offset  # ty:ignore[invalid-return-type]

class CompositeGeomObject:

    geom_objects: typing.List[GeomObject]

    def __init__(self):
        self.geom_objects = []

    def sample(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        out = np.zeros((len(xs), len(ys), len(zs)), dtype=np.float32)
        set_mask = np.zeros(out.shape, dtype=np.bool_)
        for obj in self.geom_objects:
            val_, out_mask_, crop_index_offset = obj.sample(xs, ys, zs, out_mask=set_mask)
            if out_mask_.size == 0:
                continue
            out_view = out[
                crop_index_offset[0]:crop_index_offset[0]+out_mask_.shape[0],
                crop_index_offset[1]:crop_index_offset[1]+out_mask_.shape[1],
                crop_index_offset[2]:crop_index_offset[2]+out_mask_.shape[2]
            ]
            ne.evaluate('where(out_mask_, val_, out_view)', out=out_view)
        return out

    def add_object(self, GeomObject):
        self.geom_objects.append(GeomObject)

    def addObject(self, f, typeOfObject, c, r, val, A=None, clip=None, oversampling=1):
        """Adds a geometric object to the phantom

        This function operates in two modes: (1) specifying a voxelized phantom and (2) specifying a phantom
        to be used in an analytic ray-tracing simulation (see rayTrace).
        If a volume is given (first argument of this function), then the specified object will be added to the voxel data.
        If a volume is not given then the object will be added to a stack of objects that define a phantom to be used for
        an analytic ray-tracing simulation.

        The order in which multiple object are defined is important.  Background objects must be specified first
        and foreground objects defined last.  If you reverse the order, then the foreground objects will be effectively overriden
        by the background objects.

        The CT volume or CT geometry parameters must be specified prior to running this functions.

        Args:
            f (C contiguous float32 numpy array): CT volume
            type (int): ELLIPSOID=0, PARALLELEPIPED=1, CYLINDER_X=2, CYLINDER_Y=3, CYLINDER_Z=4, CONE_X=5, CONE_Y=6, CONE_Z=7
            c (3X1 numpy array): x,y,z coordinates of the center of the object
            r (3X1 numpy array): radii in the x,y,z directions of the object
            val (float): the values to ascribe inside this object
            A (3x3 numpy array): rotation matrix to rotate the object about its center
            clip (3X1 numpy array): specifies the clipping planes, if any
        """

        if isinstance(c, int) or isinstance(c, float):
            c = np.array([c, c, c], dtype=np.float32)
        if isinstance(r, int) or isinstance(r, float):
            r = np.array([r, r, r], dtype=np.float32)

        if clip is not None:
            pass
            #print("clip is not None!")

        self.add_object(GeomObject(t=GeomObjectType(typeOfObject), c=c, r=r, val=val, A=A, clip=clip))

    def set_FORBILD(self, f=None, includeEar=True, oversampling=1):
        """Sets the FORBILD head phantom

        This function operates in two modes: (1) specifying a voxelized phantom and (2) specifying a phantom
        to be used in an analytic ray-tracing simulation (see rayTrace).
        If a volume is given (first argument of this function), then the specified object will be added to the voxel data.
        If a volume is not given then all the objects of the FORBILD head phantom will be added to a stack of objects
        that define a phantom to be used for an analytic ray-tracing simulation.

        Note that the values of the FORBILD head phantom are all scaled by 0.02
        which is the LAC of water at around 60 keV.  The FOV is about [-96, 96, -120, 120, -125, 125]

        Args:
            f (C contiguous float32 numpy array): CT volume
            includeEar (boolean): specifies whether the air bubbles in the ear are to be included or not
            oversampling (int): the oversampling factor of the voxelization

        """

        # Note: all the distances are in millimeters

        self.geom_objects = []

        self.addObject(f, 0, 10.0*np.array([0.0, 0.0, 0.0]), 10.0*np.array([9.6, 12.0, 12.5]), 1.800*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([0.0, 0.0, 0.0]), 10.0*np.array([9.0, 11.4, 11.9]), 1.050*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([-4.7, 4.3, 0.872]), 10.0*np.array([2.0, 2.0, 2.0]), 1.060*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([4.7, 4.3, 0.872]), 10.0*np.array([2.0, 2.0, 2.0]), 1.060*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([-1.08, -9, 0.0]), 10.0*np.array([0.4, 0.4, 0.4]), 1.0525*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([1.08, -9, 0.0]), 10.0*np.array([0.4, 0.4, 0.4]), 1.0475*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([0.0, 8.4, 0.0]), 10.0*np.array([1.8, 3.0, 3.0]), 0.0, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([-1.9, 5.4, 0.0]), 10.0*np.array([1.206483*np.cos(15*np.pi/180.0), 0.420276*np.cos(15*np.pi/180.0), 3.0]), 1.800*0.02, R.from_euler("xyz", [0, 0, -120], degrees=True).as_matrix(), None, oversampling)
        self.addObject(f, 0, 10.0*np.array([1.9, 5.4, 0.0]), 10.0*np.array([1.2*np.cos(15*np.pi/180.0), 0.42*np.cos(15*np.pi/180.0), 3.0]), 1.800*0.02, R.from_euler("xyz", [0, 0, 120], degrees=True).as_matrix(), None, oversampling)
        self.addObject(f, 4, 10.0*np.array([-4.3, 6.8, -1.0]), 10.0*np.array([1.8, 0.24, 2.0]), 1.800*0.02, R.from_euler("xyz", [0, 0, -150], degrees=True).as_matrix(), None, oversampling)
        self.addObject(f, 4, 10.0*np.array([4.3, 6.8, -1.0]), 10.0*np.array([1.8, 0.24, 2.0]), 1.800*0.02, R.from_euler("xyz", [0, 0, -30], degrees=True).as_matrix(), None, oversampling)
        self.addObject(f, 0, 10.0*np.array([0.0, -3.6, 0.0]), 10.0*np.array([1.8, 3.6, 3.6]), 1.045*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([6.393945, -6.393945, 0.0]), 10.0*np.array([1.2, 0.42, 1.2]), 1.055*0.02, R.from_euler("xyz", [0, 0, -58.1], degrees=True).as_matrix(), None, oversampling)
        self.addObject(f, 4, 10.0*np.array([0.0, 3.6, 0.0]), 10.0*np.array([1.2, 4.0, 0.25*np.cos(15*np.pi/180.0)]), 1.800*0.02, R.from_euler("xyz", [60, 0, 0], degrees=True).as_matrix(), None, oversampling)
        self.addObject(f, 2, 10.0*np.array([0.0, 9.6, 0.0]), 10.0*np.array([0.525561/2.0, 2.0, 0.4]), 1.800*0.02, R.from_euler("xyz", [-60, 0, 0], degrees=True).as_matrix(), None, oversampling)
        self.addObject(f, 6, 10.0*np.array([0.0, -11.15, -0.2]), 10.0*np.array([0.5, 0.75, 0.2]), 1.800*0.02, None, None, oversampling)
        self.addObject(f, 6, 10.0*np.array([0.0, -11.15, 0.2]), 10.0*np.array([0.5, 0.75, 0.2]), 1.800*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([9.1, 0.0, 0.0]), 10.0*np.array([4.2, 1.8, 1.8]), 1.800*0.02, None, np.array([91.0, 0.0, 0.0]), oversampling)

        #'''
        if includeEar:
            xyzs = np.array([8.80, -1.03920, -1.03920,  8.40, -1.03920, -1.03920,  8.0, -1.03920, -1.03920,  7.60, -1.03920, -1.03920,  8.60, -0.69280, -1.03920,  8.20, -0.69280, -1.03920,  7.80, -0.69280, -1.03920,  7.40, -0.69280, -1.03920,  7.0, -0.69280, -1.03920,  8.80, -0.34640, -1.03920,  8.40, -0.34640, -1.03920,  8.0, -0.34640, -1.03920,  7.60, -0.34640, -1.03920,  7.20, -0.34640, -1.03920,  6.80, -0.34640, -1.03920,  8.80, 1.03920, -1.03920,  8.40, 1.03920, -1.03920,  8.0, 1.03920, -1.03920,  7.60, 1.03920, -1.03920,  8.60, 0.69280, -1.03920,  8.20, 0.69280, -1.03920,  7.80, 0.69280, -1.03920,  7.40, 0.69280, -1.03920,  7.0, 0.69280, -1.03920,  8.80, 0.34640, -1.03920,  8.40, 0.34640, -1.03920,  8.0, 0.34640, -1.03920,  7.60, 0.34640, -1.03920,  7.20, 0.34640, -1.03920,  6.80, 0.34640, -1.03920,  8.60, 0.0, -1.03920,  8.20, 0.0, -1.03920,  7.80, 0.0, -1.03920,  7.40, 0.0, -1.03920,  7.0, 0.0, -1.03920,  6.60, 0.0, -1.03920,  8.80, -1.03920, 1.03920,  8.40, -1.03920, 1.03920,  8.0, -1.03920, 1.03920,  7.60, -1.03920, 1.03920,  8.60, -0.69280, 1.03920,  8.20, -0.69280, 1.03920,  7.80, -0.69280, 1.03920,  7.40, -0.69280, 1.03920,  7.0, -0.69280, 1.03920,  8.80, -0.34640, 1.03920,  8.40, -0.34640, 1.03920,  8.0, -0.34640, 1.03920,  7.60, -0.34640, 1.03920,  7.20, -0.34640, 1.03920,  6.80, -0.34640, 1.03920,  8.80, 1.03920, 1.03920,  8.40, 1.03920, 1.03920,  8.0, 1.03920, 1.03920,  7.60, 1.03920, 1.03920,  8.60, 0.69280, 1.03920,  8.20, 0.69280, 1.03920,  7.80, 0.69280, 1.03920,  7.40, 0.69280, 1.03920,  7.0, 0.69280, 1.03920,  8.80, 0.34640, 1.03920,  8.40, 0.34640, 1.03920,  8.0, 0.34640, 1.03920,  7.60, 0.34640, 1.03920,  7.20, 0.34640, 1.03920,  6.80, 0.34640, 1.03920,  8.60, 0.0, 1.03920,  8.20, 0.0, 1.03920,  7.80, 0.0, 1.03920,  7.40, 0.0, 1.03920,  7.0, 0.0, 1.03920,  6.60, 0.0, 1.03920,  8.60, -1.03920, -0.69280,  8.20, -1.03920, -0.69280,  7.80, -1.03920, -0.69280,  7.40, -1.03920, -0.69280,  7.0, -1.03920, -0.69280,  8.80, -0.69280, -0.69280,  8.40, -0.69280, -0.69280,  8.0, -0.69280, -0.69280,  7.60, -0.69280, -0.69280,  7.20, -0.69280, -0.69280,  6.80, -0.69280, -0.69280,  6.40, -0.69280, -0.69280,  8.60, -0.34640, -0.69280,  8.20, -0.34640, -0.69280,  7.80, -0.34640, -0.69280,  7.40, -0.34640, -0.69280,  7.0, -0.34640, -0.69280,  6.60, -0.34640, -0.69280,  6.20, -0.34640, -0.69280,  8.60, 1.03920, -0.69280,  8.20, 1.03920, -0.69280,  7.80, 1.03920, -0.69280,  7.40, 1.03920, -0.69280,  7.0, 1.03920, -0.69280,  8.80, 0.69280, -0.69280,  8.40, 0.69280, -0.69280,  8.0, 0.69280, -0.69280,  7.60, 0.69280, -0.69280,  7.20, 0.69280, -0.69280,  6.80, 0.69280, -0.69280,  6.40, 0.69280, -0.69280,  8.60, 0.34640, -0.69280,  8.20, 0.34640, -0.69280,  7.80, 0.34640, -0.69280,  7.40, 0.34640, -0.69280,  7.0, 0.34640, -0.69280,  6.60, 0.34640, -0.69280,  6.20, 0.34640, -0.69280,  8.80, 0.0, -0.69280,  8.40, 0.0, -0.69280,  8.0, 0.0, -0.69280,  7.60, 0.0, -0.69280,  7.20, 0.0, -0.69280,  6.80, 0.0, -0.69280,  6.40, 0.0, -0.69280,  6.0, 0.0, -0.69280,  8.60, -1.03920, 0.69280,  8.20, -1.03920, 0.69280,  7.80, -1.03920, 0.69280,  7.40, -1.03920, 0.69280,  7.0, -1.03920, 0.69280,  8.80, -0.69280, 0.69280,  8.40, -0.69280, 0.69280,  8.0, -0.69280, 0.69280,  7.60, -0.69280, 0.69280,  7.20, -0.69280, 0.69280,  6.80, -0.69280, 0.69280,  6.40, -0.69280, 0.69280,  8.60, -0.34640, 0.69280,  8.20, -0.34640, 0.69280,  7.80, -0.34640, 0.69280,  7.40, -0.34640, 0.69280,  7.0, -0.34640, 0.69280,  6.60, -0.34640, 0.69280,  6.20, -0.34640, 0.69280,  8.60, 1.03920, 0.69280,  8.20, 1.03920, 0.69280,  7.80, 1.03920, 0.69280,  7.40, 1.03920, 0.69280,  7.0, 1.03920, 0.69280,  8.80, 0.69280, 0.69280,  8.40, 0.69280, 0.69280,  8.0, 0.69280, 0.69280,  7.60, 0.69280, 0.69280,  7.20, 0.69280, 0.69280,  6.80, 0.69280, 0.69280,  6.40, 0.69280, 0.69280,  8.60, 0.34640, 0.69280,  8.20, 0.34640, 0.69280,  7.80, 0.34640, 0.69280,  7.40, 0.34640, 0.69280,  7.0, 0.34640, 0.69280,  6.60, 0.34640, 0.69280,  6.20, 0.34640, 0.69280,  8.80, 0.0, 0.69280,  8.40, 0.0, 0.69280,  8.0, 0.0, 0.69280,  7.60, 0.0, 0.69280,  7.20, 0.0, 0.69280,  6.80, 0.0, 0.69280,  6.40, 0.0, 0.69280,  6.0, 0.0, 0.69280,  8.80, -1.03920, -0.34640,  8.40, -1.03920, -0.34640,  8.0, -1.03920, -0.34640,  7.60, -1.03920, -0.34640,  7.20, -1.03920, -0.34640,  6.80, -1.03920, -0.34640,  8.60, -0.69280, -0.34640,  8.20, -0.69280, -0.34640,  7.80, -0.69280, -0.34640,  7.40, -0.69280, -0.34640,  7.0, -0.69280, -0.34640,  6.60, -0.69280, -0.34640,  6.20, -0.69280, -0.34640,  8.80, -0.34640, -0.34640,  8.40, -0.34640, -0.34640,  8.0, -0.34640, -0.34640,  7.60, -0.34640, -0.34640,  7.20, -0.34640, -0.34640,  6.80, -0.34640, -0.34640,  6.40, -0.34640, -0.34640,  6.0, -0.34640, -0.34640,  8.80, 1.03920, -0.34640,  8.40, 1.03920, -0.34640,  8.0, 1.03920, -0.34640,  7.60, 1.03920, -0.34640,  7.20, 1.03920, -0.34640,  6.80, 1.03920, -0.34640,  8.60, 0.69280, -0.34640,  8.20, 0.69280, -0.34640,  7.80, 0.69280, -0.34640,  7.40, 0.69280, -0.34640,  7.0, 0.69280, -0.34640,  6.60, 0.69280, -0.34640,  6.20, 0.69280, -0.34640,  8.80, 0.34640, -0.34640,  8.40, 0.34640, -0.34640,  8.0, 0.34640, -0.34640,  7.60, 0.34640, -0.34640,  7.20, 0.34640, -0.34640,  6.80, 0.34640, -0.34640,  6.40, 0.34640, -0.34640,  6.0, 0.34640, -0.34640,  8.60, 0.0, -0.34640,  8.20, 0.0, -0.34640,  7.80, 0.0, -0.34640,  7.40, 0.0, -0.34640,  7.0, 0.0, -0.34640,  6.60, 0.0, -0.34640,  6.20, 0.0, -0.34640,  5.80, 0.0, -0.34640,  8.80, -1.03920, 0.34640,  8.40, -1.03920, 0.34640,  8.0, -1.03920, 0.34640,  7.60, -1.03920, 0.34640,  7.20, -1.03920, 0.34640,  6.80, -1.03920, 0.34640,  8.60, -0.69280, 0.34640,  8.20, -0.69280, 0.34640,  7.80, -0.69280, 0.34640,  7.40, -0.69280, 0.34640,  7.0, -0.69280, 0.34640,  6.60, -0.69280, 0.34640,  6.20, -0.69280, 0.34640,  8.80, -0.34640, 0.34640,  8.40, -0.34640, 0.34640,  8.0, -0.34640, 0.34640,  7.60, -0.34640, 0.34640,  7.20, -0.34640, 0.34640,  6.80, -0.34640, 0.34640,  6.40, -0.34640, 0.34640,  6.0, -0.34640, 0.34640,  8.80, 1.03920, 0.34640,  8.40, 1.03920, 0.34640,  8.0, 1.03920, 0.34640,  7.60, 1.03920, 0.34640,  7.20, 1.03920, 0.34640,  6.80, 1.03920, 0.34640,  8.60, 0.69280, 0.34640,  8.20, 0.69280, 0.34640,  7.80, 0.69280, 0.34640,  7.40, 0.69280, 0.34640,  7.0, 0.69280, 0.34640,  6.60, 0.69280, 0.34640,  6.20, 0.69280, 0.34640,  8.80, 0.34640, 0.34640,  8.40, 0.34640, 0.34640,  8.0, 0.34640, 0.34640,  7.60, 0.34640, 0.34640,  7.20, 0.34640, 0.34640,  6.80, 0.34640, 0.34640,  6.40, 0.34640, 0.34640,  6.0, 0.34640, 0.34640,  8.60, 0.0, 0.34640,  8.20, 0.0, 0.34640,  7.80, 0.0, 0.34640,  7.40, 0.0, 0.34640,  7.0, 0.0, 0.34640,  6.60, 0.0, 0.34640,  6.20, 0.0, 0.34640,  5.80, 0.0, 0.34640,  8.60, -1.03920, 0.0,  8.20, -1.03920, 0.0,  7.80, -1.03920, 0.0,  7.40, -1.03920, 0.0,  7.0, -1.03920, 0.0,  6.60, -1.03920, 0.0,  8.80, -0.69280, 0.0,  8.40, -0.69280, 0.0,  8.0, -0.69280, 0.0,  7.60, -0.69280, 0.0,  7.20, -0.69280, 0.0,  6.80, -0.69280, 0.0,  6.40, -0.69280, 0.0,  6.0, -0.69280, 0.0,  8.60, -0.34640, 0.0,  8.20, -0.34640, 0.0,  7.80, -0.34640, 0.0,  7.40, -0.34640, 0.0,  7.0, -0.34640, 0.0,  6.60, -0.34640, 0.0,  6.20, -0.34640, 0.0,  5.80, -0.34640, 0.0,  8.60, 1.03920, 0.0,  8.20, 1.03920, 0.0,  7.80, 1.03920, 0.0,  7.40, 1.03920, 0.0,  7.0, 1.03920, 0.0,  6.60, 1.03920, 0.0,  8.80, 0.69280, 0.0,  8.40, 0.69280, 0.0,  8.0, 0.69280, 0.0,  7.60, 0.69280, 0.0,  7.20, 0.69280, 0.0,  6.80, 0.69280, 0.0,  6.40, 0.69280, 0.0,  6.0, 0.69280, 0.0,  8.60, 0.34640, 0.0,  8.20, 0.34640, 0.0,  7.80, 0.34640, 0.0,  7.40, 0.34640, 0.0,  7.0, 0.34640, 0.0,  6.60, 0.34640, 0.0,  6.20, 0.34640, 0.0,  5.80, 0.34640, 0.0,  8.80, 0.0, 0.0,  8.40, 0.0, 0.0,  8.0, 0.0, 0.0,  7.60, 0.0, 0.0,  7.20, 0.0, 0.0,  6.80, 0.0, 0.0,  6.40, 0.0, 0.0,  6.0, 0.0, 0.0,  5.60, 0.0, 0.0])
            for n in range(xyzs.size//3):
                x = xyzs[3*n+0]
                y = xyzs[3*n+1]
                z = xyzs[3*n+2]
                self.addObject(f, 0, 10.0*np.array([x, y, z]), 10.0*np.array([0.15, 0.15, 0.15]), 0.0, None, None, oversampling)
        #'''

def FORBILD_HU_voxwid(n: int) -> float:
    # Need 250mm across
    # Coordinates are measured in millimeters
    # Most convenient to round n down to an even number, so that e.g. n=2400 and n=2401 give the same attenuation per voxel-length, but with different centering.
    voxel_wid_mm: float = 250/(2*(n//2))
    return voxel_wid_mm

def FORBILD_HU(n: int) -> typing.Tuple[np.ndarray, float]:
    """Returns the FORBILD phantom and the voxel width in millimeters."""
    voxel_wid_mm = FORBILD_HU_voxwid(n)

    cgo = CompositeGeomObject()
    cgo.set_FORBILD()

    xs = voxel_wid_mm * (np.arange(n,dtype=np.float32) - (n-1)/2)
    ys = xs.copy()
    zs = xs.copy()

    atten_water_is_0p02 = cgo.sample(xs, ys, zs)
    # Convert to HU
    out = ne.evaluate('(atten_water_is_0p02/0.02 - 1)*1000', out=atten_water_is_0p02)
    return out, voxel_wid_mm

