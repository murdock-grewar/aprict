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

"""A module for an iterative Landweber reconstructor object applying to flat-panel cone-beam data."""

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pydantic

from ..cb import flatpanel as cbfp
from ..lib import gpu_tools

# from ..lib.texture_tools import operations as tops

if TYPE_CHECKING:
    from ..volume import Volume


class Landweber_flatpanel:  # noqa: N801
    """An iterable Landweber reconstructor."""

    # _current_iteration_number: int
    _config: Config
    _vol: Volume
    _meas: cbfp.ProjectionSet

    _acceleration_factor: float
    _rayleigh_quotient: float
    _proj_scratch_space: cbfp.ProjectionSet

    _config_bp: cbfp.ConfigBackproject
    _config_fp: cbfp.ConfigProject

    @pydantic.dataclasses.dataclass(frozen=True, kw_only=True)
    class Config:
        """Configuration dataclass for the Landweber iterator."""

        acceleration_factor: float = 1
        fast_backproject: bool = True
        soft_from_cylinder_relrad: float | None = (
            0.95  # percentage of the cylinder radius from which softening will occur
        )
        bp_soft_from_detedge_relrad: float | None = (
            0.95  # percentage of the detector half-width from which to soften
        )
        # These not implemented yet:
        # enforce_positivity_constraint: bool     = False
        # soften_to_cylinder_relhei: float | None = 0.05  # the height of the object to reserve for softening to zero. This is expressed as a proportion of the radius of the inscribed x/y cylinder.

        # denoising: float | None = None  # Applies an approximate TV gradient descent between iterations. This is NOT accounted for in the calculation of the next step, nor the calculated and printed error metric. This denoising should be considered an approximate ad-hoc method. Suggested value range: 0 to 0.05.

    def __init__(
        self, config: Config, volume: Volume, projs: cbfp.ProjectionSet
    ) -> None:
        self._config = config
        self._vol = volume
        self._meas = projs
        self._config_fp = cbfp.ConfigProject()
        soften_det_edges_proportion = (
            0
            if self._config.bp_soft_from_detedge_relrad is None
            else (1 - self._config.bp_soft_from_detedge_relrad)
        )
        self._config_bp = cbfp.ConfigBackproject(
            fast_approx=self._config.fast_backproject,
            cyl_mask_rad_vox=min(*volume.shape[:2]) / 2,
            apply_summand_multiplier=False,  # could include a constant later
            soften_det_edges_proportion=soften_det_edges_proportion,
        )
        if projs.data is None:
            msg = "The provided ProjectionSet object has no data."
            raise ValueError(msg)

        # Compute the relaxation coefficient. First we need the Rayleight quotient
        M: Volume = volume.zeroed_copy()  # noqa: N806
        p: cbfp.ProjectionSet = projs.zeroed_copy()
        assert p.data is not None
        scale_factor: float = 1 / float(
            np.mean(volume.shape)
        )  # a factor used to scale things to prevent overflow etc
        # Generate a random positive volume
        v = np.abs(np.random.randn(*M.shape)*scale_factor).astype(np.float32)  # noqa: NPY002
        M.copy_from(v)
        self._hardmask_vol(
            M
        )  # Note: not exactly the same as the cyl_mask_rad_vox offered by the backprojection. This may matter.
        non_multiply_config_bp = cbfp.ConfigBackproject(
            fast_approx=self._config.fast_backproject,
            cyl_mask_rad_vox=min(*volume.shape[:2]) / 2,
            apply_summand_multiplier=False,  # Turn this off while we compute the preconditioner
            soften_det_edges_proportion=soften_det_edges_proportion,
        )
        for _ in range(10):  # Use N iterations of power method to find the rayleigh quotient
            # Normalize the volume
            M.multiply(1/M.norm)
            # Apply A^T A to the volume
            p.data.fill(0)
            cbfp.project(config=self._config_fp, volume=M, projs=p, block=True)
            M.fill(0)
            cbfp.backproject(config=non_multiply_config_bp, volume=M, projs=p, block=True)
            rayleigh_quotient = M.norm
            # print("Rayleigh quotient: ", rayleigh_quotient)
        self._rayleigh_quotient = rayleigh_quotient
        del M
        gpu_tools.clear_cache()
        self._proj_scratch_space = p

    def _softmask_vol(self, v: Volume) -> None:
        rr = self._config.soft_from_cylinder_relrad
        if rr is not None:
            v.soften_to_xy_cylinder(rr)

    def _hardmask_vol(self, v: Volume) -> None:
        rr = 1
        if rr is not None:
            v.soften_to_xy_cylinder(rr)

    def iterate(self) -> None:
        """Perform an iteration of the algorithm."""
        # Put the projection residual into the projection scratch space
        cp.copyto(src=self._meas.data, dst=self._proj_scratch_space.data)
        config_fp = cbfp.ConfigProject(multiply_before_adding=-1)
        cbfp.project(
            config=config_fp,
            volume=self._vol,
            projs=self._proj_scratch_space,
            block=True,
        )
        # Add the backprojection of the residual into the volume
        # Divide by relaxation factor first
        relaxation_factor = self._config.acceleration_factor/(self._rayleigh_quotient)
        cp.multiply(self._proj_scratch_space.data, relaxation_factor, out=self._proj_scratch_space.data)
        cbfp.backproject(
            config=self._config_bp,
            volume=self._vol,
            projs=self._proj_scratch_space,
            summand_multiplier=None,
            block=True,
        )

    @property
    def volume(self) -> Volume:
        """Return the volume/tomogram."""
        return self._vol
