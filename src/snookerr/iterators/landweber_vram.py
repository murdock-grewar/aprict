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

import logging
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pydantic

from .. import logtools
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
    _meas: list[cbfp.ProjectionSet]  # list of ordered subsets

    _acceleration_factor: float
    _rayleigh_quotients: list[float]
    _proj_scratch_space: list[
        cbfp.ProjectionSet
    ]  # one for each ordered subset

    _config_bp: cbfp.ConfigBackproject
    _config_fp: cbfp.ConfigProject

    _logger: logging.Logger

    @pydantic.dataclasses.dataclass(frozen=True, kw_only=True)
    class Config:
        """Configuration dataclass for the Landweber iterator."""

        rayleigh_quotient_num_iterations: int = 10
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
        self,
        config: Config,
        volume: Volume,
        projs: cbfp.ProjectionSet | list[cbfp.ProjectionSet],
        *,
        proj_zero_masks: cp.ndarray | list[cp.ndarray] | None = None,
        logger: logging.Logger | None,
    ) -> None:
        self._logger = logger if logger is not None else logging.getLogger()
        self._config = config
        self._vol = volume
        self._meas = projs if type(projs) is list else [projs]  # ty:ignore[invalid-assignment]  # ty is bugged
        if proj_zero_masks is None:
            self._proj_zero_masks = [None for p in self._meas]
        elif type(projs) is list:
            if type(proj_zero_masks) is not list:
                msg = "The input measurements were a list, but only a single mask was provided."
                raise ValueError(msg)
            self._proj_zero_masks = proj_zero_masks
        else:
            if type(proj_zero_masks) is list:
                msg = "The input measurements were not a list, but a list of masks was provided."
                raise ValueError(msg)
            self._proj_zero_masks = [proj_zero_masks]
        self._config_fp = cbfp.ConfigProject()
        soften_det_edges_proportion = (
            0
            if self._config.bp_soft_from_detedge_relrad is None
            else (1 - self._config.bp_soft_from_detedge_relrad)
        )
        self._config_bp = cbfp.ConfigBackproject(
            fast_approx=self._config.fast_backproject,
            cyl_mask_rad=min(*volume.shape[:2]) / 2 * volume.voxwid,
            apply_summand_multiplier=False,
            soften_det_edges_proportion=soften_det_edges_proportion,
        )
        for projs in self._meas:
            if projs.data is None:
                msg = "A provided ProjectionSet object has no data."
                raise ValueError(msg)
        # Make the projection scratch space. Equal in size to one ordered subset.
        self._proj_scratch_space = [
            projs.zeroed_copy() for projs in self._meas
        ]
        self._rayleigh_quotients = self.calculate_rayleigh_quotients(
            num_iterations=self._config.rayleigh_quotient_num_iterations
        )

        # Apply zero masks to projection data
        for proj, zm in zip(self._meas, self._proj_zero_masks):
            if zm is not None:
                cp.multiply(proj.data, zm, out=proj.data)

    def calculate_rayleigh_quotients(self, num_iterations: int) -> list[float]:
        """Compute the list Rayleigh quotients of A^T A (one for each ordered subset) using the iterated power method."""
        fp_config: cbfp.ConfigProject = self._config_fp
        bp_config: cbfp.ConfigBackproject = self._config_bp
        # Check if current volume is 0s. In that case, don't allocate new memory. Just use that one.
        flag_using_existing_volume = self._vol.norm == 0
        scratch_volume: Volume = (
            self._vol
            if flag_using_existing_volume
            else self._vol.zeroed_copy()
        )
        scratch_projs_list: list[cbfp.ProjectionSet] = self._proj_scratch_space
        for scratch_projs in scratch_projs_list:
            if scratch_projs.data is None:
                msg = "ProjectionSet must have its data buffer initialised."
                raise ValueError(msg)

        rayleigh_quotient_list: list[float] = []
        scale_factor: float = 1 / float(
            np.mean(scratch_volume.shape)
        )  # a factor used to scale things to prevent overflow etc
        self._logger.debug(
            "Calculating Rayleigh quotient(s) for the ordered subset(s)..."
        )
        with logtools.ProgressBar(
            desc="Calc. Rayleigh quot.:",
            maxval=num_iterations * len(scratch_projs_list),
            valunit="iteration",
            logger=self._logger,
        ) as pbar:
            for scratch_projs_index, scratch_projs in enumerate(
                scratch_projs_list
            ):
                assert (
                    scratch_projs.data is not None
                )  # useful assertion for type-checking
                # Generate a random positive volume
                v = np.abs(
                    np.random.randn(*scratch_volume.shape) * scale_factor  # noqa: NPY002
                ).astype(np.float32)
                scratch_volume.copy_from(v)
                # self._hardmask_vol(
                #     scratch_volume
                # )  # Note: not exactly the same as the cyl_mask_rad offered by the backprojection. This may matter.
                svn = scratch_volume.norm
                for numit in range(
                    1, num_iterations + 1
                ):  # Use N iterations of power method to find the rayleigh quotient
                    # Normalize the volume
                    scratch_volume.multiply(1 / svn)
                    # Apply A^T A to the volume
                    scratch_projs.data.fill(0)
                    cbfp.project(
                        config=fp_config,
                        volume=scratch_volume,
                        projs=scratch_projs,
                        block=True,
                    )
                    rayleigh_quotient = cp.square(
                        cp.linalg.norm(scratch_projs.data)
                    )
                    scratch_volume.fill(0)
                    cbfp.backproject(
                        config=bp_config,
                        volume=scratch_volume,
                        projs=scratch_projs,
                        block=True,
                        logger=self._logger,
                    )
                    svn = scratch_volume.norm
                    pbar.update_to(
                        numit + num_iterations * scratch_projs_index
                    )
                rayleigh_quotient_list.append(
                    rayleigh_quotient
                )  # Could also use svn instead of rayleigh_quotient
        self._logger.debug("...done calculating Rayleigh quotient(s).")
        if flag_using_existing_volume:
            self._vol.fill(0)  # reset the volume to 0
        del scratch_volume
        del scratch_projs_list
        gpu_tools.clear_cache()  # If new memory was allocated for the scratch space, then it can be released now
        return rayleigh_quotient_list

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
        # TODO: reduce memory usage by reusing the same projection data buffer as scratch space.  # noqa: FIX002

        # Loop over the ordered subsets
        for meas, proj_scratch_space, rq, zm in zip(
            self._meas,
            self._proj_scratch_space,
            self._rayleigh_quotients,
            self._proj_zero_masks,
        ):
            # Put the projection residual into the projection scratch space
            cp.copyto(src=meas.data, dst=proj_scratch_space.data)
            config_fp = cbfp.ConfigProject(multiply_before_adding=-1)
            cbfp.project(
                config=config_fp,
                volume=self._vol,
                projs=proj_scratch_space,
                block=True,
            )
            if zm is not None:
                cp.multiply(
                    proj_scratch_space.data, zm, out=proj_scratch_space.data
                )
            # Add the backprojection of the residual into the volume
            # Divide by relaxation factor first
            relaxation_factor = self._config.acceleration_factor / rq
            cp.multiply(
                proj_scratch_space.data,
                relaxation_factor,
                out=proj_scratch_space.data,
            )
            cbfp.backproject(
                config=self._config_bp,
                volume=self._vol,
                projs=proj_scratch_space,
                summand_multiplier=None,
                block=True,
                logger=self._logger,
            )

    @property
    def volume(self) -> Volume:
        """Return the volume/tomogram."""
        return self._vol
