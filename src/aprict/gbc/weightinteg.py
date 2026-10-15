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

"""A module to compute the expected sum of weights from a perfect cylindrical trajectory. Utilises symmetry and numerical integration."""

import numexpr as ne
import numpy as np


def cossq_soft(a: float, b: float, c: np.ndarray) -> np.ndarray:
    """Cos-squared softening function."""
    gamma = 1 / (np.sin(a - b) - np.sin(a))
    chi = gamma * np.sin(a)

    absc = ne.evaluate("abs(c)", local_dict={"c": c})
    x = ne.evaluate(
        "g * sin(absc) - chi",
        local_dict={"g": gamma, "c": c, "chi": chi, "absc": absc},
    )
    ans = ne.evaluate(
        "where((absc <= a)&(absc >= a-b), 3*x*x - 2*x*x*x, 0)",
        out=x,
        local_dict={"x": x, "absc": absc, "a": a, "b": b},
    )
    return ne.evaluate(
        "where(absc < a - b, 1, ans)",
        out=ans,
        local_dict={"a": a, "b": b, "absc": absc, "ans": ans},
    )


def backproj_gbc_weight_integ(
    r_on_R_input: np.ndarray,  # ruff: ignore[invalid-argument-name]
    soft_vartheta_v: float,
    soft_vartheta_v_softspan: float,
    soft_vartheta_h: float,
    soft_vartheta_h_softspan: float,
    num_ang_samps: int,
    num_z_samps: int,
    chunk_r_on_R_size: int | None = None,  # ruff: ignore[invalid-argument-name]
) -> np.ndarray:
    """Integrate the weight*softening over the cylindrical source locus."""
    if r_on_R_input.ndim != 1:
        msg = "r_on_R should be a 1-dimensional array."
        raise ValueError(msg)
    if chunk_r_on_R_size is None:
        chunk_r_on_R_size = r_on_R_input.shape[0]  # ruff: ignore[non-lowercase-variable-in-function]

    out = np.empty_like(r_on_R_input, dtype=np.float32)

    # Things that stay the same throughout all loops
    tstep = np.pi * (1 / num_ang_samps)
    ts = np.arange(num_ang_samps, dtype=np.float32) * tstep
    t = ts[np.newaxis, np.newaxis, :]

    sl = slice(0, chunk_r_on_R_size)
    while sl.start < out.shape[0]:
        inview = r_on_R_input[sl, np.newaxis, np.newaxis]
        r_on_R = np.empty_like(inview, dtype=np.float32)  # ruff: ignore[non-lowercase-variable-in-function]
        ne.evaluate("inview", out=r_on_R, local_dict={"inview": inview})

        maxz_on_R = 2 * np.tan(soft_vartheta_v) * (1 + r_on_R.max())  # ruff: ignore[non-lowercase-variable-in-function]
        zstep = maxz_on_R * (1 / num_z_samps)
        zs = (np.arange(num_z_samps, dtype=np.float32) + 0.5) * zstep

        z_on_R = zs[np.newaxis, :, np.newaxis]  # ruff: ignore[non-lowercase-variable-in-function]
        ld = {"z_on_R": z_on_R, "t": t, "r_on_R": r_on_R}

        ell_on_R = ne.evaluate(  # ruff: ignore[non-lowercase-variable-in-function]
            "sqrt(maximum(0, sin(t)*sin(t) + (r_on_R - cos(t))*(r_on_R - cos(t))))",
            local_dict=ld,
        )
        ld = ld | {"ell_on_R": ell_on_R}
        z_on_ell_ = ne.evaluate(
            "z_on_R/maximum(1e-9, ell_on_R)", local_dict=ld
        )
        ld = ld | {"z_on_ell_": z_on_ell_}
        cos_theta_el = ne.evaluate(
            "minimum(1, maximum(0, 1/sqrt(1 + z_on_ell_*z_on_ell_)))",
            local_dict=ld,
        )
        ld = ld | {"cos_theta_el": cos_theta_el}
        sin_theta_h = ne.evaluate(
            "r_on_R * sin(-t) / sqrt(maximum(1e-9, r_on_R*r_on_R + 1 - 2*r_on_R * cos(-t)))",
            local_dict=ld,
        )
        theta_h = ne.evaluate(
            "arcsin(maximum(-1,minimum(1, sin_theta_h)))",
            local_dict={"sin_theta_h": sin_theta_h},
        )
        ld = ld | {"theta_h": theta_h}
        c_on_R = ne.evaluate("abs(cos(theta_h))", local_dict=ld)  # ruff: ignore[non-lowercase-variable-in-function]
        ld = ld | {"c_on_R": c_on_R}
        dsigned_on_R = ne.evaluate("c_on_R - ell_on_R", local_dict=ld)  # ruff: ignore[non-lowercase-variable-in-function]
        ld = ld | {"dsigned_on_R": dsigned_on_R}

        # Two equivalent ways of writing this weight. I checked quickly that they give visually the same result.
        # I've commented out my prefered expression, in favour of the one in the GBC paper.
        # weight = ne.evaluate("""cos_theta_el*cos_theta_el*cos_theta_el * c_on_R / maximum(1e-9,
        #     c_on_R*c_on_R + dsigned_on_R*dsigned_on_R
        # )""", local_dict=ld)
        weight = ne.evaluate(
            "cos_theta_el*cos_theta_el*cos_theta_el * abs(cos(theta_h)) / (cos(2*theta_h) + r_on_R*r_on_R)",
            local_dict=ld,
        )

        softfactor = cossq_soft(
            soft_vartheta_v, soft_vartheta_v_softspan, np.arccos(cos_theta_el)
        ) * cossq_soft(soft_vartheta_h, soft_vartheta_h_softspan, theta_h)

        ans = np.mean(weight * softfactor, axis=(1, 2))

        numpoints_on_surface = 2 * np.pi * 2 * maxz_on_R
        ov = out[sl]
        ne.evaluate(
            "ans * numpoints_on_surface",
            out=ov,
            local_dict={
                "ans": ans,
                "numpoints_on_surface": numpoints_on_surface,
            },
        )
        sl = slice(sl.start + chunk_r_on_R_size, sl.stop + chunk_r_on_R_size)
    return out


def backproj_gbc_weight_integ_xygrid(
    xs: np.ndarray,
    ys: np.ndarray,
    R: float,  # ruff: ignore[invalid-argument-name]
    soft_vartheta_v: float,
    soft_vartheta_v_softspan: float,
    soft_vartheta_h: float,
    soft_vartheta_h_softspan: float,
    num_ang_samps: int = 600,
    num_z_samps: int = 600,
) -> np.ndarray:
    """Return a 2D array giving the integrated weights over an xy meshgrid."""
    relRarr = (  # ruff: ignore[non-lowercase-variable-in-function]
        np.sqrt(np.square(xs[:, np.newaxis]) + np.square(ys[np.newaxis, :]))
        / R
    )
    numsamps = 2 * max(len(xs), len(ys))
    r_on_R_input = np.linspace(  # ruff: ignore[non-lowercase-variable-in-function]
        relRarr.min(), relRarr.max(), endpoint=True, num=numsamps
    )

    weight_integ = backproj_gbc_weight_integ(
        r_on_R_input,
        soft_vartheta_v,
        soft_vartheta_v_softspan,
        soft_vartheta_h,
        soft_vartheta_h_softspan,
        num_ang_samps,
        num_z_samps,
        chunk_r_on_R_size=10,
    )
    return np.interp(
        relRarr,
        r_on_R_input,
        weight_integ,
    )
