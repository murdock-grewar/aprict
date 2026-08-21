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

# lds = low-discrepancy sequence
"""Tools to generate low-discrepancy sequences and related things."""

import itertools
import math
import typing

import numpy as np

# I am using theory (and some bits of code) developed by Martin Roberts in 2018.
# Website: https://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
# I have not checked for an article citation at this stage.


def _phi(d: int, iterations: int = 100):  # ruff: ignore[missing-return-type-private-function]
    x = np.array(2, dtype=np.float64)
    for _ in range(iterations):
        x = np.float_power(1 + x, 1 / (d + 1))
    return x


def lds_nd(
    n: int,
    range_tuples: tuple[tuple[float, float], ...] | None = None,
) -> map[tuple[float, ...]]:
    """Generate a low-discrepancy sequence on an N-dimensional rectangular prism. Only really works for cubes."""
    if range_tuples is None:
        mins = tuple(0 for _ in range(n))
        maxs = tuple(1 for _ in range(n))
    else:
        if not len(range_tuples) == n:
            raise ValueError()
        mins, maxs = zip(*range_tuples)
    ranges = tuple(max_ - min_ for max_, min_ in zip(maxs, mins))
    biggest_range = max(ranges)
    ratios_wrt_biggest = tuple(range_ / biggest_range for range_ in ranges)

    phi_n: np.float64 = _phi(n, 1000)
    as_: tuple[np.float64, ...] = tuple(
        np.float_power(phi_n, -i) for i in range(1, n + 1)
    )

    filter_predicate = lambda zero_based_tuple: all(  # ruff: ignore[lambda-assignment]
        val < ratio_wrt_biggest
        for val, ratio_wrt_biggest in zip(zero_based_tuple, ratios_wrt_biggest)
    )

    return map(  # ruff: ignore[unnecessary-map]
        lambda val_tuple: tuple(
            biggest_range * val + range_min
            for range_min, val in zip(mins, val_tuple)
        ),
        filter(
            filter_predicate,
            map(  # ruff: ignore[unnecessary-map]
                lambda m: tuple(((m * as_[i]) % 1) for i in range(n)),
                itertools.count(),  # generates 0,1,2,3,4,...
            ),
        ),
    )


def lds_cyl(
    zmin: float, zmax: float, rad: float, eccentricity: float = 1
) -> map[tuple[float, float]]:
    """Generate a low-discrepancy sequence on an unrolled cylinder."""
    return map(  # ruff: ignore[unnecessary-map]
        lambda zt: (zmin + zt[0] * eccentricity * rad, zt[1]),
        lds_nd(
            2,
            range_tuples=(
                (0, (zmax - zmin) / eccentricity / rad),
                (0, 2 * np.pi),
            ),
        ),
    )


def lds_cyl_tiled_batched_gen(
    R: float,  # cylinder radius  # ruff: ignore[invalid-argument-name]
    h: float,  # cylinder height
    zmin: float,  # cylinder min z
    N: int,  # points per batch  # ruff: ignore[invalid-argument-name]
    s: int = 1,  # parity of helix, -1 or 1
) -> typing.Generator[list[tuple[float, float]]]:
    """Create a generator of points.

    Use like so:
    g = make_point_generator(...)
    # Then:
    p1 = next(g) # p1 = (z1, phi1)
    p2 = next(g) # p2 = (z2, phi2)
    # ...
    """  # ruff: ignore[docstring-missing-yields, docstring-missing-exception]
    if s not in (-1, 1):
        msg = f"s should be -1 or 1, but it is {s}"
        raise ValueError(msg)

    # twopi = 2*math.pi    # constant # Not used
    phi2 = 1.324717957  # constant
    n = 0

    # m is the number of tiles around one sheared band
    m = math.ceil(math.sqrt(N * 2 * math.pi * R / h))
    # W is the physical sidelength of a square tile, before it is sheared
    W = 2 * math.pi * R / m  # ruff: ignore[non-lowercase-variable-in-function]
    # Stepsize
    dz = s * W / m
    dt = 2 * math.pi / m

    while True:
        # Generate next term in the 2D LDS sequence for the unit square
        x1 = n / phi2 - math.floor(n / phi2)
        x2 = n / phi2 / phi2 - math.floor(n / phi2 / phi2)
        n += 1
        # Add a shear to the square
        x2 += s * x1 / m
        # Calculate the bottom-left of the first shape
        z: float = zmin + W * x2
        t: float = 0.0 + W * x1 / R
        if s > 0:
            while z >= zmin:
                z -= dz
                t -= dt
            ps = []
            z += dz
            t += dt
            while z < zmin + h:
                ps += [(z, t)]
                z += dz
                t += dt
        else:  # opposite helix parity
            while z < zmin + h:
                z -= dz
                t -= dt
            ps = []
            z += dz
            t += dt
            while z >= zmin:
                ps += [(z, t)]
                z += dz
                t += dt
        # Add an overall twist to the cylinder to make individual batches more evenly sample phi.
        # It's unclear to me whether the added term should be positive or negative, depending on the parity 's'. A quick trial-and-error seems to yield good results when the sign is the negative of 's'. Otherwise, it appears that when the helical pitch is too high, then this interferes with the LDS and can tend to produce aligned points between batches or within batches.
        ps = [
            (z, t - s * (z - zmin) * (math.sqrt(5) - 1) / 2 / R) for z, t in ps
        ]
        yield ps


def peel_cyl_points_arr(
    zt_arr: np.ndarray | list[tuple[float, float]],
    R: float,  # ruff: ignore[invalid-argument-name]
    vZ: float,  # ruff: ignore[invalid-argument-name]
    vA: float,  # ruff: ignore[invalid-argument-name]
    phi_offset: float = 0,
    reverse: bool = False,
    reverse_parity: bool = False,
    stop_time: float = 0,
) -> np.ndarray:
    """Reorder a list of evenly-sampled source points on the cylinder to give a close-to-fastest-possible traversal between them. Returns an array like np.array([[time1,z1,phi1], [time2,z2,phi2], ...])."""
    # vZ = max speed of Z translation of the stage (units of distance per second).
    # vA = max speed of the angular rotation of the stage (radians per second).
    # reverse_parity changes the direction of the peel. (clockwise/anticlockwise)

    # Initialise inputs
    zt_arr = np.array(zt_arr)
    if zt_arr.ndim != 2:
        msg = "Input list/array has wrong dimensionality. It should be of shape (N,2)."
        raise ValueError(msg)
    if zt_arr.shape[1] != 2:
        msg = "Input list/array is malformed. It should be of shape (N,2), but the length of the second axis was not 2."
        raise ValueError(msg)

    point_density: float = zt_arr.shape[0] / (
        2 * np.pi * R * (zt_arr[:, 0].max() - zt_arr[:, 0].min())
    )
    peel_pitch = math.sqrt(3 / point_density * vZ / (R * vA))

    zt_new = zt_arr.copy()
    zs = zt_new[:, 0]
    ts = zt_new[:, 1]
    ts = np.mod(ts + phi_offset, 2 * np.pi, out=ts)
    if reverse_parity:
        zt_new[:, 1] = ts + 2 * np.pi * np.floor(
            -zs / peel_pitch - ts / (2 * np.pi)
        )
    else:
        zt_new[:, 1] = ts + 2 * np.pi * np.floor(
            zs / peel_pitch - ts / (2 * np.pi)
        )
    # Sort by the ts
    if reverse_parity:
        zt_new = np.array(
            sorted(
                zt_new.tolist(),
                key=(lambda zt_tup: zt_tup[1]),
                reverse=True,
            )
        )
    else:
        zt_new = np.array(
            sorted(zt_new.tolist(), key=(lambda zt_tup: zt_tup[1]))
        )

    if reverse:
        zt_new = np.flip(zt_new, axis=0)

    # Now compute the times between points. The first one is defined as t=0, marking the *beginning* of the first exposure.
    dzdt = np.abs(np.diff(zt_new, axis=0))
    dtimes = stop_time + np.amax(
        dzdt / np.array([[vZ, vA]], dtype=np.float32), axis=1
    )
    times = np.array([0] + np.cumsum(dtimes).tolist())  # ruff: ignore[collection-literal-concatenation]

    # Construct the full array
    tzp = np.concatenate((times[:, np.newaxis], zt_new), axis=1)
    return tzp  # ruff: ignore[unnecessary-assign]


def peel_cyl_zts_gen(  # ruff: ignore[undocumented-public-function]
    zt_gen: typing.Iterable[tuple[float, float]],
    num_per_batch: int,
    R: float,  # ruff: ignore[invalid-argument-name]
    vZ: float,  # ruff: ignore[invalid-argument-name]
    vA: float,  # ruff: ignore[invalid-argument-name]
    phi_offset: float = 0,
    reverse: bool = False,
    reverse_parity: bool = False,
    stop_time: float = 0,
) -> typing.Generator[tuple[float, float, float]]:
    zt_arrs = map(np.array, itertools.batched(zt_gen, num_per_batch))
    rv = reverse
    rp = reverse_parity

    # Need a time offset between batches. Needs to include the time taken to get from the end of one batch to the beginning of the next.
    t_offset: float | None = None
    zlast, philast = None, None
    zfirst, phifirst = None, None

    for zt_arr in zt_arrs:
        peeled_tzp = peel_cyl_points_arr(
            zt_arr,
            R=R,
            vZ=vZ,
            vA=vA,
            phi_offset=phi_offset,
            reverse=rv,
            reverse_parity=rp,
            stop_time=stop_time,
        )
        for time, z, phi in peeled_tzp:
            if zfirst is None:  # Calculate the t_offset of this batch
                zfirst, phifirst = z, phi
                if t_offset is None:
                    t_offset = 0
                else:
                    t_offset += stop_time + max(
                        abs(zfirst - zlast) / vZ, abs(phifirst - philast) / vA
                    )
            time += t_offset  # ruff: ignore[redefined-loop-name]
            yield (time, z, phi)
            # phi_last = t  # Commented out because I'm not sure how best to start the next cycle when the previous one ends. Not too important, anyway.
        zlast, philast = z, phi
        zfirst, phifirst = None, None
        # Reverse the parity and the order so that the next round starts where the previous ended, and the direction of rotation remains the same. It's the Z coordinate that bops up and down.
        rv = not rv
        rp = not rp
    return


def lds_cyl_dynamic_batched_gen(
    zmin: float,
    zmax: float,
    R: float,  # The source-to-axis distance.  # ruff: ignore[invalid-argument-name]
    vZ: float,  # Speed of translation of stage (units of distance per unit of time)  # ruff: ignore[invalid-argument-name]
    vA: float,  # Speed of rotation of stage (radians per unit of time)  # ruff: ignore[invalid-argument-name]
    num_per_batch: int,  # Number of points per complete sampling of the cylinder.
    yield_whole_batches: bool = False,
    theta_start: float = 0,
    stop_time: float = 0,
) -> (
    typing.Generator[tuple[float, float, float]] | typing.Generator[np.ndarray]
):
    """Return a generator of source points which follows a close-to-fastest traversal between points. Assumes that the rotational and translation degrees of freedom are motorised separately. Imposes a minimum number of projections per 'round' around the cylinder, which is num_per_batch. Uses Murdock's 2D LDS over the cylinder."""
    if num_per_batch < 2:
        msg = "num_per_batch must be at least 2."
        raise ValueError(msg)

    # Define the point iterator
    generator = lds_cyl(zmin=zmin, zmax=zmax, rad=R)
    # Add the theta_start
    generator = map(  # ruff: ignore[unnecessary-map]
        lambda zt_tup: (zt_tup[0], zt_tup[1] + theta_start), generator
    )
    # Reorder the points within each batch according to the 'peel' method.
    generator = peel_cyl_zts_gen(
        generator,
        num_per_batch=num_per_batch,
        R=R,
        vZ=vZ,
        vA=vA,
        phi_offset=0,
        reverse=False,
        reverse_parity=False,
        stop_time=stop_time,
    )
    if yield_whole_batches:
        generator = map(list, itertools.batched(generator, num_per_batch))

    return generator  # ty:ignore[invalid-return-type]


# @article{leopardi2006partition,
#   title={A partition of the unit sphere into regions of equal area and small diameter},
#   author={Leopardi, Paul},
#   journal={Electronic Transactions on Numerical Analysis},
#   volume={25},
#   number={12},
#   pages={309--327},
#   year={2006}
# }
def points_by_equal_area_partitioning_on_sphere(  # ruff: ignore[complex-structure, undocumented-public-function]
    N: int,  # ruff: ignore[invalid-argument-name]
    _take_half: bool = False,
) -> tuple[np.ndarray, np.ndarray]:  # Theta, Phi
    if type(N) is not int:
        msg = "'n_odd' should be of type 'int'."
        raise ValueError(msg)

    # d = 2, because we work with the 2-dimensional sphere.

    # eq. (2.9)
    def Nu(theta: float) -> float:  # ruff: ignore[invalid-function-name]
        return 4 * np.pi * np.square(np.sin(theta / 2))

    # eq. (2.10) -- Theta is the inverse of Nu.
    def Theta(x: float) -> float:  # ruff: ignore[invalid-function-name]
        return 2 * np.arcsin(np.sqrt(x / (4 * np.pi)))

    # -- Formula in Section 3 --

    # 1. Determining the colatitudes of polar caps.
    # Unnumbered equation before (3.1)
    Nu_R: float = 4 * np.pi / N  # ruff: ignore[non-lowercase-variable-in-function]
    # eq. (3.1)
    theta_c: float = Theta(Nu_R)

    # 2. Determining an ideal collar angle.
    # Unnumbered equation
    delta_I: float = np.sqrt(Nu_R)  # ruff: ignore[non-lowercase-variable-in-function]

    # 3. Determining an ideal number of collars.
    # Unnumbered equation
    n_I: float = (  # ruff: ignore[non-lowercase-variable-in-function]
        np.pi - 2 * theta_c
    ) / delta_I  # ideal number of collars

    # 4. Determining the actual number of collars.
    # Unnumbered equation
    n: int = 0 if (N == 2) else max(1, round(n_I))  # actual number of collars
    if _take_half:
        # It looks a bit better (more random, less structured) if the number of collars is odd
        # n = n + (1 - n%2)
        # I want to remove the perfect equatorial band of points, which may produce a weird artefact for the z frequencies
        # To that end, I ensure that the number of collars is even here.
        n = n + (n % 2)

    # 5. Creating a list of the ideal number of regions in each collar.
    delta_F: float = n_I / n * delta_I  # ruff: ignore[non-lowercase-variable-in-function]
    theta_F: list[float] = list(theta_c + np.arange(n + 1) * delta_F)  # ruff: ignore[non-lowercase-variable-in-function]
    y: list[float] = [
        (Nu(theta_F[i + 1]) - Nu(theta_F[i])) / Nu_R for i in range(n)
    ]

    # 6. Creating a list of the actual number of regions in each collar.
    a: list[float] = [0]
    m: list[int] = []
    for i in range(len(y)):
        m.append(round(y[i] + a[i]))
        a.append(a[-1] + y[i] - m[i])

    if _take_half:
        # Round up the m to make sure they're even
        m = [mel + (mel % 2) for mel in m]
        Nu_R = 4 * np.pi / (2 + sum(m))  # ruff: ignore[non-lowercase-variable-in-function]

    # 7. Creating a list of colatitudes of each zone.
    m_cumsum: list[float] = [0] + list(np.cumsum(m))  # ruff: ignore[collection-literal-concatenation]
    theta: list[float] = [0]
    theta.extend([Theta((1 + m_cumsum[i]) * Nu_R) for i in range(n + 1)])
    if not _take_half:
        theta.append(np.pi)

    # --------------------------

    # The way that the theta are defined is weird.
    # Rewriting theta here to give the polar angles of the centers of each collar
    theta: list[float] = [0]
    theta.extend(
        [
            Theta(((1 + m_cumsum[i]) + (1 + m_cumsum[i + 1])) / 2 * Nu_R)
            for i in range(n)
        ]
    )
    if not _take_half:
        theta.append(np.pi)

    # Create a top and bottom "collar" corresponding to the polar caps.
    m = [1] + m  # ruff: ignore[collection-literal-concatenation]
    if not _take_half:
        m.append(1)

    # Define the offsets to the phi. These are the rotations of each collar, which are free to be chosen.
    # phi_offset: np.ndarray = np.zeros((len(m),), dtype=np.float32)
    if _take_half:
        # the phi_offset needs to be an even function of (np.pi/2 - theta)
        theta_el_abs = np.abs(np.pi / 2 - np.array(theta))
        phi_offset = theta_el_abs * (1 + np.sqrt(5)) / 2
    else:
        phi_offset: np.ndarray = (
            np.arange(len(m), dtype=np.float32) * (1 + np.sqrt(5)) / 2
        )

    # num_points: int = np.sum(m)
    # Theta_out: np.ndarray = np.empty((num_points,), dtype=np.float32)
    # Phi_out: np.ndarray = np.empty((num_points,), dtype=np.float32)

    Theta_Phi: list[list[float]] = []  # ruff: ignore[non-lowercase-variable-in-function]
    for theta_, phi_offset_, m_ in zip(theta, phi_offset, m):
        phi_lst: np.ndarray = phi_offset_ + np.linspace(
            0, 2 * np.pi, endpoint=False, num=m_
        )
        if _take_half:
            phi_lst = phi_lst[: (1 if m_ == 1 else m_ // 2)]
        Theta_Phi.extend([[theta_, phi_] for phi_ in phi_lst])

    Theta_Phi_out = np.array(Theta_Phi)  # ruff: ignore[non-lowercase-variable-in-function]

    return Theta_Phi_out[:, 0], Theta_Phi_out[:, 1]


def lines_by_equal_area_partitioning_on_sphere(  # ruff: ignore[undocumented-public-function]
    N_approx: int,  # ruff: ignore[invalid-argument-name]
) -> tuple[np.ndarray, np.ndarray]:  # Theta, Phi
    if type(N_approx) is not int:
        msg = "'N_approx' should be of type 'int'."
        raise ValueError(msg)
    if N_approx == 1:
        return (
            np.array([0], dtype=np.float32),
            np.array([0], dtype=np.float32),
        )
    return points_by_equal_area_partitioning_on_sphere(
        2 * N_approx, _take_half=True
    )


# From https://arxiv.org/pdf/0912.4540.pdf
# The Roberts term is from https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/#more-3069
def uniformly_distributed_points_on_sphere_fibonacci_lattice(  # ruff: ignore[undocumented-public-function]
    n_odd: int, roberts_term: bool = False
) -> tuple[np.ndarray, np.ndarray]:  # Theta, Phi
    if type(n_odd) is not int:
        msg = "'n_odd' should be of type 'int'."
        raise ValueError(msg)
    # Must be odd. Eq. (2)
    if n_odd % 2 == 0:
        msg = "'n_odd' must be odd."
        raise ValueError(msg)
    # The Roberts term is appropriate for minimising the distance between points, whereas I think without the Roberts term it's better for having about the same area 'occupied' by each point. It's better without the Roberts term when we're trying to get the number of points to be roughly the same per unit solid angle (steradian), anywhere on the sphere.
    P = n_odd  # ruff: ignore[non-lowercase-variable-in-function]
    N = P // 2  # ruff: ignore[non-lowercase-variable-in-function]
    golden_ratio = (1 + np.sqrt(5)) / 2
    arange = np.arange(-N, N + 1)
    Theta = np.arccos(2 * (arange + (0.36 if roberts_term else 0)) / P)  # ruff: ignore[non-lowercase-variable-in-function]
    Phi = 2 * np.pi * arange / golden_ratio  # ruff: ignore[non-lowercase-variable-in-function]
    # Normalise to the conventional coordinate ranges
    Phi = np.mod(Phi, 2 * np.pi)  # ruff: ignore[non-lowercase-variable-in-function]
    return Theta, Phi
