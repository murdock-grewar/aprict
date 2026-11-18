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

"""Tests for the CB projection kernel(s)."""

# ruff: disable[undocumented-public-method,invalid-class-name,missing-return-type-undocumented-public-function,unnecessary-placeholder]

import typing

import numpy as np
import pytest
import scipy.signal

from ... import ramp_kernels


# Note: probably not a good idea to make the expected output a fixture,
# because there could be a bug in the automatic creation of those outputs
# which makes the 32-bit ramp filter worse, and makes tests pass
# inappropriately. Just keep these cases hardcoded.
@pytest.mark.parametrize(
    argnames=["ramp_kern_factory", "input", "expected_output"],
    argvalues=[
        (
            ramp_kernels.ramp_kernel_ramlak,
            np.array((0,) * 10 + (1,) * 11, dtype=np.float32),
            np.array(
                [
                    -0.099425656664798988516,
                    -0.14880815270685544566,
                    -0.137728058724876561030,
                    -0.21935994868015960043,
                    -0.205519775676528198400,
                    -0.36551948579460225453,
                    -0.347742110612748519130,
                    -0.79218614913488780095,
                    -0.768517672697465615200,
                    -4.76851744649632202030,
                    5.134144167391112017700,
                    1.13414460306385353930,
                    1.183527386779151328300,
                    0.73908296096131603726,
                    0.820715327944232669000,
                    0.66071509419243922890,
                    0.820715748386331215100,
                    0.73908286081720980640,
                    1.183527409865035366100,
                    1.13414466493162672610,
                    5.134144398474657656000,
                ],
                dtype=np.float128,
            ),
        ),
        (
            ramp_kernels.ramp_kernel_ramlak,
            np.array(
                [0.1, 0.2, -0.4, 9.9, 0.01, 0.02, -0.015, 0], dtype=np.float32
            ),
            np.array(
                [
                    -4.216241069140233916,
                    3.17187823659166014010,
                    -44.356729476037068546,
                    99.23130025819907450000,
                    -39.670190333316319550,
                    0.37917036041144375436,
                    -4.660045125459352956,
                    0.11139380250773787503,
                ],
                dtype=np.float128,
            ),
        ),
        (
            ramp_kernels.ramp_kernel_ramlak,
            np.array(
                [
                    -3.4365544490024724e-01,
                    -6.1538063215993011e-01,
                    2.8857665779109837e-01,
                    9.4972547731148860e-01,
                    -9.3614099436103937e-01,
                    9.7263844987731607e-02,
                    1.9497392591784810e-01,
                    -2.6740922979139791e-04,
                    -5.4143267389963379e-01,
                    1.2238802277001017e00,
                    7.7994750821721381e-01,
                    -1.0400369316300970e00,
                    3.9258291766553866e-01,
                    -1.5573304353176978e00,
                    -1.3501835175766077e00,
                    8.0091952308558989e-01,
                    -1.4174689714775329e-01,
                    -5.6377073386985488e-01,
                    8.9235111229551078e-01,
                    3.4100730475948365e-01,
                    -6.1578613587830311e-01,
                    1.4520127105658811e00,
                    1.7862378512589433e-01,
                    4.8770744750713158e-01,
                    -1.0378116186895261e00,
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    -1.3841302291239623146,
                    -5.4405004068263804295,
                    1.4363393022996094409,
                    12.0814079778799149870,
                    -13.2323883143607764360,
                    4.0053650861057669644,
                    0.9007298760714310615,
                    1.4851656899586399950,
                    -9.7635661599539272530,
                    11.2253266121371185010,
                    7.4334553718480669230,
                    -14.0729204797997617124,
                    13.3042819999951883450,
                    -11.7749632677982452940,
                    -10.0019397249556385180,
                    13.3765977197641277090,
                    -2.0273711962236216330,
                    -7.7211139346737142877,
                    8.8818453100757507510,
                    2.5842125938809815634,
                    -13.2140051004751849040,
                    16.2540424721306485300,
                    -6.0466349460014258500,
                    8.4452274721668205700,
                    -12.8377454763767151966,
                ],
                dtype=np.float128,
            ),
        ),
    ],
    ids=[
        "ramlak-step",
        "ramlak-pseudorandom-01",
        "ramlak-pseudorandom-02",
    ],
)
class Test_RampFilter:
    """Test ramp that 1D ramp filters are more precise than what is achieved with a 32-bit kernel."""

    @pytest.fixture(scope="function", autouse=True)
    def setup_and_teardown(
        self,
        ramp_kern_factory: typing.Callable[[int, float], np.ndarray],
        input: np.ndarray,
        expected_output: np.ndarray,
        request,  # ruff: ignore[missing-type-function-argument]
    ):
        test_id: str = request.node.callspec.id  # ruff: ignore[unused-variable]
        # Setup phase
        ws: int = (
            2 * input.shape[0] + 1
        )  # make the window big enough, and odd-lengthed
        physpixsize: float = 1
        self.kern = ramp_kern_factory(ws, physpixsize)
        self.filter_with_kern_fft = (
            lambda x, kern: scipy.signal.convolve(
                x, kern, mode="same", method="fft"
            )
        )  # it makes a big difference whether we use mode='fft' or mode='direct'. fft seems to be way more accurate than direct, when using 64-bit floats, for some reason. It could be casting down the kernel to the precision of the input, when using mode='direct'. I haven't checked.
        self.input = input
        self.ans = expected_output

        # yield to the timed test
        yield

        # Teardown phase
        pass

    def test_kernel(self) -> None:
        #  output_f16 = self.filter_with_kern(self.input, self.kern.astype(np.float16))
        output_f32 = self.filter_with_kern_fft(
            self.input, self.kern.astype(np.float32)
        )
        output_f64 = self.filter_with_kern_fft(
            self.input, self.kern.astype(np.float64)
        )
        err_f32 = np.linalg.norm(
            self.ans - output_f32
        )  # 1.3113241046815158535e-06 for fft
        err_f64 = np.linalg.norm(
            self.ans - output_f64
        )  # 4.9394008275799508944e-15 for fft
        # Uncomment to view answers
        with np.printoptions(precision=30):
            print("err_f32: ", err_f32)
            print("err_f64: ", err_f64)
        #     print(output_f32)
        #     print(output_f64)
        #     print(output_gt)

        assert (
            err_f64 < 1e-8 * err_f32
        )  # assert that error is significantly less than that obtained from a float-32 computation. The exact factor should be tuned manually by inspecting the errors.


# ruff: enable[undocumented-public-method,invalid-class-name,missing-return-type-undocumented-public-function,unnecessary-placeholder]
