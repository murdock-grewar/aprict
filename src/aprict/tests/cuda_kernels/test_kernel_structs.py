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

"""Tests for the numpy dtypes/kernel structs interface."""

import importlib.resources

import cupy as cp
import numpy as np
import pytest

from ...cuda import _generate_cusrc, header_injection, kern_structs
from ...cuda import structs as kstructs
from . import test_cusrc


@pytest.fixture
def ks() -> type[kern_structs.KernelStruct]:
    """Create a custom numpy dtype."""

    class Config_Sum(  # ruff: ignore[invalid-class-name]
        kern_structs.KernelStruct,
        struct_name="Config_Sum",
        names=["alternate_sign", "multiply_first_term"],
        dtypes=[np.bool, np.int8],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    ):
        pass

    return Config_Sum


@pytest.fixture
def module(ks: kern_structs.KernelStruct) -> cp.RawModule:
    """Create a cp.RawModule corresponding to a specific KernelStruct."""
    return cp.RawModule(
        code=(
            r"""

{struct_definition_c}
""".format(struct_definition_c=ks.struct_definition_c)
            + r"""
#include <cuda/atomic>

template<typename T>
__global__ void sumsqs(Config_Sum config, T* vec, int len, T* output_num) {
    /* ASSUMPTION: this kernel is called with grid/block dims of
                   y and z lengths equal to 1. */
    __shared__ cuda::atomic<T, cuda::thread_scope_block> sum;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len) { return; }

    bool block_leader = (tid == 0);
    if (block_leader) {
        sum = 0;
    }
    __syncthreads();

    T myterm = vec[tid];
    myterm *= myterm;
    if (config.alternate_sign) {
        if (tid%2 == 1) {
            myterm *= -1;
        }
    }
    if (tid == 0) {
        myterm *= (T)config.multiply_first_term;
    }
    sum += myterm;

    if (block_leader) {
        atomicAdd(output_num, sum);
    }
}
"""
        ),
        options=("-std=c++17",),
        name_expressions=["sumsqs<float>", "sumsqs<double>"],
    )


@pytest.mark.dependency(name="ks_01")
def test_compile_module(module) -> None:  # ruff: ignore[missing-type-function-argument]
    """Test if the given module can be compiled."""
    module.compile()


@pytest.mark.dependency(name="ks_02", depends=["ks_01"])
def test_run_kernel_float_with_kernelstruct(ks, module) -> None:  # ruff: ignore[missing-type-function-argument]
    """Test a basic kernel using a custom structure."""
    kern_float = module.get_function("sumsqs<float>")
    # Create an 'array' which is just a bare value; not really an array
    alternate_sign: bool = True
    multiply_first_term: int = -1
    config = np.array((alternate_sign, multiply_first_term), dtype=ks.dtype)
    # Prepare args to pass to the kernel
    input_float_array = cp.array([1, 2, 3, 4, 5], dtype=cp.float32)
    input_len = np.int32(5)
    output_float = cp.array([0], dtype=cp.float32)
    # Call the kernel 3 times and check that the value is as expected
    for _ in range(3):
        kern_float(
            (1,),  # gridDims: (x,y,z)
            (input_len,),  # blockDims: (x,y,z)
            (config, input_float_array, input_len, output_float),  # input args
        )
    # Verify the expected output from the kernel
    # Expected value is 39 = 3*13 = 3*(-1 - 4 + 9 - 16 + 25)
    expected = cp.array([39.0], dtype=cp.float32)
    cp.testing.assert_array_almost_equal(output_float, expected)


@pytest.mark.dependency(name="ks_03", depends=["ks_01", "ks_02"])
def test_injected_kernel_with_template(ks: kern_structs.KernelStruct) -> None:
    """Test the header injection, and a template function as kernel."""
    # Resolve the paths to the source file and the header to output
    with (
        importlib.resources.path(test_cusrc, "cust_kernel.cu") as path_cu,
        importlib.resources.path(test_cusrc, "Config_Sum.h") as path_header,
    ):
        # Output the header file
        ks.save_c_struct_to_file(path_header, overwrite=True)
        injected_module = cp.RawModule(
            code=header_injection.inject_headers(path_cu),
            options=("-std=c++17",),
            name_expressions=["sumsqs<float>", "sumsqs<double>"],
        )
        injected_module.compile()
        test_run_kernel_float_with_kernelstruct(ks, injected_module)


@pytest.mark.dependency(name="ks_04", depends=["ks_03"])
def test_generate_cusrc_data_directory() -> None:
    """Test if an exception occurs when trying to generate the cusrc dir."""
    path_to_generated_cusrc = _generate_cusrc.path_to_generated_cusrc
    _generate_cusrc.delete_generated_sources()
    assert not path_to_generated_cusrc.is_dir()
    _generate_cusrc.main()
    assert path_to_generated_cusrc.is_dir()


@pytest.mark.dependency(
    name="ks_gen_sources", depends=["ks_04"], scope="session"
)
def test_a_simple_generated_kernel() -> None:
    """Test a kernel which is generated dynamically then loaded."""
    ks = kstructs.test_subdir.test_config.Test_Config_Sum
    injected_module: cp.RawModule = header_injection.load_injected_module(
        "./test_kernel.cu",
        name_expressions=["sumsqs<float>", "sumsqs<double>"],
    )
    test_run_kernel_float_with_kernelstruct(ks, injected_module)
