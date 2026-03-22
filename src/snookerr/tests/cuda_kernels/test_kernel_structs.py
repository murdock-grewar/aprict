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

import cupy as cp
import numpy as np
import pytest

from ...cuda import kern_structs

# Compile the CUDA kernels


def _create_cuda_module_sum() -> tuple[
    kern_structs.KernelStruct, cp.RawModule
]:
    ks = kern_structs.KernelStruct(
        struct_name="Config_Sum",
        names=["alternate_sign", "multiply_first_term"],
        dtypes=[np.bool, np.int8],
    )
    cuda_module = cp.RawModule(
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

    T myterm = vec[tid];
    myterm *= myterm;
    bool block_leader = (tid == 0);
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
    return ks, cuda_module


@pytest.mark.dependency(name="ks_01")
def test_create_module() -> None:
    """Test if a basic kernel can be created."""
    _create_cuda_module_sum()


@pytest.mark.dependency(depends=["ks_01"])
def test_run_kernel_with_kernelstruct() -> None:
    """Test a basic kernel using a custom structure."""
    ks, module = _create_cuda_module_sum()
    kern_float = module.get_function("sumsqs<float>")
    # Create an 'array' which is just a bare value; not really an array
    alternate_sign: bool = True
    multiply_first_term: int = -1
    config = np.array((alternate_sign, multiply_first_term), dtype=ks.dtype)
    input_float_array = cp.array([1, 2, 3, 4, 5], dtype=cp.float32)
    input_len = np.int32(5)
    output_float = cp.array([0], dtype=cp.float32)
    kern_float(
        (1,),  # gridDims: (x,y,z)
        (input_len,),  # blockDims: (x,y,z)
        (config, input_float_array, input_len, output_float),  # input args
    )
    # Expected value is 13 = -1 - 4 + 9 - 16 + 25
    expected = cp.array([13.0], dtype=cp.float32)
    cp.testing.assert_array_almost_equal(output_float, expected)
