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

"""Classes that produce iterable objects that chunk numpy and cupy arrays, including memmapped arrays."""

import enum
import typing

import cupy as cp
import cupyx
import numpy as np

from .. import gpu_tools


class CuPyFilter:
    """A class to encapsulate chainable operations with signature cp.ndarray --> cp.ndarray. These functions may or may not modify the input array."""

    _input_layer_shape: tuple[int, ...]
    _output_layer_shape: tuple[int, ...]
    _func: typing.Callable[[cp.ndarray], cp.ndarray]
    _type: Type

    class Type(enum.Enum):
        """Type of filter. Either it filters individual layers, or whole chunks at a time."""

        SingleLayer = 1
        WholeChunk = 2

    @property
    def type(self) -> Type:
        """Return the type of this filter."""
        return self._type

    @property
    def input_layer_shape(self) -> tuple[int, ...]:
        """Return the input shape of the cp.ndarray."""
        return self._input_layer_shape

    @property
    def output_layer_shape(self) -> tuple[int, ...]:
        """Return the output shape of the cp.ndarray."""
        return self._output_layer_shape

    def __init__(
        self,
        *,
        input_layer_shape: tuple[int, ...],
        output_layer_shape: tuple[int, ...] | None = None,
        func: typing.Callable[[cp.ndarray], cp.ndarray],
    ) -> None:
        if output_layer_shape is None:
            # Attempt to infer the shape automatically
            output_layer_shape = func(
                cp.zeros((1, *input_layer_shape), dtype=cp.float32)
            ).shape[1:]
        self._input_layer_shape = input_layer_shape
        self._output_layer_shape = output_layer_shape
        self._func = func

    def __call__(self, cparr: cp.ndarray) -> tuple[cp.cuda.Event, cp.ndarray]:
        """Apply the filter to an input. Return (cudaEventUponCompletion, newArray)."""
        e = cp.cuda.Event(
            block=True, disable_timing=True, interprocess=False
        )  # block=True enables us to have the CPU wait on the event by calling e.synchronize(). It does not block otherwise.
        cp.cuda.get_current_stream().record(e)
        return e, self._func(cparr)


class ChunkedCuPy:
    """An iterable of cupy.ndarray chunks of equal size along axis 0."""

    _totlen: int
    _chunksize: int  # length of axis 0 of each chunk. Last chunk is expected to be possibly shorter.
    _filter_chain: list[
        CuPyFilter
    ]  # ordered list of filters to apply to the buffered data before outputting.

    _underlying_object: np.ndarray | cp.ndarray | np.memmap
    _underlying_object_len: int
    _start_index_within_object: int
    _current_index_within_object: int

    _buffer_stream: (
        cp.cuda.Stream
    )  # stream used to buffer data before it is yielded
    _buffer_pagelocked: np.ndarray  # a page-locked/pinned allocation of RAM in which to place data before it is copied to the buffer. This allows asynchronous copies from host to device.
    _buffer: cp.ndarray  # the array to hold the chunk input.
    _output_buffer: cp.ndarray  # the array to hold the next chunk output.
    _output: cp.ndarray  # the array to hold the current chunk output.

    _layer_streams: list[
        cp.cuda.Stream
    ]  # streams used to filter chunk layers concurrently

    _iteration_is_bootstrapped: bool

    _sl_len_previous: int

    @property
    def totlen(self) -> int | None:
        """Total length of underlying object, along axis 0."""
        return self._totlen

    @property
    def chunksize(self) -> int:
        """Maximum length of each chunk, along axis 0."""
        return self._chunksize

    def __init__(
        self,
        arr: np.ndarray | cp.ndarray | np.memmap | ChunkedCuPy,
        *,
        chunksize: int,
        filterchain: list[CuPyFilter] | None = None,
    ) -> None:
        if filterchain is None:
            filterchain = []
        self._chunksize = chunksize
        self._filter_chain = filterchain
        self._iteration_is_bootstrapped = False
        self._sl_len_previous = 0  # can be anything to start with
        # Class match pattern. See https://docs.python.org/3/reference/compound_stmts.html#class-patterns
        match arr:
            case np.ndarray() | cp.ndarray() | np.memmap():
                self._underlying_object = arr
                self._current_index_within_object = 0
            case ChunkedCuPy():
                self._underlying_object = arr._underlying_object  # noqa: SLF001
                self._current_index_within_object = (
                    arr._current_index_within_object
                )

        self._start_index_within_object = self._current_index_within_object
        self._underlying_object_len = self._underlying_object.shape[0]
        self._totlen = (
            self._underlying_object_len - self._start_index_within_object
        )
        # Create buffer and output chunks
        self._buffer = cp.empty(
            (self._chunksize, *self._underlying_object.shape[1:]),
            dtype=self._underlying_object.dtype,
        )
        output_layer_shape = self._underlying_object.shape[1:]
        if len(self._filter_chain) > 0:
            output_layer_shape = self._filter_chain[-1].output_layer_shape
        self._output_buffer = cp.empty(
            (self._chunksize, *output_layer_shape), dtype=self._buffer.dtype
        )
        self._output = cp.empty(
            (self._chunksize, *output_layer_shape), dtype=self._buffer.dtype
        )
        self._buffer_pagelocked = cupyx.empty_like_pinned(
            self._buffer
        )  # may not be used if underlying object is of type cp.ndarray
        self._buffer_stream = cp.cuda.Stream(non_blocking=True)
        # Create a separate CUDA stream for each layer. These will be used to concurrently filter the different layers of a chunk.
        self._layer_streams = [
            cp.cuda.Stream(non_blocking=True) for _ in range(self._chunksize)
        ]

        # First iteration does nothing. This is part of bootstrapping.
        next(self)

    def __iter__(self):  # noqa: ANN204, D105
        return self

    def __next__(self) -> cp.ndarray:  # noqa: C901
        """Get the next chunk."""
        # Swap the output buffer with the output.
        self._output, self._output_buffer = self._output_buffer, self._output
        # Wait for the buffer stream. This will wait for data to be finished being copied into what was previously the buffer
        self._buffer_stream.synchronize()

        if self._iteration_is_bootstrapped:
            # Increment current slice index
            self._current_index_within_object = min(
                self._current_index_within_object + self._chunksize,
                self._underlying_object_len,
            )
        if self._current_index_within_object == self._underlying_object_len:
            raise StopIteration()

        # Before returning self._output, launch the GPU tasks to fill the next buffer.

        if not self._iteration_is_bootstrapped:
            self._current_index_within_object -= self._chunksize

        # Define the slice indices
        next_index_within_object = min(
            self._current_index_within_object + self._chunksize,
            self._underlying_object_len,
        )
        sl = slice(
            next_index_within_object,
            min(
                next_index_within_object + self._chunksize,
                self._underlying_object_len,
            ),
        )
        sl_len: int = sl.stop - sl.start
        if (
            sl_len > 0
        ):  # Only launch the next GPU chunk computation if there is something to do
            sl_buffer = slice(0, sl_len)

            with self._buffer_stream as stream:
                # Fill self._buffer[sl_buffer] with the next chunk of data
                match self._underlying_object:  # Class match pattern
                    case np.ndarray() | np.memmap():
                        # Copy to page-locked RAM
                        self._buffer_pagelocked[sl_buffer] = (
                            self._underlying_object[sl]
                        )  # blocking, but should be relatively quick. If not, then could do this in another thread...
                        # Asynchronous htod copy
                        self._buffer.set(self._buffer_pagelocked)
                    case cp.ndarray():
                        # Asynchronous copy from one CuPy array to another
                        # Copy data. Not sure how to do this in a way that definitely is synchronouse with the stream
                        # self._buffer[sl_buffer].set(self._underlying_object[sl])
                        cp.multiply(
                            1,
                            self._underlying_object[sl],
                            out=self._buffer[sl_buffer],
                        )
                event_buffer_filled_prefilter = cp.cuda.Event(
                    block=False, disable_timing=True, interprocess=False
                )
                stream.record(event_buffer_filled_prefilter)

            # Apply the filters to the data buffer
            if len(self._filter_chain) > 0:
                # Commented out this working code which does one layer at a time in a different stream
                if False:
                    layers = list(
                        self._buffer
                    )  # creates a view to each layer of the buffer array along axis 0
                    for filt in self._filter_chain:
                        for index in range(sl_len):
                            with self._layer_streams[index] as stream:
                                e, layers[index] = filt(layers[index])
                                stream.wait_event(e)
                                self._buffer_stream.wait_event(e)
                    # Copy filtered output to contiguous output array
                    with self._buffer_stream:
                        for index in range(sl_len):
                            # Copy data. Not sure how to do this in a way that definitely is synchronous with the stream
                            # self._output_buffer[index].set(layers[index])
                            cp.multiply(
                                1,
                                layers[index],
                                out=self._output_buffer[index],
                            )

                buff = self._buffer[sl_buffer]
                with self._buffer_stream as stream:
                    for filt in self._filter_chain:
                        e, buff = filt(buff)
                        stream.wait_event(e)
                    # Copy data. Not sure how to do this in a way that definitely is synchronous with the stream
                    cp.multiply(1, buff, out=self._output_buffer[sl_buffer])
                    # self._output_buffer[sl_buffer] = buff
            else:
                # No filters. Just copy the data straight into the output array
                with self._buffer_stream:
                    # Copy data. Not sure how to do this in a way that definitely is synchronous with the stream
                    # self._output_buffer[sl_buffer].set(self._buffer[sl_buffer])
                    cp.multiply(
                        1,
                        self._buffer[sl_buffer],
                        out=self._output_buffer[sl_buffer],
                    )

        gpu_tools.clear_cache()
        if self._iteration_is_bootstrapped:
            # Return what was previously the buffer.
            out = self._output[: self._sl_len_previous]
            self._sl_len_previous = sl_len
            return out
        # If not bootstrapped, then enable the flag and skip. This first iteration will do nothing.
        self._iteration_is_bootstrapped = True
        self._sl_len_previous = sl_len
        return None


if __name__ == "__main__":
    import time

    import cupy as cp
    import cupyx
    import numpy as np

    import snookerr

    if False:
        b = cp.random.randn(5, 101, 101)

        def func(cparr: cp.ndarray) -> cp.ndarray:
            return cparr

        filtb = CuPyFilter(input_layer_shape=b.shape[1:], func=func)
        gb = snookerr.lib.iterables.chunkers.ChunkedCuPy(
            b, chunksize=2, filterchain=[filtb]
        )
        print(list(gb))

    # Do the test
    ncycles: int = 100
    chunksize: int = 1
    # filt = RampFilterer(proj_2d_shape=(3001, 4000), projs_per_batch=3, stream=cp.cuda.Stream(non_blocking=True))
    from snookerr.cb.ramp_filter import FBPFilterer

    cparr = cp.zeros((20, 4001, 4000), dtype=np.float32)
    prefunc = FBPFilterer(
        proj_2d_shape=cparr.shape[1:],
        projs_per_batch=chunksize,
        L=300,
        physpixwid=0.1,
    )._filter_bulk_axis1_concat

    def func(arr: cp.ndarray) -> cp.ndarray:
        # return prefunc(arr[None, ...])[0]
        return prefunc(arr)

    filt = CuPyFilter(input_layer_shape=cparr.shape[1:], func=func)
    # cparr = cp.random.randn(20, 3001, 4000).astype(cp.float32)
    cparr[:, cparr.shape[1] // 2, :] = 1
    t1 = time.time()
    for _ in range(ncycles):
        # x = filt._filter_bulk_axis1_concat(cparr)
        g = snookerr.lib.iterables.chunkers.ChunkedCuPy(
            cparr, chunksize=chunksize, filterchain=[filt]
        )
        x = list(g)
        cp.cuda.get_current_stream().synchronize()
        g._buffer_stream.synchronize()
    t2 = time.time()
    print(
        f"Milliseconds elapsed per cycle per projection: {1000 * (t2 - t1) / ncycles / cparr.shape[0]:.2f}"
    )
